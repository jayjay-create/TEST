# ================================================================================
# 1. IMPORTS & DEPENDENCIES
# ================================================================================

import io
import os
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any, Iterable, Optional, List, Dict
from dataclasses import dataclass, field
import pandas as pd
import streamlit as st
from filelock import FileLock, Timeout
import tempfile
from dotenv import load_dotenv
import hashlib
from datetime import datetime, UTC
import re
import math
from collections import Counter, defaultdict
from functools import lru_cache
from difflib import get_close_matches
import logging
import requests
from urllib.parse import urlparse
import time
from enum import Enum
from sentence_transformers import CrossEncoder

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_parse import LlamaParse
from llama_index.core.schema import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────────
# BASIC TRACING: Performance-Monitoring für LLM-Calls
# ────────────────────────────────────────────────────────────────────────────────

def trace_llm_call(func):
    """Decorator für LLM-Call Tracing"""

    def wrapper(*args, **kwargs):
        start = time.time()
        func_name = func.__name__

        # Trace-Entry
        trace_id = hashlib.md5(f"{func_name}_{start}".encode()).hexdigest()[:8]

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start

            # Erfolgreicher Call
            _log_trace(trace_id, func_name, duration, success=True)

            return result

        except Exception as e:
            duration = time.time() - start
            _log_trace(trace_id, func_name, duration, success=False, error=str(e))
            raise

    return wrapper


def _log_trace(trace_id: str, func_name: str, duration: float, success: bool, error: str = None):
    """Speichert Trace in Session State"""
    if "llm_traces" not in st.session_state:
        st.session_state.llm_traces = []

    trace = {
        "id": trace_id,
        "function": func_name,
        "duration_ms": round(duration * 1000, 2),
        "success": success,
        "error": error,
        "timestamp": datetime.now(UTC).isoformat()
    }

    st.session_state.llm_traces.append(trace)

    # Nur letzte 100 Traces behalten
    if len(st.session_state.llm_traces) > 100:
        st.session_state.llm_traces = st.session_state.llm_traces[-100:]

    # Log für Entwickler
    status = "✓" if success else "✗"
    logger.info(f"{status} [{trace_id}] {func_name}: {duration * 1000:.0f}ms")

# ────────────────────────────────────────────────────────────────────────────────
# MODEL ROUTING: Kostengünstige Models für einfache Tasks
# ────────────────────────────────────────────────────────────────────────────────

class ModelTier(Enum):
    """Model-Auswahl nach Task-Komplexität"""
    CHEAP = "cheap"  # GPT-4o-mini: Intent, Summaries
    SMART = "smart"  # GPT-4o: Questions, Evidence Eval
    PREMIUM = "premium"  # o1: Complex Reasoning (falls nötig)

MODEL_CONFIG = {
    ModelTier.CHEAP: {
        "model": "gpt-4o-mini",
        "cost_per_1k_tokens": 0.00015,  # Input
        "max_tokens": 4096
    },
    ModelTier.SMART: {
        "model": "gpt-4o",
        "cost_per_1k_tokens": 0.0025,
        "max_tokens": 8192
    },
    ModelTier.PREMIUM: {
        "model": "o1-preview",
        "cost_per_1k_tokens": 0.015,
        "max_tokens": 32768
    }
}

# ════════════════════════════════════════════════════════════════════════════════
# TOOL-USE FRAMEWORK: Function Calling für autonome Entscheidungen
# ════════════════════════════════════════════════════════════════════════════════

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_evidence",
            "description": "Durchsucht hochgeladene Evidenz-Dokumente nach spezifischen Inhalten",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Was soll gesucht werden?"
                    }
                },
                "required": ["search_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_coverage",
            "description": "Berechnet die aktuelle Abdeckung der Obligations für diese Practice",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "skip_to_next",
            "description": "Überspringt die aktuelle Obligation und geht zur nächsten",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Warum wird übersprungen?"
                    }
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_evidence_upload",
            "description": "Fordert User auf, fehlende Dokumente hochzuladen",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Welche Dokumenttypen fehlen? (z.B. 'Projektplan', 'Review-Protokoll')"
                    }
                },
                "required": ["document_types"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "next_obligation",
            "description": "Markiert aktuelle Obligation als erfüllt (mit Coverage) und geht zur nächsten",
            "parameters": {
                "type": "object",
                "properties": {
                    "coverage_percent": {
                        "type": "number",
                        "description": "Abdeckungsgrad 0-100",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Begründung warum Obligation erfüllt ist"
                    }
                },
                "required": ["coverage_percent", "reasoning"]
            }
        }
    }
]


def execute_tool(tool_name: str, arguments: dict, context: dict) -> dict:
    """
    Führt ein Tool aus und gibt Ergebnis zurück.

    Args:
        tool_name: Name des Tools
        arguments: Tool-Parameter
        context: {"process_id": str, "practice_id": str, "orchestrator": AssessmentOrchestrator}

    Returns:
        {"success": bool, "result": Any, "action": str | None}
    """
    try:
        if tool_name == "query_evidence":
            query = arguments.get("search_query", "")
            idx = st.session_state.get("evidence_index")

            if not idx:
                return {
                    "success": False,
                    "result": "Keine Evidenz-Dokumente hochgeladen",
                    "action": None
                }

            try:
                snippets = _evidence_snippets_for(query, top_k=3)
                return {
                    "success": True,
                    "result": "\n".join(f"- {s[:150]}" for s in snippets) if snippets else "Keine relevanten Treffer",
                    "action": None
                }
            except Exception as e:
                return {"success": False, "result": f"Suche fehlgeschlagen: {e}", "action": None}

        elif tool_name == "calculate_coverage":
            proc = context.get("process_id", "")
            prac = context.get("practice_id", "")
            key = f"{proc}:{prac}"

            coverage = st.session_state.get("obl_coverage", {}).get(key, [])
            total = len(coverage)
            covered = sum(1 for item in coverage if item.get("status") == "covered")
            pct = (covered / total * 100) if total > 0 else 0

            return {
                "success": True,
                "result": f"{covered}/{total} Obligations abgedeckt ({pct:.0f}%)",
                "action": None
            }

        elif tool_name == "next_obligation":
            coverage = arguments.get("coverage_percent", 75.0)
            reasoning = arguments.get("reasoning", "User provided sufficient evidence")
            orchestrator = context.get("orchestrator")

            if orchestrator:
                orchestrator._mark_current_covered(coverage, reasoning)

            return {
                "success": True,
                "result": f"Obligation als abgedeckt markiert ({coverage}%)",
                "action": "advance_next"
            }

        elif tool_name == "skip_to_next":
            reason = arguments.get("reason", "Keine Antwort möglich")
            orchestrator = context.get("orchestrator")

            if orchestrator:
                orchestrator._mark_current_skipped()

                # Negative Example speichern
                if orchestrator.state.last_question_id:
                    _store_negative_example(
                        question=orchestrator.state.last_question_id,
                        context={
                            "current_practice": orchestrator.state.current_practice,
                            "current_obligation": orchestrator.state.current_obligation
                        },
                        reason="tool_skip"
                    )

            return {
                "success": True,
                "result": f"Übersprungen: {reason}",
                "action": "advance_next"
            }

        elif tool_name == "request_evidence_upload":
            docs = arguments.get("document_types", [])
            docs_text = ", ".join(docs) if docs else "relevante Dokumente"

            return {
                "success": True,
                "result": f"Bitte laden Sie folgende Dokumente hoch: {docs_text}",
                "action": "request_upload"
            }

        else:
            return {"success": False, "result": f"Unbekanntes Tool: {tool_name}", "action": None}

    except Exception as e:
        logger.error(f"Tool execution error: {tool_name} - {e}")
        return {"success": False, "result": str(e), "action": None}


@trace_llm_call
def call_llm_with_tools(messages: list[dict], context: dict) -> tuple[str, list, str | None]:
    """
    Ruft LLM mit Tool-Support auf.

    Returns:
        (response_text, executed_tools, action)
        action: None | "advance_next" | "request_upload"
    """
    from openai import OpenAI

    executed_tools = []
    action = None
    max_iterations = 2

    for iteration in range(max_iterations):
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=AVAILABLE_TOOLS,
                tool_choice="auto",
                temperature=0.3
            )

            message = response.choices[0].message

            # Keine Tool Calls → finale Antwort
            if not message.tool_calls:
                return message.content or "", executed_tools, action

            # Tool Calls verarbeiten
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in message.tool_calls
                ]
            })

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                logger.info(f"Executing tool: {func_name}")

                result = execute_tool(func_name, func_args, context)
                executed_tools.append({
                    "name": func_name,
                    "arguments": func_args,
                    "result": result
                })

                # Tool-Result zurück an LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result.get("result", ""))
                })

                # Action speichern
                if result.get("action"):
                    action = result["action"]

        except Exception as e:
            logger.error(f"Tool-aware LLM failed: {e}")
            return "Entschuldigung, technischer Fehler.", executed_tools, action

    # Finale Antwort nach Tool-Execution
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3
        )
        return final_response.choices[0].message.content or "", executed_tools, action
    except:
        return "Tool-Ausführung abgeschlossen.", executed_tools, action

# ────────────────────────────────────────────────────────────────────────────────
# RESPONSE CACHE: Kostenersparnis bei wiederholten Fragen
# ────────────────────────────────────────────────────────────────────────────────

def _get_or_generate_response(prompt: str, tier: ModelTier, temperature: float) -> str:
    """Wrapper mit Cache-Layer"""

    # Variablen VOR dem if-Block initialisieren
    cache_key = None
    prompt_hash = None

    # Nur cachen bei deterministischen Calls (temperature < 0.3)
    if temperature < 0.3:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        cache_key = f"llm_cache_{prompt_hash}_{tier.value}_{temperature}"

        if cache_key in st.session_state:
            logger.info(f"Cache HIT for {prompt_hash}")
            return st.session_state[cache_key]

    # Cache Miss: Echter LLM Call
    resp = _chat_with_model_tier(
        [{"role": "user", "content": prompt}],
        tier=tier,
        temperature=temperature
    )
    result = _resp_text(resp).strip()

    # Speichern für nächstes Mal (nur wenn cache_key definiert wurde)
    if cache_key:
        st.session_state[cache_key] = result
        logger.info(f"Cached response for {prompt_hash}")

    return result


@trace_llm_call
def _chat_with_model_tier(
        messages: list[dict],
        tier: ModelTier = ModelTier.SMART,
        temperature: float = 0.3,
        max_retries: int = 2
) -> Any:
    """
    Optimierter LLM-Call mit:
    - Model Tiering (cost optimization)
    - Retry mit exponential backoff
    - Fallback zu günstigerem Model bei Fehler
    """
    config = MODEL_CONFIG[tier]
    model_name = config["model"]

    for attempt in range(max_retries):
        try:
            llm = OpenAI(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=temperature,
                max_tokens=config["max_tokens"]
            )

            # ChatMessages konvertieren
            role_map = {
                "system": MessageRole.SYSTEM,
                "user": MessageRole.USER,
                "assistant": MessageRole.ASSISTANT,
            }

            conv = []
            for m in messages:
                if isinstance(m, dict):
                    role = role_map.get(m.get("role", "user"), MessageRole.USER)
                    text = str(m.get("content", ""))
                    conv.append(ChatMessage(role=role, blocks=[TextBlock(text=text)]))
                else:
                    conv.append(m)

            return llm.chat(conv)

        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                # Exponential backoff
                time.sleep(2 ** attempt)
            else:
                # Letzter Versuch: Fallback zu CHEAP model
                if tier != ModelTier.CHEAP:
                    logger.info(f"Falling back to CHEAP model")
                    config = MODEL_CONFIG[ModelTier.CHEAP]
                    model_name = config["model"]
                else:
                    raise

    raise RuntimeError("All retry attempts failed")

# ================================================================================
# 2. GLOBAL CONFIGURATION & CONSTANTS
# ================================================================================

# Whitelist echter Output-Work-Product IDs - wird beim Laden der Dokumente dynamisch gefüllt.
OWP_IDS: set[str] = set()

# Konfiguration für den Listenmodus:
# True = Wenn für eine ID keine direkten Kinder des gefragten Typs gefunden werden, wird eine "Keine Treffer"-Nachricht angezeigt.
# False = Wenn keine direkten Kinder gefunden werden, wird in der Prozessfamilie nach passenden Kindern gesucht (z.B. bei anderen Base Practices desselben Prozesses).
STRICT_LIST_MODE: bool = True

# Name der Ordner pro SPICE Version, in dem der LlamaIndex-Vektorindex persistent gespeichert wird - beschleunigt den Start der Anwendung.
PERSIST_DIR_31 = "storage_llamaindex_v31"
PERSIST_DIR_40 = "storage_llamaindex_v40"

# Ingest-Profile + Sentinel-Dateien
APP_INDEX_VERSION = 1  # bei Änderungen an der Buildlogik hochzählen
INPROGRESS = "INDEX_IN_PROGRESS.tmp"
COMPLETE = "INDEX_COMPLETE.ok"

# Optionale "aktive" Workspace-Container (werden später je Query gesetzt)
INDEX_V31 = INDEX_V40 = None
DOCS_V31: list[Document] = []
DOCS_V40: list[Document] = []
ID_MAP_V31: dict[str, list[int]] = {}
ID_MAP_V40: dict[str, list[int]] = {}
PARENT_MAP_V31: dict[str, list[int]] = {}
PARENT_MAP_V40: dict[str, list[int]] = {}
DISPLAY_ID_INDEX_V31: dict[str, int] = {}
DISPLAY_ID_INDEX_V40: dict[str, int] = {}


def _collect_practice_and_children_indices(practice_id: str,
                                           id_map: dict[str, list[int]],
                                           parent_map: dict[str, list[int]],
                                           exclude_outcomes: bool = True) -> list[int]:
    """
    Liefert die Dokument-Indices der exakt benannten Practice (z. B. 'SUP.1.BP1')
    sowie ihrer direkten Children (Rules, Recommendations, Output Work Products).

    Args:
        exclude_outcomes: Wenn True, werden Outcomes ausgeschlossen (Default)
    """
    if not practice_id:
        return []
    pid_u = practice_id.upper().strip()
    base = id_map.get(pid_u, []) + id_map.get(practice_id, [])
    if not base:
        return []

    kids = parent_map.get(pid_u, []) + parent_map.get(practice_id, [])

    # Outcomes filtern, da sehr ähnlich zu practices
    if exclude_outcomes:
        # Globale DOCS-Referenz nutzen
        filtered_kids = []
        for idx in kids:
            doc_type = DOCS[idx].metadata.get("type", "")
            if doc_type != "aspice_outcome":
                filtered_kids.append(idx)
        kids = filtered_kids

    seen, ordered = set(), []
    for i in base + kids:
        if i not in seen:
            ordered.append(i)
            seen.add(i)
    return ordered


# Base directory detection
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Schwellenwerte für die Heuristik zur Erkennung von Kopf-/Fußzeilen.
FOOTER_MIN_REPEATS = 3  # Eine Zeile muss mindestens so oft vorkommen, um als wiederkehrend zu gelten.
FOOTER_MAX_LINE_LEN = 120  # Längere Zeilen werden ignoriert, da sie unwahrscheinlich Kopf-/Fußzeilen sind.
HEAD_TAIL_WINDOW = 6  # wie viele Zeilen oben/unten je Chunk als Header/Footer-Kandidaten
FUZZY_SIM = 0.88  # Ähnlichkeitsschwelle für "gleich genug"
MIN_DOC_COVERAGE = 0.5  # Kandidat muss in >=50% der Chunks der Datei auftauchen


# ────────────────────────────────────────────────────────────────────────────────
# NEGATIVE EXAMPLES: Learning from ineffective questions
# ────────────────────────────────────────────────────────────────────────────────

def _store_negative_example(question: str, context: dict, reason: str = "skipped"):
    """Speichert ineffektive Fragen für Few-Shot Learning"""
    if "negative_examples" not in st.session_state:
        st.session_state.negative_examples = []

    example = {
        "question": question,
        "obligation_title": context.get("current_obligation", {}).get("title", ""),
        "practice": context.get("current_practice", ""),
        "reason": reason,  # "skipped", "confused", "unable"
        "timestamp": datetime.now(UTC).isoformat()
    }

    st.session_state.negative_examples.append(example)

    # Nur letzte 50 behalten
    if len(st.session_state.negative_examples) > 50:
        st.session_state.negative_examples = st.session_state.negative_examples[-50:]

    logger.info(f"Negative example stored: {question[:50]}... (reason: {reason})")


def _get_negative_examples_prompt(context: dict) -> str:
    """Erstellt Prompt mit abstrahierten negativen Beispielen"""
    if "negative_examples" not in st.session_state:
        return ""

    practice = context.get("current_practice", "")
    obl_title = context.get("current_obligation", {}).get("title", "")

    relevant = [
        ex for ex in st.session_state.negative_examples[-20:]
        if ex["practice"] == practice or ex["obligation_title"] == obl_title
    ]

    if not relevant:
        return ""

    # Heuristische Prüfung: Enthält die Frage spezifische Entities?
    import re

    def _specificity_score(q: str) -> int:
        """Zählt spezifische Details (je höher, desto spezifischer)"""
        score = 0

        # URLs (eindeutig zu spezifisch)
        score += len(re.findall(r'https?://\S+', q)) * 3

        # Dateiendungen (wahrscheinlich Dateinamen)
        score += len(re.findall(r'\.\w{2,4}\b', q))

        # Großgeschriebene Wörter mitten im Satz (Eigennamen/Produkte)
        score += len(re.findall(r'\s([A-Z][a-z]+[A-Z]\w+)', q))

        # Nummern mit Kontext (Kapitel 3.2, Seite 15)
        score += len(re.findall(r'\b(Kapitel|Abschnitt|Seite|Chapter|Section)\s+\d+', q, re.I))

        # Pfade (C:/... oder /home/...)
        score += len(re.findall(r'[A-Z]:/|/\w+/', q))

        return score

    # Zähle hochspezifische Fragen
    too_specific = [ex for ex in relevant if _specificity_score(ex['question']) >= 2]

    # Nur Guidance wenn Pattern erkennbar (mindestens 2 Beispiele)
    if len(too_specific) >= 2:
        return """
**Vermeide:**
- Konkrete URLs/Pfade/Dateinamen in der Frage
- Spezifische Produktnamen oder Eigennamen
- Detaillierte Referenzen (Kapitel X.Y, Seite Z)

**Frage stattdessen allgemein:**
- "Welches Dokument beschreibt X?" statt "In welchem Kapitel von Dokument.pdf steht X?"
- "Wo ist Y dokumentiert?" statt "Ist Y in Tool Z erfasst?"
"""

    return ""

# ────────────────────────────────────────────────────────────────────────────────
# RE-RANKING: Cross-Encoder für Precision nach Embedding-Retrieval
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_cross_encoder():
    """Lädt Cross-Encoder einmalig (gecacht)"""
    try:
        return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    except Exception as e:
        logger.warning(f"Cross-Encoder konnte nicht geladen werden: {e}")
        return None

CROSS_ENCODER = _load_cross_encoder()


def _rerank_nodes(query: str, nodes: list, top_k: int = 10) -> list:
    """
    Re-Rankt Retrieval-Ergebnisse mit Cross-Encoder für höhere Precision.

    Args:
        query: User-Query
        nodes: Liste von LlamaIndex Nodes (aus retriever.retrieve)
        top_k: Anzahl der finalen Ergebnisse

    Returns:
        Re-ranked Nodes (beste zuerst)
    """
    if not CROSS_ENCODER or not nodes:
        return nodes[:top_k]

    try:
        # Paare (query, document_text) für den Cross-Encoder
        pairs = []
        for n in nodes:
            node = getattr(n, 'node', n)
            text = node.get_content() if hasattr(node, 'get_content') else getattr(node, 'text', '')
            pairs.append([query, text[:512]])  # Cross-Encoder hat meist 512 Token Limit

        # Scores berechnen (höher = relevanter)
        scores = CROSS_ENCODER.predict(pairs)

        # Nach Score sortieren
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        return [nodes[i] for i in ranked_indices[:top_k]]

    except Exception as e:
        logger.warning(f"Re-Ranking failed: {e}")
        return nodes[:top_k]

# ================================================================================
# 3. REGEX PATTERNS (alphabetisch & nach Verwendung gruppiert)
# ================================================================================

# -----------------------------------------------------------------------------
# ID-PATTERNS: Erkennung verschiedener SPICE/KGAS-Identifikatoren
# -----------------------------------------------------------------------------
ID_PATTERNS = {
    "proc": re.compile(r"\b[A-Z]{3}\.\d{1,2}\b", re.I),
    "bp": re.compile(r"\b[A-Z]{3}\.\d{1,2}\.BP\d{1,2}\b", re.I),
    "gp": re.compile(r"\bGP\s*\d\.\d\.\d\b", re.I),
    "rule_global": re.compile(r"\b(?:[A-Z]{3}|[A-Z]{2}[A-Z0-9])\.RL\.\d{1,2}\b", re.I),
    "rule_proc": re.compile(r"\b[A-Z]{3}\.\d{1,2}\.RL\.\d{1,2}\b", re.I),
    "rec_global": re.compile(r"\b(?:[A-Z]{3}|[A-Z]{2}[A-Z0-9])\.RC\.\d{1,2}\b", re.I),
    "rec_proc": re.compile(r"\b[A-Z]{3}\.\d{1,2}\.RC\.\d{1,2}\b", re.I),
    "owp": re.compile(r"\b\d{2}[-–—−]\d{2}\b"),
    "kgas": re.compile(r"\bKGAS_\d{1,4}\b", re.I),
}

# Vollständige ID-Patterns (für query_documents_only)
RX_PROC = ID_PATTERNS["proc"]
RX_BP = ID_PATTERNS["bp"]
RX_GP = ID_PATTERNS["gp"]
RX_RL_G = ID_PATTERNS["rule_global"]
RX_RL_P = ID_PATTERNS["rule_proc"]
RX_RC_G = ID_PATTERNS["rec_global"]
RX_RC_P = ID_PATTERNS["rec_proc"]
RX_KGAS = ID_PATTERNS["kgas"]
RX_OWPID = ID_PATTERNS["owp"]

# Validierungs-Patterns (Full Match)
BP_FULL = re.compile(r"^[a-z]{3}\.\d{1,2}\.bp\d{1,2}$", re.I)
RL_PROC_FULL = re.compile(r"^[a-z]{3}\.\d{1,2}\.rl\.\d{1,2}$", re.I)
RC_PROC_FULL = re.compile(r"^[a-z]{3}\.\d{1,2}\.rc\.\d{1,2}$", re.I)
RX_OWPID_STRICT = re.compile(r"^\d{2}-\d{2}$")

# Parent/Prozess-Patterns
_PROC_ONLY_RX = re.compile(r'^[a-z]{3}\.\d{1,2}$', re.I)
PROC_ID_HEAD_RX = re.compile(r"^([A-Z]{3}\.\d{1,2})", re.I)
_PROC_ID = re.compile(r"\b[A-Z]{3}\.\d{1,2}\b", re.I)

# Beliebige ID (für Multi-Pattern-Suche)
_ID_ANY = re.compile(
    r"([A-Z]{3}\.\d{1,2}(?:\.BP\d{1,2})?"
    r"|GP\s*\d\.\d\.\d"
    r"|[A-Z]{2}[A-Z0-9]\.R[LC]\.\d{1,2}"
    r"|[A-Z]{3}\.\d{1,2}\.R[LC]\.\d{1,2}"
    r"|KGAS_\d{1,4}"
    r"|\b\d{2}[-—–−]\d{2}\b)", re.I)

# -----------------------------------------------------------------------------
# VERSION & LEVEL: SPICE-Versions- und Level-Erkennung
# -----------------------------------------------------------------------------
VERSION_PATTERNS = {
    "4.0": re.compile(r"""
        \b(4(?:\.0)?|40|v\.?\s*4(?:\.0)?|v\.?\s*40|version\s*4(?:\.0)?|version\s*40|
        (?:a(?:utomotive)?[\s-]*)?(?:spice|spcie|spuce)\s*4(?:\.0)?|
        (?:a(?:utomotive)?[\s-]*)?spice\s*40|pam\s*4(?:\.0)?|pam\s*40)\b
    """, re.I | re.X),
    "3.1": re.compile(r"""
        \b(3\.1|31|v\.?\s*3(?:\.1)?|v\.?\s*31|version\s*3(?:\.1)?|version\s*31|
        (?:a(?:utomotive)?[\s-]*)?(?:spice|spcie|spuce)\s*3(?:\.1)?|
        (?:a(?:utomotive)?[\s-]*)?spice\s*31|pam\s*3(?:\.1)?|pam\s*31)\b
    """, re.I | re.X),
}

GENERIC_SPICE_ONLY = re.compile(r"\b((a(?:utomotive)?[\s-]*)?spice|pam)\b", re.I | re.X)
_CL_DIRECT_RX = re.compile(r"\bCL\s*([12])\b", re.I)
_LEVEL_RX = re.compile(r"\b(?:(?:capability\s*)?level|lvl|stufe|f(?:ä|ae)higkeitsstufe)\s*([12])\b", re.I | re.X)

# -----------------------------------------------------------------------------
# INTENT DETECTION: Erkennung von User-Absichten
# -----------------------------------------------------------------------------
# Artefakt-Typen
_RX_RULES = re.compile(r"\b(rule|rules|regel|regeln)\b", re.I)
_RX_RECS = re.compile(r"\b(recommendation|recommendations|empfehlung|empfehlungen)\b", re.I)
_RX_OUT = re.compile(r"\b(outcome|outcomes|ergebnis|ergebnisse)\b", re.I)
_RX_OWP = re.compile(r"\b(output\s*work\s*product(s)?|work\s*product(s)?|arbeitsprodukt(e)?|output\s*information\s*item(s)?|outputs?)\b", re.I)
_RX_BPWRD = re.compile(r"\b(base\s*practices?|basispraktik(?:e|en)?|basispraktiken|bps?)\b", re.I)
_RX_GPWRD = re.compile(r"\b(generic\s*practices?|generische\s*praktik(?:e|en)?|generische\s*praktiken|gps?)\b", re.I)

# Erklärung vs. Liste
RX_EXPLAIN = re.compile(r"""
    \bwie\b(?!\s+(?:viel|viele|hoch|lang|gro(?:ß|ss)))|
    \b(?:wieso|warum|weshalb|wozu|wodurch)\b|
    \b(?:erkl(?:ä|ae)r(?:e|en|t|ung)?|erl(?:ä|ae)uter(?:e|en|ung))\b|
    \bbegr(?:ü|ue)nd(?:e|en|ung)\b|\b(beschreib(?:e|en|ung))\b|
    \b(?:inwiefern|zusammenhang)\b|\b(?:worum\s+geht'?s|worum\s+geht\s+es)\b|
    \b(?:was\s+bedeutet|definition)\b|
    \bwhy\b|\bhow\b(?!\s+(?:many|much|long|high|big))|
    \b(?:explain|explanation|describe|description|reason|reasons|purpose|rationale|motivation)\b
""", re.I | re.VERBOSE)

# Vergleich
RX_DIFF = re.compile(r"""
    \b(vergleiche|vergleichen|vergleich(?!sweise|bar\w*)|im\s+vergleich|(?:zum|zur)\s+vergleich|
    vergleich\s+(?:von|zwischen)|gegen(?:ü|ueber)stell\w*|gegen(?:ü|ueber)stellung|abgrenzung|
    unterschiede?(?=.*\b(?:zwischen|von|vs\.?|versus|im\s+vergleich|zu|und|&|/)\b)|
    gemeinsamkeit(?:en)?(?=.*\b(?:zwischen|von|und|&|/)\b)|
    compare|comparison|contrast(?:s|ed|ing)?|differences?|diff|
    differ(?:s|ing|ence|ent)?(?=.*\b(?:between|from)\b)|vs\.?|versus)\b
""", re.I | re.VERBOSE)

# Bewertung/Ranking
RANKING_EVAL_RX = re.compile(r"""
    \b(bewert\w+|Bewertung|einsch(ä|ae)tz\w+|Einsch(ä|ae)tzung|absch(ä|ae)tz\w+|
    Absch(ä|ae)tzung|priorisier\w+|Priorisierung|begutacht\w+|Begutachtung|
    beurteil\w+|Beurteilung|evaluier\w+|Evaluierung|assessier\w+|sch(ä|ae)tz\w+|
    wichtigst\w+|begr(ü|ue)nd\w+|Begr(ü|ue)ndung|ranking|rank|top\s*\d+|score|
    gewicht\w+|evaluate|evaluation|assess|assessment|prioritize|prioritization|
    judge|judgement|estimate|estimation|most\s+important)\b
""", re.I | re.VERBOSE)

TOPK_RX = re.compile(r"\b(?:top|Top)\s*(\d+)|\b(\d+)\s*(wichtigsten|most important)\b", re.I)

# Output-Format
RX_CONCISE = re.compile(r"""
    \b(zusammenfassung|zusammenfassen|kurzfassung|kurzform|kurz\s*gefasst|kurzgefasst|
    kurz(?:e|er|es|en)?|k(ü|ue)rz\w*|knapp\w*|pr(ä|ae)gnant\w*|b(ü|ue)ndig|
    kompakt\w*|griffig\w*|komprimier\w*|stichpunkte?|tl;?dr|tldr|brief|briefly|
    short(?:er)?|concise|succinct|compact|summar(?:y|ize|ise)|abstract|overview|
    in\s+short|in\s+a\s+nutshell|bullet\s*points?|bulleted|condens(?:e|ed))\b
""", re.I | re.VERBOSE)

# Sprache
_LANG_DE_RX = re.compile(r"[äöüÄÖÜß]|\b(der|die|das|und|nicht|mit|ohne|bitte|welche|warum|wieso|wie|übersicht|stichpunkte?)\b", re.I)
_LANG_EN_RX = re.compile(r"\b(the|and|or|not|with|without|please|which|why|how|overview|bullet\s*points?)\b", re.I)

# -----------------------------------------------------------------------------
# FUNCTION-SPECIFIC: Spezielle Patterns für einzelne Funktionen
# -----------------------------------------------------------------------------
_SENT_RX = re.compile(r"(?<=[.!?])\s+|\n+")  # Satztrennung
_RX_GROUP = re.compile(r"""  # Cross-ID-Alignment
    ^(?:(?P<proc>[A-Z]{3}\.\d+)(?:\.BP\d+)?(?:\s*\[OUTCOME\s*\d+])?|
    (?P<gp>GP\s*\d+\.\d+(?:\.\d+)?))
""", re.X)


@dataclass
class PracticeEvidence:
    """Sammelt alle Evidenz während des Assessments"""
    practice_id: str
    user_statements: List[str] = field(default_factory=list)
    uploaded_docs: List[Dict] = field(default_factory=list)  # {filename, snippets, metadata}
    url_contents: List[Dict] = field(default_factory=list)  # {url, content}
    timestamps: List[str] = field(default_factory=list)

    def get_all_evidence_text(self) -> str:
        """Konsolidierter Evidenz-Text für LLM"""
        parts = []

        # User-Statements
        if self.user_statements:
            parts.append("**User-Antworten:**\n" + "\n\n".join(f"- {s}" for s in self.user_statements))

        # Dokumente
        if self.uploaded_docs:
            parts.append("**Hochgeladene Dokumente:**")
            for doc in self.uploaded_docs:
                parts.append(f"- {doc.get('filename', 'Dokument')}")
                for snip in doc.get('snippets', [])[:3]:
                    parts.append(f"  → {snip[:200]}")

        # URLs
        if self.url_contents:
            parts.append("**Verlinkte Inhalte:**")
            for url_data in self.url_contents:
                parts.append(f"- {url_data.get('url', 'Link')}")
                content = url_data.get('content', '')[:300]
                parts.append(f"  → {content}")

        return "\n\n".join(parts) if parts else "(keine Evidenz gesammelt)"


@dataclass
class Weakness:
    """Repräsentiert eine identifizierte Schwäche"""
    aspect: str  # Was fehlt/ist unzureichend?
    evidence_gap: str  # Welche Evidenz fehlt konkret?
    process_risk: str  # Welches Risiko entsteht?
    impact: str  # "purpose"|"product_quality"
    severity: float  # 0.0-1.0
    rule_reference: Optional[str] = None  # Optionale Rule-ID


@dataclass
class RatingRule:
    """Repräsentiert eine geparste Rule/Recommendation"""
    rule_id: str
    text: str
    type: str  # "MUST"|"SHALL"|"SHOULD"
    action: str  # "ceiling"|"downrate"|"no_downrate"
    threshold: Optional[str] = None  # "N"|"P"|"L" für Ceilings
    conditions: List[str] = field(default_factory=list)  # Bedingungen aus dem Text
    weight: float = 1.0  # Enforcement-Stärke

# ================================================================================
# 4. MAGIC NUMBERS & THRESHOLDS (benannte Konstanten)
# ================================================================================

# === Text-Verarbeitung & Anzeige ===
MAX_SNIPPET_LENGTH = 300
SNIPPET_TRUNCATE_AT = 297  # MAX_SNIPPET_LENGTH - 3 für "…"
MAX_CANDIDATE_ITEMS = 15
MAX_CANDIDATE_TEXT_LENGTH = 300

# === Retrieval & Suche ===
DEFAULT_SIMILARITY_TOP_K = 50
UNION_FUSION_K_EACH = 8
UNION_FUSION_K_FUSE = 10
MAX_CONTEXT_DOCS = 200
SUPPLEMENTARY_DOCS_COUNT = 5

# === Assessor/Ranking ===
DEFAULT_TOP_K = 5
MAX_TOP_K = 10
MIN_TOP_K = 1
DEFAULT_PER_PROCESS_CAP = 6
ASSESSOR_TOTAL_CAP_MIN = 24
ASSESSOR_TOTAL_CAP_MAX = 120

# === Chat History ===
MAX_HISTORY_MESSAGES = 10

# === Fuzzy Matching ===
FUZZY_ID_MATCH_CUTOFF = 0.85
SENTENCE_SIMILARITY_LOW = 0.45   # Komplett neu
SENTENCE_SIMILARITY_MID = 0.92   # Leicht geändert

# === Cross-ID Alignment ===
JACCARD_PAIR_THRESHOLD = 0.30
RELATIVE_SECOND_THRESHOLD = 0.90
MIN_ATOM_WORDS = 5
PHRASE_LENGTHS = (8, 10, 12)
MAX_COMPARISONS_LIMIT = 10000

# === RRF Fusion ===
RRF_K_CONSTANT = 60

# === Delta Detection ===
MAX_DELTA_ITEMS = 4
MAX_DELTA_CHANGED = 6
MAX_SEMANTIC_HIGHLIGHTS = 6

# ================== SOTA: Obligations & Questions (NEW) ==================

NUM_QUESTION_CANDIDATES = int(st.session_state.get("num_q_candidates", 3))  # 3 = guter Sweet-Spot

OBLIGATIONS_PROMPT = """
Extrahiere aus dem folgenden Kontext atomare, prüfbare Pflichten (JSON-Array).

**Ziel: 4-7 Obligations pro Base Practice**

**Quellen:**
1. Base Practice Text + NOTEs
2. Rules/Recommendations (nach Priorität)

**Priorisierung (extrahiere zuerst hohe, dann niedrige Prios bis 4-6 erreicht):**

**Prio 1 (höchste Kritikalität - immer extrahieren):**
- Base Practice Kernaktivitäten
- Rules: "If X missing/not considered → must/shall not be rated F"
- Rules: "If X missing/not considered → must/shall not be rated higher than N"
- Rules: "If X missing/not considered → must/shall not be rated higher than P"

**Prio 2 (hohe Kritikalität):**
- Rules: "If X missing/not considered → shall be downrated"
- Rules: "If X missing/not considered → must/shall not be rated higher than L"

**Prio 3 (mittlere Kritikalität):**
- Recommendations: "If X missing/not considered → should not be rated F"
- Recommendations: "If X missing/not considered → should not be rated higher than N/P"
- Recommendations: "If X missing/not considered → should be downrated"

**Prio 4 (niedrige Kritikalität):**
- Recommendations: "If X missing/not considered → should not be rated higher than L"
- Substantielle NOTEs

**KEINE Obligation extrahieren bei:**
- "must/shall not be downrated" (toleriert Fehlen)
- "should not be downrated" (toleriert Fehlen)
- "must/shall/should not be used to downrate" (explizit kein Downrate-Grund)
- Konsistenz-Rules: "If indicator A downrated, indicator B must/shall/should..." (keine neue Anforderung)

**Wichtig:** Extrahiere X aus der Bedingung "If X missing/not considered/not part of" als Obligation.

**Format-Anforderungen:**
- Title: Kurz (max 10 Wörter), beginnt mit Verb (Define, Ensure, Include, Specify, etc.)
- maps_to: Referenziere BP-ID und Rule/Recommendation-ID wenn Obligation daraus abgeleitet wurde
- key_phrases: Schlüsselbegriffe aus der Quelle
- expected_evidence: Konkrete Dokumente/Artefakte die diese Obligation belegen
- source_spans: Wörtliches Zitat aus Quelle

Schema (NUR JSON zurückgeben, KEINE Code-Blocks, keine Erklärungen):
[
  {{
    "id": "short-slug",
    "title": "Verb + kurze Beschreibung",
    "maps_to": ["BP-ID", "Rule-ID falls relevant"],
    "key_phrases": ["Begriff1", "Begriff2"],
    "expected_evidence": ["Dokument-Typ1", "Dokument-Typ2"],
    "source_spans": ["Wörtliches Zitat aus Quelle"]
  }},
  ...
]

**Base Practice + NOTEs:**
{rag}

**Rules/Recommendations (wichtig für Obligation-Ableitung):**
{rules}
"""

QUESTION_CANDIDATE_PROMPT = """
Erzeuge GENAU EINE kurze, präzise Frage (ein Satz) auf Basis der Pflicht unten.
Antworte NUR als JSON:
{{
  "question": "<natürliche, kontextspezifische Frage ohne IDs>",
  "asks_evidence": ["<konkrete erwartete Nachweise>"],
  "maps_to": {maps_to_json}
}}

Pflicht:
- Titel: {title}
- Schlüsselbegriffe: {key_phrases}
- Erwartete Nachweise: {expected_evidence}
- Quelle(n): {spans}

**Verbote:**
- KEINE IDs/Versionen/BP/GP im Fragetext
- Den Titel NICHT wörtlich wiederholen
- Formuliere die Frage natürlich und spezifisch für den Kontext
""".strip()

RATING_RULES_JSON_PROMPT = """
Fasse die folgenden verbindlichen Regeln zu einer strukturierten JSON-Zusammenfassung zusammen.
Gib NUR JSON zurück, ohne Kommentar.

Schema:
{{
  "caps": [
    {{
      "reason": "<kurz>",
      "strength": "MUST|SHALL|SHOULD",
      "if_missing_obligations": ["<id|title-fragment>", "..."]?,
      "if_weakness_keywords": ["<keyword>", "..."]?,
      "max_band": "N|P|L|F"
    }}
  ],
  "mandatory_downrates": [
    {{
      "reason": "<kurz>",
      "strength": "MUST|SHALL|SHOULD",
      "if_missing_obligations": ["<id|title-fragment>", "..."]?,
      "if_weakness_keywords": ["<keyword>", "..."]?,
      "min_band": "N|P|L|F",
      "percent_penalty": 20
    }}
  ],
  "no_downrate": [
    {{
      "reason": "<kurz>",
      "strength": "MUST|SHALL|SHOULD",
      "if_weakness_keywords": ["<keyword>", "..."]?
    }}
  ]
}}

Regeln (Rohtext):
{{rules_text}}
""".strip()


# --- Rule Strength -> Gewicht (für Coverage der Pflichten)
RULE_STRENGTH_WEIGHT = {
    ("CEIL",   "MUST"): 2.0, ("CEIL",   "SHALL"): 2.0,
    ("DOWN",   "MUST"): 1.7, ("DOWN",   "SHALL"): 1.7,
    ("CEIL", "SHOULD"): 1.5,
    ("DOWN", "SHOULD"): 1.25,
}
DEFAULT_OBL_WEIGHT = 1.0

def _norm_strength(s: str) -> str:
    s = (s or "").strip().upper()
    if "MUST" in s:  return "MUST"
    if "SHALL" in s: return "SHALL"
    if "SHOULD" in s:return "SHOULD"
    return ""  # unbekannt/leer


def _obl_matches(obl: dict, needle: str) -> bool:
    """Matcht Referenzen aus Regeln gegen Pflicht-ID/Titel."""
    n = (needle or "").strip().lower()
    if not n: return False
    oid = (obl.get("id") or "").strip().lower()
    tit = (obl.get("title") or "").strip().lower()
    return (n == oid) or (n in tit)


def _obligation_weight(obl: dict, rules: dict) -> float:
    """Ermittelt wᵢ für eine Pflicht aus Caps/Downrates (shall/must/should). Priorität: Ceil > Downrate."""
    w = DEFAULT_OBL_WEIGHT
    # 1) Ceilings prüfen
    for cap in (rules.get("caps") or []):
        needles = (cap.get("if_missing_obligations") or [])
        if any(_obl_matches(obl, n) for n in needles):
            strength = _norm_strength(cap.get("strength"))
            w = max(w, RULE_STRENGTH_WEIGHT.get(("CEIL", strength), DEFAULT_OBL_WEIGHT))
    # 2) Downrates prüfen
    for dr in (rules.get("mandatory_downrates") or []):
        needles = (dr.get("if_missing_obligations") or [])
        if any(_obl_matches(obl, n) for n in needles):
            strength = _norm_strength(dr.get("strength"))
            w = max(w, RULE_STRENGTH_WEIGHT.get(("DOWN", strength), DEFAULT_OBL_WEIGHT))
    return w


def _weighted_coverage_ratio(process_id: str, practice_id: str, rules: dict) -> float:
    """∑(wᵢ * coveredᵢ) / ∑(wᵢ)  – coveredᵢ = 1 falls status=='covered', sonst 0."""
    key = f"{process_id}:{practice_id}"
    cov = st.session_state.get("obl_coverage", {}).get(key, []) or []
    obls = st.session_state.get("obl_cache", {}).get(key, {}) or {}
    if not cov or not obls:
        return 0.0
    num, den = 0.0, 0.0
    for item in cov:
        obl_meta = obls.get(item.get("id") or "", {})
        w = _obligation_weight(obl_meta, rules)
        den += w
        if (item.get("status") or "") == "covered":
            num += w
    return (num / den) if den > 0 else 0.0


# --- RL-Helper: Regeln der aktuellen Practice sammeln & als Checkliste formatieren ---

_RX_RL_DOWNRATE     = re.compile(r"\b(shall|must)\s+be\s+downrated\b", re.I)
_RX_RL_NOT_DOWNRATE = re.compile(r"\b(shall|must)\s+not\s+be\s+downrated\b", re.I)
_RX_RL_NOT_HIGHER   = re.compile(r"\b(shall|must)\s+not\s+be\s+rated\s+higher\b", re.I)

def _extract_rule_children(process_id: str, practice_id: str) -> list[tuple[str, str]]:
    """
    Liefert [(display_id, text)] für alle direkten Children vom Typ aspice_rule
    unterhalb der angegebenen Practice (z.B. SUP.1.BP1).
    """
    use_v31 = st.session_state.get("sim_version", "3.1") == "3.1"
    DOCS_X = DOCS_V31 if use_v31 else DOCS_V40
    ID_MAP_X = ID_MAP_V31 if use_v31 else ID_MAP_V40
    PARENT_X = PARENT_MAP_V31 if use_v31 else PARENT_MAP_V40

    if not practice_id:
        return []

    pid = practice_id.strip()
    root = f"{process_id.strip()}.{pid}" if not pid.upper().startswith(process_id.upper()) else pid

    out: list[tuple[str, str]] = []
    for parent_idx in (ID_MAP_X.get(root) or ID_MAP_X.get(root.lower()) or ID_MAP_X.get(root.upper()) or []):
        did = (DOCS_X[parent_idx].metadata.get("display_id") or "").strip()
        if not did:
            continue
        for child_idx in PARENT_X.get(did.lower(), []):
            md = DOCS_X[child_idx].metadata or {}
            if (md.get("type") or "") == "aspice_rule":
                out.append(((md.get("display_id") or "").strip(),
                            (DOCS_X[child_idx].text or "").strip()))
    out.sort(key=lambda t: t[0])
    return out

def _rules_text_for_practice(process_id: str, practice_id: str) -> str:
    parts = []
    for rid, txt in _extract_rule_children(process_id, practice_id):
        s = " ".join((txt or "").split())
        parts.append(f"- {rid}: {s}")
    return "\n".join(parts) if parts else ""

@lru_cache(maxsize=512)
def _extract_rating_rules(sim_ver: str, process_id: str, practice_id: str) -> dict:
    """Erzeugt strukturierte Rules (Caps/Downrate/No-downrate) via LLM; Cache auf (sim_ver, process_id, practice_id)."""
    rules_raw = _rules_text_for_practice(process_id, practice_id)
    if not rules_raw.strip():
        return {"caps": [], "mandatory_downrates": [], "no_downrate": []}

    prompt = RATING_RULES_JSON_PROMPT.format(rules_text=rules_raw)
    msgs = [
        {"role": "system", "content": f"Arbeite ausschließlich gemäß ASPICE-Version {sim_ver}."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = _chat_with_model_tier(msgs, temperature=0.0)
        data = json.loads(_resp_text(resp))
        # minimal robust machen
        return {
            "caps": data.get("caps", []) or [],
            "mandatory_downrates": data.get("mandatory_downrates", []) or [],
            "no_downrate": data.get("no_downrate", []) or [],
        }
    except Exception:
        return {"caps": [], "mandatory_downrates": [], "no_downrate": []}

# --- kleine Utilitys für Bands/Prozent ---
_BAND_MAX = {"N": 15, "P": 50, "L": 85, "F": 100}

GRADE_ORDER = ["N", "P", "L", "F"]  # niedrig -> hoch

def _band_to_pct(band: str) -> int:
    b = (band or "").strip().upper()
    return 15 if b == "N" else 50 if b == "P" else 85 if b == "L" else 100

def _pct_to_band(p: float) -> str:
    if p <= 15:  return "N"
    if p <= 50:  return "P"
    if p <= 85:  return "L"
    return "F"

def _demote_pct_one_band(pct: float) -> float:
    """Senkt den Prozentwert genau um EIN Band (F->L, L->P, P->N, N bleibt N).
       Prozent wird dabei auf die Oberkante des neuen Bands gedeckelt."""
    cur = _pct_to_band(pct)
    idx = max(0, GRADE_ORDER.index(cur) - 1)
    new_band = GRADE_ORDER[idx]
    return min(pct, _band_to_pct(new_band))


def _context_text_for_practice(practice_id: str) -> str:
    """Kontext = Nutzerantworten + ggf. Evidenz-Snippets aus Coverage."""
    ans = "\n".join(st.session_state.get("answers_by_practice", {}).get(practice_id, []))
    cov = st.session_state.get("obl_coverage", {}).get(practice_id, {})
    spans = []
    try:
        for it in cov.get("items", []):
            for s in (it.get("snips") or []):
                if isinstance(s, str): spans.append(s)
    except Exception:
        pass
    return (ans + "\n" + "\n".join(spans)).strip()


def _missing_obligations(practice_id: str, tokens: list[str]) -> bool:
    """Sehr einfache Heuristik: wenn in Coverage keine Items mit ähnlichen IDs/Titeln auftauchen → 'missing'."""
    cov = st.session_state.get("obl_coverage", {}).get(practice_id, {}) or {}
    items = cov.get("items", []) or []
    tkn = [t.lower() for t in tokens or []]
    for t in tkn:
        hit = False
        for it in items:
            title = (it.get("title") or "").lower()
            mid   = (it.get("id") or "").lower()
            if t and (t in title or t in mid):
                hit = True; break
        if not hit:
            return True
    return False


def _keyword_hit(text: str, keywords: list[str]) -> bool:
    tl = (text or "").lower()
    for k in (keywords or []):
        if k and k.lower() in tl:
            return True
    return False


def _apply_rules(practice_id: str, base_pct: float, rules: dict) -> float:
    """
    Priorität: Ceil > No-downrate > Downrate
    - Ceil: harte Obergrenze (max_band)
    - No-downrate: blockiert passende Downrates
    - Downrate: senkt Prozent (percent_penalty) und deckelt ggf. mit min_band
    """
    pct = float(base_pct)
    text_ctx = _context_text_for_practice(practice_id)

    # 1) CEILINGS
    max_pct_cap = 100
    for cap in rules.get("caps", []) or []:
        cond_missing = _missing_obligations(practice_id, cap.get("if_missing_obligations") or [])
        cond_weak    = _keyword_hit(text_ctx, cap.get("if_weakness_keywords") or [])
        if cond_missing or cond_weak:
            max_pct_cap = min(max_pct_cap, _band_to_pct(cap.get("max_band")))
    pct = min(pct, max_pct_cap)

    # 2) NO-DOWNRATE: sammle Keywords, die Downrates blocken dürfen
    blockers = set()
    for nd in rules.get("no_downrate", []) or []:
        for kw in (nd.get("if_weakness_keywords") or []):
            if _keyword_hit(text_ctx, [kw]):
                blockers.add(kw.lower())

    # 3) DOWNRATES — genau EINE SKALA abwerten
    downrated_once = False
    for dr in (rules.get("mandatory_downrates") or []):
        if downrated_once:
            break  # nur EIN Schritt insgesamt

        cond_missing = _missing_obligations(practice_id, dr.get("if_missing_obligations") or [])
        kws = dr.get("if_weakness_keywords") or []
        cond_weak = _keyword_hit(text_ctx, kws)

        if not (cond_missing or cond_weak):
            continue

        # No-downrate-Blocker?
        if any((kw or "").lower() in blockers for kw in kws):
            continue

        # >>> EIN Band herunterstufen (F->L, L->P, P->N, N bleibt N)
        pct = _demote_pct_one_band(pct)
        downrated_once = True

        # Falls die Regel zusätzlich ein min_band vorgibt, respektiere das als Obergrenze
        mb = (dr.get("min_band") or "").strip().upper()
        if mb in {"N", "P", "L", "F"}:
            pct = min(pct, _band_to_pct(mb))

    return pct


# RISIKO-DÄMPFER (Impact & Scope)

# IDs wie SUP.1.BP1 / SWE.2.BP3 / GP 2.1.1 etc.
_ID_RX = re.compile(r"\b([A-Z]{2,4})\.\d(?:\.\d)?\.(?:BP|GP)\d\b")

def _find_ids_in_text(text: str) -> set[str]:
    return set(m.group(0) for m in _ID_RX.finditer(text or ""))

def _classify_scope_from_text(text: str, current_process: str) -> str:
    """
    Scope (deine Definition):
    - isolated:  keine Prozess-ID ungleich current_process in Text/RL, und keine andere BP/GP eines anderen Prozesses
    - area:      nur andere BP/GP innerhalb DESSELBEN Prozesses genannt
    - systemic:  mindestens eine Prozess-ID eines ANDEREN Prozesses genannt
    """
    ids = _find_ids_in_text(text)
    if not ids:
        return "isolated"

    other_proc = False
    same_proc_other_bp = False
    for id_ in ids:
        proc = id_.split(".", 1)[0]
        if proc.upper() == (current_process or "").upper():
            # gleicher Prozess ⇒ könnte eine andere BP/GP sein
            same_proc_other_bp = True
        else:
            other_proc = True

    if other_proc:
        return "systemic"
    if same_proc_other_bp:
        return "area"
    return "isolated"


def _classify_impact_from_text(text: str, *, process_id: str = "") -> str:
    """
    Impact nur entlang von:
    - Prozesszweck (Purpose des Prozesses aus JSON)
    - Produkt-/Releasequalität (schlanke Wortmuster, de/en)
    Kein sonstiges Heuristik-Keywording.
    """
    t = (text or "").strip()
    if not t:
        return "low"

    # 1) Purpose des Prozesses
    purpose = _get_process_purpose(process_id or st.session_state.get("sim_process_id", ""))
    purpose = (purpose or "").strip()
    if purpose:
        # Grobe inhaltliche Überlappung: ein paar Purpose-Schlüsselwörter müssen im Text vorkommen.
        # (Bewusst minimal gehalten – keine "magische" Heuristik.)
        p_tokens = set(w.lower() for w in re.findall(r"[a-zA-ZäöüÄÖÜß]+", purpose) if len(w) >= 5)
        t_tokens = set(w.lower() for w in re.findall(r"[a-zA-ZäöüÄÖÜß]+", t)       if len(w) >= 5)
        overlap = len(p_tokens & t_tokens)
        # Wenn der Weakness-/Befundtext substanziell im Purpose-Feld funkt, werten wir Impact als "high".
        if overlap >= 3:
            return "high"

    # 2) Produkt-/Releasequalität explizit erwähnt?
    if re.search(r"\b(produkt|product)\b.*\bqualit[aä]t|quality\b.*\b(product|release)\b", t, re.I):
        return "high"

    # sonst: konservativ "low"
    return "low"


def _risk_dampener_for(practice_id: str, process_id: str) -> float:
    """
    Erzeugt einen 0..1-Risikofaktor aus (Impact, Scope).
    Kein Likelihood/Mitigation – bewusst schlank. Pure Heuristik auf Basis
    des Context-Texts (User-Antworten + Evidenzsnips) + Rules-Text.
    """
    # 1) Kontext-Text der Practice (Antworten + Snippets)
    ctx = _context_text_for_practice(practice_id)

    # 2) Auch der Regel-Text kann Cross-Process-IDs enthalten
    try:
        rules_txt = _rules_text_for_practice(process_id, practice_id)
    except Exception:
        rules_txt = ""

    base_text = f"{ctx}\n{rules_txt}".strip()

    # Impact (Prozesszweck des aktuellen Prozesses berücksichtigen)
    impact = _classify_impact_from_text(base_text, process_id=process_id)
    if impact == "high":
        imp = 0.20
    elif impact == "medium":
        imp = 0.10
    else:
        imp = 0.05

    # Scope
    scope = _classify_scope_from_text(base_text, process_id)
    if scope == "systemic":
        sc = 1.40
    elif scope == "area":
        sc = 1.20
    else:
        sc = 1.00

    # kombinierter Risiko-Faktor (0..1), moderat gehalten
    risk = min(0.35, imp * sc)
    return float(max(0.0, min(1.0, risk)))


def compute_practice_percent_and_grade_v2(process_id: str, practice_id: str) -> tuple[float, str, dict]:
    """
    1) Basisscore = gewichtete Obligation-Coverage (shall/must/should & ceil/downrate) → %
    2) Regeln anwenden (Ceilings, No-Downrate, Downrates)
    3) Mapping nach N/P/L/F
    """
    # 1) Regeln holen
    sim_ver = st.session_state.get("sim_version", "3.1")
    rules = _extract_rating_rules(sim_ver, process_id, practice_id)

    # 2) Gewichtete Coverage
    cov_w = _weighted_coverage_ratio(process_id, practice_id, rules)  # 0..1
    base_pct = float(max(0.0, min(1.0, cov_w)) * 100.0)

    # 3) Regeln anwenden
    final_pct = _apply_rules(practice_id, base_pct, rules)
    band = _pct_to_band(final_pct)

    debug = {
        "base_pct_weighted": round(base_pct, 1),
        "final_pct_after_rules": round(final_pct, 1),
        "band": band,
        "rules_used": rules,
    }
    return final_pct, band, debug


def _format_rule_checklist(process_id: str, practice_id: str, *, max_len: int = 220) -> str:
    """
    Formatiert eine knappe 'Regel-Checkliste (verbindlich)' für die Practice.
    Markiert sowohl „MUSS downraten“, „DARF NICHT downraten“ als auch „NICHT höher bewerten“.
    """
    rules = _extract_rule_children(process_id, practice_id)

    if not rules:
        return ""
    lines = ["Regel-Checkliste (verbindlich):"]
    for rid, txt in rules:
        s = " ".join((txt or "").split())
        if len(s) > max_len:
            s = s[: max_len - 1].rstrip() + "…"

        tag = ""
        if _RX_RL_NOT_DOWNRATE.search(s):
            tag = "— **DARF NICHT downraten**"
        elif _RX_RL_NOT_HIGHER.search(s):
            tag = "— **NICHT höher bewerten**"
        elif _RX_RL_DOWNRATE.search(s):
            tag = "— **MUSS downraten**"

        lines.append(f"- **{rid}** {tag}: {s}" if tag else f"- **{rid}**: {s}")
    return "\n".join(lines)


# ================================================================================
# PRODUCTION-GRADE ASSESSMENT SYSTEM - COMPLETE REWRITE
# ================================================================================

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import json


# ────────────────────────────────────────────────────────────────────────────────
# 1. STATE MACHINE
# ────────────────────────────────────────────────────────────────────────────────

class AssessmentPhase(Enum):
    ASKING = "asking"  # Initiale Frage stellen

@dataclass
class AssessmentState:
    """Vollständiger Zustand einer Assessment-Session"""
    phase: AssessmentPhase = AssessmentPhase.ASKING
    current_practice: str = ""
    current_obligation: Optional[Dict] = None
    conversation_turns: List[Dict] = field(default_factory=list)  # strukturiert, nicht raw text
    evidence_attempts: int = 0  # wie oft nach Evidenz gefragt
    frustration_signals: int = 0  # "kann nicht", "weiß nicht"
    obligations_covered: set = field(default_factory=set)
    last_question_id: str = ""  # dedupe


# ────────────────────────────────────────────────────────────────────────────────
# 2. MEMORY SYSTEM
# ────────────────────────────────────────────────────────────────────────────────

class ConversationMemory:
    """Strukturiertes Memory mit Session Persistence"""

    def __init__(self):
        # Persistence via st.session_state (kritisch für Streamlit!)
        if "memory_turns" not in st.session_state:
            st.session_state.memory_turns = []
        if "memory_facts" not in st.session_state:
            st.session_state.memory_facts = {}
        if "memory_embeddings" not in st.session_state:
            st.session_state.memory_embeddings = {}

        self.turns = st.session_state.memory_turns
        self.semantic_facts = st.session_state.memory_facts
        self.embeddings = st.session_state.memory_embeddings  # {turn_idx: embedding}
        self.working_context = None

    def add_turn(self, role: str, content: str, metadata: dict = None):
        """Fügt einen Turn hinzu mit Batch-Summarization"""
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": metadata or {}
        }

        self.turns.append(turn)

        # Alte Turns komprimieren
        if len(self.turns) > 10:
            self._compress_old_turns()

        # Batch-Summarization (alle 5 Turns)
        self._batch_summarize_if_needed()

    def _summarize_turn(self, turn: dict) -> str:
        """Komprimiert einen Turn zu einem Fakten-Satz"""
        prev = self.turns[-1] if self.turns else None

        prompt = f"""
Komprimiere diesen Gesprächs-Turn zu 1 prägnanten Satz:

{"Assistant: " + prev["content"] if prev and prev["role"] == "assistant" else ""}
User: {turn["content"]}

Format: "User [provided/refused/clarified] [what] about [aspect]."
Beispiel: "User provided document reference (Project Plan v2.1) for scope definition."

NUR der Satz, keine Erklärung.
"""
        try:
            resp = _chat_with_model_tier(
                [{"role": "user", "content": prompt}],
                tier=ModelTier.CHEAP,  # ← Summaries sind einfach, CHEAP
                temperature=0.1
            )
            return _resp_text(resp).strip()
        except:
            return turn["content"][:100]

    def _compress_old_turns(self):
        """Komprimiert Turns 0-5 zu einer Summary"""
        if len(self.turns) <= 10:
            return

        old_turns = self.turns[:5]
        summaries = [t.get("summary", t["content"][:100]) for t in old_turns]

        compressed = {
            "role": "system",
            "content": f"[Komprimierter Verlauf: {' | '.join(summaries)}]",
            "timestamp": old_turns[0]["timestamp"],
            "metadata": {"compressed": True, "original_count": len(old_turns)}
        }

        self.turns = [compressed] + self.turns[5:]

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Holt relevantesten Kontext via Semantic Search (Embedding-basiert)"""

        if not self.turns:
            return ""

        # Leeren Query abfangen
        query = (query or "").strip()
        if not query:
            logger.warning("Empty query for get_relevant_context, using fallback")
            recent = self.turns[-k:]
            summaries = [t.get("summary", t["content"][:150]) for t in recent if t["role"] != "system"]
            return "\n".join(summaries)

        # Query embedden
        try:
            # LlamaIndex nutzt get_query_embedding (nicht embed_query)
            if hasattr(Settings.embed_model, 'get_query_embedding'):
                query_embedding = Settings.embed_model.get_query_embedding(query)
            elif hasattr(Settings.embed_model, 'embed_query'):
                query_embedding = Settings.embed_model.embed_query(query)
            else:
                raise AttributeError("Embedding model has no query method")
        except Exception as e:
            logger.warning(f"Embedding failed, fallback to last-k: {e}")
            # Fallback: letzte k
            recent = self.turns[-k:]
            summaries = [t.get("summary", t["content"][:150]) for t in recent if t["role"] != "system"]
            return "\n".join(summaries)

        # Similarity-Scores berechnen
        scores = []
        for idx, turn in enumerate(self.turns):
            if turn["role"] == "system":
                continue

            # Embedding lazy laden (nur wenn noch nicht vorhanden)
            if idx not in self.embeddings:
                try:
                    summary_text = turn.get("summary", turn["content"][:500])
                    self.embeddings[idx] = Settings.embed_model.embed_query(summary_text)
                except Exception:
                    continue

            turn_embedding = self.embeddings[idx]
            similarity = _cosine_similarity(query_embedding, turn_embedding)
            scores.append((similarity, idx, turn))

        if not scores:
            # Fallback wenn keine Embeddings
            recent = self.turns[-k:]
            summaries = [t.get("summary", t["content"][:150]) for t in recent if t["role"] != "system"]
            return "\n".join(summaries)

        # Top-K nach Relevanz sortieren
        top_turns = sorted(scores, key=lambda x: x[0], reverse=True)[:k * 2]  # Hole 2x mehr für Re-Ranking

        # Re-Ranking mit Cross-Encoder (falls verfügbar)
        if CROSS_ENCODER and len(top_turns) > k:
            try:
                # Paare für Cross-Encoder
                pairs = [[query, t[2].get('summary', t[2]['content'][:512])] for t in top_turns]
                ce_scores = CROSS_ENCODER.predict(pairs)

                # Nach Cross-Encoder Score neu sortieren
                reranked = sorted(zip(ce_scores, top_turns), key=lambda x: x[0], reverse=True)
                top_turns = [t[1] for t in reranked[:k]]
            except Exception as e:
                logger.warning(f"Memory re-ranking failed: {e}")
                top_turns = top_turns[:k]
        else:
            top_turns = top_turns[:k]

        # Rückgabe: Summaries der relevantesten Turns
        return "\n".join([
            f"{t[2]['role'].title()}: {t[2].get('summary', t[2]['content'][:150])}"
            for t in top_turns
        ])

    def extract_facts(self, obl_id: str) -> List[str]:
        """Extrahiert alle Fakten zu einer Obligation"""
        return self.semantic_facts.get(obl_id, [])

    def add_fact(self, obl_id: str, fact: str):
        """Speichert einen verifizierten Fakt"""
        self.semantic_facts.setdefault(obl_id, []).append(fact)

    def _batch_summarize_if_needed(self):
        """Batch-Summarization alle 5 Turns (Performance-Optimierung)"""

        unsummarized = [i for i, t in enumerate(self.turns) if "summary" not in t and t["role"] != "system"]

        if len(unsummarized) < 5:
            return  # Noch nicht genug für Batch

        # Batch-Prompt: Alle 5 Turns auf einmal
        batch_texts = []
        for idx in unsummarized[:5]:
            turn = self.turns[idx]
            batch_texts.append(f"Turn {idx}: {turn['role']} - {turn['content'][:200]}")

        prompt = f"""
Komprimiere diese 5 Gesprächs-Turns zu je 1 prägnanten Satz:

{chr(10).join(batch_texts)}

Format (JSON Array):
[
  "Turn 0: User provided...",
  "Turn 1: Assistant asked...",
  ...
]
"""

        try:
            resp = _chat_with_model_tier(
                [{"role": "user", "content": prompt}],
                tier=ModelTier.CHEAP,
                temperature=0.1
            )
            raw = _resp_text(resp).strip()

            import re
            match = re.search(r'\[.*]', raw, re.DOTALL)
            if match:
                summaries = json.loads(match.group(0))

                for i, summary in enumerate(summaries[:5]):
                    if i < len(unsummarized):
                        idx = unsummarized[i]
                        self.turns[idx]["summary"] = summary.split(":", 1)[-1].strip()
        except Exception as e:
            logger.warning(f"Batch summarization failed: {e}")


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Berechnet Cosine Similarity zwischen zwei Vektoren"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


# ────────────────────────────────────────────────────────────────────────────────
# 5. QUESTION GENERATOR (mit Gap Analysis)
# ────────────────────────────────────────────────────────────────────────────────
@trace_llm_call
def generate_question_with_gap_analysis(
        state: AssessmentState,
        memory: ConversationMemory,
        obligation: dict,
        user_input: str = ""
) -> str:
    """Multi-Step Question Generation (ohne Self-Correction)"""

    # Sprache erkennen
    lang = detect_query_lang(user_input) if user_input else "de"

    gaps = _identify_gaps(obligation, memory)

    if not gaps:
        return "Vielen Dank, dieser Aspekt ist ausreichend belegt."

    primary_gap = gaps[0]
    style = st.session_state.get("sim_style", "rule-oriented")

    question = _generate_styled_question(primary_gap, obligation, memory, style, state, lang)  # ← lang übergeben

    return question


def _identify_gaps(obligation: Dict, memory: ConversationMemory) -> List[str]:
    """Identifiziert, was noch fehlt"""

    obl_id = obligation.get("id", "")
    known_facts = memory.extract_facts(obl_id)

    expected = obligation.get("expected_evidence", [])
    key_phrases = obligation.get("key_phrases", [])

    # Kontext aus Memory - Sinnvoller Query statt ""
    query = obligation.get("title", "") or " ".join(key_phrases[:3]) or "evidence"
    recent_context = memory.get_relevant_context(query, k=5)

    prompt = f"""
Analysiere: Was fehlt noch für diese ASPICE-Anforderung?

**Anforderung:** {obligation.get("title", "")}
**Erwartete Nachweise:** {", ".join(expected[:3])}
**Schlüsselbegriffe:** {", ".join(key_phrases[:4])}

**Bisher bekannt:**
{chr(10).join(known_facts) if known_facts else "(nichts)"}

**Bisheriger Verlauf:**
{recent_context}

Liste 1-3 konkrete Lücken (Priorität: hoch → niedrig).
Format: Kurze Stichpunkte.

Beispiel:
- Dokumenten-Referenz fehlt
- Unklar: Wer hat reviewed
- Timing nicht genannt
"""

    try:
        resp = _chat_with_model_tier(
            [{"role": "user", "content": prompt}],
            tier=ModelTier.SMART,  # ← Analysis braucht Reasoning
            temperature=0.2
        )
        raw = _resp_text(resp).strip()

        gaps = [line.strip("- ").strip() for line in raw.split("\n") if line.strip().startswith("-")]
        return gaps[:3]
    except:
        return ["Konkrete Nachweise fehlen"]


def _generate_styled_question(
        gap: str,
        obligation: Dict,
        memory: ConversationMemory,
        style: str,
        state: AssessmentState,
        lang: str = "de"
) -> str:
    """Generiert Frage im gewählten Assessor-Stil"""

    # NEU: Sprachvorgabe
    lang_tag = (
        "Antwortsprache: Deutsch. Respond in German."
        if lang == "de"
        else "Language: English. Antworte auf Englisch."
    )

    STYLE_PERSONAS = {
        "rule-oriented": "Stelle eine direkte, präzise Kontrollfrage. Fokus: messbare Nachweise.",
        "balanced": "Stelle eine umfassende Frage, die Umsetzung UND Verständnis prüft.",
        "challenging": "Stelle eine kritische Frage, die nach Grenzfällen/Risiken fragt."
    }

    persona = STYLE_PERSONAS.get(style, STYLE_PERSONAS["rule-oriented"])
    # Sinnvoller Query statt ""
    query = f"{gap} {obligation.get('title', '')}"[:100]
    recent = memory.get_relevant_context(query, k=3)

    # Adaptive Tone basierend auf Frustration
    tone_modifier = ""
    if state.frustration_signals >= 2:
        tone_modifier = "Der Prüfling hatte Schwierigkeiten. Frage konstruktiv, biete Hilfe an."
    elif state.evidence_attempts >= 3:
        tone_modifier = "Mehrfach nach Evidenz gefragt. Sei sehr konkret, was du brauchst."

    # Negative Examples holen
    neg_examples = _get_negative_examples_prompt(
        {"current_practice": state.current_practice, "current_obligation": obligation})

    prompt = f"""
{lang_tag}

{persona}

**Fehlende Information:** {gap}
**Kontext (Anforderung):** {obligation.get("title", "")}

**Bisheriger Verlauf:**
{recent}

{tone_modifier}

{neg_examples}

Formuliere EINE natürliche Assessor-Frage zu EINEM Aspekt (1-2 Sätze). KEINE Doppelfragen mit "und" oder "sowie".

**Frage-Stil:**
- Frage nach FAKTEN, nicht nach theoretischen Überlegungen
- Nutze Präsens/Vergangenheit, NICHT Konjunktiv
- Falsch: "Welche Elemente sollten enthalten sein..."
- Richtig: "Welche Elemente sind in Ihrer QA-Strategie enthalten?"
- Richtig: "Wie stellen Sie sicher, dass..."

**Verbote:**
- Keine IDs (MAN.3.BP1) im Fragetext
- Nicht "Woran erkennen Sie..."
- Nicht den Obligation-Titel wiederholen

**Erlaubt:**
- Konkrete Fragen: "Welches Dokument beschreibt X?"
- Tool-Fragen: "Wo im Tool ist Y zu finden?"
- Prozess-Fragen: "Wer ist verantwortlich für Z?"

NUR die Frage, keine Erklärung.
"""

    try:
        resp = _chat_with_model_tier(
            [{"role": "system", "content": persona}, {"role": "user", "content": prompt}],
            tier=ModelTier.SMART,
            temperature=st.session_state.get("sim_temp_questions", 0.3)
        )
        question = _resp_text(resp).strip()

        # Dedupe: Wenn identisch zur letzten Frage → variieren
        if question == state.last_question_id:
            question = _vary_question(question)

        state.last_question_id = question
        return question
    except:
        return f"Können Sie konkret zeigen, wo/wie '{gap}' umgesetzt ist?"


def _vary_question(original: str) -> str:
    """Variiert eine Frage leicht (bei Dedupe)"""
    prompt = f'Formuliere diese Frage anders (gleiche Bedeutung): "{original}"'
    try:
        resp = _chat_with_model_tier([{"role": "user", "content": prompt}], temperature=0.7)
        return _resp_text(resp).strip()
    except:
        return original + " (bitte mit konkreter Fundstelle)"


# ────────────────────────────────────────────────────────────────────────────────
# 6. MAIN ORCHESTRATOR
# ────────────────────────────────────────────────────────────────────────────────

class AssessmentOrchestrator:
    """Haupt-Controller mit State Persistence"""

    def __init__(self, process_id: str, version: str):
        self.process_id = process_id
        self.version = version

        # State Persistence
        state_key = f"assessment_state_{process_id}"
        if state_key not in st.session_state:
            st.session_state[state_key] = AssessmentState(current_practice="")
        self.state = st.session_state[state_key]

        # Memory Persistence
        memory_key = f"assessment_memory_{process_id}"
        if memory_key not in st.session_state:
            st.session_state[memory_key] = ConversationMemory()
        self.memory = st.session_state[memory_key]

    def initialize_practice(self, practice_id: str):
        """Startet eine neue Practice"""
        self.state.current_practice = practice_id
        self.state.phase = AssessmentPhase.ASKING
        self.state.evidence_attempts = 0
        self.state.frustration_signals = 0

        logger.info(f">>> INITIALIZE_PRACTICE called for {practice_id}")
        # Obligations laden & sortieren
        _ensure_obl_coverage(self.process_id, practice_id)

        key = f"{self.process_id}:{practice_id}"
        obl_cache = st.session_state.get("obl_cache", {}).get(key, {})
        coverage = st.session_state.get("obl_coverage", {}).get(key, [])

        # Erste offene Obligation
        for item in coverage:
            if item["status"] == "open":
                self.state.current_obligation = obl_cache.get(item["id"])
                break

    def process_user_input(self, user_msg: str, has_url: bool = False) -> Tuple[str, bool]:
        """
        State-of-the-art: LLM mit Tools entscheidet alles selbst.
        KEINE Intent-Klassifikation, KEINE State-Machine.
        """
        # Automatische Nachricht bei Upload ohne Text
        if not user_msg.strip() and has_url:
            user_msg = "Dokument hochgeladen"

        if not user_msg.strip():
            return "Bitte geben Sie eine Antwort ein.", False

        # Memory aktualisieren
        self.memory.add_turn("user", user_msg, {"has_url": has_url})

        # Obligation sofort laden
        obl = self.state.current_obligation
        if not obl:
            return "Keine aktive Obligation.", False

        # Evidence-Tracking
        practice_id = self.state.current_practice
        if practice_id:
            if "practice_evidence" not in st.session_state:
                st.session_state.practice_evidence = {}

            if practice_id not in st.session_state.practice_evidence:
                st.session_state.practice_evidence[practice_id] = PracticeEvidence(practice_id=practice_id)

            st.session_state.practice_evidence[practice_id].user_statements.append(user_msg)
            st.session_state.practice_evidence[practice_id].timestamps.append(datetime.now(UTC).isoformat())

            if has_url or st.session_state.get("evidence_index"):
                try:
                    snippets = _evidence_snippets_for(obl.get("title", ""), top_k=3)
                    if snippets:
                        st.session_state.practice_evidence[practice_id].uploaded_docs.append({
                            "filename": "Evidence Query",
                            "snippets": snippets,
                            "timestamp": datetime.now(UTC).isoformat()
                        })
                except:
                    pass

        # Sprache erkennen
        lang = detect_query_lang(user_msg) if user_msg else "de"
        lang_tag = (
            "Antwortsprache: Deutsch. Respond in German."
            if lang == "de"
            else "Language: English. Antworte auf Englisch."
        )

        # Kontext sammeln
        conversation_history = self.memory.get_relevant_context(user_msg, k=5)

        # Evidence-Snippets bei Upload
        uploaded_evidence = []
        if has_url or st.session_state.get("evidence_index"):
            try:
                uploaded_evidence = _evidence_snippets_for(obl.get("title", ""), top_k=5)
            except:
                pass

        # System-Prompt: LLM entscheidet ALLES
        system_prompt = f"""{_assessor_system_prompt()}

        {lang_tag}

        **Aktuelle Situation:**
        - Practice: {self.state.current_practice}
        - Obligation: {obl.get('title', '')}
        - Erwartete Nachweise: {', '.join(obl.get('expected_evidence', [])[:3])}
        - Dokument hochgeladen: {"Ja" if uploaded_evidence else "Nein"}

        **Deine Tools:**
        1. **query_evidence**: Durchsuche hochgeladene Dokumente nach spezifischen Begriffen
        2. **next_obligation**: Markiere aktuelle Obligation als erfüllt (mit Coverage) und gehe weiter
        3. **skip_to_next**: Überspringe Obligation OHNE Coverage wenn User nicht antworten kann
        4. **request_evidence_upload**: Fordere fehlende Dokumente an

        **Entscheide selbst:**
        - User hat Dokument hochgeladen? → Nutze query_evidence SOFORT um zu prüfen
        - User gibt substantielle Antwort (>30 Zeichen)? → Nutze next_obligation mit Coverage
        - User sagt "weiß nicht" / "haben wir nicht"? → Nutze skip_to_next
        - User sagt "Moment" / "lade hoch"? → Warte, sage "Kein Problem" (KEIN Tool)
        - User gibt vage Antwort (<30 Zeichen)? → Frage nach Details

        **Wichtig:**
        - Nutze Tools PROAKTIV, frage nicht erst
        - Bei Upload: IMMER query_evidence nutzen und Ergebnis prüfen
        - Bei query_evidence-Fund → nutze next_obligation
        - Keine generischen Floskeln

        Konversation bisher:
        {conversation_history}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]

        context = {
            "process_id": self.process_id,
            "practice_id": self.state.current_practice,
            "orchestrator": self,
            "has_upload": bool(uploaded_evidence),
            "user_msg": user_msg
        }

        # LLM mit ALLEN Tools - es entscheidet selbst
        response, tools_used, action = call_llm_with_tools(messages, context)

        # Action-basierte Rückgabe
        if action == "advance_next":
            next_obl = self._load_next_obligation()
            if next_obl:
                self.state.current_obligation = next_obl
                self.state.phase = AssessmentPhase.ASKING
                next_q = generate_question_with_gap_analysis(
                    self.state, self.memory, next_obl, user_input=user_msg
                )
                return f"{response}\n\n{next_q}", False
            else:
                return "Alle Aspekte dieser Practice behandelt. Bewertung folgt.", True

        elif action == "request_upload":
            return response, False

        elif action == "complete_practice":
            return response, True

        # Kein Action → einfach Antwort zurückgeben
        return response, False


    def _mark_current_covered(self, coverage: float, reasoning: str):
        """Markiert aktuelle Obligation als covered"""
        key = f"{self.process_id}:{self.state.current_practice}"
        coverage_list = st.session_state.get("obl_coverage", {}).get(key, [])

        obl_id = self.state.current_obligation.get("id", "")
        for item in coverage_list:
            if item["id"] == obl_id:
                item["status"] = "covered"
                item["confidence"] = coverage / 100.0
                item["reasoning"] = reasoning
                break

        self.state.obligations_covered.add(obl_id)

    def _mark_current_skipped(self):
        """Markiert aktuelle Obligation als skipped"""
        key = f"{self.process_id}:{self.state.current_practice}"
        coverage_list = st.session_state.get("obl_coverage", {}).get(key, [])

        obl_id = self.state.current_obligation.get("id", "") if self.state.current_obligation else ""
        if obl_id:
            for item in coverage_list:
                if item["id"] == obl_id:
                    item["status"] = "skipped"
                    item["confidence"] = 0.0
                    break

    def _load_next_obligation(self) -> Optional[dict]:
        """Lädt die nächste offene Obligation"""
        key = f"{self.process_id}:{self.state.current_practice}"
        obl_cache = st.session_state.get("obl_cache", {}).get(key, {})
        coverage_list = st.session_state.get("obl_coverage", {}).get(key, [])

        logger.info(f"_load_next_obligation: key={key}, coverage_items={len(coverage_list)}")

        for item in coverage_list:
            logger.info(f"  Checking: {item.get('id')} - status={item.get('status')}")
            if item["status"] == "open":
                obl = obl_cache.get(item["id"])
                logger.info(f"  -> Found open: {item.get('id')}, obl_exists={obl is not None}")
                return obl

        logger.info(f"  -> No open obligations found")
        return None


# ────────────────────────────────────────────────────────────────────────────────
# 7. HELPER FUNCTIONS FOR NATURAL RESPONSES
# ────────────────────────────────────────────────────────────────────────────────

def _generate_clarification(context: dict, user_msg: str) -> str:
    """LLM-generierte Klarstellung bei Verwirrung"""
    obl = context["current_obligation"]

    prompt = f"""
Der Prüfling ist verwirrt. Erkläre in 2-3 Sätzen, was du konkret wissen möchtest.

Aktuelle Anforderung: {obl.get("title", "")}
Erwartete Nachweise: {", ".join(obl.get("expected_evidence", [])[:2])}

Letzte Antwort des Prüflings: "{user_msg}"

Formuliere eine klärende Nachfrage ohne den Aspekt-Titel wörtlich zu wiederholen.
Nenne konkrete Beispiele für akzeptable Nachweise.
"""

    try:
        resp = _chat_with_model_tier([{"role": "user", "content": prompt}], tier=ModelTier.CHEAP, temperature=0.3)
        return _resp_text(resp).strip()
    except:
        return f"Ich suche nach konkreten Nachweisen zu '{obl.get('title', 'diesem Aspekt')}'. Beispiel: Dokumentname + Kapitel, oder Tool-Ansicht."


def _generate_alternative_question(context: dict) -> str:
    """Fragt nach alternativem Aspekt"""
    obl = context["current_obligation"]

    prompt = f"""
Der Prüfling kann zu "{obl.get('title', '')}" keine Nachweise liefern.

Formuliere EINE kurze Nachfrage (1 Satz), die nach einem verwandten/alternativen Aspekt fragt.
Beispiel: "Können Sie stattdessen zeigen, wie [verwandter Aspekt] dokumentiert ist?"

Sei konstruktiv und lösungsorientiert.
"""

    try:
        resp = _chat_with_model_tier([{"role": "user", "content": prompt}], tier=ModelTier.CHEAP, temperature=0.4)
        return _resp_text(resp).strip()
    except:
        return "Können Sie alternativ zeigen, wie dieser Aspekt im Projekt behandelt wird?"


def _generate_acknowledgement(obl_title: str, snippets: list[str], coverage: float) -> str:
    """LLM-generiertes Acknowledgement"""

    snip_txt = "\n".join(f"- {s[:100]}..." for s in snippets[:2])

    prompt = f"""
Bestätige in 1-2 Sätzen, dass die Evidenz den Aspekt "{obl_title}" abdeckt.

Gefundene Evidenz-Auszüge:
{snip_txt}

Ton: professionell, anerkennend, aber nicht übertrieben.
"""

    try:
        resp = _chat_with_model_tier([{"role": "user", "content": prompt}], tier=ModelTier.CHEAP, temperature=0.3)
        base = _resp_text(resp).strip()
        return f"{base} (Coverage: {int(coverage * 100)}%)"
    except:
        return f"Danke, das deckt '{obl_title}' ab. (Coverage: {int(coverage * 100)}%)"


def _generate_targeted_followup(context: dict, match: dict, user_msg: str) -> str:
    """Follow-up bei schwacher Evidenz"""
    obl = context["current_obligation"]

    prompt = f"""
Der Prüfling hat etwas zu "{obl.get('title', '')}" gesagt, aber die Evidenz ist noch nicht ausreichend.

Seine Antwort: "{user_msg}"

Formuliere EINE gezielte Nachfrage (1 Satz), die nach einer konkreteren Fundstelle fragt.
Beispiel: "Welches Kapitel/welcher Abschnitt beschreibt dies genau?"

Sei spezifisch, aber nicht fordernd.
"""

    try:
        resp = _chat_with_model_tier([{"role": "user", "content": prompt}], tier=ModelTier.CHEAP, temperature=0.3)
        return _resp_text(resp).strip()
    except:
        return f"Gut, danke. Können Sie noch konkret die Fundstelle nennen?"


def _generate_evidence_request(context: dict, user_msg: str) -> str:
    """Evidenz-Anforderung bei unklarer Antwort"""
    obl = context["current_obligation"]

    prompt = f"""
Der Prüfling hat etwas gesagt, aber keine konkreten Nachweise geliefert.

Seine Antwort: "{user_msg}"
Erwartete Nachweise für "{obl.get('title', '')}": {", ".join(obl.get("expected_evidence", [])[:2])}

Formuliere EINE höfliche Nachfrage (1-2 Sätze), die um konkrete Belege bittet.
Nenne Beispiele für akzeptable Nachweise.

Ton: freundlich-bestimmt, nicht vorwurfsvoll.
"""

    try:
        resp = _chat_with_model_tier([{"role": "user", "content": prompt}], tier=ModelTier.CHEAP, temperature=0.3)
        return _resp_text(resp).strip()
    except:
        return "Ich brauche konkretere Nachweise: Dokumentname + Kapitel, oder Tool-Screenshot/Link."

# ────────────────────────────────────────────────────────────────────────────────
# 8. HELPER: Obligations nach BP-Reihenfolge sortieren
# ────────────────────────────────────────────────────────────────────────────────

def _sort_obligations_by_bp_order(obls: List[Dict]) -> List[Dict]:
    """Sortiert Obligations nach BP-Nummer (MAN.3.BP1 < MAN.3.BP2 < ...)"""

    def bp_sort_key(obl: dict) -> tuple:
        for mid in obl.get("maps_to", []):
            m = re.match(r"([A-Z]{3})\.(\d+)\.BP(\d+)", mid.upper())
            if m:
                return (m.group(1), int(m.group(2)), int(m.group(3)))
        # GPs nach BPs
        for mid in obl.get("maps_to", []):
            m = re.match(r"GP\s*(\d+)\.(\d+)\.(\d+)", mid.upper())
            if m:
                return ("ZZZ", int(m.group(1)), int(m.group(2)))
        return ("ZZZ", 999, 999)

    return sorted(obls, key=bp_sort_key)


def _polish_de_en_question(q: str) -> str:
    if not q: return q
    s = q.strip()
    s = re.sub(r"(?i)\bwie\s+stellen\s+sie\s+([^?]+?)\s+sicher\?", r"Wie wird \1 sichergestellt?", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if s and s[-1] not in "?!": s += "?"
    return s

def _style_angle_preference() -> list[str]:
    s = (st.session_state.get("sim_style") or "").lower()
    if "challenging" in s or "challenging" in s:
        return ["EVIDENZ", "GEGENBEISPIEL", "MECHANISMUS", "ROLLE", "TIMING", "KRITERIUM"]
    if "rule" in s:
        return ["MECHANISMUS", "KRITERIUM", "ROLLE", "EVIDENZ", "TIMING", "GEGENBEISPIEL"]
    return ["MECHANISMUS", "EVIDENZ", "ROLLE", "TIMING", "KRITERIUM", "GEGENBEISPIEL"]

def _pick_angle(i: int) -> str:
    pref = _style_angle_preference()
    return pref[i % len(pref)]


def parse_rating_rules(practice_id: str, process_id: str) -> List[RatingRule]:
    """
    Parst Rules/Recommendations aus dem Practice-Kontext.
    Nutzt die bestehende _extract_rule_children Funktion.
    """
    rules_list = []

    # Hole Rules aus den Docs (nutzt bestehende Funktion)
    raw_rules = _extract_rule_children(process_id, practice_id)

    for rule_id, rule_text in raw_rules:
        rule = RatingRule(
            rule_id=rule_id,
            text=rule_text,
            type="MUST",  # Default
            action="unknown",
            conditions=[]
        )

        text_lower = rule_text.lower()

        # Typ bestimmen
        if re.search(r'\b(must|shall)\b', text_lower):
            rule.type = "MUST"
            rule.weight = 1.0
        elif re.search(r'\bshould\b', text_lower):
            rule.type = "SHOULD"
            rule.weight = 0.7

        # Action bestimmen
        # 1. Ceilings
        ceiling_match = re.search(r'not be rated higher than ([NPLF])', rule_text, re.I)
        if ceiling_match:
            rule.action = "ceiling"
            rule.threshold = ceiling_match.group(1).upper()
        elif re.search(r'not be rated higher\b', text_lower):
            rule.action = "ceiling"
            rule.threshold = "P"  # Default wenn kein Band angegeben

        # 2. Downrates
        elif re.search(r'\bshall be downrated\b|\bmust be downrated\b|\bshould be downrated\b', text_lower):
            rule.action = "downrate"

        # 3. No-Downrate
        elif re.search(r'(shall not|must not|should not) be (downrated|used to downrate)', text_lower):
            rule.action = "no_downrate"

        # 4. Spezifische Ratings ausschließen
        elif re.search(r'(shall not|must not|should not) be rated F', text_lower):
            rule.action = "ceiling"
            rule.threshold = "L"

        # Bedingungen extrahieren
        # Suche nach "if"-Klauseln
        if_match = re.search(r'\bif\b(.+?)(?:,|then|shall|must|should|$)', rule_text, re.I)
        if if_match:
            condition_text = if_match.group(1).strip()
            rule.conditions.append(condition_text)

        # Suche nach Aufzählungen (oft mit :)
        if ':' in rule_text:
            parts = rule_text.split(':', 1)
            if len(parts) > 1:
                # Extrahiere Bullet-Points
                bullets = re.findall(r'-\s*([^\n-]+)', parts[1])
                rule.conditions.extend([b.strip() for b in bullets if len(b.strip()) > 5])

        rules_list.append(rule)

    return rules_list


def _rule_condition_met(rule: RatingRule, evidence: PracticeEvidence) -> bool:
    """
    Prüft ob die Bedingungen einer Rule erfüllt sind.

    Returns:
        True wenn Rule greifen soll
        False wenn Bedingungen nicht erfüllt
    """
    if not rule.conditions:
        # Keine Bedingungen → Rule greift immer
        return True

    evidence_text = evidence.get_all_evidence_text().lower()

    # Check: Fehlt etwas? (negative Bedingungen)
    negative_indicators = [
        'incomplete', 'missing', 'not part of', 'not established',
        'fehlt', 'unvollständig', 'nicht vorhanden'
    ]

    for condition in rule.conditions:
        cond_lower = condition.lower()

        # Extrahiere Kern-Begriffe aus Bedingung
        # z.B. "project plan" aus "if no project plan exists"
        key_terms = re.findall(r'\b[a-z]{4,}\b', cond_lower)
        key_terms = [t for t in key_terms if t not in ['shall', 'must', 'should', 'have', 'been', 'with', 'from']]

        # Prüfe ob Begriff in Evidenz fehlt
        missing_count = 0
        for term in key_terms[:3]:  # Max 3 Hauptbegriffe
            if term not in evidence_text:
                missing_count += 1

        # Wenn >50% der Begriffe fehlen UND Bedingung negativ formuliert ist
        if missing_count > len(key_terms) / 2 and any(neg in cond_lower for neg in negative_indicators):
            return True

    return False


@trace_llm_call
def detect_weaknesses(
        practice_id: str,
        process_id: str,
        obligations: List[Dict],
        evidence: PracticeEvidence,
        rules: List[RatingRule]
) -> List[Weakness]:
    """
    LLM-gestützte Weakness-Erkennung mit strikten ASPICE-Kriterien
    """

    # Process Purpose holen (für Impact-Bestimmung)
    sim_ver = st.session_state.get("sim_version", "3.1")
    purpose = _get_process_purpose(process_id, sim_ver)

    # Nur Downrate-Rules für Kontext
    downrate_rules = [r for r in rules if r.action == "downrate"]
    rules_context = "\n".join(f"- {r.rule_id}: {r.text[:200]}" for r in downrate_rules[:3])

    prompt = f"""Analysiere die Evidenz für {practice_id} und identifiziere **konkrete Schwächen**.

**ASPICE Definition Weakness:**
"A context-specific comprehensive and comprehensible explanation of the process risk. 
It delivers sufficient understanding for the assessees to understand the context-specific 
process risk as a starting point for deriving improvements. The weakness is substantiated 
by objective evidence gathered during the assessment."

**Risiko-Kriterien:**
- Beeinträchtigung des Prozesszwecks: "{purpose}"
- ODER: Beeinträchtigung der Produktqualität

**Obligations (zu erfüllen):**
{chr(10).join(f"- {o.get('title', o.get('id'))}" for o in obligations[:5])}

**Gesammelte Evidenz:**
{evidence.get_all_evidence_text()}

**Relevante Downrate-Rules (Kontext):**
{rules_context or "(keine)"}

---

**Aufgabe:**
Identifiziere 0-3 konkrete Schwächen. Für jede:

1. **aspect**: Was fehlt/ist unzureichend? (KONKRET: "Projektplan fehlt", nicht "Dokumentation unzureichend")
2. **evidence_gap**: Welche Evidenz fehlt genau?
3. **process_risk**: Warum gefährdet das den Prozesszweck ODER die Produktqualität?
4. **impact**: "purpose" oder "product_quality"
5. **severity**: 0.1-1.0 (nur >0.5 wenn echtes Risiko)

**KRITISCHE VERBOTE:**
❌ "Dokumentation ist unvollständig" (zu generisch)
❌ "BP1 fordert X, aber X wurde nicht nachgewiesen" (invertierter BP-Text)
❌ Nur Rule zitieren ohne Risiko-Erklärung

**ERLAUBT:**
✓ "Kein Projektplan mit Ressourcenzuweisung vorhanden → Teams wissen nicht wann sie was tun müssen → Termine gefährdet"
✓ "Change-Requests werden nicht getrackt → Ungewollte Änderungen am Produkt → Produktqualität gefährdet"

**Output NUR als JSON-Array (falls keine Weaknesses: leeres Array):**
[
  {{
    "aspect": "...",
    "evidence_gap": "...",
    "process_risk": "...",
    "impact": "purpose",
    "severity": 0.8
  }}
]
"""

    try:
        response = _chat_with_model_tier(
            [{"role": "user", "content": prompt}],
            tier=ModelTier.SMART,
            temperature=0.2
        )

        raw = _resp_text(response).strip()

        # Robustes JSON-Parsing
        import re
        match = re.search(r'\[.*]', raw, re.DOTALL)
        if match:
            weaknesses_json = json.loads(match.group(0))

            weaknesses = []
            for w in weaknesses_json:
                # Validierung: Severity muss sinnvoll sein
                severity = float(w.get('severity', 0))
                if severity < 0.3:
                    continue  # Zu niedrig → keine echte Weakness

                weaknesses.append(Weakness(
                    aspect=w.get('aspect', ''),
                    evidence_gap=w.get('evidence_gap', ''),
                    process_risk=w.get('process_risk', ''),
                    impact=w.get('impact', 'purpose'),
                    severity=min(1.0, severity)
                ))

            return weaknesses

        return []

    except Exception as e:
        logger.error(f"Weakness detection failed: {e}")
        return []


def calculate_practice_rating_v3(
        practice_id: str,
        process_id: str,
        evidence: PracticeEvidence,
        obligations: List[Dict],
        rules: List[RatingRule],
        weaknesses: List[Weakness]
) -> Tuple[str, float, Dict]:
    """
    Mehrstufige NPLF-Berechnung:
    1. Basis-Coverage (Obligations)
    2. Weakness-Penalty
    3. Rules (Ceiling > No-Downrate > Downrate)

    Returns: (band, percentage, debug_info)
    """

    # SCHRITT 1: Basis-Coverage
    sim_ver = st.session_state.get("sim_version", "3.1")
    rules_dict = _extract_rating_rules(sim_ver, process_id, practice_id)
    base_pct = _weighted_coverage_ratio(process_id, practice_id, rules_dict) * 100

    # SCHRITT 2: Weakness-Penalty
    weakness_penalty = sum(w.severity * 15 for w in weaknesses)
    adjusted_pct = max(0.0, base_pct - weakness_penalty)

    # SCHRITT 3: Rules anwenden
    final_pct = adjusted_pct
    applied_rules = []

    # Prio 1: CEILINGS (harte Obergrenze)
    for rule in [r for r in rules if r.action == "ceiling"]:
        if _rule_condition_met(rule, evidence):
            ceiling = _band_to_pct(rule.threshold or "P")
            if final_pct > ceiling:
                old_pct = final_pct
                final_pct = ceiling
                applied_rules.append({
                    "rule_id": rule.rule_id,
                    "type": rule.type,
                    "action": "ceiling",
                    "impact": f"Capped {old_pct:.0f}% → {ceiling}%",
                    "reason": rule.text[:100]
                })

    # Prio 2: NO-DOWNRATE (sammle blockierte Aspekte)
    blocked_conditions = set()
    for rule in [r for r in rules if r.action == "no_downrate"]:
        if _rule_condition_met(rule, evidence):
            blocked_conditions.update(rule.conditions)
            applied_rules.append({
                "rule_id": rule.rule_id,
                "type": rule.type,
                "action": "no_downrate",
                "impact": "Blocks downrates",
                "reason": rule.text[:100]
            })

    # Prio 3: DOWNRATES (nur wenn nicht blockiert)
    for rule in [r for r in rules if r.action == "downrate"]:
        if not _rule_condition_met(rule, evidence):
            continue

        # Check: Blockiert durch No-Downrate?
        is_blocked = any(
            any(bc.lower() in cond.lower() for bc in blocked_conditions)
            for cond in rule.conditions
        )

        if not is_blocked:
            penalty = 20 * rule.weight
            old_pct = final_pct
            final_pct = max(0.0, final_pct - penalty)
            applied_rules.append({
                "rule_id": rule.rule_id,
                "type": rule.type,
                "action": "downrate",
                "impact": f"-{penalty:.0f}%: {old_pct:.0f}% → {final_pct:.0f}%",
                "reason": rule.text[:100]
            })

    band = _pct_to_band(final_pct)

    debug = {
        "base_coverage": round(base_pct, 1),
        "weakness_penalty": round(weakness_penalty, 1),
        "after_weaknesses": round(adjusted_pct, 1),
        "applied_rules": applied_rules,
        "final_pct": round(final_pct, 1),
        "band": band,
        "weaknesses_count": len(weaknesses)
    }

    return band, final_pct, debug

@st.cache_data(show_spinner=False)
def _extract_obligations(sim_ver: str, process_id: str, practice_id: str, _session_id: str = "") -> list[dict]:
    # _session_id ändert sich pro Assessment-Start
    rag = _cached_rag(sim_ver, process_id, practice_id)
    rules = _format_rule_checklist_cached(sim_ver, process_id, practice_id) or "–"

    # DEBUG: Zeige was ins LLM geht
    logger.info(f"=== RAG CONTENT (first 500 chars) ===")
    logger.info(rag[:500])
    logger.info(f"=== RULES CONTENT (first 500 chars) ===")
    logger.info(rules[:500])

    msgs = [
        {"role": "system", "content": _assessor_system_prompt()},
        {"role": "user", "content": OBLIGATIONS_PROMPT.format(rag=rag, rules=rules)}
    ]
    resp = _chat_with_model_tier(msgs, temperature=0.3)
    raw = (_resp_text(resp) or "").strip()

    # DEBUG
    logger.info(f"=== RAW LLM RESPONSE FOR OBLIGATIONS ===")
    logger.info(raw[:500])  # Erste 500 Zeichen

    try:
        # Robustes Parsing: Entferne Markdown Code-Block-Syntax
        import re

        # Entferne ```json am Anfang und ``` am Ende
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)

        # Suche JSON-Array (falls noch Text drum herum)
        match = re.search(r'\[.*]', cleaned, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            logger.error("No JSON array found in LLM response")
            return []

        # Parse JSON
        data = json.loads(json_str)

        # Validierung (bleibt wie vorher)
        val = []
        for it in data if isinstance(data, list) else []:
            title = (it.get("title") or "").strip()
            maps = [m for m in (it.get("maps_to") or []) if str(m).strip()]
            if not title or not maps:
                continue
            val.append({
                "id": (it.get("id") or title[:40]).strip(),
                "title": title,
                "maps_to": maps,
                "key_phrases": [k for k in (it.get("key_phrases") or []) if str(k).strip()][:6],
                "expected_evidence": [e for e in (it.get("expected_evidence") or []) if str(e).strip()][:6],
                "source_spans": [s for s in (it.get("source_spans") or []) if str(s).strip()][:4],
            })
        return val
    except Exception as e:
        logger.error(f"Failed to parse obligations: {e}")
        logger.error(f"Raw response was: {raw[:200]}")
        return []


def _ensure_obl_coverage(process_id: str, practice_id: str):
    key = f"{process_id}:{practice_id}"
    st.session_state.setdefault("obl_cache", {})
    st.session_state.setdefault("obl_coverage", {})
    if key in st.session_state["obl_coverage"]:
        logger.info(f">>> CACHE HIT for {key} - skipping obligation extraction")
        return
    sim_ver = st.session_state.get("sim_version", "3.1")
    obls = _extract_obligations(
        sim_ver,
        process_id,
        practice_id,
        _session_id=st.session_state.get("assessment_session_id", "")
    )

    # DEBUG
    logger.info(f"=== OBLIGATIONS FOR {practice_id} ===")
    logger.info(f"Extracted: {len(obls)} obligations")
    for i, o in enumerate(obls, 1):
        logger.info(f"  {i}. {o.get('id')}: {o.get('title')}")

    # SORTIERUNG NACH BP-REIHENFOLGE
    obls = _sort_obligations_by_bp_order(obls)

    if not obls:
        obls = [{
            "id": "obl-core",
            "title": _get_practice_description(practice_id) or "Wesentliche Anforderung dieser Practice",
            "maps_to": [practice_id],
            "key_phrases": [],
            "expected_evidence": [],
            "source_spans": []
        }]
    st.session_state["obl_cache"][key] = {o["id"]: o for o in obls}
    st.session_state["obl_coverage"][key] = [{"id": o["id"], "status": "open", "confidence": 0.0} for o in obls]


def _get_obligation_meta(process_id: str, practice_id: str, obl_id: str) -> dict:
    key = f"{process_id}:{practice_id}"
    return (st.session_state.get("obl_cache", {}).get(key, {}) or {}).get(obl_id, {})

def _coverage_ratio(process_id: str, practice_id: str) -> float:
    key = f"{process_id}:{practice_id}"
    arr = st.session_state.get("obl_coverage", {}).get(key, []) or []
    if not arr: return 0.0
    covered = sum(1 for x in arr if x["status"] == "covered")
    return covered / len(arr)


def _percent_to_grade(p: float) -> str:
    if p <= 15.0: return "N"
    if p <= 50.0: return "P"
    if p <= 85.0: return "L"
    return "F"

# --- einfache Parser für "Ceiling"-Regeln (Obergrenze) aus dem Regeltext ---
_CEIL_RX = [
    re.compile(r"not be rated higher than\s+([NPLF])", re.I),
]

def _parse_ceil_from_rules(rule_text: str) -> str | None:
    t = rule_text or ""
    for rx in _CEIL_RX:
        m = rx.search(t)
        if m:
            return m.group(1).upper()
    return None


def _mark_obligation(process_id: str, practice_id: str, obl_id: str, conf: float, snips: list[str]):
    key = f"{process_id}:{practice_id}"
    arr = st.session_state.get("obl_coverage", {}).get(key, []) or []
    for x in arr:
        if x["id"] == obl_id:
            x["status"] = "covered"
            x["confidence"] = float(conf)
            x["snips"] = snips[:3]
            break
    st.session_state["obl_coverage"][key] = arr


def _match_evidence_to_obligations(user_text: str|None, process_id: str, practice_id: str) -> list[dict]:
    idx = st.session_state.get("evidence_index")
    key = f"{process_id}:{practice_id}"
    open_obls = [x["id"] for x in st.session_state.get("obl_coverage", {}).get(key, []) if x["status"]=="open"]
    if not idx or not open_obls:
        return []
    res = []
    for obl_id in open_obls:
        meta = _get_obligation_meta(process_id, practice_id, obl_id)
        if not meta: continue
        q = " / ".join([
            meta["title"],
            " ".join((meta.get("key_phrases") or [])[:4]),
            " ".join((meta.get("expected_evidence") or [])[:3]),
        ])
        if user_text: q += " " + user_text[:300]
        try:
            r = idx.as_query_engine(similarity_top_k=5).query(q)
            txt = str(getattr(r, "response", r) or "")
            snips = [s.strip("-• ").strip() for s in txt.split("\n") if s.strip()][:5]
        except Exception:
            snips = []
        conf = 0.0
        joined = " ".join(snips).lower()
        conf += min(0.4, 0.1 * sum(1 for k in meta.get("key_phrases") or [] if k.lower() in joined))
        conf += min(0.3, 0.15 * sum(1 for e in meta.get("expected_evidence") or [] if e.lower().split()[0] in joined))
        if any(((s or "").lower().split(" ",1)[0] in joined) for s in meta.get("source_spans") or []):
            conf += 0.2
        if re.search(r"(kapitel|abschnitt|section)\s+\d", (user_text or "").lower()) or re.search(r"https?://", (user_text or "")):
            conf = min(1.0, conf + 0.1)
        res.append({"obl_id": obl_id, "snips": snips, "confidence": round(conf,2)})
    return sorted(res, key=lambda x: x["confidence"], reverse=True)


def _targeted_follow_up_for_obligation(process_id: str, practice_id: str, obl_id: str) -> str:
    meta = _get_obligation_meta(process_id, practice_id, obl_id) or {}
    title = meta.get("title") or "dieses Thema"
    ev    = meta.get("expected_evidence") or []
    hint  = f" (z.B. {', '.join(ev[:2])})" if ev else ""
    return f"Zu „{title}“: Gibt es eine konkrete Fundstelle (Dokumenttitel + Kapitel/Abschnitt) oder eine Toolansicht{hint}?"


def _next_assessor_question_obl() -> str:
    """
    Neue Question Generation via Orchestrator
    """
    # Orchestrator aus Session holen oder erstellen
    if "orchestrator" not in st.session_state:
        proc = st.session_state.get("sim_process_id")
        ver = st.session_state.get("sim_version", "3.1")
        st.session_state.orchestrator = AssessmentOrchestrator(proc, ver)

    orchestrator = st.session_state.orchestrator

    # Aktuelle Practice initialisieren
    queue = st.session_state.get("practice_queue", []) or []
    idx = int(st.session_state.get("practice_idx", 0) or 0)

    if idx >= len(queue):
        return ""

    cur_id = queue[idx]["id"]

    # Scope sicherstellen
    _ensure_evidence_scope(orchestrator.process_id, st.session_state.get("sim_version"))

    # Practice initialisieren (lädt Obligations)
    orchestrator.initialize_practice(cur_id)

    # Frage generieren
    if orchestrator.state.current_obligation:
        question = generate_question_with_gap_analysis(
            orchestrator.state,
            orchestrator.memory,
            orchestrator.state.current_obligation
        )

        # In Memory speichern
        orchestrator.memory.add_turn("assistant", question, {"type": "question"})

        # UI-State aktualisieren
        st.session_state["current_aspect"] = orchestrator.state.current_obligation.get("title", "")
        st.session_state["current_aspect_maps_to"] = orchestrator.state.current_obligation.get("maps_to", [])

        return question
    else:
        return "Keine offenen Aspekte mehr für diese Practice."


def _evidence_snippets_for(question: str, top_k: int = 4) -> list[str]:
    """Holt knappe Auszüge aus dem Evidence-Index passend zur Frage."""
    try:
        idx = st.session_state.get("evidence_index")
        if not idx:
            return []
        qe = idx.as_query_engine(similarity_top_k=top_k)
        resp = qe.query(question)
        # LlamaIndex Response -> Textauszug
        txt = str(getattr(resp, "response", resp))
        # in 2–4 kurze Bullet-Snips schneiden
        snips = [s.strip() for s in txt.split("\n") if s.strip()][:top_k]
        return snips
    except Exception:
        return []


def _get_practice_description(practice_id: str) -> str:
    """
    Liefert eine kurze, sprechbare Beschreibungstextzeile zu einer Practice-ID
    (z. B. 'SWE.2.BP1'), basierend auf den bereits geladenen DOCS/ID_MAP für
    die aktuell gewählte Version (st.session_state['sim_version']).
    """
    if not practice_id:
        return ""

    # Version aus dem aktuellen Sim-State ableiten
    sim_vers = (st.session_state.get("sim_version") or "3.1").strip()

    # Arbeitsmengen zur Version wählen
    if sim_vers == "4.0":
        docs = DOCS_V40
        id_map = ID_MAP_V40
    else:
        docs = DOCS_V31
        id_map = ID_MAP_V31

    # Kandidaten-Indizes zur Practice-ID suchen (case-insensitive + raw)
    idxs = (
        (id_map.get(practice_id)
         or id_map.get(practice_id.lower())
         or id_map.get(practice_id.upper())
         or [])
    )

    # Durch die Dokumente laufen und eine kurze Beschreibung extrahieren
    for idx in idxs:
        if 0 <= idx < len(docs):
            t = (docs[idx].text or "").strip()
            if not t:
                continue
            # Heuristik: erste Zeile mit '—' enthält oft eine Kurzbeschreibung hinter dem Gedankenstrich
            lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
            for ln in lines[:5]:  # nur die ersten paar Zeilen prüfen
                if "—" in ln:
                    desc = ln.split("—", 1)[1].strip()
                    return desc[:120]  # knapp halten
            # Fallback: nimm die erste nicht-leere Zeile, entferne die ID vorn, falls vorhanden
            first = lines[0]
            # gängige ID-Präfixe rauslösen
            first = re.sub(r"^[A-Z]{3}\.\d+(?:\.BP\d+)?\s*[—\-:]\s*", "", first)
            return first[:120]

    return ""


def _generate_assessment_summary() -> str:
    """Generiert Assessment-Zusammenfassung"""
    process_id = st.session_state.get("last_process_for_eval", "")
    scores = st.session_state.get("simulation_scores", {}).get(process_id, {}).get("bp_gp", {})

    if not scores:
        return "## Assessment abgeschlossen\n\nKeine Bewertungen vorhanden."

    lines = [f"## 📊 Assessment-Zusammenfassung: {process_id}\n"]

    for practice_id, evaluation in scores.items():
        # NPLF extrahieren
        import re
        match = re.search(r'\*\*([NPLF])\*\*', evaluation)
        rating = match.group(1) if match else "?"
        lines.append(f"**{practice_id}**: {rating}")

    return "\n".join(lines)


# --- gecachte Variante (version-aware) ---
@lru_cache(maxsize=256)
def _format_rule_checklist_cached(_sim_vers: str, process_id: str, practice_id: str, max_len: int = 220) -> str:
    """
    Version-aware Cache der RL-Checkliste.
    sim_version in den Key aufnehmen, weil die zugrundeliegenden DOCS/Maps je Version wechseln.
    """
    # DEBUG
    logger.info(f"=== _format_rule_checklist_cached called ===")
    logger.info(f"sim_vers: {_sim_vers}, process: {process_id}, practice: {practice_id}")

    # wir rufen die un-cached Variante auf; der Cache-Key enthält sim_version
    result = _format_rule_checklist(process_id, practice_id, max_len=max_len)

    # DEBUG
    logger.info(f"Result length: {len(result) if result else 0}")
    logger.info(f"Result preview: {result[:200] if result else 'EMPTY/NONE'}")

    return result

@lru_cache(maxsize=1024)
def _cached_rag(_sim_vers: str, process_id: str, practice_id: str | None, k_each: int = 3) -> str:
    """
    Version-aware Cache für den deterministischen RAG-Kontext.
    _sim_vers fließt als Teil des Cache-Keys ein (die zugrunde liegenden DOCS/Maps hängen von der Version ab).
    """
    return _format_rag_context(process_id, practice_id, k_each=k_each)



# ================================================================================
# 5. STREAMLIT SETUP & CONFIGURATION
# ================================================================================

# Lädt Umgebungsvariablen aus `.env`-Datei für die lokale Entwicklung.
load_dotenv()

# Konfiguriert die Streamlit-Seite mit Titel, Icon und Layout.
st.set_page_config(page_title="Mr. Spicy - AI Chatbot", page_icon="🤖", layout="centered")


def show_branding():
    # mehrere sinnvolle Suchpfade probieren
    candidates = [
        Path(__file__).parent / "images" / "spicy_robot.png",
        Path.cwd() / "images" / "spicy_robot.png",
        Path("images/spicy_robot.png"),
    ]
    for p in candidates:
        if p.exists():
            st.image(str(p), width=500)
            return
    # Fallback: nur Texttitel (wenn das Bild fehlt)
    st.markdown("### Mr. Spicy - AI Chatbot")


# Erstellt ein zweispaltiges Layout für den Header der Anwendung.
c1, c2 = st.columns([3, 5], vertical_alignment="center")
with c1:
    show_branding()
with c2:
    st.title("Mr. Spicy - AI Chatbot")

# API-Schlüssel aus den Streamlit Secrets laden.
# Für lokale Entwicklung werden diese in der Datei `.streamlit/secrets.toml` gespeichert.
# OpenAI-Key setzen (für LLM & Embeddings) - wird vom LLM (für die Textgenerierung) und vom Embedding-Modell (für die Vektorisierung) benötigt.
if "OPENAI_API_KEY" not in os.environ:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("OPENAI_API_KEY fehlt (weder in Environment noch Streamlit Secrets)")
        st.stop()

# Llama-Key setzen (für LlamaParse zum Parsen von PDFs in Markdown-Format)
if "LLAMA_CLOUD_API_KEY" not in os.environ:
    if "LLAMA_CLOUD_API_KEY" in st.secrets:
        os.environ["LLAMA_CLOUD_API_KEY"] = st.secrets["LLAMA_CLOUD_API_KEY"]


def ensure_state():
    """
    Stellt sicher, dass der Chat-Verlauf (`st.session_state.messages`) existiert.
    """
    # 1) Immer eine Liste sicherstellen – aber ohne Intro befüllen
    if "messages" not in st.session_state or not isinstance(st.session_state.messages, list):
        st.session_state.messages = []

    # 2) Intro NUR im Chatmodus (nicht in der Simulation) und nur wenn noch leer
    if not st.session_state.get("simulation_active") and not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Hallo, ich bin Mr. SPICY!\n\n"
                "Ich gebe dir fundierte Antworten zu allen Inhalten rund um Qualität & ASPICE.\n\n"
                "Ich kann Prozess-Assessments simulieren - mit Fragen wie im echten Assessment.\n\n"
                "Ich kann deine Dokumente auf die Einhaltung von Q-Anforderungen prüfen.\n\n"
                "Ich kann dir Checklisten und Dokument-Templates für alle wichtigen Prozessschritte erstellen."
            )
        })

# Stellt sicher, dass der Chatverlauf initialisiert ist.
ensure_state()

# Injiziert benutzerdefiniertes CSS, um die Darstellung von Überschriften in Chat-Nachrichten zu vereinheitlichen.
st.markdown("""<style>
/* Headings in Chat-Nachrichten vereinheitlichen */
[data-testid="stChatMessageContent"] h1 { font-size: 1.20rem; }
[data-testid="stChatMessageContent"] h2 { font-size: 1.18rem; }
[data-testid="stChatMessageContent"] h3 { font-size: 1.16rem; }
[data-testid="stChatMessageContent"] h4 { font-size: 1.14rem; }
[data-testid="stChatMessageContent"] h5 { font-size: 1.12rem; }
[data-testid="stChatMessageContent"] h6 { font-size: 1.10rem; }
/* Optional: Abstand leicht harmonisieren */
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3,
[data-testid="stChatMessageContent"] h4,
[data-testid="stChatMessageContent"] h5,
[data-testid="stChatMessageContent"] h6 { margin: .5rem 0 .25rem; }
</style>""", unsafe_allow_html=True)

# --- Simulation: Defaults sicherstellen (idempotent) ---
for k, v in [
    ("simulation_active", False),
    ("last_assessor_question", None),
    ("practice_queue", []),
    ("practice_idx", 0),
]:
    st.session_state.setdefault(k, v)

# === Helpers used by sidebar (moved up) ===
def _init_sim_state():
    st.session_state.simulation_active = False
    st.session_state.simulation_round = 0
    st.session_state.simulation_cfg = {
        "version": "3.1",
        "capability_level": "CL 1",
        "processes": [],
    }
    st.session_state.simulation_scores = {}
    st.session_state.simulation_history = []
    st.session_state.last_assessor_question = None
    st.session_state.last_process_for_eval = None
    # Stil + getrennte Temperaturen initialisieren
    if "sim_style" not in st.session_state:
        st.session_state.sim_style = "rule-oriented"  # steuert NUR die Frage-Kreativität
    if "sim_temp_questions" not in st.session_state:
        st.session_state.sim_temp_questions = 0.3  # Startwert; wird unten vom Stil überschrieben
    # Bewertung immer fix (nicht einstellbar)
    st.session_state.setdefault("sim_temp_evaluation", 0.3)
    # Evidenz-Index NICHT zurücksetzen
    st.session_state.setdefault("evidence_index", None)
    st.session_state.setdefault("evidence_doc_hashes", set())  # Hash-Set für Dedupe
    # optional: initialer Scope
    st.session_state.setdefault("evidence_scope", (
    st.session_state.get("sim_version", "3.1"), st.session_state.get("sim_process_id", "")))


def _ingest_evidence_files(files) -> int:
    """
    Liest hochgeladene Dateien ein und hängt sie an den Evidence-Index an.
    Erstellt bei Bedarf einen neuen Index. Gibt die Anzahl neu hinzugefügter
    Dokumente zurück.
    """
    if not files:
        return 0

    tmp_dir = Path(tempfile.mkdtemp(prefix="evidence_"))
    new_docs = []

    for f in files:
        p = tmp_dir / f.name
        p.write_bytes(f.read())
        try:
            # PDFs bevorzugt via LlamaParse (falls Key vorhanden)
            if p.suffix.lower() == ".pdf" and (os.getenv("LLAMA_CLOUD_API_KEY") or st.secrets.get("LLAMA_CLOUD_API_KEY", None)):
                parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY") or st.secrets.get("LLAMA_CLOUD_API_KEY"))
                for d in parser.load_data(str(p)):
                    md = dict(d.metadata or {}); md.setdefault("file_name", p.name)
                    new_docs.append(Document(text=d.text, metadata=md))
            else:
                # Generischer Reader
                for d in SimpleDirectoryReader(input_files=[str(p)]).load_data():
                    md = dict(d.metadata or {}); md.setdefault("file_name", p.name)
                    new_docs.append(Document(text=d.text, metadata=md))
        except Exception as e:
            st.warning(f"Evidenz konnte nicht geladen werden: {p.name} → {e}")

    if not new_docs:
        return 0

    # --- inhaltliche Dedupe über Text-Hash (auch gegen bestehendes) ---
    st.session_state.setdefault("evidence_doc_hashes", set())
    deduped_docs = []
    for d in new_docs:
        # Text holen (robust, je nach LlamaIndex-Version)
        content = d.get_content() if hasattr(d, "get_content") else getattr(d, "text", "")
        norm = " ".join((content or "").split())
        h = hashlib.sha1(norm[:20000].encode("utf-8")).hexdigest()
        if h in st.session_state.evidence_doc_hashes:
            continue  # bereits vorhanden → überspringen
        md = dict(d.metadata or {})
        md.setdefault("content_sha1", h)
        d.metadata = md
        deduped_docs.append(d)
        st.session_state.evidence_doc_hashes.add(h)

    if not deduped_docs:
        return 0

    # Index neu anlegen oder anhängen
    if not st.session_state.get("evidence_index"):
        st.session_state.evidence_index = VectorStoreIndex.from_documents(deduped_docs)
        return len(deduped_docs)
    else:
        added = 0
        for d in deduped_docs:
            try:
                st.session_state.evidence_index.insert(d)
                added += 1
            except Exception as e:
                st.warning(f"Evidenz konnte nicht hinzugefügt werden: {d.metadata.get('file_name', '?')} → {e}")
        return added


def _clear_evidence():
    """Alle Evidenz-Artefakte für das aktuelle Assessment entfernen."""
    st.session_state["evidence_index"] = None
    st.session_state["evidence_doc_hashes"] = set()
    st.session_state.pop("last_evidence_upload_sig", None)
    st.session_state.pop("evidence_uploader_nonce", None)

def _ensure_evidence_scope(process_id: str | None, sim_version: str | None):
    """
    Wenn Prozess oder Version wechselt, wird die Evidenz isoliert:
    - Bei Scope-Wechsel: Evidenz leeren und neuen Scope setzen.
    """
    cur_scope = st.session_state.get("evidence_scope")  # Tuple (ver, proc)
    new_scope = ((sim_version or "3.1"), (process_id or ""))
    if cur_scope != new_scope:
        _clear_evidence()
        st.session_state["evidence_scope"] = new_scope

def _end_assessment(clear_evidence: bool = True):
    """Assessment sauber beenden (State + optional Evidenz leeren)."""
    keys = [
        "practice_queue", "practice_idx",
        "last_assessor_question", "current_aspect", "current_aspect_maps_to",
        "question_meta", "obl_cache", "obl_coverage"
    ]
    for k in keys:
        st.session_state.pop(k, None)
    if clear_evidence:
        _clear_evidence()
    st.session_state.pop("evidence_scope", None)
    st.session_state["assessment_ended_at"] = time.time()


def fetch_url_content(url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Holt Inhalt von einer URL.
    Returns: (content, error_message)
    """
    try:
        parsed = urlparse(url)
        if not parsed.scheme in ['http', 'https']:
            return None, "Nur HTTP/HTTPS URLs unterstützt"

        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; MrSpicy/1.0)'
        })
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()

        # Text/HTML/JSON verarbeiten
        if any(t in content_type for t in ['text', 'json', 'html', 'xml']):
            content = response.text[:100000]  # Max 100k Zeichen
            return content, None
        # PDFs als Binary markieren
        elif 'pdf' in content_type:
            return None, "PDF-Dokumente bitte direkt hochladen"
        else:
            return None, f"Dateityp nicht unterstützt: {content_type}"

    except requests.Timeout:
        return None, "Timeout beim Abrufen"
    except requests.RequestException as e:
        return None, f"Fehler: {str(e)}"
    except Exception as e:
        return None, f"Unerwarteter Fehler: {str(e)}"


def process_urls_in_message(text: str) -> list[tuple[str, str]]:
    """
    Extrahiert URLs aus Text und holt deren Inhalt.
    Returns: Liste von (url, content) Tupeln
    """
    import re
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = list(set(re.findall(url_pattern, text)))[:5]  # Max 5 URLs, keine Duplikate

    results = []
    for url in urls:
        content, error = fetch_url_content(url)
        if content:
            results.append((url, content))
        elif error:
            results.append((url, f"[Fehler: {error}]"))

    return results


BP_RX = re.compile(r"\b([A-Z]{2,}\.\d+\.)?(BP\d{1,2})\b", re.IGNORECASE)
GP_RX = re.compile(r"\b(GP\d{1,2})\b", re.IGNORECASE)

def _get_process_doc(process_id: str, version: str):
    docs = DOCS_V31 if version == "3.1" else DOCS_V40
    for d in (docs or []):
        md = d.metadata or {}
        if (md.get("type") == "aspice_process") and ((md.get("display_id") or "").strip().lower() == process_id.lower()):
            return d
    return None


def _get_process_purpose(process_id: str, version: str | None = None) -> str:
    """
    Liefert den 'Purpose:'-Text des Prozesses (aus dem JSON-Ingest),
    z.B. 'The purpose of the ... is to ...'
    """
    ver = version or st.session_state.get("sim_version", "3.1")
    d = _get_process_doc(process_id, ver)
    if not d:
        return ""
    body = d.text or ""
    # Der Parser schreibt 'Purpose: ...' in die Prozess-Docs
    m = re.search(r"(?im)^\s*purpose\s*:\s*(.+)$", body)
    return (m.group(1).strip() if m else "")


def _extract_practices(process_id: str, version: str, capability_level: int | str) -> list[dict]:
    """Queue nur aus Mapping: erst alle BPs (in Dokumentreihenfolge), bei CL2 danach alle GPs."""
    # Versionierte Arbeitsmengen wählen
    if str(version).strip() == "4.0":
        DOCS_LOC, ID_MAP_LOC, PARENT_MAP_LOC = DOCS_V40, ID_MAP_V40, PARENT_MAP_V40
    else:
        DOCS_LOC, ID_MAP_LOC, PARENT_MAP_LOC = DOCS_V31, ID_MAP_V31, PARENT_MAP_V31

    # Prozess-Index ermitteln (display_id → Index) über DISPLAY_ID_INDEX
    pid_key = (process_id or "").strip().lower()
    idx_map = DISPLAY_ID_INDEX_V40 if str(version).strip() == "4.0" else DISPLAY_ID_INDEX_V31
    proc_idx = idx_map.get(pid_key)
    if proc_idx is None:
        # defensive: tolerante Suche (case-insensitive)
        for did, ix in (idx_map or {}).items():
            if (did or "").strip().lower() == pid_key:
                proc_idx = ix
                break
    if proc_idx is None:
        return []

    # Direkte Kinder des Prozesses (IDs in PARENT_MAP sind display_ids)
    # → Wir sammeln die Indizes und filtern nach Typ.
    proc_did = (DOCS_LOC[proc_idx].metadata.get("display_id") or "").strip().lower()
    child_idxs = list(PARENT_MAP_LOC.get(proc_did, []))

    # Reihenfolge beibehalten, Duplikate vermeiden
    seen: set[str] = set()
    bps: list[str] = []
    gps: list[str] = []

    # CL normalisieren
    try:
        cl = int(capability_level)
    except Exception:
        cl = 1

    for i in child_idxs:
        d = DOCS_LOC[i]
        md = d.metadata or {}
        typ = md.get("type")
        did = (md.get("display_id") or md.get("id") or "").strip().upper()
        if not did or did in seen:
            continue
        if typ == "aspice_base_practice":
            bps.append(did)
            seen.add(did)
        elif typ == "aspice_generic_practice" and cl >= 2:
            gps.append(did)
            seen.add(did)

    return [{"type": "BP", "id": x} for x in bps] + [{"type": "GP", "id": x} for x in gps]


def _children_of(process_id: str, version: str, type_filter: set[str]) -> list[tuple[str, int, dict]]:
    """
    Liefert eine Liste der Kinder-Dokumente (display_id, doc_index, metadata)
    für einen Prozess – ausschließlich via PARENT_MAP & DOCS, ohne Regex.
    """
    if version == "3.1":
        docs = DOCS_V31; pmap = PARENT_MAP_V31
    else:
        docs = DOCS_V40; pmap = PARENT_MAP_V40

    pid = (process_id or "").strip()
    out: list[tuple[str, int, dict]] = []
    for i in (pmap.get(pid) or pmap.get(pid.upper()) or pmap.get(pid.lower()) or []):
        d = docs[i]
        md = d.metadata or {}
        t  = (md.get("type") or "")
        if t in type_filter:
            did = (md.get("display_id") or md.get("id") or "").strip()
            if did:
                out.append((did, i, md))
    return out

def _generic_practice_ids(version: str, capability_level: str) -> list[str]:
    """
    Globaler Zugriff auf GPs (ohne Prozessbezug) aus DOCS_{version}, nur Metadaten.
    """
    if version == "3.1":
        docs = DOCS_V31
    else:
        docs = DOCS_V40

    try:
        cl_int = int(str(capability_level).strip()[:1] or "1")
    except Exception:
        cl_int = 1

    if cl_int < 2 or not docs:
        return []

    out, seen = [], set()
    for d in docs:
        md = (d.metadata or {})
        if md.get("type") == "aspice_generic_practice":
            did = (md.get("display_id") or md.get("id") or "").strip()
            gp_cl = str(md.get("capability_level") or md.get("cl") or "").strip()
            if gp_cl and gp_cl != "2":
                continue
            if did and did not in seen:
                seen.add(did)
                out.append(did)
    return out


# =====================================================================
# 5.x ASSESSMENT SIMULATION: PROMPTS, STILE & HELPERS
# =====================================================================
STYLE_HEADERS = {
    "rule-oriented": (
        "Du bist ein präziser ASPICE-Assessor.\n\n"
        "**Arbeitsweise:** Orientiere dich eng am Regelwerk (BPs, GPs, Rules, Recommendations).\n"
        "**Fragen:** Direkt aus dem Kontext abgeleitet, messbar, regelbasiert.\n"
        "**Beispiele:**\n"
        "- 'Welches Dokument erfüllt Anforderung X aus BP Y?'\n"
        "- 'Wo ist der Nachweis für Rule Z dokumentiert?'\n\n"
        "Bleibe nah am expliziten Regelwerk."
    ),
    "balanced": (
        "Du bist ein ganzheitlicher Quality Manager.\n\n"
        "**Arbeitsweise:** Regelwerk als Basis, aber denke auch in Zusammenhängen.\n"
        "**Fragen:** Verbinde Requirements mit Prozess-Realität, erkunde Schnittstellen.\n"
        "**Beispiele:**\n"
        "- 'Wie stellt Dokument X sicher, dass Anforderung Y prozessual umgesetzt wird?'\n"
        "- 'Welche Abhängigkeiten bestehen zwischen A und B?'\n\n"
        "Regelwerk + Kontext."
    ),
    "challenging": (
        "Du bist ein kritischer Assessor (Devil's Advocate).\n\n"
        "**Arbeitsweise:** Regelwerk als Startpunkt, aber denke kreativ über Grenzen hinaus.\n"
        "**Fragen:** Grenzfälle, Risiken, 'Was-wäre-wenn'-Szenarien.\n"
        "**Beispiele:**\n"
        "- 'Was passiert wenn Prozess X unter Zeitdruck kompromittiert wird?'\n"
        "- 'Wie verhindern Sie Umgehungen von Regel Y?'\n\n"
        "Regelwerk + kreatives Hinterfragen."
    )
}

ASSESSOR_SYSTEM_BASE = (
    "Sprich in der Sprache der letzten Nutzernachricht, professionell aber freundlich.\n"
    "Rolle: Erfahrene:r Automotive-SPICE Assessor:in.\n"
    "Vorgehen: Stelle präzise, prüfende Fragen und fordere **konkrete Nachweise** "
    "(Dokumentname/ID, Kapitel/Abschnitt, Tool-Screenshot/Link). "
    "Bohre nach, wenn Antworten allgemein bleiben.\n"
    "Frage systematisch BP/Outcomes/OWP (CL1) sowie — falls CL≥2 — auch die relevanten GP ab."
)

EVAL_PROMPT_TEMPLATE = """
Du bewertest eine Antwort in einer ASPICE-Assessment-Simulation.

Rahmen:
- Version: {version}; Capability Level: {capability_level}; Prozess: {process_id}

**Version-Awareness**
- Du bewertest **ausschließlich nach Version {sim_version}**.
- Nutze nur Inhalte aus dieser Version.
- Achte auf versionsspezifische Terminologie (z. B. **3.1: "evidence"** vs. **4.0: "record"**) und bewerte konsistent in dieser Terminologie.

Kontext (Soll & ggf. User-Evidenz):
{rag_context}

Rating Rules (verbindlich – strikt befolgen, wenn zutreffend):
{rule_checklist}

Fokussiere (falls relevant) diese Normstellen:
{focus_ids}

Nutzer-Antwort:
\"\"\"{user_answer}\"\"\"

**WICHTIG: Bewerte NUR die aktuell abgefragte Practice, nicht den gesamten Prozess.**

Gib ein kompaktes Markdown-Resultat:
1) Kurzfeedback zur Antwortqualität (max. 3 Sätze).
2) Objektive Evidenz-Zusammenfassung (welche Nachweise wurden konkret genannt/gefunden).
3) Lücken (nur evidenzbasiert; keine allgemeinen Aussagen).
4) NPLF-Bewertung für die aktuelle Practice:
   - **Nur die aktuell abgefragte Practice**: **N | P | L | F** + 1-Satz-Begründung.
   - **KEINE** Bewertung anderer BPs/GPs des Prozesses.
   - **KEINE** Gesamtbewertung des Prozesses.

WICHTIG – Wenn eine Schwäche/Lücke benannt wird ist Folgendes zu beachten:
"A weakness is a context-specific comprehensive and comprehensible explanation of the process risk. It must deliver sufficient understanding for the assessees to understand the context-specific process risk as a starting point for deriving improvements. The weakness must be substantiated by objective evidence gathered during the assessment. General statements or inverted restatements of BP/GP text are not acceptable. A rating rule cannot replace a comprehensive weakness statement; it may only support it."

**Wende die Rating Rules so an, wie sie formuliert sind**:
- „shall/must be downrated“ ⇒ entsprechende Indikatoren **downraten**.
- „shall/must not be downrated“ ⇒ **kein Downrating** aus diesem Grund.
- „shall/must not be rated higher than …“ ⇒ **Obergrenze** beachten.

Bewerte nur Inhalte aus dem gegebenen Kontext (Soll/BP/GP/WP und ggf. User-Evidenz).
""".strip()

def _last_user_language() -> str:
    msgs = st.session_state.get("messages", [])
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    text = (user_msgs[-1]["content"] if user_msgs else "") or ""
    if _LANG_DE_RX.search(text):
        return "Deutsch"
    if _LANG_EN_RX.search(text):
        return "English"
    return "Deutsch"


def _assessor_system_prompt() -> str:
    style = STYLE_HEADERS.get(st.session_state.get("sim_style", "rule-oriented"), "")
    return f"{style}\n\n{ASSESSOR_SYSTEM_BASE}\n(Hinweis: Sprache = {_last_user_language()})"


def _ensure_practice_aspects(process_id: str, practice_id: str):
    """
    State-of-the-art: Obligations = Aspekte. LLM-Fallback mit OBLIGATIONS_PROMPT.
    """
    key = f"{process_id}:{practice_id}"
    pa = st.session_state.get("practice_aspects") or {}
    if key in pa and pa[key].get("open"):
        return

    _ensure_obl_coverage(process_id, practice_id)
    obls = st.session_state["obl_cache"][key].values()
    aspects = [o["title"] for o in obls]
    meta = {o["title"]: {"maps_to": o["maps_to"]} for o in obls}

    if not aspects:
        # LLM-Fallback mit OBLIGATIONS_PROMPT
        _sim_vers = st.session_state.get("sim_version", "3.1")
        rag = _cached_rag(_sim_vers, process_id, practice_id)
        rules = _format_rule_checklist_cached(_sim_vers, process_id, practice_id) or ""

        msgs = [
            {"role": "system", "content": _assessor_system_prompt()},
            {"role": "user", "content": OBLIGATIONS_PROMPT.format(rag=rag, rules=rules)}
        ]
        resp = _chat_with_model_tier(msgs, temperature=0.3, tier=ModelTier.SMART)
        raw = _resp_text(resp).strip()

        try:
            # Parse JSON
            json_match = re.search(r'\[.*]', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                aspects = [o.get("title") for o in data if o.get("title")]
                meta = {o["title"]: {"maps_to": o.get("maps_to", [])} for o in data if o.get("title")}
        except:
            pass

        # Letzter Fallback
        if not aspects:
            human = _get_practice_description(practice_id) or "Wesentliche Anforderung dieser Practice"
            aspects = [human]
            meta = {human: {"maps_to": [practice_id]}}

    st.session_state.setdefault("practice_aspects", {})
    st.session_state["practice_aspects"][key] = {"open": aspects, "meta": meta}


def _append_evidence(lines: list[str], meta: dict | None, snippet: str, *, max_len: int = 600):
    """
    Hängt einen strukturierten Evidenz-Eintrag an (JSON-String).
    UI kann daraus klickbare Links rendern.
    """
    s = " ".join((snippet or "").split())
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"

    m = meta or {}
    src = (m.get("file_name")  # Haupt-Key vom Upload
           or m.get("source")  # LlamaIndex Standard
           or m.get("source_name")  # Fallback 1
           or m.get("display_name")  # Fallback 2
           or m.get("doc_name")  # Fallback 3
           or m.get("title")  # Fallback 4
           or "Unbekannte Quelle")
    url = (m.get("url") or m.get("link") or "")

    payload = {
        "type": "EVIDENZ",
        "source": src,
        "url": url,
        "snippet": s,
    }
    lines.append(json.dumps(payload, ensure_ascii=False))


def _format_rag_context(process_id: str, practice_id: str | None = None, k_each: int = 3) -> str:
    """
    Baut einen deterministischen RAG-Kontext ausschließlich aus:
      - der exakt adressierten Practice (z. B. 'SUP.1.BP1')
      - deren direkten Kindern (Outcomes, Rules, Output Work Products)
    plus optional knappen Evidenz-Schnipseln.
    """
    use_v31 = st.session_state.get("sim_version", "3.1") == "3.1"

    DOCS = DOCS_V31 if use_v31 else DOCS_V40
    ID_MAP = ID_MAP_V31 if use_v31 else ID_MAP_V40
    PARENT_MAP = PARENT_MAP_V31 if use_v31 else PARENT_MAP_V40

    # practice_id ist jetzt optional
    pid = (practice_id or "").strip()
    if pid:
        root_id = f"{process_id.strip()}.{pid}" if not pid.upper().startswith(process_id.upper()) else pid
        wanted = _collect_practice_and_children_indices(root_id, ID_MAP, PARENT_MAP)
    else:
        # Prozess-only: kein BP-Kontext → nur Prozessknoten verwenden
        root_id = process_id.strip()
        wanted = ID_MAP.get(root_id, [])

    lines, size, max_chars = [], 0, 6000
    for i in wanted:
        if i < 0 or i >= len(DOCS):
            continue
        t = (DOCS[i].text or "").strip()
        if not t:
            continue
        s = " ".join(t.split())
        if len(s) > 600:
            s = s[:597].rstrip() + "…"
        if size + len(s) > max_chars:
            s = s[: max(0, max_chars - size)]
        lines.append(f"- [SOLL] {s}")
        size += len(s)
        if size >= max_chars:
            break

    # optionale Evidenz mit Relevanz-Suche
    evid = st.session_state.get("evidence_index")
    if evid:
        try:
            r2 = evid.as_retriever(similarity_top_k=k_each * 2)

            # Dynamische Query-Erstellung aus SOLL-Kontext
            evidence_query = root_id  # Default: Practice-ID
            if lines:
                # Erste SOLL-Zeile als Kontext-Quelle nutzen
                first_soll = next((line for line in lines if line.startswith("- [SOLL]")), "")
                if first_soll:
                    # Practice-ID und Trennzeichen entfernen
                    clean_soll = re.sub(r'^- \[SOLL]\s*[A-Z0-9.]+\s*[—–-]\s*', '', first_soll)
                    # Relevante Wörter extrahieren (mindestens 4 Zeichen, keine Stoppwörter)
                    key_words = [w for w in re.findall(r'\b\w{4,}\b', clean_soll)
                                 if w.lower() not in {'that', 'with', 'from', 'they', 'have', 'this', 'will', 'been',
                                                      'were', 'than'}][:4]
                    if key_words:
                        evidence_query = f"{root_id} {' '.join(key_words)}"

            for node in r2.retrieve(evidence_query)[:k_each]:
                # Robuste Text-Extraktion (aus Fix 3)
                if hasattr(node, 'node'):
                    actual_node = node.node
                    t = actual_node.get_content() if hasattr(actual_node, 'get_content') else getattr(actual_node,
                                                                                                      'text', '')
                    meta = getattr(actual_node, 'metadata', {})
                else:
                    t = node.get_content() if hasattr(node, 'get_content') else getattr(node, 'text', '')
                    meta = getattr(node, 'metadata', {})

                t = (t or "").strip()
                if not t:
                    continue
                s = " ".join(t.split())
                if len(s) > 300:
                    s = s[:297].rstrip() + "…"

                # statt Plain-Text: strukturierte Evidenz mit Quelle/URL
                _append_evidence(lines, meta or {}, s)
        except Exception:
            pass

    # Fallback: falls keine Kinder/Evidenz – liefere wenigstens den Root-Text
    if not lines:
        for i in ID_MAP.get(root_id, []):
            if 0 <= i < len(DOCS):
                t = (DOCS[i].text or "").strip()
                if t:
                    s = " ".join(t.split())
                    if len(s) > 600:
                        s = s[:597].rstrip() + "…"
                    lines.append(f"- [SOLL] {s}")
                    break
        return "\n".join(lines) if lines else "n/a"

    # >>> Finales Return für den Normalpfad <<<
    return "\n".join(lines) if lines else "n/a"


def _render_chat_block(text: str):
    """Rendert Textzeilen; JSON-EVIDENZ-Zeilen werden klickbar angezeigt."""
    import json
    for row in (text or "").splitlines():
        r = row.strip()
        if r.startswith("{") and r.endswith("}"):
            try:
                obj = json.loads(r)
                if obj.get("type") == "EVIDENZ":
                    src = obj.get("source") or "Evidenz"
                    url = obj.get("url") or ""
                    snip = obj.get("snippet") or ""
                    if url:
                        st.markdown(f"📄 **[{src}]({url})**  \n_{snip}_")
                    else:
                        st.markdown(f"📄 **{src}**  \n_{snip}_")
                    continue
            except Exception:
                pass
        # Fallback: normale Zeile
        st.markdown(r)


# --- JSON-Schema-Validatoren für LLM-Outputs ---

def _validate_aspects_payload(raw: str):
    """
    Erwartet ein JSON-Array von Objekten:
      [{"aspect": str, "maps_to": [str,...]?}, ...]
    Rückgabe:
      (aspects_list, meta_map)
        aspects_list: Liste[str]  (nur Aspekt-Texte)
        meta_map:     Dict[str, Dict]  (per Aspekt-Text die Zusatzinfos)
    """
    if not isinstance(raw, str):
        return None, None

    s = raw.strip()

    # 1) Codefences entfernen (```json ... ``` oder ``` ... ```)
    if s.startswith("```"):
        s = s.strip("`")
        # Zeile mit 'json' am Anfang entfernen, falls vorhanden
        s = "\n".join(line for line in s.splitlines() if not line.strip().lower().startswith("json")).strip()

    # 2) Versuchen, den JSON-Array-Teil zu isolieren
    import re, json
    m = re.search(r"\[.*]", s, flags=re.S)
    if m:
        s = m.group(0)

    # 3) Laden versuchen
    try:
        data = json.loads(s)
    except Exception:
        return None, None

    # 4) Optionaler Sonderfall: ein Objekt mit "items"/"aspects"
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            data = data["items"]
        elif "aspects" in data and isinstance(data["aspects"], list):
            data = data["aspects"]
        else:
            return None, None

    if not isinstance(data, list):
        return None, None

    aspects, meta = [], {}
    for item in data:
        if not isinstance(item, dict):
            # Auch reine Strings als Aspekte akzeptieren
            if isinstance(item, str) and item.strip():
                a = item.strip()
                aspects.append(a)
                meta[a] = {"maps_to": []}
            continue

        # keys case-insensitive
        lower = {str(k).lower(): v for k, v in item.items()}
        a = lower.get("aspect")
        if not isinstance(a, str) or not a.strip():
            continue
        a = a.strip()

        # maps_to erlauben: Liste ODER einzelner String
        mts = lower.get("maps_to", [])
        if isinstance(mts, str) and mts.strip():
            mts = [mts.strip()]
        elif isinstance(mts, list):
            mts = [str(x).strip() for x in mts if str(x).strip()]
        else:
            mts = []

        meta[a] = {"maps_to": mts}
        aspects.append(a)

    # Duplikate entfernen (Reihenfolge stabil)
    seen, uniq = set(), []
    for a in aspects:
        if a not in seen:
            uniq.append(a); seen.add(a)

    if not uniq:
        return None, None
    return uniq, meta



def _validate_question_payload(raw: str):
    """
    Erwartet ein JSON-Objekt:
      {"question": str, "asks_evidence": [str,...]?, "maps_to": [str,...]?}
    Rückgabe:
      (question_text, meta_dict)  oder (None, None) bei Fehler.
    """
    try:
        obj = json.loads(raw)
    except Exception:
        return None, None
    if not isinstance(obj, dict):
        return None, None
    q = obj.get("question")
    if not isinstance(q, str) or not q.strip():
        return None, None
    q = q.strip()
    asks = obj.get("asks_evidence") if isinstance(obj.get("asks_evidence"), list) else []
    maps = obj.get("maps_to") if isinstance(obj.get("maps_to"), list) else []
    return q, {"asks_evidence": [str(x) for x in asks], "maps_to": [str(x) for x in maps]}


def _chat_with_temp(messages, temperature: float):
    model_name = os.getenv("OPENAI_MODEL", "gpt-5-chat-latest")
    llm = OpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"), temperature=temperature)

    role_map = {
        "system": MessageRole.SYSTEM,
        "user": MessageRole.USER,
        "assistant": MessageRole.ASSISTANT,
    }

    conv = []
    for m in messages:
        if isinstance(m, dict):
            role = role_map.get(m.get("role", "user"), MessageRole.USER)
            text = str(m.get("content", ""))
            conv.append(ChatMessage(role=role, blocks=[TextBlock(text=text)]))
        elif hasattr(m, "blocks"):
            conv.append(m)  # bereits ChatMessage
        else:
            conv.append(ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=str(m))]))

    return llm.chat(conv)


def _resp_text(resp) -> str:
    try:
        m = getattr(resp, "message", None)
        if m is None:
            return str(resp)
        if getattr(m, "content", None):
            return m.content
        blocks = getattr(m, "blocks", None)
        if blocks:
            parts = []
            for b in blocks:
                t = getattr(b, "text", None)
                if t:
                    parts.append(t)
            if parts:
                return "\n".join(parts)
        return str(resp)
    except Exception:
        return str(resp)


@st.cache_resource
def configure_global_settings():
    """ Konfiguriert die globalen Einstellungen für LlamaIndex. `@st.cache_resource` sorgt dafür, dass die Initialisierung
        (z.B. das Erstellen der API-Clients) nur einmal pro Session erfolgt und nicht bei jeder Nutzerinteraktion.
        """
    # LLM-Konfiguration:
    model_name = os.getenv("OPENAI_MODEL", "gpt-5-chat-latest")
    Settings.llm = OpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))

    # Embedding-Modell-Konfiguration - wandelt Text in numerische Vektoren um, die im Index gespeichert und für die Ähnlichkeitssuche (RAG) verwendet werden.
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY"),
        embed_batch_size=8,  # Ein konservativer Wert, um Rate-Limits der API zu vermeiden.
    )


# Führt die globale Konfiguration aus.
configure_global_settings()

# Globale Workspace-Variablen für aktiven Kontext (werden später gesetzt)
index = None
DOCS = []
ID_MAP = {}
PARENT_MAP = {}

# ================================================================================
# 6. INGEST PROFILE & PERSISTENCE HELPERS
# ================================================================================

def _compute_ingest_profile(doc_dir: Path) -> tuple[dict, str]:
    had_llama = bool(os.getenv("LLAMA_CLOUD_API_KEY") or st.secrets.get("LLAMA_CLOUD_API_KEY", None))
    pdfs = [p for p in doc_dir.rglob("*.pdf")]
    profile = {
        "had_llama_key": had_llama,
        "parser": "llamaparse" if had_llama else "pypdf",
        "parser_version": os.getenv("LLAMA_PARSE_VERSION", "n/a"),
        "clean_pdf_headers": True,
        "embed_model": "text-embedding-3-large",
        "embed_batch_size": 8,
        "app_index_version": APP_INDEX_VERSION,
        "pdf_count_total": len(pdfs),
    }
    s = json.dumps(profile, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return profile, h


def _profile_subdir(base: Path, profile_hash: str) -> Path:
    d = base / profile_hash[:12]
    d.mkdir(parents=True, exist_ok=True)
    return d


# ================================================================================
# 7. PDF PREPROCESSING - HEADER/FOOTER REMOVAL
# ================================================================================

# Vorkompilierte reguläre Ausdrücke zur Erkennung von Seitenzahlen und typischen Fußzeilen-Texten.
_page_patterns = [
    re.compile(r"^\s*\d{1,4}\s*$"),
    re.compile(r"^\s*(Seite|Page)\s+\d+(\s*/\s*\d+)?\s*$", re.I)
]
_generic_footer_patterns = [
    re.compile(r"©"),
    re.compile(r"all rights reserved", re.I)
]


def _is_pdf_doc(d) -> bool:
    """Überprüft anhand der Metadaten, ob ein LlamaIndex-Dokument aus einer PDF-Datei stammt"""
    md = d.metadata or {}
    name = (md.get("file_path") or md.get("file_name") or "").lower()
    mime = (md.get("source_mime") or md.get("content_type") or "").lower()
    return name.endswith(".pdf") or "pdf" in mime


def _normalize_line(line: str) -> str:
    """Bereinigt eine einzelne Zeile: entfernt führende/nachgestellte Leerzeichen und normalisiert Whitespace"""
    s = line.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _looks_like_page_or_footer(line: str) -> bool:
    """Prüft, ob eine Zeile wie eine Seitenzahl oder eine generische Fußzeile aussieht, basierend auf den Regex-Mustern"""
    for pat in _page_patterns:
        if pat.match(line):
            return True
    for pat in _generic_footer_patterns:
        if pat.search(line):
            return True
    return False


def remove_headers_and_footers_grouped(docs):
    """
    Entfernt wiederkehrende Kopf-/Fußzeilen aus PDF-Texten.
    - Kandidaten werden nur aus den ersten/letzten HEAD_TAIL_WINDOW Zeilen je Chunk gebildet.
    - Ähnliche Zeilen werden geclustert (SequenceMatcher).
    - Ein Kandidat gilt als Header/Footer, wenn er in >= MIN_DOC_COVERAGE der Chunks auftaucht.
    - Zusätzlich werden klassische Seitenzahlen/© etc. entfernt (_looks_like_page_or_footer).
    """
    groups = defaultdict(list)
    for d in docs:
        src = d.metadata.get("file_path") or d.metadata.get("file_name") or "unknown"
        groups[src].append(d)

    # Silbentrennung am Zeilenende entfernen
    def _preclean_text(t: str) -> str:
        return re.sub(r"-\n", "", t or "")

    # Fasse ähnliche Zeilen zu Clustern zusammen und gib je Cluster einen Repräsentanten zurück
    def _cluster(lines: list[str]) -> list[str]:
        reps: list[str] = []
        clusters: list[list[str]] = []
        for s in lines:
            added = False
            for c in clusters:
                if max(SequenceMatcher(a=s, b=x).ratio() for x in c) >= FUZZY_SIM:
                    c.append(s)
                    added = True
                    break
            if not added:
                clusters.append([s])

        for c in clusters:
            # nimm den längsten String als stabilen Repräsentanten
            reps.append(max(c, key=len))
        return reps

    cleaned_docs: list[Document] = []

    for src, group_docs in groups.items():
        # Zeilenlisten je Chunk
        per_doc_lines: list[list[str]] = []
        # Kandidaten aus Kopf- und Fußbereich
        head_cands: list[str] = []
        foot_cands: list[str] = []

        for d in group_docs:
            lines = _preclean_text(d.text).splitlines()
            per_doc_lines.append(lines)

            # Kandidaten nur aus Kopf/Fuß sammeln (normiert + Länge prüfen)
            head_zone = [_normalize_line(x) for x in lines[:HEAD_TAIL_WINDOW] if x.strip()]
            foot_zone = [_normalize_line(x) for x in lines[-HEAD_TAIL_WINDOW:] if x.strip()]

            head_cands.extend([s for s in head_zone if len(s) <= FOOTER_MAX_LINE_LEN])
            foot_cands.extend([s for s in foot_zone if len(s) <= FOOTER_MAX_LINE_LEN])

        # Fuzzy-Cluster sind Repräsentanten
        head_reps = _cluster([s for s in head_cands if s and _looks_like_page_or_footer(s)])
        foot_reps = _cluster([s for s in foot_cands if s and _looks_like_page_or_footer(s)])

        # Deckung über Chunks messen (wie oft kommt ein Rep in der jeweiligen Zone vor)
        def _coverage(rep: str, lines: list[list[str]], is_head: bool) -> int:
            hits = 0
            for L in lines:
                zone = L[:HEAD_TAIL_WINDOW] if is_head else L[-HEAD_TAIL_WINDOW:]
                zone_norm = [_normalize_line(x) for x in zone]
                if any(SequenceMatcher(a=rep, b=z).ratio() >= FUZZY_SIM for z in zone_norm):
                    hits += 1
            return hits

        need = max(FOOTER_MIN_REPEATS, int(len(per_doc_lines) * MIN_DOC_COVERAGE))
        head_final = {rep for rep in head_reps if _coverage(rep, per_doc_lines, True) >= need}
        foot_final = {rep for rep in foot_reps if _coverage(rep, per_doc_lines, False) >= need}

        # Zusätzlich: alte "repeated line" Heuristik als weicher Fallback
        freq = Counter()
        for L in per_doc_lines:
            for ln in L:
                n = _normalize_line(ln)
                if n and len(n) <= FOOTER_MAX_LINE_LEN:
                    freq[n] += 1
        repeated = {line for line, count in freq.items() if count >= FOOTER_MIN_REPEATS}

        # Rekonstruktion ohne Header/Footer
        for idx, d in enumerate(group_docs):
            L = per_doc_lines[idx]
            new_lines: list[str] = []
            N = len(L)

            for i, ln in enumerate(L):
                n = _normalize_line(ln)
                in_head_tail = (i < HEAD_TAIL_WINDOW) or (i >= N - HEAD_TAIL_WINDOW)

                if _looks_like_page_or_footer(n):
                    continue

                if in_head_tail and (
                        any(SequenceMatcher(a=n, b=rep).ratio() >= FUZZY_SIM for rep in head_final) or
                        any(SequenceMatcher(a=n, b=rep).ratio() >= FUZZY_SIM for rep in foot_final)
                ):
                    continue

                if n in repeated:
                    continue

                new_lines.append(ln)

            new_text = "\n".join(new_lines).strip()
            if new_text:
                cleaned_docs.append(Document(text=new_text, metadata=d.metadata, doc_id=d.doc_id))

    return cleaned_docs


# ================================================================================
# 8. ID RECOGNITION & NORMALIZATION
# ================================================================================

def _normalize_dashes(s: str) -> str:
    """Ersetzt verschiedene Arten von Bindestrichen durch einen Standard-Bindestrich"""
    return (s or "").replace("—", "-").replace("–", "-").replace("−", "-")

def _normalize_query(q: str) -> str:
    """Bereinigt und normalisiert den Eingabe-Query des Nutzers."""
    q = _normalize_dashes(q)
    q = re.sub(r"\s+", " ", q.strip())
    return q

def normalize_tokens(q: str) -> str:
    """
    Robuste Normalisierung des Nutzertexts: Leerzeichen/Punkte tolerieren,
    Varianten vereinheitlichen (SWE1→SWE.1, SWE.1BP2→SWE.1.BP2, GP2.1.1→GP 2.1.1, ...),
    Version-Kürzel erkennen (v4→4.0, 31→3.1 bei SPICE/PAM-Kontext).
    """
    s = (q or "").strip()
    s = s.replace("—", "-").replace("–", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s)

    # GP: "gp2.1.1" / "GP2.1.1" / "GP 2.1.1" -> "GP 2.1.1"
    s = re.sub(r"\bgp\s*([23])\s*\.?\s*([0-9])\s*\.?\s*([0-9]{1,2})\b",
               r"GP \1.\2.\3", s, flags=re.I)

    # Komprimierte GP-Schreibweise "GP2.11" → "GP 2.1.1"
    s = re.sub(r"\bgp\s*([23])\.(\d)(\d)\b", r"GP \1.\2.\3", s, flags=re.I)

    # Toleranz auch ohne Space vor GP: "gp2.11" → "GP 2.1.1"
    s = re.sub(r"\bgp([23])\.(\d)(\d)\b", r"GP \1.\2.\3", s, flags=re.I)

    # Prozesse: "SWE1" -> "SWE.1"
    s = re.sub(r"\b([A-Za-z]{3})\s*([0-9]{1,2})\b", r"\1.\2", s)

    # BP: "SWE.1BP2" / "SWE.2.BP.5" -> "SWE.1.BP2" / "SWE.2.BP5"
    s = re.sub(r"\b([A-Za-z]{3}\.[0-9]{1,2})\.?\s*bp\.?\s*([0-9]{1,2})\b",
               r"\1.BP\2", s, flags=re.I)

    # RL/RC: "SWE.1RL3" -> "SWE.1.RL.3"  (auch "SUP.9.RC.2" bleibt gleich)
    s = re.sub(r"\b([A-Za-z]{3}\.[0-9]{1,2})\.?\s*(RL|RC)\.?\s*([0-9]{1,2})\b",
               r"\1.\2.\3", s, flags=re.I)

    # OWP: "0404" / "04.04" -> "04-04"
    s = re.sub(r"\b([0-9]{2})\s*[-—–.]?\s*([0-9]{2})\b", r"\1-\2", s)

    # Level: "level1"/"stufe2"/"cl2" -> normiert
    s = re.sub(r"\b(?:level|stufe)\s*([12])\b", r"Level \1", s, flags=re.I)
    s = re.sub(r"\bCL\s*([12])\b", r"CL \1", s, flags=re.I)

    # Versionen: nur bei Kontext (v/Version/SPICE/PAM) vereinheitlichen
    if re.search(r"\b(v(?:ersion)?|spice|aspice|pam)\b", s, re.I):
        s = re.sub(r"\bv\.?\s*4(?:\.0)?\b|\b40\b", "4.0", s, flags=re.I)
        s = re.sub(r"\b3\.?1\b|\b31\b", "3.1", s, flags=re.I)

    return s


def _id_aliases(did: str):
    """Erzeugt mögliche Aliase für alle ID-Typen (BP, GP, etc.)"""
    if not did:
        return []
    s = _normalize_dashes(did.strip())
    # Alle Case-Varianten hinzufügen
    out = [s, s.lower(), s.upper()]

    # GP-spezielle Behandlung für alle Varianten
    for variant in [s, s.lower(), s.upper()]:
        if variant.lower().startswith("gp "):
            out.append(variant.replace(" ", ""))
        elif variant.lower().startswith("gp") and " " not in variant:
            m = re.match(r"^gp(\d\.\d\.\d)$", variant, re.I)
            if m:
                out.append(f"GP {m.group(1)}")
                out.append(f"gp {m.group(1)}")

    # eindeutige Reihenfolge
    uniq = []
    seen = set()
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

def canon_id(s: str) -> str:
    """Kanonische Stringform für Fuzzy-Abgleich."""
    s = (s or "").strip().lower().replace(" ", "")
    s = s.replace(".bp.", ".bp")
    s = re.sub(r"gp\s*([0-4])\.?([0-9])\.?([0-9]{1,2})", r"gp\1.\2.\3", s)
    return s

def fuzzy_map_ids(tokens: List[str]) -> List[str]:
    """Mappt evtl. fehlerhafte Tokens auf bekannte display_id-Aliase aus ID_MAP"""
    known = list(ID_MAP.keys())  # aktuelle Workspace-ID_MAP (V31/V40 je nach Auswahl)
    known_canon = {canon_id(k): k for k in known}
    out: List[str] = []

    for t in tokens:
        c = canon_id(t)
        if not c:
            continue
        if c in known_canon:
            out.append(known_canon[c])
            continue
        cand = get_close_matches(c, list(known_canon.keys()), n=1, cutoff=FUZZY_ID_MATCH_CUTOFF)
        if cand:
            out.append(known_canon[cand[0]])

    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res

def extract_ids_from_query(q: str) -> List[str]:
    """
    Extrahiert alle erkannten IDs aus dem Nutzer-Query und führt ein "Pruning" durch:
    Wenn der Query "Zeige mir SWE.1.BP1 und SWE.1" enthält, wird `SWE.1` entfernt.
    Der Bot konzentriert sich dann auf die spezifischere Anfrage (`SWE.1.BP1`) - verhindert redundante oder widersprüchliche Ausgaben.
    """
    qn = _normalize_query(q)
    raw = set()

    # Findet alle potenziellen IDs mit den definierten Mustern.
    for rx in ID_PATTERNS.values():
        for tok in rx.findall(qn):
            raw.add(_normalize_dashes(tok).strip().lower())

    # Fügt Aliase hinzu (z.B. 'gp2.2.3' für 'gp 2.2.3').
    tokens = set(raw)
    for t in list(raw):
        if t.startswith("gp "):
            tokens.add(t.replace(" ", ""))

    # Pruning-Logik: Behalte nur die spezifischsten IDs.
    bps = [t for t in tokens if BP_FULL.fullmatch(t)]
    r_proc = [t for t in tokens if RL_PROC_FULL.fullmatch(t)]
    rc_proc = [t for t in tokens if RC_PROC_FULL.fullmatch(t)]

    # (a) Wenn eine BP (`swe.1.bp1`) gefunden wurde, entferne den zugehörigen Prozess (`swe.1`).
    for bp in bps:
        proc = ".".join(bp.split(".")[:2])  # z. B. "sup.1"
        if proc in tokens:
            tokens.discard(proc)

    # (b) Wenn eine prozessgebundene Rule/Recommendation (`swe.1.rl.1`) gefunden wurde, entferne den Prozess.
    for rr in (r_proc + rc_proc):
        proc = ".".join(rr.split(".")[:2])
        if proc in tokens:
            tokens.discard(proc)

    return sorted(tokens)

# ================================================================================
# 9. VERSION DETECTION & CAPABILITY LEVELS
# ================================================================================

def detect_version(user_q: str) -> str | None:
    """
    Gibt '3.1', '4.0' oder 'both' zurück, wenn explizit erkennbar.
    Wenn nur '(Automotive) SPICE' / 'ASPICE' / 'PAM' ohne Versionszahl vorkommt → None.
    """
    q = (user_q or "").strip()
    has40 = bool(VERSION_PATTERNS["4.0"].search(q))
    has31 = bool(VERSION_PATTERNS["3.1"].search(q))

    if has40 and has31:
        return "both"
    if has40:
        return "4.0"
    if has31:
        return "3.1"

    # Default: 3.1
    return "3.1"


def detect_capability_level(user_q: str) -> int | None:
    q = (user_q or "").strip()
    m = _CL_DIRECT_RX.search(q)
    if m:
        return int(m.group(1))
    m = _LEVEL_RX.search(q)
    if m:
        return int(m.group(1))
    return None

# ================================================================================
# 10. HELPER FUNCTIONS FOR ID PROCESSING
# ================================================================================

def _parent_process_ids_for_doc(idx: int) -> set[str]:
    """Extrahiert aus den Metadaten eines Dokuments die zugehörigen Prozess-IDs (z.B. 'swe.1')."""
    out = set()
    for p in (DOCS[idx].metadata.get("parents") or []):
        pl = (p or "").strip().lower()
        if _PROC_ONLY_RX.match(pl):
            out.add(pl)
        elif BP_FULL.match(pl):
            # "swe.3.bp7" -> "swe.3"
            out.add(pl.split('.bp', 1)[0])
    return out


def _balanced_sample_by_process(indices: list[int], target_procs: list[str], total_cap: int) -> list[int]:
    """
    Stellt einen ausbalancierten Kandidatenpool für den Assessor-Modus zusammen.
    Wenn der Nutzer z.B. nach einer Bewertung für SWE.1 und SUP.10 fragt, sorgt diese Funktion dafür,
    dass Kandidaten aus beiden Prozessen berücksichtigt werden (Round-Robin-Verfahren), anstatt
    dass ein Prozess die Ergebnisse dominiert.
    """
    tset = [p.lower() for p in target_procs]
    buckets = {p: [] for p in tset}
    no_owner = []

    # Verteilt die Dokument-Indizes auf die jeweiligen Prozess-"Buckets".
    for i in indices:
        owners = _parent_process_ids_for_doc(i) & set(tset)
        if owners:
            for p in owners:
                buckets[p].append(i)
        else:
            no_owner.append(i)

    # Wählt abwechselnd (Round Robin) Kandidaten aus jedem Bucket aus.
    picked = []
    k = 0
    while len(picked) < total_cap:
        moved = False
        for p in tset:
            b = buckets.get(p, [])
            if k < len(b):
                picked.append(b[k])
                moved = True
                if len(picked) >= total_cap:
                    break
        if not moved:
            break
        k += 1

    # Füllt mit Kandidaten ohne klaren Prozess-Bezug auf, falls noch Platz ist.
    for i in no_owner:
        if len(picked) >= total_cap:
            break
        picked.append(i)

    # Entfernt Duplikate und bewahrt die Reihenfolge.
    seen = set()
    res = []
    for i in picked:
        if i not in seen:
            seen.add(i)
            res.append(i)
    return res


def _to_proc_id(any_id: str) -> str | None:
    """Extrahiert die reine Prozess-ID (z.B. 'swe.1') aus einer längeren ID."""
    m = PROC_ID_HEAD_RX.match((any_id or "").strip())
    return m.group(1).lower() if m else None


def _has_bp_parent(doc_index: int, proc_id: str) -> bool:
    """
    Überprüft, ob ein Dokument (via `doc_index`) eine Base Practice des angegebenen Prozesses (`proc_id`) als Parent hat.
    Dies ist entscheidend für die deterministische Relationssuche (z.B. "Was verbindet MAN.3 und SUP.9?").
    """
    parents = (DOCS[doc_index].metadata.get("parents") or [])
    pfx = (proc_id or "").lower() + ".bp"
    for p in parents:
        s = (p or "").strip().lower()
        if s.startswith(pfx) and BP_FULL.match(s):
            return True
    return False


def _descendant_dids_for(parent_id: str, max_depth: int = 2) -> set[str]:
    """
    Sammelt alle display_ids (z.B. SWE.1.BP1) der Nachkommen einer gegebenen `parent_id` bis zu einer bestimmten Tiefe.
    Nutzt die globalen `PARENT_MAP` und `DOCS` für eine schnelle Traversierung des Hierarchie-Graphen.
    - `max_depth=1`: Direkte Kinder (z.B. Prozess -> BPs, GPs).
    - `max_depth=2`: Kinder der Kinder (z.B. Prozess -> BPs -> Rules, Recommendations).
    """
    parent_id = (parent_id or "").strip().lower()
    if not parent_id:
        return set()

    dids: set[str] = set()
    visited: set[str] = {parent_id}
    frontier: list[str] = [parent_id]
    depth = 0

    while frontier and depth < max_depth:
        next_frontier: list[str] = []
        for pid in frontier:
            for idx in PARENT_MAP.get(pid, []):
                d = DOCS[idx]
                did = (d.metadata.get("display_id") or "").strip().lower()
                if not did:
                    continue

                # Ergebnismenge
                dids.add(did)

                # Für die nächste Ebene weiterlaufen
                if did not in visited:
                    visited.add(did)
                    next_frontier.append(did)

        frontier = next_frontier
        depth += 1

    return dids

# ================================================================================
# 11. JSON PARSERS FOR STRUCTURED DATA
# ================================================================================

# Schlüssel, die auf textuellen Inhalt in generischen JSON-Objekten hinweisen
_TEXTY_KEYS = {"description", "text", "content", "statement", "purpose", "name", "title", "characteristics"}


def _norm(s: str) -> str:
    """Normalisiert einen String (Whitespace)."""
    return re.sub(r"\s+", " ", (s or "").strip())


def _to_text_block(s: Any) -> str:
    """Wandelt einen Wert (String, Liste, etc.) in einen sauberen, mehrzeiligen Textblock um."""
    if isinstance(s, list):
        flat: list[str] = []
        for item in s:
            if item is None:
                continue
            if isinstance(item, (list, tuple)):
                flat.extend(str(x) for x in item if x is not None)
            else:
                flat.append(str(item))
        s = "\n".join(flat)
    else:
        s = str(s or "")

    # "\n" aus JSON-escapes normalisieren
    s = s.replace("\\n", "\n")
    return s.strip()


def _is_kgas_item(o: Dict[str, Any]) -> bool:
    """Prüft, ob ein Dictionary-Objekt ein KGAS-Item ist (anhand der ID)."""
    return isinstance(o, dict) and "id" in o and str(o["id"]).startswith("KGAS_")


def _parse_spice_json(data: Dict[str, Any], src: str) -> Iterable[Document]:
    """
    Ein spezialisierter Parser, der eine JSON-Struktur gemäß dem Automotive SPICE-Schema verarbeitet.
    Er iteriert durch Prozesse, Base Practices, Generic Practices etc. und erzeugt für jedes Element ein LlamaIndex `Document`-Objekt.
    Bildet die Hierarchie in den Metadaten ab (`parents`-Feld). Beispiel: Eine Rule unter einer Base Practice bekommt sowohl die BP als auch den Prozess als Parent.
    Diese Metadaten sind die Grundlage für die deterministische Navigation.
    """
    processes = data.get("processes") or data.get("Processes") or []
    top_level_gps = data.get("generic_practices") or data.get("GenericPractices") or []

    # Top-Level Generic Practices werden pro Prozess unten eingehängt.
    top_level_gps = data.get("generic_practices") or data.get("GenericPractices") or []

    # Verarbeitet Prozesse und ihre untergeordneten Elemente
    for proc in processes:
        pid = _norm(proc.get("id", ""))
        pname = _norm(proc.get("name", ""))
        purpose = _to_text_block(proc.get("purpose", ""))
        if pid:
            yield Document(
                text=f"{pid} — {pname}\nPurpose: {purpose}",
                metadata={
                    "type": "aspice_process",
                    "display_id": pid,
                    "title": pname,
                    "parents": [],
                    "source": src,
                    "file_name": src,
                    "id_path": f"processes[{pid}]"
                }
            )

        # Verarbeitet Base Practices unter einem Prozess
        for bp in proc.get("base_practices", []) or []:
            bpid = _norm(bp.get("id", ""))
            bptxt = _to_text_block(bp.get("description", ""))
            if bpid:
                yield Document(
                    text=f"{bpid} — {bptxt}",
                    metadata={
                        "type": "aspice_base_practice",
                        "display_id": bpid,
                        "parents": [pid],
                        "source": src,
                        "file_name": src,
                        "id_path": f"{pid}.base_practices[{bpid}]"
                    }
                )

            for oc in bp.get("outcomes", []) or []:
                ocid = _norm(oc.get("id", ""))
                octxt = _to_text_block(oc.get("description", ""))
                if ocid:
                    yield Document(
                        text=f"{bpid} {ocid} — {octxt}",
                        metadata={
                            "type": "aspice_outcome",
                            "display_id": f"{bpid} {ocid}",
                            "parents": [pid, bpid],
                            "source": src,
                            "file_name": src,
                            "id_path": f"{pid}.{bpid}.outcomes[{ocid}]"
                        }
                    )

            for rl in bp.get("rules", []) or []:
                rlid = _norm(rl.get("id", ""))
                rltxt = _to_text_block(rl.get("description", ""))
                if rlid:
                    yield Document(
                        text=f"{rlid} — {rltxt}",
                        metadata={
                            "type": "aspice_rule",
                            "display_id": rlid,
                            "parents": [pid, bpid],
                            "source": src,
                            "file_name": src,
                            "id_path": f"{pid}.{bpid}.rules[{rlid}]"
                        }
                    )

            for rc in bp.get("recommendations", []) or []:
                rcid = _norm(rc.get("id", ""))
                rctxt = _to_text_block(rc.get("description", ""))
                if rcid:
                    yield Document(
                        text=f"{rcid} — {rctxt}",
                        metadata={
                            "type": "aspice_recommendation",
                            "display_id": rcid,
                            "parents": [pid, bpid],
                            "source": src,
                            "file_name": src,
                            "id_path": f"{pid}.{bpid}.recommendations[{rcid}]"
                        }
                    )

        # Verarbeitet Generic Practices
        # Fallback auf Top-Level GPs, wenn der Prozess keine eigenen hat
        proc_gps = proc.get("generic_practices", []) or (top_level_gps or [])

        for gp in proc_gps:
            gid = _norm(gp.get("id", ""))
            gtxt = _to_text_block(gp.get("description", ""))
            if gid:
                yield Document(
                    text=f"{gid} — {gtxt}",
                    metadata={
                        "type": "aspice_generic_practice",
                        "display_id": gid,
                        "parents": [pid],
                        "source": src,
                        "file_name": src,
                        "id_path": f"{pid}.generic_practices[{gid}]"
                    }
                )
            for rl in gp.get("rules", []) or []:
                rlid = _norm(rl.get("id", ""))
                rtxt = _to_text_block(rl.get("description", ""))
                if rlid:
                    yield Document(
                        text=f"{rlid} — {rtxt}",
                        metadata={
                            "type": "aspice_rule",
                            "display_id": rlid,
                            "parents": [pid, gid],
                            "source": src,
                            "file_name": src,
                            "id_path": f"{pid}.{gid}.rules[{rlid}]"
                        }
                    )
            for rc in gp.get("recommendations", []) or []:
                rcid = _norm(rc.get("id", ""))
                rctxt = _to_text_block(rc.get("description", ""))
                if rcid:
                    yield Document(
                        text=f"{rcid} — {rctxt}",
                        metadata={
                            "type": "aspice_recommendation",
                            "display_id": rcid,
                            "parents": [pid, gid],
                            "source": src,
                            "file_name": src,
                            "id_path": f"{pid}.{gid}.recommendations[{rcid}]"
                        }
                    )

        # Verarbeitet Output Work Products
        owp_list = (proc.get("output_work_products")
                    or proc.get("output work products")
                    or proc.get("outputWorkProducts")
                    or [])
        for owp in owp_list:
            wid = _norm(owp.get("id", ""))
            wname = _norm(owp.get("name", ""))
            raw_chars = owp.get("characteristics", [])
            wchars = _to_text_block(raw_chars)
            if wid or wname or wchars:
                yield Document(
                    text=f"{wid} — {wname}\n\nCharacteristics:\n{wchars}".strip(),
                    metadata={
                        "type": "aspice_output_work_product",
                        "display_id": _normalize_dashes(wid or wname),
                        "title": wname,
                        "parents": [pid],
                        "source": src,
                        "file_name": src,
                        "id_path": f"{pid}.output_work_products[{wid or wname}]",
                        "owp_characteristics_list": raw_chars if isinstance(raw_chars, list) else []
                    }
                )


def _parse_kgas_json(data: Any, src: str) -> Iterable[Document]:
    """Spezialisierter Parser für JSON-Dateien, die KGAS-Anforderungen enthalten."""
    items = data if isinstance(data, list) else data.get("kgas_requirements") or data.get("requirements") or []
    for item in items or []:
        if not _is_kgas_item(item):
            continue
        kid = _norm(item.get("id", ""))
        desc = _to_text_block(item.get("description", ""))
        acc = str(item.get("requirement_accepted", "")).strip()
        body = f"{kid} — {desc}".strip()
        if acc:
            body += f"\n\naccepted: {acc}"
        yield Document(
            text=body,
            metadata={
                "type": "kgas_requirement",
                "display_id": kid,
                "parents": ["KGAS"],
                "source": src,
                "file_name": src,
                "id_path": f"kgas_requirements[{kid}]"
            }
        )


def _parse_json_generic(data: Any, src: str) -> Iterable[Document]:
    """Ein Fallback-Parser für beliebige JSON-Dateien, die nicht dem SPICE- oder KGAS-Schema entsprechen."""

    def iter_records(o: Any, path=""):
        if isinstance(o, dict):
            # Erkennt ein "Record", wenn es eine "id" und einen textuellen Schlüssel hat.
            if "id" in o and any(k in o for k in _TEXTY_KEYS):
                yield path, o
            for k, v in o.items():
                yield from iter_records(v, f"{path}.{k}" if path else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from iter_records(v, f"{path}[{i}]")

    for p, rec in iter_records(data):
        rid = _norm(str(rec.get("id", "")))
        title = _norm(rec.get("title") or rec.get("name") or "")
        parts = []
        if rid:
            parts.append(rid)
        if title:
            parts.append(title)
        for k in ["description", "text", "content", "purpose", "statement", "characteristics"]:
            if k in rec and rec[k]:
                parts.append(_to_text_block(rec[k]))

        head = " — ".join(filter(None, [rid, title]))
        text = (head + ("\n\n" + "\n".join(parts[2:]) if parts[2:] else "")).strip()
        yield Document(
            text=text,
            metadata={
                "type": "generic_json_record",
                "display_id": rid or title or p,
                "title": title,
                "parents": [],
                "source": src,
                "file_name": src,
                "id_path": p
            }
        )



# ================================================================================
# 12. DOCUMENT LOADING & INDEXING
# ================================================================================

@st.cache_resource
def load_and_index_data(documents_path: str):
    """
    Zentrale Funktion zum Laden, Verarbeiten und Indexieren aller Quelldokumente.
    Wird von Streamlit gecacht (`@st.cache_resource`), um sie nur einmal auszuführen.

    Ablauf:
    1. Durchläuft den `documents_path` und verarbeitet jede Datei je nach Typ.
    2. PDFs werden mit `LlamaParse` zu Markdown konvertiert.
    3. Excel/CSV werden mit `pandas` zeilenweise in Text umgewandelt.
    4. JSON-Dateien werden an die entsprechenden Parser (SPICE, KGAS, generisch) weitergeleitet.
    5. PDF-Texte werden mit `remove_headers_and_footers_grouped` bereinigt.
    6. Prüft, ob ein persistenter Index im `PERSIST_DIR` existiert.
        - Ja: Lädt den fertigen Index (schnell).
        - Nein: Erstellt einen neuen Vektorindex aus den Dokumenten und speichert ihn (langsam).
    7. Erstellt zwei Datenstrukturen für die deterministische Logik:
        - `id_map`: Mappt eine `display_id` auf den numerischen Index des Dokuments in der `documents`-Liste.
        - `parent_map`: Mappt eine Parent-ID auf eine Liste der Indizes ihrer Kinder.

    Gibt den Index, die Liste der Dokumente und die beiden Maps zurück.
    """
    llama_key = os.getenv("LLAMA_CLOUD_API_KEY") or st.secrets.get("LLAMA_CLOUD_API_KEY", None)
    pdf_parser = LlamaParse(api_key=llama_key, result_type="markdown", verbose=True) if llama_key else None
    documents: List[Document] = []
    dir_path = Path(documents_path)

    # ---- FAST-PATH: Index sofort laden, wenn persistente Artefakte + Manifest passen ----
    # Signatur des Quellordners
    def _dir_signature(root: Path) -> dict:
        n, total_bytes, latest_mtime = 0, 0, 0.0
        for p in root.rglob('*'):
            if p.is_file():
                try:
                    st_ = p.stat()
                except Exception:
                    continue
                n += 1
                total_bytes += st_.st_size
                if st_.st_mtime > latest_mtime:
                    latest_mtime = st_.st_mtime
        return {"files": n, "bytes": int(total_bytes), "latest_mtime": int(latest_mtime)}

    # Profil berechnen und Unterordner wählen
    profile, profile_hash = _compute_ingest_profile(dir_path)
    base_31 = (BASE_DIR / PERSIST_DIR_31)
    base_40 = (BASE_DIR / PERSIST_DIR_40)
    persist_dir_31 = _profile_subdir(base_31, profile_hash)
    persist_dir_40 = _profile_subdir(base_40, profile_hash)

    def _has_any_vecstore(persist_dir: Path) -> bool:
        return persist_dir.exists() and any(persist_dir.glob("*vector_store.json"))

    def _try_fastload(persist_dir: Path):
        docstore = persist_dir / "docstore.json"
        idxstore = persist_dir / "index_store.json"
        manifest_file = persist_dir / "manifest.json"
        docs_file = persist_dir / "documents.jsonl"
        complete_file = persist_dir / COMPLETE

        if not (docstore.exists() and idxstore.exists() and _has_any_vecstore(persist_dir)
                and manifest_file.exists() and docs_file.exists() and complete_file.exists()):
            return None

        try:
            man = json.loads(manifest_file.read_text(encoding="utf-8"))

            # Quelle unverändert?
            if man.get("source_signature") != _dir_signature(dir_path):
                return None

            # Build-Konfiguration muss passen
            cfg_current = {
                "embed_model": "text-embedding-3-large",
                "embed_batch_size": 8,
                "clean_pdf_headers": True,
                "app_index_version": APP_INDEX_VERSION,
            }
            if man.get("build_config") != cfg_current:
                return None

            # Profil muss passen
            if man.get("ingest_profile_hash") != profile_hash:
                return None

            storage = StorageContext.from_defaults(persist_dir=str(persist_dir))
            index = load_index_from_storage(storage)

            loaded_docs: List[Document] = []
            with docs_file.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    loaded_docs.append(Document(text=rec["text"], metadata=rec["metadata"]))

            id_map = man.get("id_map") or {}
            parent_map = man.get("parent_map") or {}

            # Im Manifest ist auch display_id_index gespeichert
            display_id_index = man.get("display_id_index") or {}

            return index, loaded_docs, id_map, parent_map, display_id_index

        except Exception:
            return None

    _fast31 = _try_fastload(persist_dir_31)
    _fast40 = _try_fastload(persist_dir_40)

    if _fast31 or _fast40:
        index_v31, documents_v31, id_map_v31, parent_map_v31, display_id_index_v31 = (_fast31 or (None, [], {}, {}, {}))
        index_v40, documents_v40, id_map_v40, parent_map_v40, display_id_index_v40 = (_fast40 or (None, [], {}, {}, {}))

        return (
            index_v31, documents_v31, id_map_v31, parent_map_v31, display_id_index_v31,
            index_v40, documents_v40, id_map_v40, parent_map_v40, display_id_index_v40,
        )

    # Build als "in progress" markieren
    for d in (persist_dir_31, persist_dir_40):
        (d / INPROGRESS).write_text("building", encoding="utf-8")
        # ein evtl. altes COMPLETE-Signal entfernen
        (d / COMPLETE).unlink(missing_ok=True)

    skipped_pdfs = 0

    # Alle Dateien im Dokumentenordner verarbeiten
    for file_path in dir_path.rglob('*'):
        if file_path.is_dir() or file_path.name.startswith('~'):
            continue

        try:
            ext = file_path.suffix.lower()

            if ext == ".pdf":
                if not pdf_parser:
                    skipped_pdfs += 1
                    continue
                try:
                    pdf_docs = pdf_parser.load_data(str(file_path))
                    for d in pdf_docs:
                        md = dict(d.metadata or {})
                        md.setdefault("file_name", file_path.name)
                        documents.append(Document(text=d.text, metadata=md))
                except Exception as e:
                    logger.warning(f"PDF parsing failed: {file_path.name}", exc_info=True)
                    st.warning(f"PDF konnte nicht geparst werden: {file_path.name} → {e}")
                continue

            elif ext == ".xlsx":
                # Verarbeitung von Excel-Dateien
                xls = pd.ExcelFile(file_path, engine='openpyxl')
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name).dropna(how='all')
                    for i, row in df.iterrows():
                        text_parts = [f"{col}: {val}" for col, val in row.items()
                                      if pd.notna(val) and str(val).strip()]
                        text = f"Tabellenblatt: {sheet_name}; Zeile: {i + 2}; {'; '.join(text_parts)}"
                        metadata = {"file_name": file_path.name, "sheet_name": sheet_name,
                                    "row_index": i + 2, **row.astype(str).to_dict()}
                        documents.append(Document(text=text, metadata=metadata))

            elif ext == ".csv":
                # Verarbeitung von CSV-Dateien
                df = pd.read_csv(file_path).dropna(how='all')
                for i, row in df.iterrows():
                    text_parts = [f"{col}: {val}" for col, val in row.items()
                                  if pd.notna(val) and str(val).strip()]
                    text = f"Zeile: {i + 2}; {'; '.join(text_parts)}"
                    metadata = {"file_name": file_path.name, "row_index": i + 2,
                                **row.astype(str).to_dict()}
                    documents.append(Document(text=text, metadata=metadata))

            elif ext == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                src = file_path.name
                is_spice = isinstance(data, dict) and (
                        "processes" in data or "Processes" in data or "generic_practices" in data
                )
                is_kgas = (isinstance(data, list) and any(_is_kgas_item(x) for x in data)) or \
                          (isinstance(data, dict) and any(_is_kgas_item(x) for x in (
                                  data.get("kgas_requirements") or data.get("requirements") or [])))

                if is_spice:
                    parsed = list(_parse_spice_json(data, src))
                    version = str(data.get("spice_version", "")).strip() or "unknown"
                    for d in parsed:
                        md = dict(d.metadata or {})
                        md["spice_version"] = version
                        d.metadata = md
                    documents.extend(parsed)
                elif is_kgas:
                    documents.extend(list(_parse_kgas_json(data, src)))
                else:
                    documents.extend(list(_parse_json_generic(data, src)))

            else:
                # Fallback für andere Textdateien (.txt, .md etc.)
                other_docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
                for d in other_docs:
                    md = dict(d.metadata or {})
                    md.setdefault("file_name", file_path.name)
                    documents.append(Document(text=d.text, metadata=md))

        except Exception as e:
            logger.warning(f"Could not parse file: {file_path.name}", exc_info=True)
            st.warning(f"Konnte Datei '{file_path.name}' nicht verarbeiten: {e}")

    if skipped_pdfs:
        logger.warning(f"PDF parsing failed - no LLama Cloud Key set: {skipped_pdfs}", exc_info=True)
        st.warning(f"{skipped_pdfs} PDF(s) wurden übersprungen (kein LLAMA_CLOUD_API_KEY gesetzt).")

    # PDF-Dokumente bereinigen
    pdf_docs = [d for d in documents if _is_pdf_doc(d)]
    other_docs = [d for d in documents if not _is_pdf_doc(d)]

    if pdf_docs:
        pdf_docs = remove_headers_and_footers_grouped(pdf_docs)

    documents = pdf_docs + other_docs

    if not documents:
        st.error("Es wurden keine Dokumente geladen. Bitte überprüfe den 'documents'-Ordner.")
        st.stop()

    # Baut die globale Whitelist für OWP-IDs aus den geladenen Dokumenten auf.
    global OWP_IDS
    OWP_IDS = {
        (d.metadata.get("display_id") or "").strip().lower()
        for d in documents
        if (d.metadata.get("type") or "") == "aspice_output_work_product"
    }

    # Persistenz/Index je SPICE Version
    def _build_maps(docs: List[Document]) -> tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, int]]:
        id_map: Dict[str, List[int]] = {}
        display_id_index: Dict[str, int] = {}

        for i, d in enumerate(docs):
            did = str(d.metadata.get("display_id", "")).strip()

            # Display-ID → Index Mapping (für schnelle Einzelsuche)
            if did:
                display_id_index[did.lower()] = i

            # Aliase für ID_MAP (bleibt wie vorher)
            for alias in _id_aliases(did):
                id_map.setdefault(alias, []).append(i)

        parent_map: Dict[str, List[int]] = defaultdict(list)
        for i, d in enumerate(docs):
            for p in d.metadata.get("parents", []) or []:
                parent_map[p.lower()].append(i)

        return id_map, parent_map, display_id_index

    documents_v31 = [d for d in documents if (d.metadata.get("spice_version") == "3.1")]
    documents_v40 = [d for d in documents if (d.metadata.get("spice_version") == "4.0")]

    # Index 3.1
    index_v31 = None
    if documents_v31:
        docstore_31 = persist_dir_31 / "docstore.json"
        idxstore_31 = persist_dir_31 / "index_store.json"

        if docstore_31.exists() and idxstore_31.exists() and _has_any_vecstore(persist_dir_31):
            storage = StorageContext.from_defaults(persist_dir=str(persist_dir_31))
            index_v31 = load_index_from_storage(storage)
        else:
            persist_dir_31.mkdir(parents=True, exist_ok=True)
            build_lock = FileLock(str(BASE_DIR / ".index_build.lock"))
            try:
                with build_lock.acquire(timeout=900):
                    # Double-check: hat ein anderer Prozess inzwischen persistiert?
                    if docstore_31.exists() and idxstore_31.exists() and _has_any_vecstore(persist_dir_31):
                        storage = StorageContext.from_defaults(persist_dir=str(persist_dir_31))
                        index_v31 = load_index_from_storage(storage)
                    else:
                        storage = StorageContext.from_defaults()
                        index_v31 = VectorStoreIndex.from_documents(documents_v31, storage_context=storage)
                        storage.persist(persist_dir=str(persist_dir_31))
            except Timeout:
                logger.warning("Another process is creating 3.1 index - try loading again later" , exc_info=True)
                st.warning("Ein anderer Prozess erstellt gerade den 3.1-Index. Bitte später neu laden.")

    # Index 4.0
    index_v40 = None
    if documents_v40:
        docstore_40 = persist_dir_40 / "docstore.json"
        idxstore_40 = persist_dir_40 / "index_store.json"

        if docstore_40.exists() and idxstore_40.exists() and _has_any_vecstore(persist_dir_40):
            storage = StorageContext.from_defaults(persist_dir=str(persist_dir_40))
            index_v40 = load_index_from_storage(storage)
        else:
            persist_dir_40.mkdir(parents=True, exist_ok=True)
            build_lock = FileLock(str(BASE_DIR / ".index_build.lock"))
            try:
                with build_lock.acquire(timeout=900):
                    # Double-check: hat ein anderer Prozess inzwischen persistiert?
                    if docstore_40.exists() and idxstore_40.exists() and _has_any_vecstore(persist_dir_40):
                        storage = StorageContext.from_defaults(persist_dir=str(persist_dir_40))
                        index_v40 = load_index_from_storage(storage)
                    else:
                        storage = StorageContext.from_defaults()
                        index_v40 = VectorStoreIndex.from_documents(documents_v40, storage_context=storage)
                        storage.persist(persist_dir=str(persist_dir_40))
            except Timeout:
                logger.warning("Another process is creating 4.0 index - try loading again later", exc_info=True)
                st.warning("Ein anderer Prozess erstellt gerade den 4.0-Index. Bitte später neu laden.")

    # Maps je Version (mit Display-ID-Index)
    id_map_v31, parent_map_v31, display_id_index_v31 = _build_maps(documents_v31)
    id_map_v40, parent_map_v40, display_id_index_v40 = _build_maps(documents_v40)

    # Dokumente + Manifest schreiben (für Fast-Path beim nächsten Start)
    def _dump_docs_jsonl(docs: List[Document], out_path: Path):
        with out_path.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({"text": d.text, "metadata": d.metadata}, ensure_ascii=False) + "\n")

    _cfg = {
        "embed_model": "text-embedding-3-large",
        "embed_batch_size": 8,
        "clean_pdf_headers": True,
        "app_index_version": 1,
    }
    _sig = _dir_signature(dir_path)

    if documents_v31:
        _dump_docs_jsonl(documents_v31, persist_dir_31 / "documents.jsonl")
        (persist_dir_31 / "manifest.json").write_text(json.dumps({
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "source_signature": _sig,
            "build_config": _cfg,
            "ingest_profile": profile,
            "ingest_profile_hash": profile_hash,
            "pdf_count_indexed": sum(1 for d in documents_v31 if _is_pdf_doc(d)),
            "skipped_pdfs": int(skipped_pdfs),
            "id_map": id_map_v31,
            "parent_map": parent_map_v31,
            "display_id_index": display_id_index_v31,
        }, ensure_ascii=False), encoding="utf-8")

        # COMPLETE nur setzen, wenn keine PDFs übersprungen wurden
        if skipped_pdfs == 0:
            (persist_dir_31 / COMPLETE).write_text(json.dumps({"ok": True}), encoding="utf-8")
        # IN_PROGRESS immer entfernen
        (persist_dir_31 / INPROGRESS).unlink(missing_ok=True)

    if documents_v40:
        _dump_docs_jsonl(documents_v40, persist_dir_40 / "documents.jsonl")
        (persist_dir_40 / "manifest.json").write_text(json.dumps({
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "source_signature": _sig,
            "build_config": _cfg,
            "ingest_profile": profile,
            "ingest_profile_hash": profile_hash,
            "pdf_count_indexed": sum(1 for d in documents_v40 if _is_pdf_doc(d)),
            "skipped_pdfs": int(skipped_pdfs),
            "id_map": id_map_v40,
            "parent_map": parent_map_v40,
            "display_id_index": display_id_index_v40,
        }, ensure_ascii=False), encoding="utf-8")

        if skipped_pdfs == 0:
            (persist_dir_40 / COMPLETE).write_text(json.dumps({"ok": True}), encoding="utf-8")
        (persist_dir_40 / INPROGRESS).unlink(missing_ok=True)

    return (
        index_v31, documents_v31, id_map_v31, parent_map_v31, display_id_index_v31,
        index_v40, documents_v40, id_map_v40, parent_map_v40, display_id_index_v40,
    )


# Pfad zum Ordner mit den Quelldokumenten
documents_folder_path = str(BASE_DIR / "documents")
if not os.path.exists(documents_folder_path):
    st.error(f"Der Ordner '{documents_folder_path}' wurde nicht gefunden. Bitte erstellen.")
    st.stop()

# Lädt die Daten und initialisiert die versionierten Workspaces.
(
    INDEX_V31, DOCS_V31, ID_MAP_V31, PARENT_MAP_V31, DISPLAY_ID_INDEX_V31,
    INDEX_V40, DOCS_V40, ID_MAP_V40, PARENT_MAP_V40, DISPLAY_ID_INDEX_V40,
) = load_and_index_data(documents_folder_path)

# Setze eine sinnvolle Default-"aktive" Umgebung
if DOCS_V31:
    index, DOCS, ID_MAP, PARENT_MAP = INDEX_V31, DOCS_V31, ID_MAP_V31, PARENT_MAP_V31
elif DOCS_V40:
    index, DOCS, ID_MAP, PARENT_MAP = INDEX_V40, DOCS_V40, ID_MAP_V40, PARENT_MAP_V40
else:
    index, DOCS, ID_MAP, PARENT_MAP = None, [], {}, {}

# Stellt sicher, dass die globalen Variablen verfügbar sind
globals().update({
    'index': index,
    'DOCS': DOCS,
    'ID_MAP': ID_MAP,
    'PARENT_MAP': PARENT_MAP
})

# === Sidebar: Performance Tracing (Debug) ===
if st.session_state.get("llm_traces"):
    with st.sidebar.expander("⏱️ Performance Traces", expanded=False):
        traces = st.session_state.llm_traces[-10:]  # Letzte 10

        total_time = sum(t["duration_ms"] for t in traces)
        avg_time = total_time / len(traces) if traces else 0

        st.metric("Avg Call Time", f"{avg_time:.0f}ms")
        st.metric("Total Calls", len(st.session_state.llm_traces))

        st.caption("Letzte Calls:")
        for t in reversed(traces):
            icon = "✅" if t["success"] else "❌"
            st.text(f"{icon} {t['function']}: {t['duration_ms']}ms")

        # Tool-Use Stats
        tool_traces = [t for t in st.session_state.llm_traces if "tool" in t.get("function", "").lower()]
        if tool_traces:
            st.divider()
            st.caption("Tool Calls:")
            tool_counts = {}
            for t in tool_traces:
                name = t.get("function", "unknown")
                tool_counts[name] = tool_counts.get(name, 0) + 1

            for tool, count in tool_counts.items():
                st.text(f"  {tool}: {count}x")

# === Sidebar: Assessment-Simulation (nur wenn NICHT aktiv) ===
if not st.session_state.get("simulation_active"):
    with st.sidebar.expander("⚙️ AI Assessor", expanded=False):
        STYLE_TEMP = {"rule-oriented": 0.3, "balanced": 0.6, "challenging": 0.9}

        # 1. Assessor-Typ
        st.session_state.setdefault("sim_style", "rule-oriented")
        sel_style = st.radio(
            "Assessor-Typ",
            options=list(STYLE_TEMP.keys()),
            index=["rule-oriented", "balanced", "challenging"].index(st.session_state.get("sim_style", "rule-oriented")),
            horizontal=True,
        )
        st.session_state["sim_style"] = sel_style
        st.session_state.sim_temp_questions = STYLE_TEMP[sel_style]
        st.session_state.setdefault("sim_temp_evaluation", 0.3)

        # 2. SPICE-Version
        st.session_state.setdefault("sim_version", "3.1")
        sel_version = st.radio(
            "SPICE-Version",
            options=["3.1", "4.0"],
            index=["3.1", "4.0"].index(st.session_state.get("sim_version", "3.1")),
            horizontal=True,
        )
        st.session_state["sim_version"] = sel_version

        # 3. Capability Level
        st.session_state.setdefault("sim_capability_level", "1")
        sel_cl = st.radio(
            "Capability Level",
            options=["1", "2"],
            index=["1", "2"].index(st.session_state.get("sim_capability_level", "1")),
            horizontal=True,
        )
        st.session_state["sim_capability_level"] = sel_cl

        # 4. Prozess-Auswahl
        docs_src = DOCS_V31 if sel_version == "3.1" else DOCS_V40
        idx_map = DISPLAY_ID_INDEX_V31 if sel_version == "3.1" else DISPLAY_ID_INDEX_V40


        def natural_sort_key(process_id):
            parts = process_id.upper().split('.')
            if len(parts) == 2:
                prefix = parts[0]
                try:
                    num = int(parts[1])
                    return (prefix, num)
                except ValueError:
                    return (prefix, 999)
            return (process_id.upper(), 0)


        _proc_choices = []
        for pid, di in (idx_map or {}).items():
            try:
                md = (docs_src[di].metadata or {})
                if md.get("type") == "aspice_process":
                    title = (md.get("title") or pid)
                    pid_upper = pid.upper()
                    _proc_choices.append((pid_upper, title))
            except Exception:
                continue

        _proc_choices.sort(key=lambda x: natural_sort_key(x[0]))

        if _proc_choices:
            labels = [f"{pid} — {title}" for pid, title in _proc_choices]
            picked = st.selectbox(
                "Prozess",
                labels,
                index=0,
            )
            process_id = picked.split(" — ", 1)[0]
        else:
            st.info("Keine Prozesse für diese SPICE-Version gefunden.")
            process_id = None

        # 5. Start-Button
        if st.button("▶️ Assessment starten", use_container_width=True, disabled=not bool(process_id)):
            if process_id:
                # WICHTIG: Evidenz-Scope vor dem Start prüfen/isolieren
                _ensure_evidence_scope(process_id, sel_version)

                # alten Manager weg, saubere Sim-States setzen
                st.session_state.pop("assessment_manager", None)
                _init_sim_state()

                # KEIN Wiederherstellen eines alten evidence_index hier!
                st.session_state.sim_version = sel_version
                st.session_state.sim_capability_level = sel_cl
                st.session_state.sim_process_id = process_id
                st.session_state.simulation_cfg.update({
                    "version": sel_version,
                    "capability_level": sel_cl,
                    "processes": [process_id],
                })
                st.session_state.practice_queue = _extract_practices(
                    process_id=process_id,
                    version=sel_version,
                    capability_level=int(sel_cl),
                )
                st.session_state.practice_idx = 0
                st.session_state.qa_count_for_practice = 0
                st.session_state.answers_by_practice = {}
                st.session_state.simulation_active = True
                st.session_state.last_assessor_question = None
                st.session_state.last_process_for_eval = process_id

                # Strukturen für Aspekte & Frage-Metadaten bereitstellen
                st.session_state.setdefault("practice_aspects", {})
                st.session_state.setdefault("question_meta", {})

                st.rerun()

# === Sidebar: Tool-Use Toggle ===
st.sidebar.divider()
st.session_state.setdefault("enable_tool_use", True)
enable_tools = st.sidebar.checkbox(
    "🔧 Tool-Use aktivieren",
    value=st.session_state.get("enable_tool_use", True),
    help="LLM kann autonom Tools nutzen (Evidence-Suche, Skip, etc.)"
)
st.session_state.enable_tool_use = enable_tools

# ================================================================================
# 13. INTENT DETECTION & LANGUAGE PROCESSING
# ================================================================================

# Konstanten für die verschiedenen Dokument-Typen aus dem SPICE-Schema.
TYPE_RULE = "aspice_rule"
TYPE_REC = "aspice_recommendation"
TYPE_OUT = "aspice_outcome"
TYPE_OWP = "aspice_output_work_product"
TYPE_BP = "aspice_base_practice"
TYPE_GP = "aspice_generic_practice"


def detect_query_lang(q: str) -> str:
    q = (q or "").strip()
    if _LANG_DE_RX.search(q):
        return "de"
    if _LANG_EN_RX.search(q):
        return "en"
    # Fallback: UI ist deutsch
    return "de"


def make_out_hint(lang: str, concise: bool) -> str:
    if lang == "en":
        return (
            "Answer briefly and to the point."
            if concise else
            "Answer in detail and well structured in Markdown (headings, paragraphs, bullets)."
        )
    # de (Default)
    return (
        "Antworte kurz und bündig."
        if concise else
        "Antworte ausführlich und gut strukturiert in Markdown (Überschriften, Absätze, Bullets)."
    )


def compute_out_hint_and_flag(query_text: str):
    """
    Analysiert den Query, um festzustellen, ob eine knappe Antwort gewünscht wird und in welcher Sprache.
    Gibt einen passenden System-Prompt-Teil (`out_hint`), ein Flag (`want_concise`) und einen
    Sprachhinweis (`lang_tag`) zurück, um das LLM zu steuern.
    """
    want_concise = bool(RX_CONCISE.search(query_text or ""))
    lang = detect_query_lang(query_text)
    out_hint = make_out_hint(lang, want_concise)

    # Harte Sprachansage — in DE und EN formuliert, damit Modelle sie sicher respektieren.
    lang_tag = (
        "Antwortsprache: Deutsch. Respond in German."
        if lang == "de"
        else "Language: English. Antworte auf Englisch."
    )
    return out_hint, want_concise, lang_tag


def detect_intent_targets(query_text: str) -> Optional[set]:
    """
    Ermittelt, ob der Nutzer eine deterministische Liste anfordert.
    Bedingungen: Mindestens eine ID muss im Query sein, ein Schlüsselwort für einen Artefakttyp
    (z.B. "rules") muss vorkommen und es darf keine Erklärung-Frage sein.
    Gibt ein Set der gewünschten Artefakt-Typen zurück (z.B. `{TYPE_RULE}`).
    """
    q = " ".join((_normalize_query(query_text) or "").split())
    if not _ID_ANY.search(q):
        return None
    if RX_EXPLAIN.search(q):
        return None

    targets = set()
    if _RX_RULES.search(q):
        targets.add(TYPE_RULE)
    if _RX_RECS.search(q):
        targets.add(TYPE_REC)
    if _RX_OUT.search(q):
        targets.add(TYPE_OUT)
    if _RX_OWP.search(q):
        targets.add(TYPE_OWP)

    has_proc_id = bool(_PROC_ID.search(q))
    if has_proc_id and _RX_BPWRD.search(q):
        targets.add(TYPE_BP)
    if has_proc_id and _RX_GPWRD.search(q):
        targets.add(TYPE_GP)

    return targets or None


# Level-abhängiger Dokumenttyp-Filter
ALLOWED_TYPES_L1 = {
    "aspice_base_practice", "aspice_outcome", "aspice_output_work_product",
    "aspice_rule", "aspice_recommendation", "aspice_process"}
ALLOWED_TYPES_L2 = ALLOWED_TYPES_L1 | {"aspice_generic_practice"}
RULE_LIKE = {"aspice_rule", "aspice_recommendation"}


def _extract_topk(q: str, default_k: int = DEFAULT_TOP_K) -> int:
    """Extrahiert die gewünschte Anzahl K (5) für eine Top-K-Bewertung aus dem Query."""
    m = TOPK_RX.search(q or "")
    if not m:
        return default_k
    for g in m.groups():
        if g and g.isdigit():
            try:
                k = int(g)
                return max(1, min(10, k))  # clamp 1..10
            except Exception:
                pass
    return default_k


def _extract_goal_text(q: str) -> str:
    """Extrahiert das Bewertungskriterium aus dem Query. Beispiel: "Bewerte SWE.1 anhand des Kriteriums Security-Relevanz" -> "Security-Relevanz"""

    qn = q.strip()

    # Nimm ALLES nach dem Keyword bis zum ersten Stopp-Signal
    m = re.search(
        r'(anhand|auf\s+basis\s+von|auf\s+basis|basierend\s+auf|based\s+on|'
        r'with\s+respect\s+to|regarding|with\s+regards?\s+to|according\s+to|'
        r'nach|for|hinsichtlich|bzgl\.?|concerning|focused\s+on|fokussiert\s+auf)\s+'
        r'(?:des\s+)?(?:kriteriums?\s+)?'
        r'(.+?)(?:\s+(?:und|oder|für|bei|in|von|zeig|bewert|prioris)\b|$)',
        qn, re.I
    )
    if m:
        criterion = m.group(2).strip(' .,:;')
        if len(criterion) > 2:
            return criterion

    # Fallback bei "Qualität" auch ohne Keyword
    if re.search(r"qualit(ä|ae)t|quality", qn, re.I):
        return "wichtigste Items hinsichtlich Produkt- und Prozessqualität"

    # Generischer Fallback
    return "wichtigste Items im Projektkontext (Experteneinschätzung)"

# ================================================================================
# 14. COMPARISON & ASSESSMENT FUNCTIONS
# ================================================================================

def _split_sents(t: str) -> list[str]:
    t = (t or "").strip()
    return [s.strip() for s in _SENT_RX.split(t) if s.strip()]


MAX_SHORT_TEXT = 160

def _short(s: str, n: int = MAX_SHORT_TEXT) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def _word_diff(a: str, b: str) -> str:
    from difflib import SequenceMatcher
    out = []
    a_words = (a or "").split()
    b_words = (b or "").split()
    sm = SequenceMatcher(None, a_words, b_words)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete"):
            out.append("- " + _short(" ".join(a_words[i1:i2]), 80))
        if tag in ("replace", "insert"):
            out.append("+ " + _short(" ".join(b_words[j1:j2]), 80))

    return "; ".join(out) if out else "↔ (nur kleinere Umformulierungen)"


def _extract_key_deltas(text31: str, text40: str) -> dict[str, list[str]]:
    from difflib import SequenceMatcher
    s31 = _split_sents(text31)
    s40 = _split_sents(text40)

    best31_for40 = []
    for s in s40:
        scores = [(SequenceMatcher(None, s, t).ratio(), t) for t in s31] or [(0.0, "")]
        r, tbest = max(scores, key=lambda x: x[0])
        best31_for40.append((s, tbest, r))

    added = [_short(s) for (s, t, r) in best31_for40 if r < SENTENCE_SIMILARITY_LOW]
    changed_pairs = [(t, s) for (s, t, r) in best31_for40 if SENTENCE_SIMILARITY_LOW <= r < SENTENCE_SIMILARITY_MID]

    best40_for31 = []
    for t in s31:
        scores = [(SequenceMatcher(None, t, s).ratio(), s) for s in s40] or [(0.0, "")]
        r, sbest = max(scores, key=lambda x: x[0])
        best40_for31.append((t, sbest, r))

    removed = [_short(t) for (t, s, r) in best40_for31 if r < 0.45]

    changed = []
    for old, new in changed_pairs[:4]:
        changed.append(_short(_word_diff(old, new)))

    return {"added": added[:MAX_DELTA_ITEMS], "removed": removed[:MAX_DELTA_ITEMS],
            "changed": changed[:MAX_DELTA_CHANGED]}

# Semantik-Normalisierung und Ähnlichkeit
def _normalize_for_match(s: str) -> str:
    """
    Prozessneutrale Normalisierung für semantische Ähnlichkeit:
    - Kleinbuchstaben, Unicode-Striche vereinheitlichen
    - Outcome-/Note-/Example-Zeilen entfernen
    - IDs (SUP.1.BP3, SWE.4.BP1, GP 2.1.3) entfernen
    - Satzzeichen/Mehrfach-Whitespace normalisieren
    - sehr simple Wortstamm-Normalisierung (ing/ed/es/s)
    - nur allgemeine Funktionswörter filtern (keine domänenspezifischen Synonyme!)
    """
    s = (s or "").lower()
    # 0) offensichtliche Format-/Meta-Bestandteile raus
    s = s.replace("—", "-").replace("–", "-").replace("/", " ")

    # 1) Outcome-/Note-/Example-Zeilen und Outcome-Tags entfernen
    s = re.sub(r"\[outcome\s*\d+]", " ", s, flags=re.I)
    s = re.sub(r"(?mi)^(?:note|notes?|example|examples?)\s*\d*\s*:\s.*$", " ", s)

    # 2) IDs entfernen (SUP.1.BP3 / SWE.4.BP1 / GP 2.1.3 / HWE.2.BP1 …)
    s = re.sub(r"\b[a-z]{3}\.\d+(?:\.[a-z]{2}\d+)?\b", " ", s, flags=re.I)  # SUP.1.BP3 etc.
    s = re.sub(r"\bgp\s*\d+\.\d+(?:\.\d+)?\b", " ", s, flags=re.I)  # GP 2.1.3

    # 3) sonstige Satzzeichen entfernen
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    s = re.sub(r"\s+-\s+", " ", s)

    # 4) simple Normalisierung von Plural-/Verbformen (bewusst konservativ)
    toks = s.split()
    norm = []
    for t in toks:
        if len(t) > 4 and t.endswith("ing"):
            t = t[:-3]
        elif len(t) > 3 and t.endswith("ed"):
            t = t[:-2]
        elif len(t) > 3 and t.endswith("es"):
            t = t[:-2]
        elif len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        norm.append(t)

    # 5) nur allgemeine Funktionswörter filtern (keine Domänenbegriffe!)
    STOP = {
        "the", "a", "an", "and", "or", "of", "for", "to", "with", "in", "on", "by", "as",
        "is", "are", "be", "being", "been", "this", "that", "these", "those",
        "from", "at", "it", "its", "their", "there", "here", "which", "who", "whom",
        "within", "without", "into", "onto", "than", "then", "so", "such", "per"
    }
    norm = [t for t in norm if t not in STOP]

    return " ".join(norm)


# Semantische Ähnlichkeit per Embeddings
def _cosine(u: list[float], v: list[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    return (dot / (nu * nv)) if (nu > 0 and nv > 0) else 0.0


@lru_cache(maxsize=4096)
def _embed_normed(text: str) -> tuple[float, ...]:
    t = _normalize_for_match(text or "")
    if len(t) < 20:
        t = (t + " context").strip()
    try:
        # LlamaIndex API
        if hasattr(Settings.embed_model, 'get_query_embedding'):
            vec = Settings.embed_model.get_query_embedding(t)
        elif hasattr(Settings.embed_model, 'embed_query'):
            vec = Settings.embed_model.embed_query(t)
        else:
            return tuple()
        return tuple(float(x) for x in (vec or []))
    except Exception:
        return tuple()


def _align_cross_ids(*, bp31, gp31, bp40, gp40, i31, i40):
    """
    Content-first Alignment (deterministisch, ID-unabhängig):
      - Text -> Atome (normierte Sätze/Teilsätze)
      - IDF-gewichtete Überschneidung (Jaccard-IDF)
      - Greedy Matching (best-first), BP↔GP erlaubt
      - Splits/Merges via relatives Top-2/Top-n
    Rückgabe: (moved, moved_changed, residual_added, residual_removed, common_sameid, splits, merges)
    """
    # 0) Hilfen: Normalisierung & Atome
    _sent_split = re.compile(r"[.;:\n]+")  # einfache, robuste Satztrenner

    def _normalize_txt(t: str) -> str:
        t = (t or "").strip().lower()
        # IDs/Prefix raus, Zahlen normalisieren, Mehrfach-Whitespace glätten
        t = re.sub(r"\b(?:acq|spl|sup|man|eng|sys|swe|hwe|mle|val|ver)\.\d+(?:\.bp\d+)?\b", " ", t)
        t = re.sub(r"\bgp\s*\d+(?:\.\d+){1,2}\b", " ", t)
        t = re.sub(r"\[(?:outcome|rule|recommendation)\s*\d+]", " ", t)
        t = re.sub(r"\d+", " 0 ", t)
        t = re.sub(r"[^\w\s\-]", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    # liefert normalisierte 'Satz-Atome' (>= 5 Wörter) + lange Phrasen (8-12 Wörter sliding).
    def _atoms(t: str) -> set[str]:
        t = _normalize_txt(t)
        atoms = set()

        # Sätze
        for s in _sent_split.split(t):
            s = s.strip()
            if not s:
                continue
            if len(s.split()) >= MIN_ATOM_WORDS:
                atoms.add(s)

        # Sliding Phrasen (robuster gegen Umformulierungen)
        words = t.split()
        for n in PHRASE_LENGTHS:
            if len(words) >= n:
                for i in range(0, len(words) - n + 1):
                    atoms.add(" ".join(words[i:i + n]))

        return atoms

    # 1) Pools je Version (nur BP/GP, nur dieser Prozess)
    keys31 = sorted((bp31 | gp31))
    keys40 = sorted((bp40 | gp40))
    pool31 = [(typ, did, i31.get((typ, did), "") or "") for (typ, did) in keys31]
    pool40 = [(typ, did, i40.get((typ, did), "") or "") for (typ, did) in keys40]

    # map -> Atome
    A31 = {(t, d): _atoms(txt) for (t, d, txt) in pool31}
    A40 = {(t, d): _atoms(txt) for (t, d, txt) in pool40}

    # 2) IDF-Gewichte (über alle Schlüssel beider Versionen)
    df = Counter()
    for key, atoms in A31.items():
        for a in set(atoms):
            df[a] += 1
    for key, atoms in A40.items():
        for a in set(atoms):
            df[a] += 1

    Ndocs = len(A31) + len(A40) or 1

    def _idf(a: str) -> float:
        # +1 im Nenner gegen Div/0 und zur Glättung
        return math.log(1.0 + Ndocs / (1.0 + df[a]))

    def _wjaccard(setA: set[str], setB: set[str]) -> float:
        if not setA or not setB:
            return 0.0
        inter = setA & setB
        union = setA | setB
        wI = sum(_idf(a) for a in inter)
        wU = sum(_idf(a) for a in union)
        return 0.0 if wU == 0 else (wI / wU)

    # 3) Ähnlichkeiten berechnen (nur sinnvolle Paare)
    # Schwellen
    REL_SECOND = 0.90  # Zweitkandidat akzeptieren, wenn >= 90% des Besten (für Split/Merge)

    # alle Kandidaten scoren
    scores = []  # (score, (ot,od), (nt,nd))
    for (ot, od, _txtA) in pool31:
        a = A31[(ot, od)]
        if not a:
            continue
        for (nt, nd, _txtB) in pool40:
            b = A40[(nt, nd)]
            if not b:
                continue
            s = _wjaccard(a, b)
            if s >= JACCARD_PAIR_THRESHOLD:
                scores.append((s, (ot, od), (nt, nd)))

    scores.sort(reverse=True)

    # 4) Greedy Matching (best-first), Mehrfachkanten danach
    used_old, used_new = set(), set()
    pairs = []  # (score, (ot,od), (nt,nd))
    for s, oldk, newk in scores:
        if oldk in used_old or newk in used_new:
            continue
        pairs.append((s, oldk, newk))
        used_old.add(oldk)
        used_new.add(newk)

    # 5) Split/Merge erkennen (relative Nächstbeste)
    # Index von best scores pro Quelle/Ziel
    by_old = defaultdict(list)
    by_new = defaultdict(list)
    for s, oldk, newk in scores:
        by_old[oldk].append((s, newk))
        by_new[newk].append((s, oldk))

    for v in by_old.values():
        v.sort(reverse=True)
    for v in by_new.values():
        v.sort(reverse=True)

    splits = []  # [(old_typ, old_id, [(new_typ,new_id), ...])]
    merges = []  # [([(old_typ,old_id), ...], (new_typ,new_id))]

    # Split: ein alter passt gut zu mehreren neuen
    for (ot, od) in keys31:
        cands = by_old.get((ot, od), [])
        if len(cands) >= 2 and cands[1][0] >= REL_SECOND * cands[0][0]:
            tgt = [(nt, nd) for (_s, (nt, nd)) in cands[:2] if (nt, nd) not in used_new]
            if len(tgt) >= 2:
                splits.append((ot, od, tgt))

    # Merge: mehrere alte passen gut zu einem neuen
    for (nt, nd) in keys40:
        cands = by_new.get((nt, nd), [])
        if len(cands) >= 2 and cands[1][0] >= REL_SECOND * cands[0][0]:
            src = [(ot, od) for (_s, (ot, od)) in cands[:2] if (ot, od) not in used_old]
            if len(src) >= 2:
                merges.append((src, (nt, nd)))

    # 6) Ergebnisse klassifizieren
    moved, moved_changed, common_sameid = [], [], []

    # common_sameid (nur Info) — ID gleich in beiden Versionen
    set31 = set(keys31)
    set40 = set(keys40)
    for key in sorted(set31 & set40):
        (typ, did) = key
        a = i31.get(key, "")
        b = i40.get(key, "")
        delta = _extract_key_deltas(a, b)
        common_sameid.append((typ, did, delta))

    # akzeptierte Paare -> moved/moved_changed
    for s, (ot, od), (nt, nd) in pairs:
        a = i31.get((ot, od), "")
        b = i40.get((nt, nd), "")
        delta = _extract_key_deltas(a, b)
        if delta["added"] or delta["removed"] or delta["changed"]:
            moved_changed.append((ot, od, nt, nd, delta))
        else:
            moved.append((ot, od, nt, nd))

    # Residuen (neu/entfallen)
    claimed_old = {oldk for _, oldk, _ in pairs} | {(t, d) for (t, d, _) in pool31 if not A31[(t, d)]}
    claimed_new = {newk for _, _, newk in pairs} | {(t, d) for (t, d, _) in pool40 if not A40[(t, d)]}

    residual_removed = sorted((t, d) for (t, d) in keys31 if (t, d) not in claimed_old and (t, d) not in set40)
    residual_added = sorted((t, d) for (t, d) in keys40 if (t, d) not in claimed_new and (t, d) not in set31)

    return moved, moved_changed, residual_added, residual_removed, common_sameid, splits, merges


def _build_candidate_block(indices: list[int], max_items: int = MAX_CANDIDATE_ITEMS) -> str:
    """
    Erstellt aus einer Liste von Dokument-Indizes einen formatierten Textblock.
    Dieser Block wird dem LLM im Assessor-Prompt als "Kandidatenliste" zur Verfügung gestellt.
    Er enthält die ID und die erste Zeile des Textes als Kurzbeschreibung.
    """
    seen = set()
    lines = []
    for i in indices:
        md = DOCS[i].metadata or {}
        did = (md.get("display_id") or "").strip()
        if not did or did in seen:
            continue
        seen.add(did)

        # 1. Zeile/erster Satz als Kurzbeschreibung
        txt = (DOCS[i].text or "").strip().splitlines()[0]
        if len(txt) > 300:
            txt = txt[:297].rstrip() + "…"
        lines.append(f"{did} — {txt}")

        if len(lines) >= max_items:
            break
    return "\n".join(lines)


# Prompt-Template, das das LLM in die Rolle eines Assessors versetzt.
ASSESSOR_PROMPT_TMPL = """Rolle: Du bist ein neutraler Automotive-SPICE/Qualitäts-Assessor.

Ziel der Bewertung:
- Nutzerziel (wörtlich/nahe am Wortlaut): "{zieltext}"
- Top-K: {top_k}
- Gegenstand: ausschließlich die unter "Kandidaten" gelisteten Items.

Verbindliche Regeln:
1) Kriterien:
   • WICHTIG: Beginne deine Antwort DIREKT mit der Bewertung. Keine Vorbemerkungen, keine Wiederholung der Aufgabe.
   • Antworte ausschließlich auf Basis des Zieltextes. Leite KEINE eigenen Kriterien ab.
   • Wenn der Zieltext ein konkretes Kriterium nennt (z.B. „… anhand des Kriteriums X …" oder Text in Anführungszeichen), verwende X exakt im Wortlaut als Beurteilungsmaßstab. Paraphrasiere nicht und füge keine weiteren Kriterien hinzu.

2) Mehrere Prozesse:
   • Wenn in der Frage mehrere Prozess-IDs vorkommen, betrachte ALLE genannten Prozesse gleichberechtigt.
   • Die Top-Liste darf nicht ausschließlich aus einem Prozess stammen, sofern vergleichbare Kandidaten aus mehreren Prozessen vorliegen. Bei knappen Abständen begründe die Auswahl kurz.

3) Ausschlüsse/Negationen:
   • Beachte sprachliche Ausschlüsse wie „ohne …", „without …", „exclude …", „excluding …", „nicht mit …". Schließe solche Inhalte aus der Auswahl aus.

4) Quellen & Wissen:
   • Primär: Text der Kandidaten (IDs + Kurztexte im Block „Kandidaten").
   • Erweitert: Du darfst übliches Domänenwissen verwenden. Markiere alles, was NICHT direkt aus den Kandidaten stammt, mit „Expert Insight:".
   • Erfinde keine IDs/Kandidaten. Bei Widerspruch gilt der Kandidatentext.

5) Auswahl & Begründung:
   • Begründe jede Auswahl in 1—2 Sätzen (prägnant, faktennah, projektrelevant).
   • Sortiere absteigend nach erwarteter Relevanz im Sinne des Zieltextes und wähle Top-{top_k}.

6) Empfehlungen:
   • Formuliere anschließend 2—4 prägnante Empfehlungen für die Praxis (ausschließlich auf Basis der Top-Ergebnisse).

Ausgabehinweis:
- {out_hint}
- Es gibt kein fixes Layout.

Hinweise:
- Wenn keine passenden Kandidaten vorhanden sind: Antworte exakt „Keine relevanten Informationen in den Dokumenten gefunden."

Kandidaten:
{candidate_block}""".strip()


# ================================================================================
# 15. HELPER FUNCTIONS FOR DOCUMENT PROCESSING
# ================================================================================

def _semantic_highlights_from_txt(k: int = MAX_SEMANTIC_HIGHLIGHTS) -> list[str]:
    """Liefert 6 inhaltliche Highlights aus den SPICE TXTs - Fokus auf Definitions-/Terminologie-/Modell-Abschnitte"""

    # 1) TXT je Version sammeln
    def _collect_txt(docs):
        chunks = []
        for d in docs or []:
            md = d.metadata or {}
            fn = (md.get("file_name") or "").lower()
            if fn.endswith(".txt"):
                chunks.append(d.text or "")
        return "\n\n".join(chunks)

    t31 = _collect_txt(DOCS_V31)
    t40 = _collect_txt(DOCS_V40)

    if not t31 or not t40:
        return []

    # 2) Relevante Sektionen heuristisch cutten (Überschriften-Orientierung)
    def _sections(t: str):
        # simple Headline-Split: Zeilen in ALL CAPS / Title Case / mit ':' als Header werten
        blocks = {}
        cur = None
        buf = []
        for line in (t or "").splitlines():
            l = line.strip()
            if not l:
                continue
            if re.match(
                    r"^(purpose|definitions?|terminology|model|intro|scope|measurement framework|process capability determination|"
                    r"process reference model|process assessment model|process performance indicators|process capability indicators|process capability levels)\b",
                    l.lower()) or l.endswith(":") or l.isupper():
                if cur and buf:
                    blocks[cur] = "\n".join(buf).strip()
                cur = l.strip(": ")
                buf = []
            else:
                buf.append(l)
        if cur and buf:
            blocks[cur] = "\n".join(buf).strip()
        return blocks

    s31 = _sections(t31)
    s40 = _sections(t40)

    # 3) Gleiche Konzepte paaren (Header-Name-Fuzzy); feingranularer Vergleich pro Paar
    headers31 = list(s31.keys())
    headers40 = list(s40.keys())

    def _best(h, pool):
        # einfacher Fuzzy-Match auf Header
        from difflib import SequenceMatcher
        best, score = None, 0.0
        for q in pool:
            r = SequenceMatcher(None, h.lower(), q.lower()).ratio()
            if r > score:
                best, score = q, r
        return best, score

    candidates = []
    used40 = set()
    for h in headers31:
        h40, sc = _best(h, headers40)
        if h40 and sc >= 0.65:
            used40.add(h40)
            candidates.append((h, h40, s31[h], s40[h40]))

    # 4) Pro gepaartem Konzept: Satz-Delta → aber semantisch bündeln
    bullets = []

    def _core_stems(t: str) -> set[str]:
        # sehr schlank: nur wenige, robuste Wortstämme
        toks = re.findall(r"[a-z]+", (t or "").lower())
        keys = ("communicat", "report", "independ", "assur", "work", "product",
                "evaluat", "monitor", "resourc", "interfac", "escalat")
        return {k for w in toks for k in keys if w.startswith(k)}

    for (h31, h40, b31, b40) in candidates:
        deltas = _extract_key_deltas(b31, b40)
        plus = [s for s in (deltas.get("added") or []) if len(s) > 25][:2]
        minus = [s for s in (deltas.get("removed") or []) if len(s) > 25][:2]
        chg = [s for s in (deltas.get("changed") or []) if len(s) > 25][:2]

        if not (plus or minus or chg):
            continue

        label = h40 if len(h40) <= 40 else h40[:37] + "…"
        parts = []
        if plus:
            parts.append("neu: " + "; ".join(plus))

        # Guard: Wenn Kernkonzepte in beiden Versionen vorkommen, NICHT "entfällt", sondern "bleibt erhalten, aber geändert" (nutzt changed/plus falls vorhanden).
        if minus:
            s31 = _core_stems(b31)
            s40 = _core_stems(b40)
            if s31 & s40:
                if chg:
                    parts.append("bleibt erhalten, aber geändert: " + "; ".join(chg))
                    chg = []
                else:
                    parts.append("bleibt erhalten, aber geändert: " + "; ".join(minus))
            else:
                parts.append("entfällt: " + "; ".join(minus))

        if chg:
            parts.append("präzisiert: " + "; ".join(chg))

        bullets.append(f"{label} — " + " | ".join(parts))

        if len(bullets) >= k:
            break

    # 5) Falls nichts Brauchbares gefunden wurde → leer zurück
    return [f"- {b}" for b in bullets]


def _collect_by_id(ids: list[str], DOCS_X, ID_MAP_X, PARENT_MAP_X, level: int | None):
    """Wie in PFAD 3: Basis + deterministische Kinder + Level-Filter."""
    base = [i for _id in ids for i in ID_MAP_X.get(_id, [])]
    seen = set()
    kids = []
    for b in base:
        did = (DOCS_X[b].metadata.get("display_id") or "").lower()
        for cidx in PARENT_MAP_X.get(did, []):
            if cidx not in seen:
                seen.add(cidx)
                kids.append(cidx)

    idxs = base + kids
    docs = [DOCS_X[i] for i in idxs if 0 <= i < len(DOCS_X)]

    if level in (1, 2):
        allowed = ALLOWED_TYPES_L1 if level == 1 else ALLOWED_TYPES_L2

        def ok(d):
            t = (d.metadata.get("type") or "")
            if t in allowed:
                return True
            if t in RULE_LIKE:
                for p in (d.metadata.get("parents") or []):
                    for pi in ID_MAP_X.get((p or "").lower(), []):
                        if (DOCS_X[pi].metadata.get("type") or "") in allowed:
                            return True
            return False

        docs = [d for d in docs if ok(d)]

    return docs


def _index_docs_by_key(docs):
    out = {}
    for d in docs or []:
        md = d.metadata or {}
        key = ((md.get("type") or ""), (md.get("display_id") or "").strip())
        if key[1]:
            out[key] = (d.text or "")
    return out


def _resolve_title(proc_id: str, ver: str) -> Optional[str]:
    """Liefert den offiziellen Titel aus den Index-Metadaten (falls vorhanden) - ver: "3.1" oder "4.0"""
    pid = (proc_id or "").strip().lower()
    if not pid:
        return None

    if ver == "4.0":
        idmap, docs = ID_MAP_V40, DOCS_V40
    else:
        idmap, docs = ID_MAP_V31, DOCS_V31

    for i in (idmap.get(pid, []) or []):
        try:
            t = (docs[i].metadata or {}).get("title")
            if t and t.strip() and not re.fullmatch(r"[A-Z]{3}\.\d+(?:\.BP\d+)?", t.strip()):
                return t.strip()
        except Exception:
            continue
    return None


def _fmt_id_with_title(proc_id: str, ver: str) -> str:
    t = _resolve_title(proc_id, ver)
    return f"{proc_id} — {t}" if t else proc_id


def collect_docs_for_ids(ids: List[str], id_map: Dict[str, List[int]]) -> List[int]:
    out = []
    seen = set()
    for key in ids:
        for idx in id_map.get(key, []):
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
    return out


def expand_children_of_ids(base_idxs: List[int], documents: List[Document], parent_map: Dict[str, List[int]]) -> List[
    int]:
    out = list(base_idxs)
    seen = set(base_idxs)
    for i in base_idxs:
        did = (documents[i].metadata.get("display_id") or "").lower()
        if not did:
            continue
        for cidx in parent_map.get(did, []):
            if cidx not in seen:
                out.append(cidx)
                seen.add(cidx)
    return out


def sort_indices(idxs: List[int], documents: List[Document], wanted_types: Optional[set]) -> List[int]:
    def natural_key(s: str) -> List[Any]:
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s or "")]

    def score(i: int) -> Tuple[int, List[Any]]:
        typ = documents[i].metadata.get("type", "")
        did = documents[i].metadata.get("display_id", "")
        if wanted_types:
            prio = 0 if typ in wanted_types else 1
        else:
            order = [TYPE_RULE, TYPE_REC, TYPE_OUT, TYPE_OWP, "aspice_base_practice",
                     "aspice_generic_practice", "aspice_process", "kgas_requirement", "generic_json_record"]
            prio = order.index(typ) if typ in order else 99
        return prio, natural_key(did)

    return sorted(idxs, key=score)


def _filter_indices_by_level(indices: list[int], level: int | None) -> list[int]:
    if not level:
        return indices
    allowed = ALLOWED_TYPES_L1 if level == 1 else ALLOWED_TYPES_L2
    res: list[int] = []
    for i in indices:
        t = (DOCS[i].metadata.get("type") or "")
        if t in allowed:
            res.append(i)
            continue
        if t in RULE_LIKE:
            ok = False
            for p in (DOCS[i].metadata.get("parents") or []):
                for pi in ID_MAP.get(p.lower(), []):
                    if (DOCS[pi].metadata.get("type") or "") in allowed:
                        ok = True
                        break
                if ok:
                    break
            if ok:
                res.append(i)
    return res


def _collect_children_for_id(_id: str, *, only_types: set[str] | None = None) -> list[Document]:
    base = [i for i in ID_MAP.get(_id.lower(), [])]
    seen = set(base)
    child = []

    for b in base:
        did = (DOCS[b].metadata.get("display_id") or "").lower()
        for c in PARENT_MAP.get(did, []):
            if c in seen:
                continue
            if (not only_types) or ((DOCS[c].metadata.get("type") or "") in only_types):
                child.append(DOCS[c])
                seen.add(c)

    return child


# ================================================================================
# 16. UNION/FUSION AND COMPARISON FUNCTIONS
# ================================================================================

def answer_both_union_fusion(query_text: str, k_each: int = UNION_FUSION_K_EACH, k_fuse: int = UNION_FUSION_K_FUSE) -> str:
    """Durchsucht v31 und v40 parallel, fusioniert Trefferlisten (RRF), de-dupliziert semantisch und erzeugt EINE konsolidierte Antwort (Kein Vergleich/Delta)"""
    # 0) Output-/Sprach-Hinweise wie im normalen Pfad
    out_hint, want_concise, lang_tag = compute_out_hint_and_flag(query_text)

    # 1) Capability Level ermitteln (für denselben Typ-Filter wie im Einzel-Pfad)
    level = detect_capability_level(query_text)
    allowed = None
    if level in (1, 2):
        allowed = ALLOWED_TYPES_L1 if level == 1 else ALLOWED_TYPES_L2

    # 2) Parallel-Retrieval beider Indizes
    def _retrieve_from(_index, _docs):
        if not _index or not _docs:
            return []
        try:
            retr = _index.as_retriever(similarity_top_k=k_each)
            nodes = retr.retrieve(query_text) or []
        except Exception:
            nodes = []

        hits = []
        for rank, n in enumerate(nodes, start=1):
            node = getattr(n, "node", n)
            txt = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
            md = getattr(node, "metadata", {}) or {}

            # Level-Filter (wie im Einzel-Pfad)
            if allowed:
                t = (md.get("type") or "")
                if t not in allowed:
                    # Rule/Recommendation darf bleiben, wenn Parent erlaubt ist
                    if t in RULE_LIKE:
                        keep = False
                        for p in (md.get("parents") or []):
                            for pi in ID_MAP.get((p or "").lower(), []):
                                if (_docs[pi].metadata.get("type") or "") in allowed:
                                    keep = True
                                    break
                            if keep:
                                break
                        if not keep:
                            continue
                    else:
                        continue

            hits.append({"rank": rank, "text": txt, "meta": md})
        return hits

    hits31 = _retrieve_from(INDEX_V31, DOCS_V31)
    hits40 = _retrieve_from(INDEX_V40, DOCS_V40)

    # 3) RRF-Fusion + Deduplizierung (falls 3.1 und 4.0 gleiche Inhalte haben)
    k_rrf = 60
    bucket = {}

    def _key(md: dict, txt: str) -> str:
        # bevorzugter Schlüssel: display_id/id; fallback: normalisierter Titel/erste Zeile
        did = (md.get("display_id") or md.get("id") or "").strip().lower()
        if did:
            return did
        title = (md.get("title") or "").strip().lower()
        if title:
            return title
        # fallback: erste Zeile des Texts
        return (txt.strip().splitlines()[0] if txt else "").lower()

    def _accumulate(hits, is40: bool):
        for h in hits:
            key = _key(h["meta"], h["text"])
            b = bucket.setdefault(key, {
                "title": h["meta"].get("title") or key,
                "text31": "", "text40": "",
                "score": 0.0,
                "from31": False, "from40": False,
                "meta": h["meta"],
            })
            b["score"] += 1.0 / (k_rrf + h["rank"])
            if is40:
                if len(h["text"]) > len(b["text40"]):
                    b["text40"] = h["text"]
                b["from40"] = True
            else:
                if len(h["text"]) > len(b["text31"]):
                    b["text31"] = h["text"]
                b["from31"] = True

    _accumulate(hits31, is40=False)
    _accumulate(hits40, is40=True)

    fused = sorted(bucket.values(), key=lambda x: x["score"], reverse=True)[:k_fuse]

    # 4) Kontext zusammenstellen (konsolidiert) - wenn aus beiden Versionen vorhanden, kurze Markierung
    context_blocks = []
    for f in fused:
        tags = []
        if f["from31"]:
            tags.append("[v3.1]")
        if f["from40"]:
            tags.append("[v4.0]")
        header = f"### {f['title']} {' '.join(tags)}".rstrip()

        # bevorzugt längeren Text; wenn beide vorhanden, nimm 3.1 als „lead" und 4.0 als Ergänzung
        if f["text31"] and f["text40"]:
            txt = (f["text31"].strip() + "\n\n— 4.0 (ergänzend) —\n" + f["text40"].strip()).strip()
        else:
            txt = (f["text31"] or f["text40"] or "").strip()

        if txt:
            context_blocks.append(header + "\n" + txt)

    context_txt = "\n\n---\n\n".join(context_blocks).strip()

    # 5) Synthese-Prompt (analog zu deinem allgemeinen Non-Listen-Pfad)
    history_txt = _get_history_for_prompt(10)
    if not context_txt:
        # wenn der Kontext leer ist, sauberer Fallback
        return "Keine relevanten Informationen in den Dokumenten gefunden."

    prompt = f"""{lang_tag}

Kontext:
{context_txt}

Arbeitsregeln:
- Antworte ausschließlich auf Basis des Kontexts.
- IDs exakt wiedergeben, keine neuen Kennungen erfinden.
- {out_hint}

Gespräch (gekürzt):
{history_txt}"""

    try:
        return (Settings.llm.complete(prompt).text or "").strip()
    except Exception:
        return "Keine relevanten Informationen in den Dokumenten gefunden."


def answer_compare(query_text: str) -> str:
    """Liefert EINE konsolidierte Vergleichs-Antwort. Mit ID(s): Narrativer Diff mit wenigen Beispiel-IDs - Ohne ID: Narrative Gesamtzusammenfassung"""
    # Sprache/Knappheit
    out_hint, want_concise, lang_tag = compute_out_hint_and_flag(query_text)

    # IDs extrahieren (inkl. Pruning/Aliase wie im Hauptpfad)
    ids_in_prompt = sorted(extract_ids_from_query(query_text))

    # OWP-Whitelist
    if OWP_IDS:
        ids_in_prompt = [i for i in ids_in_prompt if not RX_OWPID_STRICT.fullmatch(i) or i in OWP_IDS]

    level = detect_capability_level(query_text)

    # A) Mit IDs → gezielter Diff (REIN LLM, keine Listen)
    if ids_in_prompt:
        docs31 = _collect_by_id(ids_in_prompt, DOCS_V31, ID_MAP_V31, PARENT_MAP_V31, level) if DOCS_V31 else []
        docs40 = _collect_by_id(ids_in_prompt, DOCS_V40, ID_MAP_V40, PARENT_MAP_V40, level) if DOCS_V40 else []

        i31 = _index_docs_by_key(docs31)
        i40 = _index_docs_by_key(docs40)

        # Scope/Filter basierend auf Query
        consider_outcomes = bool(_RX_OUT.search(query_text))
        # Outcomes standardmäßig entfernen (außer explizit angefragt)
        if not consider_outcomes:
            i31 = {k: v for k, v in i31.items() if k[0] != TYPE_OUT}
            i40 = {k: v for k, v in i40.items() if k[0] != TYPE_OUT}

        # Für die spätere Zuordnung: standardmäßig BPs+GPs; bei Level 1 nur BPs
        MAP_TYPES = {TYPE_BP, TYPE_GP} if level != 1 else {TYPE_BP}

        # 1) Pärchen: gleiche IDs (für Delta/Beispiele)
        pairs = [(k, i31[k], i40[k]) for k in (set(i31.keys()) & set(i40.keys()))]

        # BP/GP-Mengen bilden
        bp31_set = {k for k in i31.keys() if k[0] == TYPE_BP}
        gp31_set = {k for k in i31.keys() if k[0] == TYPE_GP}
        bp40_set = {k for k in i40.keys() if k[0] == TYPE_BP}
        gp40_set = {k for k in i40.keys() if k[0] == TYPE_GP}

        # CL1: GPs vollständig aus dem Matcher entfernen
        if level == 1:
            gp31_set = set()
            gp40_set = set()

        moved, moved_changed, residual_added, residual_removed, common_sameid, splits, merges = _align_cross_ids(
            bp31=bp31_set, gp31=gp31_set, bp40=bp40_set, gp40=gp40_set, i31=i31, i40=i40
        )

        # Cross-ID-Paare als Kontext & Preamble vorbereiten
        xmap_lines: list[str] = []

        def _head_snip(txt: str) -> str:
            # sehr kurze Inhalts-Überschrift erzeugen
            t = (txt or "").strip().split("\n", 1)[0]
            t = re.sub(r"\s+", " ", t)
            return t[:90] + ("…" if len(t) > 90 else "")

        for (o_typ, o_did, n_typ, n_did) in (moved or []):
            if o_typ in MAP_TYPES and n_typ in MAP_TYPES:
                t31 = i31.get((o_typ, o_did), "")
                t40 = i40.get((n_typ, n_did), "")
                xmap_lines.append(
                    f"{o_did} → {n_did} · {_head_snip(t31)} → {_head_snip(t40)}"
                )

        for (o_typ, o_did, n_typ, n_did, _delta) in (moved_changed or []):
            if o_typ in MAP_TYPES and n_typ in MAP_TYPES:
                t31 = i31.get((o_typ, o_did), "")
                t40 = i40.get((n_typ, n_did), "")
                xmap_lines.append(
                    f"{o_did} → {n_did} (angepasst) · {_head_snip(t31)} → {_head_snip(t40)}"
                )

        xmap_ctx = "\n".join(f"- {ln}" for ln in xmap_lines[:8]) or "- (keine)"

        # 3) Scoring für „markante" Beispiele
        scored = []
        for (typ, did), t31, t40 in pairs:
            kd = _extract_key_deltas(t31, t40)
            score = len(kd.get("added", [])) + len(kd.get("removed", [])) + len(kd.get("changed", []))
            scored.append((score, (typ, did), kd, t31, t40))

        # moved/moved_changed ebenfalls für den Kontext berücksichtigen (ohne harte Ausgabe)
        for (o_typ, o_did, n_typ, n_did) in (moved or []):
            t31 = i31.get((o_typ, o_did), "")
            t40 = i40.get((n_typ, n_did), "")
            kd = _extract_key_deltas(t31, t40)
            score = len(kd.get("added", [])) + len(kd.get("removed", [])) + len(kd.get("changed", []))
            scored.append((score, (n_typ, n_did), kd, t31, t40))

        # moved_changed: [(old_typ, old_id, new_typ, new_id, delta)]
        for (o_typ, o_did, n_typ, n_did, _delta) in (moved_changed or []):
            t31 = i31.get((o_typ, o_did), "")
            t40 = i40.get((n_typ, n_did), "")
            kd = _extract_key_deltas(t31, t40)
            score = len(kd.get("added", [])) + len(kd.get("removed", [])) + len(kd.get("changed", []))
            scored.append((score, (n_typ, n_did), kd, t31, t40))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Explizite Move-Hinweise für den LLM-Kontext
        move_notes = []
        for (o_typ, o_did, n_typ, n_did) in (moved or [])[:6]:
            move_notes.append(f"Verschoben/umbenannt: {o_did} → {n_did} ({(o_typ or '').replace('aspice_', '')})")

        for (o_typ, o_did, n_typ, n_did, _delta) in (moved_changed or [])[:6]:
            move_notes.append(f"Geändert: {o_did} → {n_did} ({(o_typ or '').replace('aspice_', '')})")

        # Diese Hinweise ganz vorne in die Kandidaten setzen
        candidates = []
        if move_notes:
            candidates.extend(move_notes)

        # 4) Kompakter Delta-Kontext (nur Beispiele, keine Listen in der Ausgabe)
        for score, key, kd, t31, t40 in scored[:8]:
            typ, did = key if isinstance(key, tuple) else (key[0], key[1])
            typ_short = (typ or "").replace("aspice_", "")
            bits = []
            if kd.get("added"):
                bits.append("+ " + _short("; ".join(kd["added"][:1]), 120))
            if kd.get("removed"):
                bits.append("- " + _short("; ".join(kd["removed"][:1]), 120))
            if kd.get("changed"):
                bits.append("± " + _short("; ".join(kd["changed"][:1]), 120))
            if bits:
                candidates.append(f"{did} ({typ_short}): " + "; ".join(bits))

        delta_ctx = "\n".join(f"- {c}" for c in candidates) or "- (keine markanten Deltas extrahiert)"

        # 5) Semantische Hinweise aus TXT-Dateien hinzufügen
        txt_ctx = "\n".join(_semantic_highlights_from_txt(k=4) or [])

        # 6) EIN LLM-Text
        prompt = f"""{lang_tag}

        {out_hint}

        Erkläre die inhaltlichen Unterschiede für {", ".join(ids_in_prompt).upper()} zwischen Automotive SPICE 3.1 und 4.0.

        - **Wichtig:** Vergleiche nicht nach Gleichheit der IDs. Richte dich allein nach den Cross-ID-Mappings (inkl. BP↔GP, Umnummern, Split/Merge).
        - Nutze **ausschließlich** die bereitgestellten Cross-ID-Zuordnungen und Delta-Anker als Grundlage und beschreibe daraus **in Prosa** die fachlichen Änderungen (z. B. Verschiebung von Strategie→Unabhängigkeit, Prozess→Work-Product-Fokus, neu/entfällt/verschmolzen/aufgeteilt).
        - Benenne **Verschiebungen** explizit (z. B. „früher BPx → jetzt GP y"), **ohne** daraus Tabellen oder starre Listen zu machen.
        - **Keine** Meta-Sätze, **keine** Rollen/Selbstansprachen, **keine** zusätzlichen IDs erfinden, **keine** Outcomes außer explizit gefragt.
        - Verwende ausschließlich die im Kontext gelieferten Benennungen (ID und ggf. offizieller Titel) - erfinde keinen Langnamen.
        - Antwort in sauberem Markdown (Überschriften, kurze Absätze, Bullets nur wo nötig).

        Nutzerfrage:
        {query_text}

        Cross-ID-Zuordnungen (Kontext, nicht replizieren):
        {xmap_ctx}

        Interne Delta-Anker (Kontext, nicht replizieren):
        {delta_ctx}

        Terminologie-/Modell-Hinweise aus Begleit-TXT (Kontext):
        {txt_ctx}
        """

        try:
            llm_text = (Settings.llm.complete(prompt).text or "").strip()
            return (llm_text or "").strip()
        except Exception:
            # Minimaler Fallback: kurzer Absatz aus den Delta-Ankern
            lead = f"Unterschiede für {', '.join(ids_in_prompt).upper()} (3.1 → 4.0):"
            body = " ".join(
                x.split(': ', 1)[-1] for x in candidates[:4]) or "Keine aussagekräftigen Änderungen extrahiert."
            return f"{lead}\n\n— {body}"

    # B) Ohne IDs → globaler Vergleich
    # Globale Delta-Anker für 3.1 vs 4.0 (ohne IDs, ohne Statistiken)
    i31 = _index_docs_by_key(DOCS_V31 or [])
    i40 = _index_docs_by_key(DOCS_V40 or [])

    # Outcomes standardmäßig entfernen (außer explizit angefragt)
    consider_outcomes = bool(_RX_OUT.search(query_text))
    if not consider_outcomes:
        i31 = {k: v for k, v in i31.items() if k[0] != TYPE_OUT}
        i40 = {k: v for k, v in i40.items() if k[0] != TYPE_OUT}

    # CL1-Sonderfall: bei Level 1 NUR BPs, sonst BPs+GPs
    allowed_types = {TYPE_BP} if detect_capability_level(query_text) == 1 else {TYPE_BP, TYPE_GP}
    i31 = {k: v for k, v in i31.items() if k[0] in allowed_types}
    i40 = {k: v for k, v in i40.items() if k[0] in allowed_types}

    # Rein inhaltsbasierter Vergleich (ohne IDs)
    def _collect_corpus_text(i_map: dict[tuple, str]) -> str:
        parts = []
        for (_typ, _did), txt in i_map.items():
            if txt:
                parts.append(_normalize_for_match(txt))
        return "\n".join(parts).strip()

    text31_all = _collect_corpus_text(i31)
    text40_all = _collect_corpus_text(i40)

    # Globale inhaltliche Deltas (Satz-/Satzfragmentebene)
    kd_global = _extract_key_deltas(text31_all, text40_all)  # {"added":[], "removed":[], "changed":[]}

    # Thematische Kurzanker (ohne IDs) aus dem globalen Delta
    MAX_THEMATIC_ANCHORS = 10

    def _mk_bits_from_kd(kd: dict[str, list[str]]) -> list[str]:
        bits = []
        if kd.get("added"):
            bits.append("+ " + _short("; ".join(kd["added"][:2]), 160))
        if kd.get("removed"):
            bits.append("- " + _short("; ".join(kd["removed"][:2]), 160))
        if kd.get("changed"):
            bits.append("± " + _short("; ".join(kd["changed"][:3]), 160))
        return bits

    # 1) Delta-Ankerzeilen (nur Inhalt, keine IDs)
    delta_anchor_lines: list[str] = []
    for b in _mk_bits_from_kd(kd_global):
        delta_anchor_lines.append(b)

    # ggf. auf Top-N kappen
    delta_anchor_lines = delta_anchor_lines[:MAX_THEMATIC_ANCHORS]

    # Vollständige Liste neuer/entfallener Prozesse (deterministisch)
    def _all_process_ids(docs):
        out = []
        for d in docs or []:
            md = d.metadata or {}
            if md.get("type") == "aspice_process":
                did = (md.get("display_id") or "").strip()
                if did:
                    out.append(did)
        return sorted(set(out))

    proc_31 = set(_all_process_ids(DOCS_V31))
    proc_40 = set(_all_process_ids(DOCS_V40))
    full_removed_processes = sorted(proc_31 - proc_40)  # entfallen ggü. 3.1
    full_added_processes = sorted(proc_40 - proc_31)  # neu in 4.0

    # (a) Pre-Candidates (kurze Prozessanker) für den LLM-Kontext
    _added_proc = full_added_processes[:10]
    _removed_proc = full_removed_processes[:10]
    pre_candidates = []
    if _added_proc:
        pre_candidates.append("Neue Prozesse in 4.0: " + ", ".join(_added_proc))
    if _removed_proc:
        pre_candidates.append("Entfallene Prozesse vs 3.1: " + ", ".join(_removed_proc))

    # (b) Must-Facts (Neue/entfernte Prozesse, die später in den LLM-Output injiziert werden)
    must_facts_lines = []
    if full_added_processes:
        must_facts_lines.append(
            "Neue Prozesse in 4.0: " + ", ".join(_fmt_id_with_title(p, "4.0") for p in full_added_processes)
        )
    if full_removed_processes:
        must_facts_lines.append(
            "Entfallene Prozesse (vs 3.1): " + ", ".join(_fmt_id_with_title(p, "3.1") for p in full_removed_processes)
        )

    # (c) Für den Prompt (vollständige Liste, kein Auszug)
    ctx_full_added = "\n".join(f"- {_fmt_id_with_title(p, '4.0')}" for p in full_added_processes) or "- (keine)"
    ctx_full_removed = "\n".join(f"- {_fmt_id_with_title(p, '3.1')}" for p in full_removed_processes) or "- (keine)"

    # 2) Pre-Candidates (kurze Prozessanker)
    candidates: list[str] = pre_candidates[:]
    candidates.extend(delta_anchor_lines)

    # 3) Endgültiger Delta-Kontext (ohne IDs)
    delta_ctx = "\n".join(f"- {c}" for c in candidates) or "- (keine markanten Deltas extrahiert)"

    # zusätzliche semantische Hinweise aus den TXT-Begleitdateien (ohne sie sichtbar zu listen)
    txt_bullets = _semantic_highlights_from_txt(k=6)
    txt_ctx = "\n".join(txt_bullets or [])

    # LLM: eine prägnante Prosa-Erklärung der Modelländerungen.
    prompt = f"""{lang_tag}
    {out_hint}

    Schreibe eine prägnante, inhaltliche Gegenüberstellung der relevanten Änderungen von 3.1 zu 4.0 für die im Kontext sichtbaren Items.

    - Bleibe sachlich und neutral (**Keine** Ich-Form, **keine** Rollen- oder Persona-Formulierungen, **keine** Vorreden).
    - Keine BP-/GP-Kennungen nennen (Prozess-IDs dürfen genannt werden); fasse ausschließlich thematische/fachliche Änderungen zusammen.
    - Benenne klar, was neu betont, abgeschwächt, entfernt oder umdefiniert wurde (Schwerpunkte, Anforderungen, Nachweise, Verantwortlichkeiten).
    - Nutze ausschließlich den gelieferten Kontext; keine zusätzlichen Kennungen erfinden.
    - Antworte gut strukturiert in Markdown.

    Added processes (4.0 only):
    {ctx_full_added}

    Removed processes (not in 4.0):
    {ctx_full_removed}

    Interne Delta-Anker (Kontext, nicht replizieren):
    {delta_ctx}

    Terminologie-/Modell-Hinweise aus Begleit-TXT (Kontext):
    {txt_ctx}
    """

    try:
        llm_text = Settings.llm.complete(prompt).text.strip()
        facts = "\n".join(f"- {ln}" for ln in must_facts_lines)
        if facts:
            # 1) direkt nach "Überblick"
            new_text = re.sub(r"(?im)(^#?\s*überblick\s*$\n+)", r"\1" + facts + "\n\n", llm_text, count=1)
            # 2) falls "Überblick" nicht gefunden: nach der ersten Markdown-Überschrift einfügen
            if new_text == llm_text:
                m = re.search(r"(?m)^#{1,3}\s.*\n+", llm_text)
                if m:
                    pos = m.end()
                    new_text = llm_text[:pos] + facts + "\n\n" + llm_text[pos:]
            # 3) Fallback: ganz oben einfügen
            if new_text == llm_text:
                new_text = facts + "\n\n" + llm_text
            llm_text = new_text
        return (llm_text or "").strip()

    # Fallback: kurze, textuelle Zusammenfassung nur aus den Delta-Ankern + Fix-Fakten
    except Exception:
        lead = "Automotive SPICE 4.0 vs 3.1:"
        body = "\n".join(f"- {c}" for c in candidates) or "- Keine aussagekräftigen Änderungen extrahiert."
        preamble = "\n\n".join(must_facts_lines) or ""
        return f"{lead}\n\n{preamble}\n\n{body}".strip()


# ================================================================================
# 17. QUERY PROCESSING & ROUTING
# ================================================================================

def _get_history_for_prompt(max_msgs: int = MAX_HISTORY_MESSAGES) -> str:
    ensure_state()
    msgs = [m for m in st.session_state.get("messages", []) if m.get("role") in ("user", "assistant")]
    if msgs and msgs[0].get("role") == "assistant" and "Hallo, ich bin Mr. SPICY" in msgs[0].get("content", ""):
        msgs = msgs[1:]
    msgs = msgs[-max_msgs:]
    return "\n".join(
        [("User: " if m.get("role") == "user" else "Assistant: ") + m.get("content", "") for m in msgs]).strip()


def rewrite_to_standalone_question(history: str, user_query: str) -> str:
    """Macht aus einer Folgefrage eine eigenständige Frage. Gibt die Original-Frage unverändert zurück, wenn sie schon eigenständig ist."""
    try:
        prompt = f"""Formuliere die Nutzerfrage als eigenständige Frage, nur wenn sie ohne Kontext unverständlich wäre.
Wenn sie bereits eigenständig ist, gib sie exakt unverändert zurück.

Verlauf:
{history}

Frage:
{user_query}"""
        rewritten = Settings.llm.complete(prompt).text.strip().strip("`'\" ")
        return rewritten or user_query
    except Exception:
        return user_query


def _answer_compare_via_pf3(ids_left: list[str], ids_right: list[str], query_text: str) -> str:
    """PFAD-3-artiger Vergleich für zwei IDs (semantisch, narrativ)"""
    # Sprache/Kürze wie im restlichen System
    out_hint, want_concise, lang_tag = compute_out_hint_and_flag(query_text)
    level = detect_capability_level(query_text)

    def _pf3_context_for(ids: list[str]) -> str:
        # 1) Deterministischer Kern: Eltern + Kinder
        base_idxs = collect_docs_for_ids(ids, ID_MAP)
        child_idxs = expand_children_of_ids(base_idxs, DOCS, PARENT_MAP)
        det = list(dict.fromkeys(base_idxs + child_idxs))[:200]

        # 2) Level-Filter (CL1/CL2)
        if level in (1, 2):
            det = _filter_indices_by_level(det, level)

        core_docs = [DOCS[i] for i in det]

        # 3) Kleiner RAG-Zuschlag
        supp = []
        try:
            retr = index.as_retriever(similarity_top_k=SUPPLEMENTARY_DOCS_COUNT)
            nodes = retr.retrieve(" ".join(ids))
            for n in nodes or []:
                node = getattr(n, "node", n)
                txt = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
                if txt and not any(txt == d.text for d in core_docs):
                    supp.append(Document(text=txt, metadata=getattr(node, "metadata", {})))
        except Exception:
            pass

        final_docs = core_docs + supp
        return "\n\n---\n\n".join((d.text or "").strip() for d in final_docs if (d.text or "").strip())

    ctx_left = _pf3_context_for(ids_left)
    ctx_right = _pf3_context_for(ids_right)

    # Wenn eine Seite leer wäre, lieber normale Erklärung fahren damit kein „leerer" Vergleich entsteht.
    if not (ctx_left.strip() and ctx_right.strip()):
        side = ids_left if ctx_left.strip() else ids_right
        return query_documents_only("Erkläre " + ", ".join(side))

    prompt = f"""Vergleiche inhaltlich die beiden Kontexte A vs. B. Keine Statistiken zur Anzahl Elemente,
sondern fachliche Unterschiede, Überschneidungen und Abgrenzungen. Beziehe dich auf Begriffe/IDs
genau wie im Kontext. Antworte in der Sprache des Nutzers.

Bleibe sachlich und neutral (**Keine** Ich-Form, **keine** Rollen- oder Persona-Formulierungen, **keine** Vorreden, **kein** "im Folgenden")

Ausgabehinweis:
- {out_hint}
- {lang_tag}
{(" - Bitte eher kurz/prägnant." if want_concise else "")}

Frage des Nutzers:
{query_text}

### Kontext A
{ctx_left}

### Kontext B
{ctx_right}"""

    try:
        return Settings.llm.complete(prompt).text
    except Exception as e:
        return f"Fehler bei der Vergleichsantwort: {e}"


def compare_router(query_text: str, qn: str, ids_in_prompt: list[str]) -> str | None:
    # (a) Versionen 3.1 ↔ 4.0 nur bei explizitem Wunsch
    if RX_DIFF.search(query_text or "") and detect_version(query_text) == "both":
        return answer_compare(query_text)

    # (b) BP ↔ GP (gleicher Prozess) → semantischer PFAD-3-Vergleich der Kinder
    if RX_DIFF.search(query_text or "") and _RX_BPWRD.search(qn) and _RX_GPWRD.search(qn):
        proc_ids = [t for t in ids_in_prompt if _PROC_ONLY_RX.fullmatch(t)]
        if proc_ids:
            proc = proc_ids[0].lower()
            bps = _collect_children_for_id(proc, only_types={"aspice_base_practice"})
            gps = _collect_children_for_id(proc, only_types={"aspice_generic_practice"})
            bp_ids = [d.metadata.get("display_id", "") for d in bps]
            gp_ids = [d.metadata.get("display_id", "") for d in gps]
            if bp_ids and gp_ids:
                # einheitlicher, narrativer Vergleich (wiederverwendet PFAD-3-Style)
                return _answer_compare_via_pf3(bp_ids, gp_ids, query_text)
            # Falls eine Seite leer ist → normaler Fluss erklärt die gefundenen IDs
            return None

    # (c) ID ↔ ID: allgemeiner Side-by-Side Vergleich
    if RX_DIFF.search(query_text or "") and len(ids_in_prompt) == 2:
        a, b = ids_in_prompt[0], ids_in_prompt[1]
        return _answer_compare_via_pf3([a], [b], query_text)

    # (d) Prozess ↔ Prozess: wenn 2 Prozess-IDs extrahiert wurden
    if RX_DIFF.search(query_text or ""):
        procs = [t for t in ids_in_prompt if re.fullmatch(r"[A-Z]{3}\.\d{1,2}", t, flags=re.I)]
        if len(procs) == 2:
            p1, p2 = procs[0].lower(), procs[1].lower()
            return _answer_compare_via_pf3([p1], [p2], query_text)

    # kein Compare-Fall → Router gibt an den normalen Fluss zurück
    return None


# ================================================================================
# 18. QUERY ROUTING - MODULAR PFADE
# ================================================================================

def _route_list_mode(query_text: str, ids: List[str], wanted_types: set, level: Optional[int]) -> Optional[str]:
    """
    PFAD 1: Deterministischer Listenmodus
    Gibt formatierte Liste zurück oder None wenn nicht anwendbar.
    """
    if not ids or not wanted_types:
        return None

    # Globale Referenzen (werden von query_documents_only gesetzt)
    global index, DOCS, ID_MAP, PARENT_MAP

    # DIREKTTREFFER-PRÜFUNG (PFAD 1.1)
    if len(ids) == 1:
        base_idxs = collect_docs_for_ids(ids, ID_MAP)
        if base_idxs:
            doc_type_of_id = DOCS[base_idxs[0]].metadata.get("type", "")
            if doc_type_of_id in wanted_types:
                d = DOCS[base_idxs[0]]
                did = (d.metadata.get("display_id") or "").strip()
                body = (d.text or "").strip()
                after = body.split("—", 1)[-1].lstrip() if "—" in body else body
                title = (d.metadata.get("title") or "").strip()
                first_ln = after.split("\n", 1)[0].strip()
                headline = title if title else first_ln
                details = after.split("\n", 1)[1].strip() if "\n" in after else ""

                if not details and (d.metadata.get("type") or "") == "aspice_output_work_product":
                    lst = d.metadata.get("owp_characteristics_list") or []
                    if isinstance(lst, list) and lst:
                        details = "\n".join(s for s in (str(x).strip() for x in lst if x is not None) if s)

                header = f"**Informationen zu {did}:**\n\n- **{did}** — {headline}" if headline else f"**Informationen zu {did}:**\n\n- **{did}**"
                return f"{header}\n{details}" if details else header

    # KINDER SAMMELN
    base_idxs = collect_docs_for_ids(ids, ID_MAP)
    child_idxs = []
    seen = set()
    for bidx in base_idxs:
        canonical = (DOCS[bidx].metadata.get("display_id") or "").lower()
        for cidx in PARENT_MAP.get(canonical, []):
            if cidx in seen:
                continue
            if DOCS[cidx].metadata.get("type") in wanted_types:
                seen.add(cidx)
                child_idxs.append(cidx)

    # ASSESSOR-MODUS (PFAD 1.2)
    if child_idxs and RANKING_EVAL_RX.search(query_text or ""):
        if level in (1, 2):
            child_idxs = _filter_indices_by_level(child_idxs, level)
            if not child_idxs:
                pass

        top_k = _extract_topk(query_text, default_k=DEFAULT_TOP_K)
        zieltxt = _extract_goal_text(query_text)
        proc_ids = [pid for pid in ids if _PROC_ONLY_RX.match(pid)]

        per_proc_cap = max(DEFAULT_PER_PROCESS_CAP, 2 * top_k)
        total_cap = min(max(ASSESSOR_TOTAL_CAP_MIN, per_proc_cap * max(1, len(proc_ids))), ASSESSOR_TOTAL_CAP_MAX)

        try:
            if proc_ids:
                candidate_idxs = _balanced_sample_by_process(child_idxs, proc_ids, total_cap)
            else:
                candidate_idxs = child_idxs[:total_cap]
        except NameError:
            candidate_idxs = child_idxs[:total_cap]

        candidate_block = _build_candidate_block(candidate_idxs, max_items=25)

        if candidate_block.strip():
            out_hint, _, _ = compute_out_hint_and_flag(query_text)
            prompt = ASSESSOR_PROMPT_TMPL.format(
                zieltext=zieltxt,
                top_k=top_k,
                candidate_block=candidate_block,
                out_hint=out_hint
            )
            try:
                return Settings.llm.complete(prompt).text
            except Exception:
                pass

    # LISTEN-AUSGABE (PFAD 1.3)
    if child_idxs:
        child_idxs = _filter_indices_by_level(child_idxs, level)
        child_idxs = sort_indices(child_idxs, DOCS, wanted_types)

        label = {
            frozenset({TYPE_RULE}): "Rules",
            frozenset({TYPE_REC}): "Recommendations",
            frozenset({TYPE_OUT}): "Outcomes",
            frozenset({TYPE_OWP}): "Output Work Products",
            frozenset({TYPE_BP}): "Base Practices",
            frozenset({TYPE_GP}): "Generic Practices",
        }.get(frozenset(wanted_types), "Results")

        out_hint, want_concise, lang_tag = compute_out_hint_and_flag(query_text)

        if want_concise:
            texts = []
            for i in child_idxs[:MAX_CONTEXT_DOCS]:
                d = DOCS[i]
                txt = (d.text or "").strip()
                if txt:
                    texts.append(txt)

            summary_ctx = "\n\n---\n\n".join(texts)
            prompt = f"""        
Beantworte die Frage ausschließlich anhand des Dokumentenkontexts.
{out_hint}
{lang_tag}

Liefere eine knappe Zusammenfassung der {label} für {', '.join(ids)} in 5–8 Stichpunkten.
- Keine langen Zitate.

Wenn Information fehlt, antworte exakt: "Keine relevanten Informationen in den Dokumenten gefunden."

Kontext:
{summary_ctx}
""".strip()

            try:
                return Settings.llm.complete(prompt).text
            except Exception:
                lines = [f"**{label} for {', '.join(ids)}:**"]
                for i in child_idxs:
                    d = DOCS[i]
                    did = (d.metadata.get("display_id") or "").strip()
                    body = (d.text or "").strip()
                    after = body.split("—", 1)[-1].lstrip() if "—" in body else body
                    title = (d.metadata.get("title") or "").strip()
                    first_ln = after.split("\n", 1)[0].strip()
                    headline = title if title else first_ln
                    lines.append(f"- **{did}** — {headline}".rstrip())
                return "\n".join(lines)

        # Ausführliche Liste
        lines = [f"**{label} for {', '.join(ids)}:**"]
        for i in child_idxs:
            d = DOCS[i]
            did = (d.metadata.get("display_id") or "").strip()
            body = (d.text or "").strip()
            after = body.split("—", 1)[-1].lstrip() if "—" in body else body
            title = (d.metadata.get("title") or "").strip()
            first_ln = after.split("\n", 1)[0].strip()
            headline = title if title else first_ln
            details = after.split("\n", 1)[1].strip() if "\n" in after else ""

            inline_tail = ""
            if title:
                if first_ln.lower().startswith(title.lower()):
                    inline_tail = first_ln[len(title):].lstrip("—-:• ").strip()
            else:
                pos = first_ln.find("•")
                if pos != -1:
                    inline_tail = first_ln[pos:].strip()

            if not details and inline_tail:
                details = inline_tail

            if not details and (d.metadata.get("type") or "") == "aspice_output_work_product":
                lst = d.metadata.get("owp_characteristics_list") or []
                if isinstance(lst, list) and lst:
                    details = "\n".join(s for s in (str(x).strip() for x in lst if x is not None) if s)

            header = f"- **{did}** — {headline}" if headline else f"- **{did}**"
            lines.append(f"{header}\n{details}" if details else header)

        return "\n".join(lines)

    # KEINE TREFFER (PFAD 1.4)
    kind = ("rules" if TYPE_RULE in wanted_types else
            "recommendations" if TYPE_REC in wanted_types else
            "outcomes" if TYPE_OUT in wanted_types else
            "output work products")

    if STRICT_LIST_MODE:
        return f"Keine {kind} für {', '.join(ids)} gefunden."

    # Soft-Fallback
    def _process_of(id_str: str) -> str | None:
        m = re.match(r"^([a-z]{3}\.\d{1,2})(?:\.bp\d{1,2})?$", id_str, re.I)
        if m:
            return m.group(1).lower()
        m = re.match(r"^([a-z]{3}\.\d{1,2})\.r[lc]\.\d{1,2}$", id_str, re.I)
        if m:
            return m.group(1).lower()
        return None

    family_procs = {p for _id in ids for p in ([_process_of(_id)] if _process_of(_id) else [])}
    if not family_procs:
        return f"Keine {kind} für {', '.join(ids)} gefunden."

    anchor_idxs = []
    for proc in family_procs:
        for idx in PARENT_MAP.get(proc, []):
            typ = (DOCS[idx].metadata.get("type") or "")
            if typ in {"aspice_base_practice", "aspice_generic_practice"}:
                anchor_idxs.append(idx)

    wanted_set = set(wanted_types)
    cand = []
    seen = set()
    for aidx in anchor_idxs:
        did = (DOCS[aidx].metadata.get("display_id") or "").lower()
        for cidx in PARENT_MAP.get(did, []):
            if cidx in seen:
                continue
            if DOCS[cidx].metadata.get("type") in wanted_set:
                seen.add(cidx)
                cand.append(cidx)

    cand = _filter_indices_by_level(cand, level)

    def _prio(i: int) -> int:
        t = DOCS[i].metadata.get("type", "")
        order = ["aspice_recommendation", "aspice_rule", "aspice_outcome", "aspice_output_work_product",
                 "aspice_base_practice", "aspice_generic_practice", "aspice_process"]
        return order.index(t) if t in order else 99

    seen_did = set()
    uniq = []
    for i in sorted(cand, key=_prio):
        did = (DOCS[i].metadata.get("display_id") or "").strip().lower()
        if did and did not in seen_did:
            seen_did.add(did)
            uniq.append(i)

    uniq = uniq[:20]

    if uniq:
        out_hint, want_concise, lang_tag = compute_out_hint_and_flag(query_text)
        lines = [f"**{kind} (Fallback) für {', '.join(ids)}:**"]
        for i in uniq:
            d = DOCS[i]
            did = (d.metadata.get("display_id") or "").strip()
            body = (d.text or "").strip()
            after = body.split("—", 1)[-1].lstrip() if "—" in body else body
            title = (d.metadata.get("title") or "").strip()
            first_ln = after.split("\n", 1)[0].strip()
            headline = title if title else first_ln

            if want_concise:
                lines.append(f"- **{did}** — {headline}".rstrip())
            else:
                details = after.split("\n", 1)[1].strip() if "\n" in after else ""
                inline_tail = ""
                if title:
                    if first_ln.lower().startswith(title.lower()):
                        inline_tail = first_ln[len(title):].lstrip("—-:• ").strip()
                else:
                    pos = first_ln.find("•")
                    if pos != -1:
                        inline_tail = first_ln[pos:].strip()

                if not details and inline_tail:
                    details = inline_tail

                if not details and (d.metadata.get("type") or "") == "aspice_output_work_product":
                    lst = d.metadata.get("owp_characteristics_list") or []
                    if isinstance(lst, list) and lst:
                        details = "\n".join(s for s in (str(x).strip() for x in lst if x is not None) if s)

                header = f"- **{did}** — {headline}" if headline else f"- **{did}**"
                lines.append(f"{header}\n{details}" if details else header)

        return "\n".join(lines)

    return f"Keine {kind} für {', '.join(ids)} gefunden."


def _route_relation_mode(query_text: str, ids: List[str]) -> Optional[str]:
    """
    PFAD 2: Deterministischer Relationenmodus
    Findet Beziehungen zwischen mehreren IDs.
    """
    if len(ids) < 2:
        return None

    global index, DOCS, ID_MAP, PARENT_MAP

    target_procs = []
    for raw in ids:
        p = _to_proc_id(raw)
        if p:
            target_procs.append(p)
    target_procs = sorted(set(target_procs))

    if len(target_procs) < 2:
        return None

    sets_per_proc = [_descendant_dids_for(pid, max_depth=2) for pid in target_procs]
    common_dids = set.intersection(*sets_per_proc) if sets_per_proc else set()

    if not common_dids:
        sets_per_proc_1 = [_descendant_dids_for(pid, max_depth=1) for pid in target_procs]
        common_dids = set.intersection(*sets_per_proc_1) if sets_per_proc_1 else set()

    candidate_idxs: list[int] = []
    for did in common_dids:
        idxs = [i for i, d in enumerate(DOCS)
                if (d.metadata.get("display_id", "").strip().lower() == did)]
        candidate_idxs.extend(idxs)

    # Harte Filter
    filtered_by_bp_bridge: list[int] = []
    for i in candidate_idxs:
        if all(_has_bp_parent(i, proc) for proc in target_procs):
            filtered_by_bp_bridge.append(i)

    if not filtered_by_bp_bridge:
        proc_prefixes = tuple([p + "." for p in target_procs])
        for i in candidate_idxs:
            did = (DOCS[i].metadata.get("display_id") or "").strip().lower()
            if not did.startswith(proc_prefixes):
                continue
            if any(_has_bp_parent(i, p) for p in target_procs):
                filtered_by_bp_bridge.append(i)

    # Deduplizierung + Sortierung
    type_order = {
        "aspice_recommendation": 0,
        "aspice_rule": 1,
        "aspice_outcome": 2,
        "aspice_output_work_product": 3,
        "aspice_base_practice": 4,
        "aspice_generic_practice": 5,
        "aspice_process": 6,
    }

    seen_did = set()
    unique_candidates: list[int] = []
    for i in sorted(filtered_by_bp_bridge, key=lambda k: type_order.get(DOCS[k].metadata.get("type", ""), 99)):
        did = (DOCS[i].metadata.get("display_id") or "").strip().lower()
        if did and did not in seen_did:
            seen_did.add(did)
            unique_candidates.append(i)

    unique_candidates = unique_candidates[:20]

    if unique_candidates:
        nice_ids = ", ".join(ids)
        lines = [f"**Bezüge zwischen {nice_ids}:**"]

        out_hint, want_concise, lang_tag = compute_out_hint_and_flag(query_text)

        for i in unique_candidates:
            d = DOCS[i]
            did = (d.metadata.get("display_id") or "").strip()
            body = (d.text or "").strip()
            after = body.split("—", 1)[-1].lstrip() if "—" in body else body
            title = (d.metadata.get("title") or "").strip()
            first_ln = after.split("\n", 1)[0].strip()
            headline = title if title else first_ln

            if want_concise:
                lines.append(f"- **{did}** — {headline}".rstrip())
            else:
                details = after.split("\n", 1)[1].strip() if "\n" in after else ""
                inline_tail = ""
                if title:
                    if first_ln.lower().startswith(title.lower()):
                        inline_tail = first_ln[len(title):].lstrip("—-:• ").strip()
                else:
                    pos = first_ln.find("•")
                    if pos != -1:
                        inline_tail = first_ln[pos:].strip()

                if not details and inline_tail:
                    details = inline_tail

                if not details and (d.metadata.get("type") or "") == "aspice_output_work_product":
                    lst = d.metadata.get("owp_characteristics_list") or []
                    if isinstance(lst, list) and lst:
                        details = "\n".join(s for s in (str(x).strip() for x in lst if x is not None) if s)

                header = f"- **{did}** — {headline}" if headline else f"- **{did}**"
                lines.append(f"{header}\n{details}" if details else header)

        return "\n".join(lines)

    return None


def _route_hybrid_mode(query_text: str, ids: List[str], level: Optional[int]) -> str:
    """
    PFAD 3: Hybridmodus (deterministisch + semantisch)
    Für offene Fragen zu IDs.
    """
    global index, DOCS, ID_MAP, PARENT_MAP

    base_idxs = collect_docs_for_ids(ids, ID_MAP)
    child_idxs = expand_children_of_ids(base_idxs, DOCS, PARENT_MAP)

    deterministic_idxs_set = set(base_idxs + child_idxs)
    deterministic_idxs = [i for i in (base_idxs + child_idxs) if
                          i in deterministic_idxs_set and deterministic_idxs_set.remove(i) is None]

    core_docs = [DOCS[i] for i in deterministic_idxs[:MAX_CONTEXT_DOCS]]

    # Level-Filter
    if level in (1, 2):
        allowed = ALLOWED_TYPES_L1 if level == 1 else ALLOWED_TYPES_L2

        def _ok_doc(d: Document) -> bool:
            t = (d.metadata.get("type") or "")
            if t in allowed:
                return True
            if t in RULE_LIKE:
                for p in (d.metadata.get("parents") or []):
                    for pi in ID_MAP.get((p or "").lower(), []):
                        if (DOCS[pi].metadata.get("type") or "") in allowed:
                            return True
            return False

        core_docs = [d for d in core_docs if _ok_doc(d)]

    # Semantische Anreicherung
    supplementary_docs = []
    try:
        retr = index.as_retriever(similarity_top_k=SUPPLEMENTARY_DOCS_COUNT)
        nodes = retr.retrieve(query_text)
        for n in nodes or []:
            node = getattr(n, "node", n)
            if not any(node.get_content() == d.text for d in core_docs):
                supplementary_docs.append(Document(text=node.get_content(), metadata=getattr(node, "metadata", {})))
    except Exception:
        pass

    final_docs = core_docs + supplementary_docs
    context = "\n\n---\n\n".join([(d.text or "").strip() for d in final_docs if (d.text or "").strip()])

    if not context.strip():
        return "Keine relevanten Informationen in den Dokumenten gefunden."

    out_hint, _, lang_tag = compute_out_hint_and_flag(query_text)
    history_txt = _get_history_for_prompt(MAX_HISTORY_MESSAGES)

    prompt = f"""    
Beantworte die Frage des Nutzers präzise und umfassend. Nutze dafür ausschließlich den bereitgestellten Dokumentenkontext.
Strukturiere deine Antwort gut mit Markdown (Überschriften, Listen), um die Lesbarkeit zu maximieren.
Wenn der Kontext Informationen zu der in der Frage genannten ID enthält, stelle diese prominent dar.

Ausgabehinweis:
- {out_hint}
- {lang_tag}

Gespräch (gekürzt):
{history_txt}

Frage:
{query_text}

Kontext:
{context}
"""

    try:
        return Settings.llm.complete(prompt).text
    except Exception as e:
        return f"Fehler bei der Antwortgenerierung: {e}"


def _route_semantic_mode(query_text: str, level: Optional[int]) -> str:
    """
    PFAD 4: Globaler Semantikmodus (klassisches RAG)
    Fallback für Fragen ohne IDs.
    """
    global index, DOCS, ID_MAP, PARENT_MAP

    history_txt = _get_history_for_prompt(MAX_HISTORY_MESSAGES)
    query_text = rewrite_to_standalone_question(history_txt, query_text)

    try:
        # Hole mehr Kandidaten für Re-Ranking
        retr = index.as_retriever(similarity_top_k=DEFAULT_SIMILARITY_TOP_K * 2)
        nodes = retr.retrieve(query_text)

        # Re-Ranking mit Cross-Encoder
        nodes = _rerank_nodes(query_text, nodes, top_k=DEFAULT_SIMILARITY_TOP_K)
        context_docs = []
        for n in nodes or []:
            node = getattr(n, "node", n)
            txt = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
            md = getattr(node, "metadata", {}) or {}
            context_docs.append(Document(text=txt, metadata=md))
    except Exception:
        context_docs = []

    # Level-Filter
    if level in (1, 2) and context_docs:
        allowed = ALLOWED_TYPES_L1 if level == 1 else ALLOWED_TYPES_L2

        def _ctx_ok(md: dict) -> bool:
            t = (md.get("type") or "")
            if t in allowed:
                return True
            if t in RULE_LIKE:
                for p in (md.get("parents") or []):
                    for pi in ID_MAP.get(p.lower(), []):
                        if (DOCS[pi].metadata.get("type") or "") in allowed:
                            return True
            return False

        context_docs = [d for d in context_docs if _ctx_ok(d.metadata)]

    if not context_docs:
        context = ""
    else:
        context = "\n\n---\n\n".join([(d.text or "").strip() for d in context_docs[:MAX_CONTEXT_DOCS]])

    if not context.strip() and not (history_txt or "").strip():
        return "Keine relevanten Informationen in den Dokumenten gefunden."

    out_hint, _, lang_tag = compute_out_hint_and_flag(query_text)

    prompt = f"""Du bist ein präziser Assistent für Qualität und insbesondere Automotive SPICE & KGAS Inhalte.

Ausgabehinweis:
- {out_hint}
- {lang_tag}
- Es gibt kein fixes Layout

Arbeitsregeln:
1) Nutze vorrangig den Dokumentenkontext unter „Kontext:". Wenn die Frage eine Folgewunsch ist (siehe Erkennung unten) und der Dokumentenkontext leer/zu knapp ist, darfst du stattdessen den Inhalt aus „Gespräch (gekürzt)" verwenden. Erfinde keine Inhalte.
2) Antworte in der Sprache der Frage (Deutsch oder Englisch).
3) Nenne IDs exakt (Groß/Kleinschreibung egal), ändere keine Kennungen.
4) Wenn keine ausreichenden Informationen vorhanden sind, antworte exakt mit "Keine relevanten Informationen in den Dokumenten gefunden."
5) Zeige keine Zwischenschritte/Überlegungen.
6) Keine pauschalen Trend-Behauptungen (z. B. „Kommunikation ist nicht länger der Kern"), wenn das Kernkonzept (z. B. communicate/report) in beiden Versionen vorkommt. In diesem Fall formuliere „bleibt erhalten, aber …".

Stil:
- Wenn knapp gefordert (Kurz-Signal) → antworte kurz und bündig.
- Sonst (Standard) → antworte ausführlich und gut strukturiert in Markdown (Überschriften, Absätze, Bullets).

Gespräch (gekürzt):
{history_txt}

Frage:
{query_text}

Kontext:
{context}"""

    try:
        return Settings.llm.complete(prompt).text
    except Exception as e:
        return f"Fehler bei der Antwortgenerierung: {e}"


def query_documents_only(query_text: str) -> str:
    """
    Haupt-Router für Dokumentenabfragen.
    Delegiert an spezialisierte Pfad-Funktionen.
    """
    # Globale Referenzen sicherstellen
    global index, DOCS, ID_MAP, PARENT_MAP

    # ID-Extraktion & Normalisierung
    ids_in_prompt = sorted(extract_ids_from_query(query_text))

    # OWP-Whitelist-Filter
    if OWP_IDS:
        ids_in_prompt = [i for i in ids_in_prompt if not RX_OWPID_STRICT.fullmatch(i) or i in OWP_IDS]

    # Intent & Level Detection
    wanted_types = detect_intent_targets(query_text)
    level = detect_capability_level(query_text)

    # Compare-Router (hat Vorrang)
    qn = normalize_tokens(query_text)
    routed = compare_router(query_text, qn, ids_in_prompt)
    if routed is not None:
        return routed

    # Pfad-Routing
    if ids_in_prompt and wanted_types:
        result = _route_list_mode(query_text, ids_in_prompt, wanted_types, level)
        if result:
            return result

    if len(ids_in_prompt) >= 2 and not wanted_types:
        result = _route_relation_mode(query_text, ids_in_prompt)
        if result:
            return result

    if ids_in_prompt:
        return _route_hybrid_mode(query_text, ids_in_prompt, level)

    return _route_semantic_mode(query_text, level)


# ================================================================================
# 19. EXPORT & PROTOCOL FUNCTIONALITY
# ================================================================================

def create_excel_export(return_df: bool = False):
    """Erstellt eine Excel-Datei aus dem aktuellen Chat-Verlauf im `session_state`."""
    ensure_state()
    messages = st.session_state.get("messages", [])
    user_questions = [m.get("content", "") for m in messages if m.get("role") == "user"]
    assistant_answers = [m.get("content", "") for m in messages if m.get("role") == "assistant"]

    # Eröffnungsnachricht rausfiltern
    if assistant_answers and "Hallo, ich bin Mr. SPICY" in assistant_answers[0]:
        assistant_answers.pop(0)

    min_len = min(len(user_questions), len(assistant_answers))
    df = pd.DataFrame({"question": user_questions[:min_len], "answer": assistant_answers[:min_len]})

    if len(user_questions) > len(assistant_answers):
        df.loc[len(df)] = [user_questions[len(assistant_answers)], "Noch keine Antwort generiert."]

    if return_df:
        return df

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Chat-Verlauf')
    return output.getvalue()


def save_chat_to_protocol(question: str, answer: str):
    """Speichert die aktuelle Frage und Antwort in einer persistenten Protokolldatei (`chat_protocol.xlsx`)"""
    protocol_folder = Path("./protocol")
    protocol_file = protocol_folder / "chat_protocol.xlsx"
    lock_file = protocol_folder / "chat_protocol.lock"
    protocol_folder.mkdir(parents=True, exist_ok=True)

    try:
        with FileLock(str(lock_file), timeout=10):
            # Bestehende Daten laden (robust) oder neues DF anlegen
            try:
                if protocol_file.exists():
                    protokoll_df = pd.read_excel(protocol_file, sheet_name='Protokoll', engine='openpyxl')
                else:
                    protokoll_df = pd.DataFrame(columns=['question', 'answer'])
            except Exception:
                protokoll_df = pd.DataFrame(columns=['question', 'answer'])

            # Neuen Eintrag anhängen
            new_entry_df = pd.DataFrame([{"question": question, "answer": answer}])
            protokoll_df = pd.concat([protokoll_df, new_entry_df], ignore_index=True)

            # Statistik bauen
            unanswered_text = "Keine relevanten Informationen in den Dokumenten gefunden."
            total_questions = len(protokoll_df)
            unanswered_df = protokoll_df[protokoll_df['answer'].str.strip() == unanswered_text]
            num_unanswered = len(unanswered_df)
            num_answered = total_questions - num_unanswered
            percentage_answered = (num_answered / total_questions) * 100 if total_questions > 0 else 0

            stats_summary_df = pd.DataFrame({
                "Statistik": ["Fragen gesamt", "Beantwortet", "Nicht beantwortet", "Erfolgsquote"],
                "Wert": [total_questions, num_answered, num_unanswered, f"{percentage_answered:.1f}%"]
            })

            unanswered_list_df = unanswered_df[['question']].copy().rename(
                columns={'question': 'Nicht beantwortete Fragen'})

            # Atomisch schreiben: erst Temp-Datei, dann os.replace
            with tempfile.NamedTemporaryFile("wb", suffix=".xlsx", delete=False, dir=str(protocol_folder)) as tmp:
                tmp_name = tmp.name

            try:
                with pd.ExcelWriter(tmp_name, engine='openpyxl') as writer:
                    protokoll_df.to_excel(writer, sheet_name='Protokoll', index=False)
                    stats_summary_df.to_excel(writer, sheet_name='Statistik', index=False)
                    unanswered_list_df.to_excel(writer, sheet_name='Statistik', index=False,
                                                startrow=stats_summary_df.shape[0] + 2)
                os.replace(tmp_name, protocol_file)
            finally:
                # Aufräumen, falls etwas schiefging
                if os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except OSError:
                        pass

    except Timeout:
        logger.warning("Protocol is blocked - writing data is skipped", exc_info=True)
        st.warning("Protokolldatei ist gerade belegt. Der Eintrag wird übersprungen.")
    except Exception as e:
        st.error(f"Protokoll konnte nicht gespeichert werden: {e}")


# ================================================================================
# 20. MAIN UI & INTERACTION LOOP
# ================================================================================

df_chat = create_excel_export(return_df=True)
has_conversation = not df_chat.empty
excel_data = create_excel_export() if has_conversation else b""

c_empty, c_btn = st.columns([0.75, 0.25])
with c_btn:
    st.download_button("Chat-Verlauf 📥", data=excel_data, file_name="mrspicy_chat_verlauf.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       disabled=not has_conversation, use_container_width=True)


def _evaluate_practice(practice_id: str) -> str:
    """
    Vollständige NPLF-Bewertung mit:
    - Weakness Detection
    - Rule-basierter Berechnung
    - LLM-generierter strukturierter Ausgabe
    """

    # 1. Prozess-Kontext
    proc = (st.session_state.get("sim_process_id")
            or st.session_state.get("last_process_for_eval")
            or (st.session_state.get("simulation_cfg", {}).get("processes") or [""])[0])

    sim_ver = st.session_state.get("sim_version", "3.1")

    # 2. Evidenz holen (falls vorhanden, sonst leer)
    evidence = st.session_state.get("practice_evidence", {}).get(practice_id)
    if not evidence:
        evidence = PracticeEvidence(practice_id=practice_id)

    # 3. Obligations & Rules laden
    key = f"{proc}:{practice_id}"
    obl_cache = st.session_state.get("obl_cache", {}).get(key, {})
    obligations = list(obl_cache.values())

    rules = parse_rating_rules(practice_id, proc)

    # 4. Weaknesses erkennen
    weaknesses = detect_weaknesses(practice_id, proc, obligations, evidence, rules)

    # 5. Rating berechnen
    band, pct, debug = calculate_practice_rating_v3(
        practice_id, proc, evidence, obligations, rules, weaknesses
    )

    # 6. Kontext für LLM sammeln
    rag = _cached_rag(sim_ver, proc, practice_id)
    rl_block = _format_rule_checklist_cached(sim_ver, proc, practice_id) or "—"

    # Band-Labels
    BAND_LABELS = {
        "N": "Not achieved",
        "P": "Partially achieved",
        "L": "Largely achieved",
        "F": "Fully achieved"
    }

    # 7. Weaknesses formatieren
    weakness_section = ""
    if weaknesses:
        weakness_lines = []
        for w in weaknesses:
            weakness_lines.append(f"**{w.aspect}**")
            weakness_lines.append(f"- Evidence Gap: {w.evidence_gap}")
            weakness_lines.append(f"- Process Risk: {w.process_risk}")
            weakness_lines.append(f"- Impact: {w.impact}, Severity: {w.severity:.1f}")
            weakness_lines.append("")
        weakness_section = "\n".join(weakness_lines)
    else:
        weakness_section = "(Keine signifikanten Schwächen identifiziert)"

    # 8. Applied Rules formatieren
    rules_section = ""
    if debug.get('applied_rules'):
        rules_lines = []
        for r in debug['applied_rules']:
            rules_lines.append(f"- **{r['rule_id']}** ({r['type']}): {r['action']} → {r['impact']}")
        rules_section = "\n".join(rules_lines)
    else:
        rules_section = "(Keine Rules angewendet)"

    # 9. LLM-Prompt für strukturierte Ausgabe
    eval_prompt = f"""Verfasse eine vollständige ASPICE-Bewertung für {practice_id}.

**Berechnetes Rating:** {band} ({pct:.0f}%)

**ASPICE-Kontext (SOLL):**
{rag[:1000]}

**Rating Rules:**
{rl_block[:500]}

**Gesammelte Evidenz:**
{evidence.get_all_evidence_text()}

**Identifizierte Weaknesses:**
{weakness_section}

**Angewandte Rules:**
{rules_section}

---

**Aufgabe:**
Verfasse eine strukturierte Bewertung im folgenden Format:

**Bewertung {practice_id}**

**1) Kurzfeedback zur Antwortqualität**
(2-3 Sätze: Wie vollständig, konkret, klar waren die User-Antworten? Gab es Uploads/Links?)

**2) Evidenz-Zusammenfassung**
(Objektive Aufzählung: Welche konkreten Dokumente, Statements, Nachweise wurden bereitgestellt?)

**3) Schwächen**
{weakness_section if weaknesses else "(Keine signifikanten Schwächen identifiziert)"}

**4) Bewertung**
**{band} – {BAND_LABELS[band]}**

*Begründung:* (2-3 Sätze warum genau dieses Band. Beziehe dich auf Coverage, Weaknesses und angewandte Rules)

**Details:**
- Basis-Coverage: {debug['base_coverage']:.0f}%
- Weakness-Penalty: -{debug['weakness_penalty']:.0f}%
- Nach Rules: {debug['final_pct']:.0f}%
- Angewandte Rules: {len(debug.get('applied_rules', []))}

---

**WICHTIG:**
- Bei Schwächen: Erkläre das konkrete Risiko (nicht nur "fehlt")
- Bei Rules: Erkläre warum sie gegriffen haben
- Sei präzise aber fair
"""

    try:
        msgs = [
            {"role": "system",
             "content": f"Bewerte streng nach ASPICE {sim_ver}. Beachte die Weakness-Definition: kontextspezifisch & evidenzgestützt."},
            {"role": "user", "content": eval_prompt}
        ]

        response = _chat_with_model_tier(msgs, tier=ModelTier.SMART, temperature=0.3)
        return _resp_text(response).strip()

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

        # Fallback: Minimale Bewertung
        fallback = f"""**Bewertung {practice_id}**

**Rating:** {band} – {BAND_LABELS[band]}

**Coverage:** {pct:.0f}%

**Weaknesses:**
{weakness_section}

**Details:**
- Basis-Coverage: {debug['base_coverage']:.0f}%
- Weakness-Penalty: -{debug['weakness_penalty']:.0f}%
- Nach Rules: {debug['final_pct']:.0f}%

(Detaillierte Begründung konnte nicht generiert werden: {e})
"""
        return fallback


# === Helper: EIN Upload-Widget im Simulationsmodus ===
def _render_assessment_evidence_uploader():
    # rotierender Key leert den Uploader nach erfolgreichem Ingest
    nonce = st.session_state.get("evidence_uploader_nonce", 0)
    uploaded = st.file_uploader(
        "📎 Evidenz hochladen oder Links in den Chat pasten",
        type=["pdf", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
        key=f"assessment_evidence_upload_main_{nonce}"
    )

    if not uploaded:
        return

    # Signatur der aktuellen Auswahl (Name+Größe reicht in der Praxis)
    try:
        sig = "|".join(sorted(f"{f.name}:{getattr(f, 'size', 0)}" for f in uploaded))
    except Exception:
        sig = "|".join(sorted(f.name for f in uploaded))

    # Wenn identische Auswahl schon verarbeitet wurde → nichts tun
    if sig and sig == st.session_state.get("last_evidence_upload_sig"):
        return

    added = _ingest_evidence_files(uploaded)
    if added > 0:
        st.session_state.last_evidence_upload_sig = sig

        # Bestätigung anzeigen
        with st.chat_message("assistant"):
            st.markdown(f"✅ {added} Dokument(e) zur Evidenz hinzugefügt")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"✅ {added} Dokument(e) zur Evidenz hinzugefügt"
        })

        # Auto-Message setzen damit Chat-Flow triggert
        if st.session_state.get("simulation_active"):
            st.session_state.auto_user_message = "Dokument hochgeladen"

        # Uploader leeren (neuer Key) und re-rendern
        st.session_state.evidence_uploader_nonce = nonce + 1
        st.rerun()


# === Helper: Assessment starten (saubere Initialisierung + Re-Run) ===
def _start_assessment(process_id: str, version: str, capability_level: str, practice_queue: list):
    # Einzigartige Session-ID für dieses Assessment
    st.session_state.assessment_session_id = datetime.now(UTC).isoformat()

    # Flags & Parameter für die neue Session
    st.session_state.simulation_active = True
    st.session_state.sim_process_id = process_id
    st.session_state.sim_version = version
    st.session_state.sim_capability_level = capability_level

    # Queue sauber setzen
    st.session_state.practice_queue = practice_queue or []
    st.session_state.practice_idx = 0
    st.session_state.last_assessor_question = None

    # Arbeitszustände leeren (keine Chat-Messages oder Evidenz löschen!)
    st.session_state.answers_by_practice = {}
    st.session_state.practice_aspects = {}
    st.session_state.question_meta = {}

    # ALTEN Manager verwerfen, sonst klebt alter Zustand
    st.session_state.pop("assessment_manager", None)

    # Sofort neu rendern -> Header/Stop-Button direkt sichtbar
    st.rerun()

# =====================================================================
# ASSESSMENT SIMULATION: CHAT-FLOW (konsolidiert)
# =====================================================================
if st.session_state.get("simulation_active"):

    # --- Orchestrator initialisieren (ersetzt alten Manager) ---
    if "orchestrator" not in st.session_state:
        proc = st.session_state.get("sim_process_id", "")
        ver = st.session_state.get("sim_version", "3.1")
        st.session_state.orchestrator = AssessmentOrchestrator(proc, ver)

    orchestrator = st.session_state.orchestrator

    # --- gesamte Chat-Historie anzeigen (Intro im Sim-Modus ausblenden) ---
    msgs = st.session_state.get("messages") or []
    if msgs and msgs[0].get("role") == "assistant" and "Hallo, ich bin Mr. SPICY" in msgs[0].get("content", ""):
        msgs = msgs[1:]

    for m in msgs:
        with st.chat_message(m.get("role", "assistant")):
            st.markdown(m.get("content", ""))
            # Fokus persistent anzeigen
            if m.get("focus"):
                st.caption(m["focus"])

    # --- Simulations-Header (Option A): direkt über der ersten Assessor-Frage ---
    process_id = st.session_state.get("sim_process_id") or st.session_state.get("last_process_for_eval", "")
    version = st.session_state.get("sim_version", "3.1")
    cl = st.session_state.get("sim_capability_level", "1")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"**🎯 Assessment läuft** – {process_id or 'N/A'} | SPICE {version} | CL {cl}")
    with col2:
        if st.button("⏹️ Beenden", type="secondary", use_container_width=True, key="stop_assessment_btn"):
            # sauber stoppen (Evidenzindex bleibt unangetastet)
            st.session_state.simulation_active = False
            st.session_state.simulation_round = 0
            st.session_state.simulation_history = []
            st.session_state.last_assessor_question = None
            st.session_state.practice_queue = []
            st.session_state.practice_idx = 0
            st.session_state.answers_by_practice = {}
            st.session_state.practice_aspects = {}
            st.session_state.question_meta = {}
            st.session_state.pop("orchestrator", None)  # ← GEÄNDERT
            st.rerun()

    # Queue für Fokus-Anzeige laden
    queue = st.session_state.get("practice_queue", []) or []

    # EIN Upload-Widget (unter der Historie; gilt nur im Sim-Modus)
    _render_assessment_evidence_uploader()
    st.divider()

    # --- erste Frage nur einmal stellen ---
    if not st.session_state.get("last_assessor_question"):
        q = _next_assessor_question_obl()

        # Fokus-Info sammeln
        queue = st.session_state.get("practice_queue", []) or []
        idx = int(st.session_state.get("practice_idx", 0) or 0)
        focus_caption = ""

        if idx < len(queue):
            proc = st.session_state.get("sim_process_id")
            cur_pr = queue[idx]["id"]
            key = f"{proc}:{cur_pr}"
            obl_cache = st.session_state.get("obl_cache", {}).get(key, {})
            coverage = st.session_state.get("obl_coverage", {}).get(key, [])

            for item in coverage:
                if item["status"] == "open":
                    obl_meta = obl_cache.get(item["id"], {})
                    cur_asp = obl_meta.get("title", "")
                    if cur_asp:
                        focus_caption = f"🔍 **Practice:** {cur_pr} | **Aspekt:** {cur_asp}"
                    break

        with st.chat_message("assistant"):
            st.markdown(q)
            if focus_caption:
                st.caption(focus_caption)

        st.session_state.messages.append({
            "role": "assistant",
            "content": q,
            "focus": focus_caption
        })
        st.session_state.last_assessor_question = q

    # --- Nutzer-Eingabe ---
    user_msg_sim = st.chat_input("Deine Antwort an den Assessor …")

    # Auto-Message von File-Upload übernehmen
    if not user_msg_sim and st.session_state.get("auto_user_message"):
        user_msg_sim = st.session_state.pop("auto_user_message")

    if user_msg_sim:
        # 1) Nutzer-Nachricht anzeigen
        with st.chat_message("user"):
            st.markdown(user_msg_sim)
        st.session_state.messages.append({"role": "user", "content": user_msg_sim})

        # 2) URL-Handling + File-Upload-Erkennung
        has_url_evidence = False

        # ✅ Prüfe zuerst ob Files hochgeladen wurden
        if st.session_state.get("evidence_index"):
            has_url_evidence = True
            logger.info("✅ Evidence index exists - file was uploaded")

        # URLs im Text verarbeiten
        url_contents = process_urls_in_message(user_msg_sim)
        if url_contents:
            success_count = 0
            for url, content in url_contents:
                if not content.startswith("[Fehler:"):
                    st.session_state.setdefault("evidence_doc_hashes", set())
                    norm = " ".join((content or "").split())
                    h = hashlib.sha1(norm[:20000].encode("utf-8")).hexdigest()
                    if h in st.session_state.evidence_doc_hashes:
                        continue

                    doc = Document(
                        text=content,
                        metadata={"file_name": f"Web: {url}", "source": url, "url": url, "content_sha1": h}
                    )
                    if not st.session_state.get("evidence_index"):
                        st.session_state.evidence_index = VectorStoreIndex.from_documents([doc])
                        st.session_state.evidence_doc_hashes.add(h)
                        success_count += 1
                    else:
                        try:
                            st.session_state.evidence_index.insert(doc)
                            st.session_state.evidence_doc_hashes.add(h)
                            success_count += 1
                        except Exception:
                            pass

            if success_count > 0:
                has_url_evidence = True
                ack = f"🌐 {success_count} Link-Inhalt als Evidenz hinzugefügt"
                with st.chat_message("assistant"):
                    st.markdown(ack)
                st.session_state.messages.append({"role": "assistant", "content": ack})

        # Auto-Message bei File-Upload ohne Text
        if not user_msg_sim.strip() and has_url_evidence:
            user_msg_sim = "Dokument hochgeladen"
            logger.info("✅ Auto-generated message for file upload")

        # 3) Orchestrator holen
        if "orchestrator" not in st.session_state:
            proc = st.session_state.get("sim_process_id")
            ver = st.session_state.get("sim_version", "3.1")
            st.session_state.orchestrator = AssessmentOrchestrator(proc, ver)

        orchestrator = st.session_state.orchestrator

        # 4) Queue-Check
        queue = st.session_state.get("practice_queue", []) or []
        idx = int(st.session_state.get("practice_idx", 0) or 0)

        if idx >= len(queue):
            # Assessment abgeschlossen
            summary = _generate_assessment_summary()
            with st.chat_message("assistant"):
                st.markdown(summary)
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.session_state.simulation_active = False
            st.rerun()

        # 5) User-Input verarbeiten
        try:
            result = orchestrator.process_user_input(user_msg_sim, has_url_evidence)

            logger.info(f">>> RAW RESULT type: {type(result)}, value: {result}")

            # Robust: Falls Tuple returnt wird
            if isinstance(result, tuple):
                response = result[0]
                should_advance = result[1] if len(result) > 1 else False
            else:
                response = str(result)
                should_advance = False

            logger.info(f">>> AFTER EXTRACTION: response type={type(response)}, should_advance={should_advance}")

            # KRITISCH: Prüfe ob response SELBST ein Tuple ist (doppelt verpackt)
            if isinstance(response, tuple):
                logger.warning(f">>> DOUBLE TUPLE DETECTED! Unwrapping: {response}")
                response = response[0]
                should_advance = response[1] if len(response) > 1 else should_advance

            # KRITISCH: Prüfe ob response ein STRING-REPRÄSENTATION eines Tuples ist
            if isinstance(response, str) and response.startswith("('") and response.endswith("')"):
                logger.error(f">>> TUPLE-STRING DETECTED! Raw: {response}")
                # Versuche zu parsen
                import ast

                try:
                    parsed = ast.literal_eval(response)
                    if isinstance(parsed, tuple):
                        response = parsed[0]
                        should_advance = parsed[1] if len(parsed) > 1 else should_advance
                        logger.info(f">>> FIXED: response={response}, should_advance={should_advance}")
                except:
                    # Fallback: Nimm den String wie er ist
                    pass

            # Final: Sicherstellen dass response ein reiner String ist
            response = str(response)

        except Exception as e:
            logger.error(f"process_user_input failed: {e}", exc_info=True)
            response = "Entschuldigung, es gab einen technischen Fehler."
            should_advance = False

        logger.info(f">>> FINAL: should_advance={should_advance}, response='{response[:80]}'")

        # 6) Response anzeigen
        with st.chat_message("assistant"):
            st.markdown(response)

        # Fokus für nächste Frage vorbereiten
        focus_caption = ""
        if not should_advance:
            queue = st.session_state.get("practice_queue", []) or []
            idx = int(st.session_state.get("practice_idx", 0) or 0)
            if idx < len(queue):
                proc = st.session_state.get("sim_process_id")
                cur_pr = queue[idx]["id"]
                key = f"{proc}:{cur_pr}"
                obl_cache = st.session_state.get("obl_cache", {}).get(key, {})
                coverage = st.session_state.get("obl_coverage", {}).get(key, [])

                for item in coverage:
                    if item["status"] == "open":
                        obl_meta = obl_cache.get(item["id"], {})
                        cur_asp = obl_meta.get("title", "")
                        if cur_asp:
                            focus_caption = f"🔍 **Practice:** {cur_pr} | **Aspekt:** {cur_asp}"
                        break

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "focus": focus_caption
        })

        # 7) Practice-Wechsel?
        if should_advance:
            logger.info("=" * 80)
            logger.info("SHOULD_ADVANCE = TRUE - Starting evaluation")

            queue = st.session_state.get("practice_queue", []) or []
            idx = int(st.session_state.get("practice_idx", 0) or 0)

            logger.info(f"Queue length: {len(queue)}, Current idx: {idx}")

            if idx >= len(queue):
                logger.info("Assessment complete - all practices done")
                # Assessment komplett fertig
                summary = _generate_assessment_summary()
                with st.chat_message("assistant"):
                    st.markdown(summary)
                st.session_state.messages.append({"role": "assistant", "content": summary})
                st.session_state.simulation_active = False
                st.rerun()

            # Bewertung für aktuelle Practice
            current_practice = queue[idx]["id"]
            logger.info(f"Evaluating practice: {current_practice}")

            try:
                fb = _evaluate_practice(current_practice)
            except Exception as e:
                logger.error(f"Evaluation failed for {current_practice}: {e}", exc_info=True)
                fb = f"**Bewertung für {current_practice}:** Konnte nicht erstellt werden (Fehler: {e})"

            with st.chat_message("assistant"):
                st.markdown(fb)
            st.session_state.messages.append({"role": "assistant", "content": fb})

            # Nächste Practice
            st.session_state.practice_idx = idx + 1
            st.session_state.last_assessor_question = None

            if st.session_state.practice_idx < len(queue):
                # Neue Practice initialisieren
                next_practice = queue[st.session_state.practice_idx]["id"]
                orchestrator.initialize_practice(next_practice)

                # Erste Frage
                if orchestrator.state.current_obligation:
                    try:
                        next_q = generate_question_with_gap_analysis(
                            orchestrator.state,
                            orchestrator.memory,
                            orchestrator.state.current_obligation
                        )

                        # Fokus für neue Practice
                        cur_asp = orchestrator.state.current_obligation.get("title", "")
                        new_focus = f"🔍 **Practice:** {next_practice} | **Aspekt:** {cur_asp}" if cur_asp else ""

                        with st.chat_message("assistant"):
                            st.markdown(next_q)
                            if new_focus:
                                st.caption(new_focus)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": next_q,
                            "focus": new_focus
                        })
                        st.session_state.last_assessor_question = next_q

                    except Exception as e:
                        logger.error(f"Question generation failed: {e}", exc_info=True)
            else:
                # Finales Summary
                summary = _generate_assessment_summary()
                with st.chat_message("assistant"):
                    st.markdown(summary)
                st.session_state.messages.append({"role": "assistant", "content": summary})
                st.session_state.simulation_active = False

        st.rerun()

# Normale Chat-Historie anzeigen, wenn **kein** Assessment läuft
if not st.session_state.get("simulation_active"):
    for m in st.session_state.get("messages", []):
        with st.chat_message(m.get("role", "assistant")):
            st.markdown(m.get("content", ""))
            # Fokus anzeigen falls vorhanden
            if m.get("focus"):
                st.caption(m["focus"])

# Eingabefeld für die Nutzerfrage.
prompt = st.chat_input("Frag mich was.")

if prompt:
    ensure_state()

    # 1. Nutzereingabe sofort in der UI anzeigen.
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Platzhalter mit Spinner anzeigen, während die Antwort generiert wird.
    with st.chat_message("assistant"):
        ph = st.empty()
        with st.spinner("Mr. SPICY denkt nach…"):
            ver = detect_version(prompt)

            if ver == "both":
                # Vergleich oder Union/Fusion
                if RX_DIFF.search(prompt or ""):
                    answer = answer_compare(prompt)
                else:
                    answer = answer_both_union_fusion(prompt)

            elif ver == "4.0" and DOCS_V40:
                # Explizit 4.0 gewählt
                index, DOCS, ID_MAP, PARENT_MAP = INDEX_V40, DOCS_V40, ID_MAP_V40, PARENT_MAP_V40
                answer = query_documents_only(prompt)

            elif DOCS_V31:
                # 3.1 (entweder explizit oder Default)
                index, DOCS, ID_MAP, PARENT_MAP = INDEX_V31, DOCS_V31, ID_MAP_V31, PARENT_MAP_V31
                answer = query_documents_only(prompt)

            elif DOCS_V40:
                # Fallback auf 4.0 wenn 3.1 nicht verfügbar
                index, DOCS, ID_MAP, PARENT_MAP = INDEX_V40, DOCS_V40, ID_MAP_V40, PARENT_MAP_V40
                answer = query_documents_only(prompt)

            else:
                # Keine Dokumente geladen
                answer = "Keine relevanten Informationen in den Dokumenten gefunden."

        ph.markdown(answer)

    # 4. Antwort im Verlauf und im Protokoll speichern.
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_chat_to_protocol(question=prompt, answer=answer)

    # 5. App neu ausführen, um den Zustand zu aktualisieren.
    st.rerun()

# =====================================================================
# ZUSAMMENFASSUNG – nach dem normalen Chat-Flow anhängen
# =====================================================================
def _render_simulation_summary():
    if not st.session_state.get("simulation_history"):
        return
    rows = []
    for proc, data in st.session_state.get("simulation_scores", {}).items():
        for note in data.get("notes", []):
            rows.append((proc, note))
    if not rows:
        return
    st.markdown("---")
    st.subheader("📊 AI Assessor – Zusammenfassung")
    st.markdown(
        "Bewertungen nach **N/P/L/F** je BP/GP und **Gesamt-NPLF** pro Prozess. "
        "Begründungen folgen der Weakness-Regel (kontextspezifisch & evidenzgestützt)."
    )
    for proc, md in rows:
        with st.expander(f"Ergebnisse für {proc.upper()}"):
            st.markdown(md)


if not st.session_state.get("simulation_active", False):
    _render_simulation_summary()

