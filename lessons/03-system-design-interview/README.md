# Lesson 3 — System Design Interview Masterclass

Comprehensive interview preparation for Staff/Senior Engineer roles at Google, Meta, Amazon, Microsoft, and Nvidia.
7 interactive modules covering data modeling, ML system design, RAG pipelines, security, and agentic AI.

---

## Overview

This lesson turns system design theory into hands-on practice. Each module provides:
- An interactive architecture explorer with live Plotly diagrams
- Simulation playgrounds (RAG retrieval, ReAct agent traces, multi-agent patterns)
- LLM-powered feedback on your design answers (optional Gemini API key)
- Trade-off analysis with scoring rubrics that mirror real FAANG interview criteria

---

## Learning Path

| # | Module | Level | Topics |
|---|---|---|---|
| 01 | Data Model Design | 🟡 INTERMEDIATE | Relational vs. NoSQL, schema design, partition keys, CAP theorem |
| 02 | Data Architecture | 🟡 INTERMEDIATE | Lambda/Kappa architecture, CDC, data lakes, OLTP vs. OLAP |
| 03 | ML System Design | 🟡 INTERMEDIATE | Feature stores, training pipelines, serving, monitoring, MLOps |
| 04 | RAG Design | 🔴 ADVANCED | Chunking, embedding, BM25, dense retrieval, hybrid RRF, reranking |
| 05 | RAG Security | 🔴 ADVANCED | Prompt injection, data poisoning, PII extraction, defense in depth |
| 06 | Agentic AI | 🔴 ADVANCED | ReAct loops, tool use, memory, sandboxing, failure handling |
| 07 | Multi-Agent Systems | 🔴 ADVANCED | Orchestration patterns, message passing, consensus, fault tolerance |

---

## Prerequisites

- Python 3.10 or higher
- Lesson 2 (ML with Gradio) completed — helpful but not required
- Familiarity with basic SQL and REST APIs
- Optional: [Google Gemini API key](https://aistudio.google.com/app/apikey) for live LLM feedback

---

## Quick Start

**1. Install dependencies (from project root):**

```bash
uv pip install -r lessons/03-system-design-interview/app/requirements.txt
```

Or with pip:

```bash
pip install -r lessons/03-system-design-interview/app/requirements.txt
```

**2. Run the app:**

```bash
cd lessons/03-system-design-interview/app && python app.py
```

**3. Open in browser:**

```
http://localhost:7862
```

---

## Module Overview

### 🗄️ 01 — Data Model Design
Compare relational, document, wide-column, and graph databases. Design schemas for real interview prompts (social feeds, inventory systems, ride-sharing). Visualize trade-offs with interactive radar charts.

### 🏗️ 02 — Data Architecture
Explore Lambda vs. Kappa architecture, CDC pipelines, and data warehouse patterns. Compare Snowflake, BigQuery, and Redshift. Walk through a streaming data flow diagram.

### 🤖 03 — ML System Design
End-to-end ML system explorer: feature stores → training pipelines → serving → monitoring. Simulate model drift detection and automated retraining triggers.

### 🔍 04 — RAG Design
Live RAG playground: paste your own corpus, tune chunk size and overlap, compare BM25 vs. dense vs. hybrid retrieval, and visualize how retrieved chunks change with query.

### 🔒 05 — RAG Security
Attack simulation lab: see how prompt injection, data poisoning, and PII extraction work against a naive RAG system, then apply defenses and compare outputs.

### ⚡ 06 — Agentic AI
Visualize ReAct agent traces step by step. Configure tool sets, adjust iteration limits, and see how the agent recovers from tool failures.

### 🕸️ 07 — Multi-Agent Systems
Simulate hierarchical, peer-to-peer, and pipeline orchestration patterns. Vary the number of agents, inject a failure, and compare message counts and latency across patterns.

---

## LLM API Integration

All modules work offline with simulated LLM responses. To unlock live AI-powered feedback:

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. In any module, select **"Google Gemini"** from the LLM Provider dropdown
3. Enter your API key in the password field
4. Ask a design question — responses come from the live Gemini API

**Supported models:** gemini-2.5-flash (fast), gemini-2.5-pro (detailed), gemini-3-flash, gemini-3.1-pro

API keys are never stored — they're only held in memory for the current session.

---

## Interview Coverage

This lesson prepares you for these interview formats:

| Format | Covered |
|---|---|
| System Design (60 min whiteboard) | All 7 modules |
| ML System Design (Meta/Google ML role) | Modules 03, 04 |
| AI Infrastructure (Nvidia/Google DeepMind) | Modules 04, 05, 06, 07 |
| Data Engineering (Amazon/Microsoft) | Modules 01, 02 |
| Behavioral: "Tell me about a system you designed" | Exercises in each module |

### What FAANG Interviewers Look For
- **Requirements first** — ask about scale, consistency, latency before designing
- **Trade-off awareness** — show you know the cost of every choice
- **Concrete numbers** — "10K QPS" not "a lot of traffic"
- **Failure modes** — what breaks first and how do you handle it?
- **Iterative refinement** — start simple, then scale up

---

## Slash Commands

These Claude Code slash commands are available at the project root:

**Setup:**
```
/l3-setup
```
Installs all Python dependencies for Lesson 3.

**Run app:**
```
/l3-run
```
Checks dependencies and launches the Gradio app on port 7862.

---

## Exercises

Structured practice problems are in `exercises/`:

| File | Covers |
|---|---|
| `data_model_exercises.md` | 5 exercises on data modeling (Modules 01–02) |
| `ml_system_exercises.md` | 5 exercises on ML systems (Module 03) |
| `rag_agentic_exercises.md` | 6 exercises on RAG + Agentic AI (Modules 04–07) |

Each exercise includes a scoring rubric aligned to FAANG interview expectations.

---

## Project Structure

```
lessons/03-system-design-interview/
├── README.md
├── app/
│   ├── app.py               # Gradio app entry point (port 7862)
│   ├── config.py            # Colors, model lists, constants
│   ├── requirements.txt     # Python dependencies
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── m01_data_model_design.py
│   │   ├── m02_data_architecture.py
│   │   ├── m03_ml_system_design.py
│   │   ├── m04_rag_system_design.py
│   │   ├── m05_rag_security.py
│   │   ├── m06_agentic_ai_design.py
│   │   └── m07_multi_agent_design.py
│   └── utils/
│       ├── __init__.py
│       ├── diagram_utils.py      # Plotly architecture diagram helpers
│       ├── llm_utils.py          # LLM call abstraction (simulated + Gemini)
│       └── simulation_utils.py   # RAG, ReAct, multi-agent simulators
└── exercises/
    ├── data_model_exercises.md
    ├── ml_system_exercises.md
    └── rag_agentic_exercises.md
```

---

## Dependencies

| Package | Purpose |
|---|---|
| gradio>=4.0.0 | Interactive UI framework |
| plotly>=5.0.0 | Architecture diagrams and charts |
| scikit-learn>=1.3.0 | TF-IDF for dense retrieval simulation |
| numpy>=1.24.0 | Numerical computations |
| pandas>=2.0.0 | Data manipulation |
| scipy>=1.11.0 | Statistical utilities |
| rank_bm25>=0.2.2 | BM25 retrieval simulation |
| google-generativeai>=0.8.0 | Optional Gemini API integration |
