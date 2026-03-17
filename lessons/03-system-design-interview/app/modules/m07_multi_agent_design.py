"""Module 07 — Multi-Agent System Design
Level: Advanced"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import COLORS, GEMINI_MODELS, LLM_PROVIDERS

try:
    from utils.llm_utils import call_llm
except Exception:
    call_llm = None  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
## Multi-Agent System Design — Interview Preparation Guide

Multi-agent systems are the frontier topic at staff-level AI system design
interviews. Google, Meta, LinkedIn, Uber, and Microsoft all have production
multi-agent deployments. This module gives you the depth to answer with
confidence and the vocabulary to signal senior thinking.

---

## Section 1 — Why Multi-Agent?

Single agents hit three hard limits that multi-agent architectures solve:

**1. Context window constraints.** A 200K-token context window sounds large,
but a full codebase, long research corpus, or multi-day conversation quickly
overflows it. Multi-agent systems distribute state: each agent maintains
its own focused context and hands off structured summaries at boundaries.

**2. Task complexity.** Some tasks require heterogeneous skills — research,
writing, code generation, security review — that cannot be optimally served
by one generalist model. Specialized models, each fine-tuned for a narrow
domain, consistently outperform a single generalist on complex pipelines.

**3. Parallelism.** Sequential reasoning is a bottleneck. When sub-tasks are
independent — reviewing five code files, querying three data sources, running
five unit tests — a multi-agent system executes them in parallel, reducing
wall-clock time by O(n) in the best case.

**The core trade-off:** coordination overhead grows with the number of agents.
A well-designed system minimizes coordination cost while maximizing parallelism
benefit. The interview signal is knowing *when* to add agents vs. *when* the
overhead isn't worth it (for simple linear tasks, a single agent is cheaper).

---

## Section 2 — Orchestration Patterns

This is the core interview topic. You must know all four patterns by name,
understand their communication complexity, and be able to recommend the right
one for a given scenario.

| Pattern | Structure | Comms | Bottleneck | Failure mode | Best for |
|---------|-----------|-------|-----------|--------------|----------|
| **Hierarchical** | Tree: Manager → Supervisor → Executor | O(depth) | Manager node | Manager failure = full stop | Complex workflows, clear task decomposition |
| **Peer-to-Peer (Mesh)** | All agents talk to all | O(n²) | None (decentralized) | Split-brain, consensus hard | Distributed voting, collaborative tasks |
| **Pipeline** | Sequential: A → B → C → D | O(n) | Slowest step | Upstream failure cascades | Data transformation, linear workflows |
| **Event-Driven** | Agents react to events (Kafka-like) | Async events | Event broker | Message ordering issues | Real-time systems, reactive workflows |

### When to use each pattern

**Hierarchical:** "Design a research assistant" — manager decomposes task →
assigns to research, write, and review agents → aggregates results → returns
final answer. Use when: task has clear decomposition, agents have distinct
roles, and you need a single authoritative result.

**Pipeline:** "Build an ETL data pipeline" — extract agent reads raw data →
transform agent cleans and enriches → load agent writes to warehouse. Use
when: workflow is linear with natural handoff points, each stage is pure
function (output of A is input to B), and you want clear retry boundaries.

**Peer-to-Peer:** "Code review system" — linter agent, security agent, style
agent, and performance agent all review the same PR independently → aggregator
merges opinions → majority verdict. Use when: you need multiple perspectives,
no agent should be an authority, and you can afford N × LLM calls per query.

**Event-Driven:** "Real-time infrastructure monitoring" — metric agent publishes
CPU/memory/latency alerts to event bus → remediation agents subscribe and react
(scale up, restart service, page on-call). Use when: system is reactive
(not request-driven), agents are loosely coupled, and you need horizontal scale.

---

## Section 3 — State Management

State is the hardest part of multi-agent system design. Three main approaches:

**Shared State (Blackboard Pattern)**
All agents read/write a common state store (Redis, DynamoDB, Postgres).
- Pros: fast reads/writes, agents always see latest state
- Cons: race conditions when two agents update the same field; requires atomic
  operations (Redis SETNX, DynamoDB conditional writes) or optimistic locking
- Use: LangGraph uses a typed TypedDict as shared state; edges define
  conditional routing based on state values

**Message Passing**
Agents communicate exclusively through messages (queues, event buses).
- Pros: decoupled, auditable (message log is the full execution trace), safe
- Cons: eventual consistency; harder to query "current state" globally
- Use: Redis Pub/Sub for < 1K agents/sec; Apache Kafka for > 10K agents/sec

**Consensus (Distributed Agreement)**
When agents need to agree on a decision (voting, leader election).
- Raft protocol for strong consistency: one leader per term, majority quorum
  required for writes; used by etcd (Kubernetes) and CockroachDB
- For LLM applications: simpler majority voting — each agent outputs a verdict,
  aggregator takes the most common verdict; works for 3–7 agents
- Cost: N × LLM calls + aggregation overhead; worth it when accuracy > speed

**LangGraph State Pattern** (know this by name):
```
TypedDict state → node functions mutate state → conditional edges route flow
```
State is a typed dictionary. Each node (agent) receives the full state and
returns a partial update. The graph engine merges updates. This eliminates
shared-state race conditions while keeping a centralized state object.

---

## Section 4 — Agent Roles & Specialization

Production multi-agent systems use clearly defined roles. Interviewers expect
you to name these precisely:

**Manager / Orchestrator**
Receives the high-level task. Decomposes it into sub-tasks. Routes each
sub-task to the appropriate specialist. Aggregates results. Returns the
final synthesized answer. The manager should be a capable planning model
(Gemini 2.5 Pro, GPT-4o) but does not need domain knowledge.

**Researcher**
Executes web searches, reads documents, retrieves from vector stores,
summarizes source material. Optimized for information gathering, not generation.
Often paired with a Tavily or Perplexity search API. Key capability: structured
output (JSON summaries) to hand off to downstream agents.

**Executor / Coder**
Generates code, executes API calls, performs data transformations. Runs in a
sandboxed environment (Docker container with CPU/memory limits). Must be
idempotent — caller retries on failure without side effects.

**Validator / Critic**
Performs quality control: fact-checking, security review, style enforcement,
test execution. Receives the executor's output and returns a structured verdict:
approve / revise / reject with specific feedback. Critical for agentic systems
that take real-world actions.

**Planner**
Handles long-horizon planning, dependency analysis, and task ordering.
Determines which sub-tasks can run in parallel vs. must be sequential.
Produces a DAG (directed acyclic graph) of tasks that the orchestrator executes.

---

## Section 5 — Failure Modes & Resilience

Multi-agent failure modes are a common deep-dive at staff interviews. Know the
four canonical failure modes and their mitigations:

**Deadlock**
Two agents each waiting for the other to produce output.
- Detection: timeout per agent call (e.g., 30 seconds); if no response,
  declare deadlock and break the cycle
- Resolution: force the lower-priority agent to produce a partial result;
  retry the chain from the last successful checkpoint

**Cascading Failure**
Upstream agent fails → all downstream agents block on missing input.
- Mitigation: circuit breaker pattern per agent-to-agent connection
- Circuit breaker states: CLOSED (normal) → OPEN (blocked after N failures)
  → HALF-OPEN (test one request) → CLOSED (if successful)
- Fallback: route to a simpler fallback agent when the primary is unavailable

**Split-Brain**
Network partition causes two groups of agents to reach conflicting decisions.
- Mitigation: leader election via Raft; only the leader's decisions are applied
- For LLM applications: route all writes through a single coordinator; use
  idempotent operations so duplicate writes are harmless

**Byzantine Failure**
One agent produces incorrect or adversarially crafted output.
- Mitigation: majority voting (3 agents, take the majority verdict eliminates
  one Byzantine agent); output schema validation (reject malformed outputs)
- For code generation: run generated code in a sandbox and validate output
  against expected schema before accepting

**Mitigation Toolkit (know all of these):**
- Heartbeat monitoring: each agent pings a health endpoint every N seconds;
  orchestrator marks agent as failed if N consecutive pings missed
- Retry with exponential backoff + jitter: prevents thundering herd on recovery
- Circuit breaker: stops sending requests to a failing agent; preserves capacity
- Checkpoint/replay: persist state after each successful step; resume from
  last checkpoint on failure (important for long multi-hour workflows)
- Human-in-the-loop: for irreversible actions (send email, deploy to prod),
  require explicit human confirmation before execution

---

## Section 6 — Real-World Implementations

Name-dropping specific frameworks and companies signals production experience:

**LangGraph (LangChain)** — stateful multi-agent with typed state, conditional
edges, and built-in human-in-the-loop. Used by LinkedIn (job recommendation
agent), Uber (route optimization), and 400+ companies in production as of 2025.
The key insight: graph topology replaces ad-hoc if/else routing logic.

**AutoGen (Microsoft)** — conversational multi-agent framework. In 2025,
Microsoft merged AutoGen with Semantic Kernel into a unified agent SDK.
Strength: simulates multi-agent conversations; agents can debate, correct each
other, and reach consensus through dialogue.

**CrewAI** — role-based agents with explicit task delegation. Good for
enterprise workflows where you want to define teams of agents with defined
responsibilities. Less flexible than LangGraph but easier to set up for
straightforward task delegation.

**Google AgentOrchestra (2025)** — hierarchical framework for general-purpose
task solving. Designed for long-horizon tasks requiring planning, research,
and multi-step execution across Google services (Search, Workspace, Cloud).

**LangGraph Code Pattern** (know this structure):

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class TeamState(TypedDict):
    task: str
    research: str
    draft: str
    review: str
    final: str

graph = StateGraph(TeamState)
graph.add_node("researcher", research_agent)
graph.add_node("writer", writing_agent)
graph.add_node("reviewer", review_agent)
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_conditional_edges("reviewer", should_revise,
    {"revise": "writer", "approve": END})
```

---

## Section 7 — Communication Protocol

**TEA Protocol** — the industry standard for agent interoperability:
- **Tool:** what actions the agent can take (function signatures)
- **Environment:** shared resources accessible to the agent (files, APIs, DBs)
- **Agent:** metadata, role, constraints, and capability boundaries

**Message Format** (know the fields):
- `agent_id`: unique identifier of the sender
- `task_id`: groups messages belonging to one task (for distributed tracing)
- `message_type`: request / response / event / error
- `content`: the payload (structured JSON, not free text for machine-to-machine)
- `timestamp`: ISO 8601, used for ordering and timeout detection

**Idempotency:** Every message has a unique `msg_id`. Receivers deduplicate
by storing processed `msg_id` values in a set (Redis SET). If the same message
arrives twice (at-least-once delivery guarantee), the receiver ignores the
duplicate. This is critical for retry-safe distributed systems.

**Message bus scaling:**
- Redis Pub/Sub: < 1K agents, < 100K messages/sec, single datacenter
- Apache Kafka: > 10K agents, millions of messages/sec, multi-datacenter,
  durable message log (replay-based debugging becomes trivial)

---

## Section 8 — Interview Questions & Red/Green Flags

### Common Interview Questions

**"Design a multi-agent system for automated code review"**
Best answer: Peer-to-peer pattern. Four specialist agents review the same PR
independently: linter agent (syntax/style), security agent (OWASP Top 10),
performance agent (complexity analysis), test coverage agent. An aggregator
agent merges all four outputs and produces a final structured report.
Communication complexity: O(n) from PR to each agent, O(n) back to aggregator.

**"How do you handle agent failure in a hierarchical system?"**
Best answer: Three layers of resilience. First, heartbeat monitoring — every
agent pings the orchestrator every 10 seconds; after 3 missed heartbeats,
the agent is marked failed and a backup agent is spawned. Second, circuit
breaker — after 5 consecutive failures, stop routing to that agent and serve
a degraded response. Third, checkpoint/replay — persist orchestrator state
after each successful sub-task completion; on failure, resume from last
checkpoint rather than restarting the full workflow.

**"What communication protocol do you use?"**
Mention TEA protocol, structured JSON messages with `msg_id` for idempotency,
and choose Redis vs. Kafka based on scale. Mentioning message deduplication
is a strong signal — most candidates miss it.

### Red Flags (instant fail)
- No failure handling: "agents just retry indefinitely"
- O(n²) communication without justification: choosing mesh for a sequential ETL pipeline
- No stopping conditions: agent loop with no max_iterations guard (infinite loops in prod)
- Synchronous blocking everywhere: no async/parallel execution of independent sub-tasks
- No state persistence: restart from zero on any failure

### Green Flags (strong positive signals)
- Specifying communication complexity (O(n), O(n²), O(depth))
- Failure mode analysis: naming deadlock, cascading failure, Byzantine failure
- Using LangGraph / AutoGen / CrewAI by name with specific trade-offs
- Mentioning idempotency and message deduplication
- Discussing the coordinator/worker separation and how to avoid single-point-of-failure
- Proposing a specific evaluation harness: task completion rate, step efficiency, cost per task
"""

# ─────────────────────────────────────────────────────────────────────────────
# CODE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

CODE_EXAMPLE = '''from typing import TypedDict, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid

# ── Agent Role Definition ─────────────────────────────────────────────────────
@dataclass
class AgentSpec:
    agent_id: str
    role: str  # "manager" | "researcher" | "executor" | "validator"
    capabilities: list[str]
    max_retries: int = 3

# ── Message Protocol ──────────────────────────────────────────────────────────
@dataclass
class AgentMessage:
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender_id: str = ""
    receiver_id: str = ""
    msg_type: str = "request"  # "request" | "response" | "event" | "error"
    content: str = ""
    task_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

# ── Hierarchical Orchestrator ─────────────────────────────────────────────────
class HierarchicalOrchestrator:
    def __init__(self, agents: list[AgentSpec]):
        self.agents = {a.agent_id: a for a in agents}
        self.message_log: list[AgentMessage] = []

    def decompose_task(self, task: str) -> list[dict]:
        """Manager decomposes task into sub-tasks for specialists."""
        return [
            {"subtask": task, "assigned_to": a.agent_id, "status": "pending"}
            for a in self.agents.values()
            if a.role != "manager"
        ]

    def route_message(self, msg: AgentMessage) -> str:
        """Route message to appropriate agent."""
        self.message_log.append(msg)
        receiver = self.agents.get(msg.receiver_id)
        if receiver is None:
            return f"ERROR: Agent {msg.receiver_id} not found"
        return f"Delivered to {receiver.role} ({msg.receiver_id})"

    def check_consensus(self, results: list[dict]) -> str:
        """Aggregate results (majority vote for validation)."""
        votes = [r.get("verdict", "unknown") for r in results]
        return max(set(votes), key=votes.count)

# ── Circuit Breaker (failure resilience) ──────────────────────────────────────
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, timeout_sec: int = 30):
        self.threshold = failure_threshold
        self.timeout = timeout_sec
        self.failures = 0
        self.state = "CLOSED"  # CLOSED (normal) | OPEN (blocked) | HALF-OPEN (testing)
        self.last_failure_time = None

    def call(self, func: Callable, *args, **kwargs):
        if self.state == "OPEN":
            elapsed = (datetime.now() - self.last_failure_time).seconds
            if elapsed < self.timeout:
                raise RuntimeError(
                    f"Circuit OPEN: agent unavailable for {self.timeout - elapsed}s more"
                )
            self.state = "HALF-OPEN"
        try:
            result = func(*args, **kwargs)
            self.failures = 0
            self.state = "CLOSED"
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = datetime.now()
            if self.failures >= self.threshold:
                self.state = "OPEN"
            raise

# ── LangGraph-style State Pattern ─────────────────────────────────────────────
# Conceptual — requires: pip install langgraph
#
# from langgraph.graph import StateGraph, END
# from typing import TypedDict
#
# class TeamState(TypedDict):
#     task: str
#     research: str
#     draft: str
#     review: str
#     final: str
#
# def should_revise(state: TeamState) -> str:
#     return "revise" if "NEEDS REVISION" in state["review"] else "approve"
#
# graph = StateGraph(TeamState)
# graph.add_node("researcher", research_agent)
# graph.add_node("writer",     writing_agent)
# graph.add_node("reviewer",   review_agent)
# graph.add_edge("researcher", "writer")
# graph.add_edge("writer",     "reviewer")
# graph.add_conditional_edges("reviewer", should_revise,
#     {"revise": "writer", "approve": END})
# chain = graph.compile()
# result = chain.invoke({"task": "Write a report on AI safety"})

# ── Usage Example ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agents = [
        AgentSpec("mgr-01",        "manager",    ["decompose", "aggregate"]),
        AgentSpec("researcher-01", "researcher", ["search", "summarize"]),
        AgentSpec("writer-01",     "executor",   ["write", "format"]),
        AgentSpec("reviewer-01",   "validator",  ["review", "fact-check"]),
    ]
    orch = HierarchicalOrchestrator(agents)
    subtasks = orch.decompose_task("Write a report on AI safety")
    for st in subtasks:
        msg = AgentMessage(
            sender_id="mgr-01",
            receiver_id=st["assigned_to"],
            msg_type="request",
            content=st["subtask"],
            task_id="task-001",
        )
        print(orch.route_message(msg))
'''

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION DATA
# ─────────────────────────────────────────────────────────────────────────────

SIMULATION_TRACES = {
    ("Hierarchical", "Research + Write Article"): [
        {"from": "manager",    "to": "researcher", "msg": "Research AI safety papers from 2024-2025", "step": 1},
        {"from": "researcher", "to": "manager",    "msg": "Found 5 relevant papers on alignment and RLHF", "step": 2},
        {"from": "manager",    "to": "writer",     "msg": "Write 800-word summary using these papers", "step": 3},
        {"from": "writer",     "to": "reviewer",   "msg": "Draft complete — please review for accuracy", "step": 4},
        {"from": "reviewer",   "to": "manager",    "msg": "Approved with minor citation edits", "step": 5},
        {"from": "manager",    "to": "writer",     "msg": "Apply reviewer feedback and finalize", "step": 6},
    ],
    ("Hierarchical", "Multi-stage Code Review"): [
        {"from": "manager",  "to": "linter",    "msg": "Check syntax and style for PR #142", "step": 1},
        {"from": "manager",  "to": "security",  "msg": "Run OWASP security analysis on PR #142", "step": 2},
        {"from": "linter",   "to": "manager",   "msg": "3 style violations found in auth.py", "step": 3},
        {"from": "security", "to": "manager",   "msg": "SQL injection risk detected in query builder", "step": 4},
        {"from": "manager",  "to": "reviewer",  "msg": "Aggregate findings and write final report", "step": 5},
        {"from": "reviewer", "to": "manager",   "msg": "Report complete — PR requires changes", "step": 6},
    ],
    ("Hierarchical", "Data Analysis Pipeline"): [
        {"from": "manager",     "to": "extractor",  "msg": "Extract last 30 days of sales data from warehouse", "step": 1},
        {"from": "extractor",   "to": "manager",    "msg": "Extracted 2.3M rows — schema validated", "step": 2},
        {"from": "manager",     "to": "analyst",    "msg": "Compute revenue trends and anomaly detection", "step": 3},
        {"from": "analyst",     "to": "manager",    "msg": "3 anomalous days detected — revenue spike on day 14", "step": 4},
        {"from": "manager",     "to": "visualizer", "msg": "Generate executive dashboard from analysis", "step": 5},
        {"from": "visualizer",  "to": "manager",    "msg": "Dashboard ready — 4 charts generated", "step": 6},
    ],
    ("Hierarchical", "Real-time Alert Processing"): [
        {"from": "manager",    "to": "monitor",    "msg": "Monitor CPU/memory/latency for prod cluster", "step": 1},
        {"from": "monitor",    "to": "manager",    "msg": "ALERT: p99 latency spiked to 3.2s (threshold: 1s)", "step": 2},
        {"from": "manager",    "to": "diagnoser",  "msg": "Diagnose root cause of latency spike", "step": 3},
        {"from": "diagnoser",  "to": "manager",    "msg": "Root cause: database connection pool exhausted", "step": 4},
        {"from": "manager",    "to": "remediator", "msg": "Scale up connection pool and restart affected pods", "step": 5},
        {"from": "remediator", "to": "manager",    "msg": "Remediation applied — latency back to 180ms", "step": 6},
    ],
    ("Pipeline", "Research + Write Article"): [
        {"from": "researcher",  "to": "fact_checker", "msg": "Research output: 5 papers on AI safety", "step": 1},
        {"from": "fact_checker","to": "writer",        "msg": "Verified sources — 4 pass, 1 flagged low credibility", "step": 2},
        {"from": "writer",      "to": "editor",        "msg": "800-word draft using 4 verified sources", "step": 3},
        {"from": "editor",      "to": "publisher",     "msg": "Edited article — grammar and flow improved", "step": 4},
        {"from": "publisher",   "to": None,            "msg": "Article published to CMS", "step": 5},
    ],
    ("Pipeline", "Multi-stage Code Review"): [
        {"from": "linter",   "to": "security",  "msg": "Style pass complete — 2 warnings", "step": 1},
        {"from": "security", "to": "tester",    "msg": "Security scan complete — 1 critical issue found", "step": 2},
        {"from": "tester",   "to": "reviewer",  "msg": "Unit tests run — 94% coverage", "step": 3},
        {"from": "reviewer", "to": "approver",  "msg": "Human-readable review summary ready", "step": 4},
        {"from": "approver", "to": None,        "msg": "PR rejected: security issue must be fixed first", "step": 5},
    ],
    ("Pipeline", "Data Analysis Pipeline"): [
        {"from": "extractor",  "to": "transformer", "msg": "Raw CSV extracted: 500K rows", "step": 1},
        {"from": "transformer","to": "enricher",    "msg": "Cleaned data: nulls filled, types cast", "step": 2},
        {"from": "enricher",   "to": "aggregator",  "msg": "Joined with customer dimension table", "step": 3},
        {"from": "aggregator", "to": "loader",      "msg": "Revenue aggregates computed by region/product", "step": 4},
        {"from": "loader",     "to": None,          "msg": "Results written to Redshift OLAP table", "step": 5},
    ],
    ("Pipeline", "Real-time Alert Processing"): [
        {"from": "ingestor",   "to": "filter",     "msg": "Raw metrics: 10K events/sec from Prometheus", "step": 1},
        {"from": "filter",     "to": "classifier", "msg": "Filtered to 42 anomalous events", "step": 2},
        {"from": "classifier", "to": "prioritizer","msg": "3 CRITICAL, 8 HIGH, 31 LOW severity alerts", "step": 3},
        {"from": "prioritizer","to": "dispatcher", "msg": "CRITICAL alerts prioritized for immediate action", "step": 4},
        {"from": "dispatcher", "to": None,         "msg": "PagerDuty triggered for 3 on-call engineers", "step": 5},
    ],
    ("Peer-to-Peer", "Research + Write Article"): [
        {"from": "researcher_a","to": "aggregator", "msg": "Research angle: technical AI safety methods", "step": 1},
        {"from": "researcher_b","to": "aggregator", "msg": "Research angle: policy and governance aspects", "step": 2},
        {"from": "researcher_c","to": "aggregator", "msg": "Research angle: industry adoption and case studies", "step": 3},
        {"from": "aggregator",  "to": "writer",     "msg": "Merged perspectives — 3 sections assigned", "step": 4},
        {"from": "writer",      "to": "all_agents", "msg": "Draft ready — all agents vote: approve/revise?", "step": 5},
        {"from": "all_agents",  "to": "aggregator", "msg": "Vote: 3x approve → article published", "step": 6},
    ],
    ("Peer-to-Peer", "Multi-stage Code Review"): [
        {"from": "linter_agent",   "to": "vote_bus", "msg": "Vote: NEEDS_CHANGES — 3 style issues", "step": 1},
        {"from": "security_agent", "to": "vote_bus", "msg": "Vote: NEEDS_CHANGES — SQL injection risk", "step": 2},
        {"from": "perf_agent",     "to": "vote_bus", "msg": "Vote: APPROVED — O(n log n) complexity acceptable", "step": 3},
        {"from": "test_agent",     "to": "vote_bus", "msg": "Vote: APPROVED — 96% test coverage", "step": 4},
        {"from": "vote_bus",       "to": "reporter", "msg": "Consensus: NEEDS_CHANGES (3/4 agents)", "step": 5},
        {"from": "reporter",       "to": None,       "msg": "Final verdict delivered to PR author", "step": 6},
    ],
    ("Peer-to-Peer", "Data Analysis Pipeline"): [
        {"from": "analyst_a", "to": "consensus_bus", "msg": "Statistical analysis: revenue up 12% MoM", "step": 1},
        {"from": "analyst_b", "to": "consensus_bus", "msg": "Trend analysis: growth driven by APAC region", "step": 2},
        {"from": "analyst_c", "to": "consensus_bus", "msg": "Anomaly detection: 2 outlier days flagged", "step": 3},
        {"from": "consensus_bus","to": "validator",  "msg": "Merging 3 analytical perspectives", "step": 4},
        {"from": "validator",  "to": "reporter",    "msg": "Cross-validated — findings consistent", "step": 5},
        {"from": "reporter",   "to": None,          "msg": "Executive summary delivered", "step": 6},
    ],
    ("Peer-to-Peer", "Real-time Alert Processing"): [
        {"from": "node_monitor_a", "to": "event_bus", "msg": "Node A: CPU at 98%, memory at 87%", "step": 1},
        {"from": "node_monitor_b", "to": "event_bus", "msg": "Node B: CPU at 45%, memory at 62%", "step": 2},
        {"from": "node_monitor_c", "to": "event_bus", "msg": "Node C: CPU at 91%, memory at 79%", "step": 3},
        {"from": "event_bus",      "to": "correlator","msg": "2/3 nodes critical — cluster-wide issue", "step": 4},
        {"from": "correlator",     "to": "all_nodes", "msg": "Vote: trigger auto-scale? 2/3 yes", "step": 5},
        {"from": "all_nodes",      "to": "scheduler", "msg": "Consensus reached — adding 4 nodes", "step": 6},
    ],
    ("Event-Driven", "Research + Write Article"): [
        {"from": "event_broker", "to": "researcher",  "msg": "EVENT: new_topic_requested {topic: 'AI safety'}", "step": 1},
        {"from": "researcher",   "to": "event_broker","msg": "EVENT: research_complete {sources: 5}", "step": 2},
        {"from": "event_broker", "to": "writer",      "msg": "EVENT: writing_ready {context: 'research_output'}", "step": 3},
        {"from": "writer",       "to": "event_broker","msg": "EVENT: draft_complete {word_count: 820}", "step": 4},
        {"from": "event_broker", "to": "reviewer",    "msg": "EVENT: review_requested {draft_id: 'doc-042'}", "step": 5},
        {"from": "reviewer",     "to": "event_broker","msg": "EVENT: review_approved {verdict: 'publish'}", "step": 6},
    ],
    ("Event-Driven", "Multi-stage Code Review"): [
        {"from": "event_broker", "to": "linter",    "msg": "EVENT: pr_opened {pr_id: 142, files: 8}", "step": 1},
        {"from": "event_broker", "to": "security",  "msg": "EVENT: pr_opened {pr_id: 142, files: 8}", "step": 2},
        {"from": "linter",       "to": "event_broker","msg": "EVENT: lint_done {issues: 2, severity: 'warning'}", "step": 3},
        {"from": "security",     "to": "event_broker","msg": "EVENT: sec_done {issues: 1, severity: 'critical'}", "step": 4},
        {"from": "event_broker", "to": "aggregator", "msg": "EVENT: all_checks_done — aggregating results", "step": 5},
        {"from": "aggregator",   "to": "event_broker","msg": "EVENT: review_complete {verdict: 'changes_required'}", "step": 6},
    ],
    ("Event-Driven", "Data Analysis Pipeline"): [
        {"from": "event_broker", "to": "ingestor",  "msg": "EVENT: data_upload_complete {rows: 2.3M}", "step": 1},
        {"from": "ingestor",     "to": "event_broker","msg": "EVENT: ingestion_done {table: 'raw_sales'}", "step": 2},
        {"from": "event_broker", "to": "transformer","msg": "EVENT: raw_data_ready {table: 'raw_sales'}", "step": 3},
        {"from": "transformer",  "to": "event_broker","msg": "EVENT: transform_done {table: 'clean_sales'}", "step": 4},
        {"from": "event_broker", "to": "analyst",   "msg": "EVENT: clean_data_ready — trigger analysis", "step": 5},
        {"from": "analyst",      "to": "event_broker","msg": "EVENT: analysis_complete {dashboard_id: 'dash-7'}", "step": 6},
    ],
    ("Event-Driven", "Real-time Alert Processing"): [
        {"from": "prometheus",   "to": "event_broker","msg": "EVENT: metric_alert {metric: 'p99_latency', value: 3.2}", "step": 1},
        {"from": "event_broker", "to": "diagnoser",  "msg": "EVENT: alert_received {severity: 'critical'}", "step": 2},
        {"from": "diagnoser",    "to": "event_broker","msg": "EVENT: root_cause_found {cause: 'db_pool_exhausted'}", "step": 3},
        {"from": "event_broker", "to": "remediator", "msg": "EVENT: remediation_needed {action: 'scale_db_pool'}", "step": 4},
        {"from": "remediator",   "to": "event_broker","msg": "EVENT: remediation_applied {new_pool_size: 100}", "step": 5},
        {"from": "event_broker", "to": "notifier",   "msg": "EVENT: incident_resolved {duration_min: 4}", "step": 6},
    ],
}

# Pattern metadata
PATTERN_META = {
    "Hierarchical": {
        "complexity": "O(depth)",
        "strengths": "Clear task decomposition, single authoritative result, easy to debug",
        "weaknesses": "Manager is a single point of failure; manager must be a capable planning model",
        "real_world": "LinkedIn job recommendation agent (LangGraph), Google AgentOrchestra",
        "recovery": "fallback manager + checkpoint/replay",
    },
    "Pipeline": {
        "complexity": "O(n)",
        "strengths": "Simple to reason about, clear retry boundaries, pure functional handoffs",
        "weaknesses": "Upstream failure blocks all downstream agents; no parallelism",
        "real_world": "Databricks ETL pipelines, Airflow DAG-based agent chains",
        "recovery": "circuit breaker per stage + retry with exponential backoff",
    },
    "Peer-to-Peer": {
        "complexity": "O(n²)",
        "strengths": "No single point of failure, multiple perspectives reduce hallucination",
        "weaknesses": "O(n²) communication cost; consensus is hard; N × LLM calls per task",
        "real_world": "AutoGen debate/consensus patterns, CrewAI peer review workflows",
        "recovery": "majority voting eliminates Byzantine agents; timeout breaks split-brain",
    },
    "Event-Driven": {
        "complexity": "O(n) via broker",
        "strengths": "Highly decoupled, horizontal scale, async execution, replay debugging",
        "weaknesses": "Event ordering can be tricky; broker is now the single point of failure",
        "real_world": "Uber real-time dispatch agents (Kafka), AWS EventBridge agentic workflows",
        "recovery": "event broker HA (Kafka replication), dead-letter queues for failed events",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# AGENT LAYOUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_agent_roles(pattern: str, n_agents: int, task: str) -> list[dict]:
    """
    Return a list of agent dicts with id, label, role, x, y coordinates.
    Layout differs per pattern.
    """
    # Core roles per pattern
    if pattern == "Hierarchical":
        roles_pool = ["manager", "researcher", "writer", "reviewer", "validator",
                      "coder", "planner", "summarizer"]
        roles = ["manager"] + roles_pool[1: n_agents]
    elif pattern == "Pipeline":
        roles_pool = ["extractor", "transformer", "enricher", "analyzer",
                      "validator", "formatter", "loader", "publisher"]
        roles = roles_pool[:n_agents]
    elif pattern == "Peer-to-Peer":
        roles_pool = ["analyst-A", "analyst-B", "analyst-C", "analyst-D",
                      "analyst-E", "analyst-F", "analyst-G", "aggregator"]
        roles = roles_pool[:n_agents - 1] + ["aggregator"]
    else:  # Event-Driven
        roles_pool = ["monitor", "diagnoser", "remediator", "notifier",
                      "scaler", "logger", "reporter"]
        roles = ["event-broker"] + roles_pool[:n_agents - 1]

    agents = []
    for i, role in enumerate(roles):
        if pattern == "Hierarchical":
            if i == 0:  # manager at top
                x, y = 0.5, 0.9
            else:
                # spread workers across bottom rows
                worker_count = len(roles) - 1
                x = (i - 1) / max(worker_count - 1, 1) if worker_count > 1 else 0.5
                y = 0.3 + 0.2 * (1 if worker_count <= 4 else (0 if i <= worker_count // 2 else 0.4))
        elif pattern == "Pipeline":
            x = i / max(len(roles) - 1, 1)
            y = 0.5
        elif pattern == "Peer-to-Peer":
            if i == len(roles) - 1:  # aggregator in center
                x, y = 0.5, 0.5
            else:
                angle = 2 * math.pi * i / (len(roles) - 1)
                x = 0.5 + 0.38 * math.cos(angle)
                y = 0.5 + 0.38 * math.sin(angle)
        else:  # Event-Driven: broker center, others around it
            if i == 0:
                x, y = 0.5, 0.5
            else:
                angle = 2 * math.pi * (i - 1) / (len(roles) - 1)
                x = 0.5 + 0.38 * math.cos(angle)
                y = 0.5 + 0.38 * math.sin(angle)

        agents.append({"id": f"agent-{i}", "label": role, "role": role, "x": x, "y": y})
    return agents


def _role_color(role: str) -> str:
    """Assign a color based on role type."""
    r = role.lower()
    if "manager" in r or "orchestrator" in r:
        return COLORS["info"]          # blue
    if "research" in r or "analyst" in r or "monitor" in r:
        return COLORS["palette"][5]    # purple
    if "writer" in r or "executor" in r or "coder" in r or "transform" in r:
        return COLORS["success"]       # green
    if "review" in r or "valid" in r or "security" in r or "check" in r:
        return COLORS["warning"]       # orange
    if "broker" in r or "event" in r or "aggrega" in r:
        return COLORS["palette"][6]    # cyan
    return COLORS["primary"]           # indigo default


def _get_edges(pattern: str, agents: list[dict], trace: list[dict]) -> list[dict]:
    """
    Derive edges to draw based on the simulation trace.
    For peer-to-peer, also add background all-to-all edges.
    """
    label_to_idx = {a["label"]: i for i, a in enumerate(agents)}
    edges = []

    # Background topology edges for P2P
    if pattern == "Peer-to-Peer":
        for i in range(len(agents) - 1):
            for j in range(i + 1, len(agents)):
                edges.append({"from": i, "to": j, "step": 0, "dashed": True, "msg": ""})

    # Trace edges
    for t in trace:
        from_label = t["from"]
        to_label = t["to"]
        if to_label is None:
            continue
        # fuzzy match: partial label match
        from_idx = next((i for i, a in enumerate(agents) if from_label in a["label"] or a["label"] in from_label), None)
        to_idx = next((i for i, a in enumerate(agents) if to_label in a["label"] or a["label"] in to_label), None)
        if from_idx is not None and to_idx is not None:
            edges.append({"from": from_idx, "to": to_idx, "step": t["step"], "dashed": False, "msg": t["msg"]})

    return edges


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────

def _build_orchestration_figure(
    pattern: str,
    n_agents: int,
    task: str,
    inject_failure: bool,
    agents: list[dict],
    edges: list[dict],
    trace: list[dict],
) -> go.Figure:
    """Build the Plotly network graph for the orchestration visualizer."""
    fig = go.Figure()

    # ── Draw background edges (dashed topology lines) ──────────────────────
    for edge in edges:
        if not edge["dashed"]:
            continue
        a0 = agents[edge["from"]]
        a1 = agents[edge["to"]]
        fig.add_trace(go.Scatter(
            x=[a0["x"], a1["x"], None],
            y=[a0["y"], a1["y"], None],
            mode="lines",
            line=dict(color="rgba(100,100,150,0.2)", width=1, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # ── Determine failed agent (inject_failure: second-to-last non-manager) ─
    failed_idx = None
    if inject_failure and len(agents) > 1:
        failed_idx = len(agents) - 2  # second to last

    # ── Draw trace edges (active message flows) ────────────────────────────
    active_edges = [e for e in edges if not e["dashed"]]
    for edge in active_edges:
        a0 = agents[edge["from"]]
        a1 = agents[edge["to"]]
        is_failed = inject_failure and (edge["from"] == failed_idx or edge["to"] == failed_idx)
        line_color = COLORS["danger"] if is_failed else COLORS["success"]
        dash = "dash" if is_failed else "solid"
        fig.add_trace(go.Scatter(
            x=[a0["x"], a1["x"], None],
            y=[a0["y"], a1["y"], None],
            mode="lines",
            line=dict(color=line_color, width=2.5, dash=dash),
            hoverinfo="text",
            hovertext=f"Step {edge['step']}: {edge['msg'][:60]}",
            showlegend=False,
        ))
        # Step label at midpoint
        mid_x = (a0["x"] + a1["x"]) / 2
        mid_y = (a0["y"] + a1["y"]) / 2
        fig.add_annotation(
            x=mid_x, y=mid_y,
            text=f"<b>{edge['step']}</b>",
            showarrow=False,
            font=dict(size=9, color="#CBD5E1"),
            bgcolor="rgba(30,30,50,0.7)",
            borderpad=2,
        )

    # ── Draw agent nodes ────────────────────────────────────────────────────
    for i, agent in enumerate(agents):
        is_failed = inject_failure and i == failed_idx
        node_color = COLORS["danger"] if is_failed else _role_color(agent["role"])
        node_symbol = "x-open" if is_failed else "circle"
        border_color = "#EF4444" if is_failed else "#1E1E32"

        fig.add_trace(go.Scatter(
            x=[agent["x"]],
            y=[agent["y"]],
            mode="markers+text",
            marker=dict(
                size=40,
                color=node_color,
                symbol=node_symbol,
                line=dict(color=border_color, width=3),
                opacity=0.5 if is_failed else 1.0,
            ),
            text=[agent["label"]],
            textposition="bottom center",
            textfont=dict(size=11, color="#E2E8F0"),
            hoverinfo="text",
            hovertext=f"<b>{agent['label']}</b><br>Role: {agent['role']}<br>{'FAILED' if is_failed else 'Active'}",
            showlegend=False,
        ))

        # Failure badge
        if is_failed:
            fig.add_annotation(
                x=agent["x"], y=agent["y"] + 0.06,
                text="FAILED",
                showarrow=False,
                font=dict(size=9, color=COLORS["danger"]),
                bgcolor="rgba(239,68,68,0.15)",
                bordercolor=COLORS["danger"],
                borderwidth=1,
                borderpad=2,
            )

    # ── Title ───────────────────────────────────────────────────────────────
    failure_note = " — Failure injected" if inject_failure else ""
    fig.update_layout(
        title=dict(
            text=f"{pattern} Pattern — {task}{failure_note}",
            font=dict(size=14, color="#E2E8F0"),
        ),
        xaxis=dict(visible=False, range=[-0.1, 1.1]),
        yaxis=dict(visible=False, range=[0.0, 1.05]),
        plot_bgcolor="#1E1E32",
        paper_bgcolor="#1E1E32",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def _build_metrics_md(
    pattern: str,
    n_agents: int,
    task: str,
    inject_failure: bool,
    trace: list[dict],
    provider: str,
    model: str,
    api_key: str,
) -> str:
    """Build the orchestration results markdown."""
    meta = PATTERN_META[pattern]
    total_messages = len(trace)

    # LLM augmentation: ask Gemini to give a more detailed analysis
    llm_section = ""
    if call_llm is not None:
        prompt = (
            f"You are a staff-level AI system design interviewer. "
            f"In 3-4 concise bullet points, explain the key trade-offs of the "
            f"{pattern} multi-agent orchestration pattern for the task: '{task}'. "
            f"Focus on what a senior engineer must know for a system design interview. "
            f"Use technical vocabulary. Be brief."
        )
        raw = call_llm(prompt, provider, model, api_key)
        llm_section = f"\n\n**AI Analysis ({provider}):**\n{raw}"

    failure_note = (
        f"Yes — agent failure simulated; recovered via **{meta['recovery']}**"
        if inject_failure else "No failure injected"
    )

    steps_md = "\n".join(
        f"{t['step']}. **{t['from']}** → **{t['to'] or 'OUTPUT'}**: _{t['msg']}_"
        for t in trace
    )

    return f"""### Orchestration Results — {pattern} Pattern

| Metric | Value |
|--------|-------|
| Pattern | {pattern} |
| Agents | {n_agents} |
| Messages | {total_messages} |
| Communication complexity | {meta['complexity']} |
| Task | {task} |
| Agent failure | {failure_note} |

**Execution Steps:**
{steps_md}

**Pattern Trade-offs:**
- Strengths: {meta['strengths']}
- Weaknesses: {meta['weaknesses']}
- Real-world: {meta['real_world']}

**Failure Handling:** {failure_note}
{llm_section}"""


def run_orchestration_demo(
    pattern: str,
    n_agents: int,
    task: str,
    inject_failure: bool,
    provider: str,
    model: str,
    api_key: str,
) -> tuple:
    try:
        agents = _get_agent_roles(pattern, n_agents, task)
        key = (pattern, task)
        trace = SIMULATION_TRACES.get(key, SIMULATION_TRACES[(pattern, "Research + Write Article")])
        edges = _get_edges(pattern, agents, trace)

        fig = _build_orchestration_figure(pattern, n_agents, task, inject_failure, agents, edges, trace)
        metrics_md = _build_metrics_md(pattern, n_agents, task, inject_failure, trace, provider, model, api_key)
        return fig, metrics_md
    except Exception:
        import traceback
        return go.Figure(), f"**Error:** {traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def _msg_count(pattern: str, n: int) -> int:
    if pattern == "Peer-to-Peer":
        return n * (n - 1) // 2  # O(n²)
    if pattern == "Hierarchical":
        return 2 * (n - 1)  # manager ↔ each worker (down + up)
    if pattern == "Pipeline":
        return n - 1  # A→B→C→D
    if pattern == "Event-Driven":
        return 2 * (n - 1)  # each agent → broker → each agent
    return n


def _build_comparison_figure(n_agents_max: int) -> go.Figure:
    """Build a 2-panel comparison: message count line chart + radar."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Message Volume vs. Agent Count", "Pattern Capabilities (5-axis Radar)"),
        specs=[[{"type": "scatter"}, {"type": "polar"}]],
        horizontal_spacing=0.12,
    )

    ns = list(range(2, n_agents_max + 1))
    patterns = ["Hierarchical", "Pipeline", "Peer-to-Peer", "Event-Driven"]
    line_colors = [COLORS["info"], COLORS["success"], COLORS["danger"], COLORS["warning"]]

    # ── Left: line chart ────────────────────────────────────────────────────
    for pat, col in zip(patterns, line_colors):
        counts = [_msg_count(pat, n) for n in ns]
        fig.add_trace(go.Scatter(
            x=ns, y=counts,
            mode="lines+markers",
            name=pat,
            line=dict(color=col, width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{pat}</b><br>Agents: %{{x}}<br>Messages: %{{y}}<extra></extra>",
        ), row=1, col=1)

    # ── Right: radar chart ───────────────────────────────────────────────────
    # Scores: Scalability, Resilience, Simplicity, Latency, Parallelism (0–10)
    radar_scores = {
        "Hierarchical":  [7, 6, 8, 7, 6],
        "Pipeline":      [7, 5, 9, 6, 4],
        "Peer-to-Peer":  [5, 9, 4, 4, 9],
        "Event-Driven":  [9, 7, 6, 8, 8],
    }
    axes = ["Scalability", "Resilience", "Simplicity", "Latency", "Parallelism"]

    for pat, col in zip(patterns, line_colors):
        vals = radar_scores[pat]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=axes + [axes[0]],
            name=pat,
            line=dict(color=col, width=2),
            fill="toself",
            fillcolor=col.replace(")", ", 0.1)").replace("rgb(", "rgba(") if col.startswith("rgb(") else col,
            opacity=0.7,
            hovertemplate=f"<b>{pat}</b><br>%{{theta}}: %{{r}}/10<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(color="#94A3B8", size=8)),
            angularaxis=dict(tickfont=dict(color="#E2E8F0", size=10)),
            bgcolor="#1E1E32",
        ),
        plot_bgcolor="#1E1E32",
        paper_bgcolor="#1E1E32",
        font=dict(color="#E2E8F0"),
        legend=dict(
            font=dict(color="#E2E8F0"),
            bgcolor="rgba(30,30,50,0.8)",
        ),
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(
        title_text="Number of Agents",
        tickfont=dict(color="#94A3B8"),
        gridcolor="#2D2D48",
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Number of Messages",
        tickfont=dict(color="#94A3B8"),
        gridcolor="#2D2D48",
        row=1, col=1,
    )
    return fig


def _build_decision_table_md(n_agents: int) -> str:
    msgs = {p: _msg_count(p, n_agents) for p in ["Hierarchical", "Pipeline", "Peer-to-Peer", "Event-Driven"]}
    return f"""### Pattern Decision Guide (n = {n_agents} agents)

| Pattern | Messages at n={n_agents} | Use when | Avoid when |
|---------|--------------------------|----------|------------|
| **Hierarchical** | {msgs['Hierarchical']} | Task has clear decomposition; need single authoritative result | No clear "manager" role; agents are peers |
| **Pipeline** | {msgs['Pipeline']} | Workflow is linear; each stage is a pure transformation | Tasks can run in parallel; need resilience to upstream failure |
| **Peer-to-Peer** | {msgs['Peer-to-Peer']} | Need multiple independent perspectives; no single authority | Cost-sensitive; n > 6 agents (O(n²) cost explodes) |
| **Event-Driven** | {msgs['Event-Driven']} | Real-time reactive system; loose coupling required | Strict ordering required; simple sequential tasks |

### Communication Complexity
- **Pipeline**: O(n) — {msgs['Pipeline']} messages for {n_agents} agents
- **Hierarchical**: O(n) — {msgs['Hierarchical']} messages for {n_agents} agents
- **Event-Driven**: O(n) via broker — {msgs['Event-Driven']} messages for {n_agents} agents
- **Peer-to-Peer**: O(n²) — {msgs['Peer-to-Peer']} messages for {n_agents} agents

### Interview Signal
When an interviewer asks "how do your agents communicate?", mention:
1. **Communication complexity** (O(n) vs O(n²))
2. **Message format** (structured JSON with msg_id for idempotency)
3. **Transport** (Redis Pub/Sub for < 1K agents/sec; Kafka for > 10K)
4. **Failure handling** (circuit breaker + retry + heartbeat)
"""


def run_pattern_comparison(n_agents: int) -> tuple:
    try:
        fig = _build_comparison_figure(n_agents)
        md = _build_decision_table_md(n_agents)
        return fig, md
    except Exception:
        import traceback
        return go.Figure(), f"**Error:** {traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_tab():
    gr.Markdown(
        "# Module 07 — Multi-Agent System Design\n"
        "*Level: Advanced — Google, Meta, LinkedIn, Uber, Microsoft production patterns*"
    )

    with gr.Tabs():
        # ── Sub-tab 1: Orchestration Visualizer ───────────────────────────
        with gr.Tab("Orchestration Visualizer"):
            gr.Markdown(
                "### Orchestration Pattern Visualizer\n"
                "Select a pattern, task, and agent count to see how messages flow "
                "between agents. Inject a failure to see resilience mechanisms."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    pattern_rd = gr.Radio(
                        choices=["Hierarchical", "Pipeline", "Peer-to-Peer", "Event-Driven"],
                        value="Hierarchical",
                        label="Orchestration Pattern",
                    )
                    n_agents_sl = gr.Slider(
                        minimum=2,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Number of Agents",
                    )
                    task_dd = gr.Dropdown(
                        choices=[
                            "Research + Write Article",
                            "Multi-stage Code Review",
                            "Data Analysis Pipeline",
                            "Real-time Alert Processing",
                        ],
                        value="Research + Write Article",
                        label="Task Type",
                    )
                    inject_failure_cb = gr.Checkbox(
                        value=False,
                        label="Inject agent failure (test resilience)",
                    )
                    with gr.Accordion("LLM Provider (optional)", open=False):
                        provider_dd = gr.Dropdown(
                            choices=LLM_PROVIDERS,
                            value=LLM_PROVIDERS[0],
                            label="LLM Provider",
                        )
                        model_dd = gr.Dropdown(
                            choices=GEMINI_MODELS,
                            value=GEMINI_MODELS[0],
                            label="Gemini Model",
                            visible=False,
                        )
                        api_key_tb = gr.Textbox(
                            label="Gemini API Key",
                            placeholder="AIza...",
                            type="password",
                            visible=False,
                        )

                        def _toggle_gemini(prov):
                            show = prov == "Google Gemini"
                            return gr.update(visible=show), gr.update(visible=show)

                        provider_dd.change(
                            _toggle_gemini,
                            inputs=[provider_dd],
                            outputs=[model_dd, api_key_tb],
                        )

                    run_btn = gr.Button("Run Orchestration", variant="primary")

                with gr.Column(scale=2):
                    orch_fig = gr.Plot(label="Agent Communication Graph")

            metrics_md = gr.Markdown(label="Orchestration Results")

            run_btn.click(
                fn=run_orchestration_demo,
                inputs=[pattern_rd, n_agents_sl, task_dd,
                        inject_failure_cb, provider_dd, model_dd, api_key_tb],
                outputs=[orch_fig, metrics_md],
            )

        # ── Sub-tab 2: Pattern Comparison ─────────────────────────────────
        with gr.Tab("Pattern Comparison"):
            gr.Markdown(
                "### Pattern Comparison\n"
                "Compare message volume and capability trade-offs across all four "
                "orchestration patterns. Move the slider to see how O(n²) vs O(n) "
                "communication complexity diverges as agent count grows."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    n_agents_compare_sl = gr.Slider(
                        minimum=2,
                        maximum=10,
                        value=6,
                        step=1,
                        label="Number of Agents",
                    )
                    compare_btn = gr.Button("Compare Patterns", variant="primary")
                with gr.Column(scale=2):
                    compare_fig = gr.Plot(label="Pattern Comparison Charts")
            compare_md = gr.Markdown(label="Decision Guide")

            compare_btn.click(
                fn=run_pattern_comparison,
                inputs=[n_agents_compare_sl],
                outputs=[compare_fig, compare_md],
            )

            # Auto-run on slider change
            n_agents_compare_sl.change(
                fn=run_pattern_comparison,
                inputs=[n_agents_compare_sl],
                outputs=[compare_fig, compare_md],
            )

    # ── Theory accordion ────────────────────────────────────────────────────
    with gr.Accordion("Theory — Multi-Agent System Design", open=False):
        gr.Markdown(THEORY)

    # ── Code reference accordion ─────────────────────────────────────────────
    with gr.Accordion("Code Reference — Multi-Agent Patterns", open=False):
        gr.Code(
            value=CODE_EXAMPLE,
            language="python",
            label="Production Multi-Agent Patterns",
        )
