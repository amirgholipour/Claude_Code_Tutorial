# RAG & Agentic AI System Design Exercises

Practice exercises for Modules 04–07 (RAG Design, RAG Security, Agentic AI, Multi-Agent Systems).
Target companies: Google DeepMind, OpenAI, Anthropic, Microsoft Copilot, Amazon Bedrock.

---

## Exercise 1 — Design an Enterprise Knowledge Assistant (Microsoft Copilot Level)

**Prompt:**
Design a RAG-based knowledge assistant for a Fortune 500 company:
- 500,000 internal documents: PDFs, Word, Confluence, Jira, email threads
- 20,000 employees querying simultaneously at peak
- Strict access control: employees only see documents they're permitted to read
- Answers must cite sources with document title, section, and relevance score
- Hallucination rate must be < 1% on verifiable facts

**Your task:**
1. Design the full ingestion pipeline (crawl → chunk → embed → index)
2. What chunking strategy would you use for different document types?
3. Design the access-control-aware retrieval layer — how do you enforce permissions at query time?
4. How would you evaluate and minimize hallucination?
5. Design the citation and provenance system
6. Describe your strategy for keeping the index fresh as documents are updated/deleted

**Key concepts to demonstrate:**
- Permission-filtered vector search (metadata filtering)
- Sentence-window vs. hierarchical chunking
- Faithfulness evaluation (RAGAS or similar)
- Delta indexing and tombstoning deleted documents

---

## Exercise 2 — Design a Real-Time Web RAG System (Perplexity Level)

**Prompt:**
Design a RAG system that retrieves live web content to answer queries:
- Queries require information published in the last 24 hours (news, stock prices, sports scores)
- 1M queries per day
- Must synthesize information from 5–10 web sources per query
- Source credibility and freshness matter (Reuters > random blog)
- Response latency target: < 3 seconds end-to-end

**Your task:**
1. Design the real-time web crawling and indexing pipeline
2. How would you rank sources by credibility? Design a source quality scoring model
3. Describe your strategy for deduplicating near-identical content from multiple sources
4. Design the query decomposition for multi-hop questions ("What did the CEO of NVIDIA say today about AI chips?")
5. How would you handle contradictory information across sources?
6. Design a freshness-aware retrieval scoring function

**Key concepts to demonstrate:**
- Real-time vs. batch indexing
- Multi-hop question decomposition
- Source credibility signals (domain authority, publication date, author reputation)
- Contradiction detection and resolution

---

## Exercise 3 — Design a Secure RAG System for Healthcare (HIPAA Level)

**Prompt:**
Design a RAG system for a hospital network that helps clinicians access patient records and medical literature:
- 500GB of patient records (PHI — Protected Health Information)
- 2TB of medical literature (PubMed, clinical guidelines)
- HIPAA compliance is mandatory — any PII/PHI leak is a critical failure
- Queries like: "Summarize patient John Doe's last 3 cardiology visits" and "What's the recommended treatment for X?"
- Audit log required: who queried what, when, and what was retrieved

**Your task:**
1. Design the data classification and PII detection pipeline at ingestion
2. How do you enforce patient record access at the retrieval layer?
3. Design the query routing: clinical literature queries vs. patient record queries
4. Describe your output scanning pipeline to catch PHI in generated responses
5. Design the audit logging system — what fields do you capture?
6. How would you respond to a data breach (what's your incident response for RAG systems)?

**Key concepts to demonstrate:**
- PHI/PII detection (NER-based: Presidio, spaCy)
- Role-based and attribute-based access control (RBAC/ABAC)
- Output filtering and redaction
- Immutable audit log design

---

## Exercise 4 — Design a Code Generation Agent (GitHub Copilot / Cursor Level)

**Prompt:**
Design an agentic AI system for automated code generation and debugging:
- Developer describes a feature in natural language; agent writes code, tests it, iterates
- Agent has access to: code interpreter (Docker sandbox), file system, Git, test runner, documentation search
- Agent must handle multi-file changes coherently
- Safety: agent cannot push to `main` branch directly
- Iteration budget: max 10 tool-call cycles per task

**Your task:**
1. Design the ReAct loop architecture for this agent
2. What tools would you give the agent, and what guardrails for each?
3. How would you design the code execution sandbox for safety?
4. Describe the agent's memory design: what state does it need to maintain across iterations?
5. How would you detect and break infinite loops (agent repeating the same failed action)?
6. Design the human-in-the-loop checkpoint — when must the agent pause for human approval?

**Key concepts to demonstrate:**
- ReAct (Reason + Act) pattern
- Tool sandboxing (Docker, seccomp, network isolation)
- Loop detection (hash of Thought+Action to detect repetition)
- Human-in-the-loop triggers (file deletion, external API calls, branch merges)

---

## Exercise 5 — Design a Multi-Agent Research System (Google DeepResearch Level)

**Prompt:**
Design a multi-agent system that can autonomously conduct a 3-hour deep research task:
- Input: "Write a comprehensive analysis of the electric vehicle battery supply chain, with citations"
- Agents: Orchestrator, Web Researcher, Data Analyst, Fact Checker, Writer, Reviewer
- Total research time budget: < 5 minutes (parallel agent execution)
- Output: 5,000-word report with citations, charts, and an executive summary
- Failure handling: if one agent fails, the system should degrade gracefully

**Your task:**
1. Design the task decomposition and orchestration logic
2. Choose the coordination pattern (hierarchical vs. pipeline vs. peer-to-peer) and justify
3. Design the shared memory / message bus between agents
4. How would you prevent duplicate research (Agent A and Agent B both searching the same source)?
5. Describe your failure handling: what if the Fact Checker agent times out?
6. How would you evaluate output quality automatically (before showing to the user)?

**Key concepts to demonstrate:**
- Task DAG (Directed Acyclic Graph) for parallel agent execution
- Shared scratchpad vs. message-passing communication
- Idempotent task execution (safe to retry)
- Self-evaluation / critique agents

---

## Exercise 6 — Design a RAG Security Red-Team System

**Prompt:**
You are hired to find vulnerabilities in a company's RAG-powered customer support chatbot before it goes to production. The chatbot has access to all customer support documentation and a CRM tool to look up customer accounts.

**Your task — play both attacker and defender:**

**As an attacker:**
1. Enumerate at least 4 attack vectors specific to this RAG + tool-calling setup
2. Write concrete adversarial prompts for each attack vector
3. Estimate the blast radius of each attack (data exposed, actions taken)

**As a defender:**
1. For each attack, design a specific technical countermeasure
2. Prioritize the defenses by effort vs. impact
3. Design a continuous red-teaming schedule (how often, what tests, who reviews results?)
4. What's your security SLA? (e.g., critical vulnerabilities patched within 24h)

**Key concepts to demonstrate:**
- Prompt injection (direct and indirect)
- Tool-calling abuse (CRM lookup for unauthorized accounts)
- PII exfiltration via crafted queries
- Jailbreak resilience
- Defense in depth (no single point of failure)

---

## Scoring Rubric

| Criterion | Description | Max Points |
|---|---|---|
| Architecture diagram | Clear components with data flow described | 3 |
| Security & access control | Explicit about who can access what data | 3 |
| Failure mode coverage | At least 3 failure scenarios with mitigations | 3 |
| Evaluation strategy | How do you know it works? Metrics defined | 3 |
| Trade-off articulation | Acknowledged at least 2 design trade-offs | 3 |

**Target for FAANG AI/ML Engineer interview:** 12/15 or higher

---

## Further Reading

- [RAGAS: Evaluation Framework for RAG](https://docs.ragas.io/)
- [Anthropic: Building with Claude — Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Microsoft GraphRAG](https://microsoft.github.io/graphrag/)
- [LangGraph: Multi-Agent Orchestration](https://langchain-ai.github.io/langgraph/)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
