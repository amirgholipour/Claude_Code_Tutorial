"""Module 04 — RAG System Design
Level: Advanced"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
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
## Retrieval-Augmented Generation (RAG) — System Design Masterclass

RAG is one of the most commonly asked topics in ML and AI system design interviews
at Google, Meta, Amazon, and Databricks. This module covers the full pipeline —
from document chunking through to production deployment — with the depth expected
at staff-level interviews.

---

## Section 1 — Why RAG?

Large language models have two fundamental limitations that RAG directly addresses:

**1. Static knowledge cutoffs.** A model trained through October 2023 has no knowledge
of events after that date. Retraining is expensive ($1M–$100M+), so keeping models
current via retraining is impractical for most teams.

**2. Hallucination.** LLMs generate plausible-sounding text even when they lack
factual knowledge. Without grounding in retrieved documents, models confidently
fabricate citations, statistics, and technical details.

RAG solves both: at query time, relevant documents are retrieved from an up-to-date
index and injected into the prompt as context. The model generates an answer
*grounded in retrieved evidence*, with citations.

**The two phases of RAG:**

```
OFFLINE (Indexing):
  Raw Documents → Chunker → Embedding Model → Vector Database

ONLINE (Retrieval + Generation):
  Query → Retriever → Reranker → LLM with Context → Answer
```

**Key components:**
- **Document store:** S3/Blob — raw source documents at rest
- **Chunker:** splits documents into passages the embedding model can encode
- **Embedding model:** encodes text into dense vectors (e.g., 384–3072 dimensions)
- **Vector database:** indexes vectors for ANN (approximate nearest neighbor) search
- **Retriever:** fetches top-k candidate chunks given a query vector
- **Reranker:** cross-encoder that re-scores candidates for precision
- **LLM:** generates a grounded answer from the retrieved context

---

## Section 2 — Chunking Strategies

Chunking is the first leverage point for RAG quality. Poor chunking guarantees
poor retrieval — no downstream component can recover from it.

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Fixed-size (512 tokens)** | Split every N tokens | Simple, predictable | Cuts mid-sentence |
| **Sentence-aware** | Split on sentence boundaries | Preserves meaning | Variable sizes |
| **Semantic** | Split where topic changes (embedding similarity drops) | Best quality | Expensive to compute |
| **Hierarchical** | Parent chunks with child chunks | Preserves context | Complex retrieval |
| **Recursive** | Try paragraph → sentence → word boundaries | Flexible | Multiple passes |

**Key parameter: overlap** (typically 10–20% of chunk size). When a chunk boundary
cuts across a key sentence, overlap ensures the surrounding context appears in at
least one chunk. Without overlap, boundary sentences are under-represented in
retrieval.

**Real-world examples:**
- Legal document (10 pages): fixed-size=512, overlap=50 is a solid baseline. Legal
  prose has consistent sentence density and formal boundaries.
- Technical manual: hierarchical chunking (chapter as parent → section as child)
  gives better context preservation. The parent chunk carries structural context
  (which chapter/section) while child chunks carry the specific technical detail.
- Customer support transcripts: sentence-aware chunking preserves the natural
  question-answer turn structure that gets destroyed by fixed-size splitting.

---

## Section 3 — Embedding Models

| Model | Dimensions | Size | Notes |
|-------|-----------|------|-------|
| **all-MiniLM-L6-v2** | 384 | 80 MB | Fast, good baseline, SentenceTransformers |
| **BGE-M3** | 1024 | 570 MB | Multilingual, strong performance/cost ratio |
| **OpenAI text-embedding-3-large** | 3072 | API | Best MTEB benchmark, $0.00013/1K tokens |
| **Cohere embed-v3** | 1024 | API | Strong retrieval, supports input type hints |

**Key insight:** domain-specific fine-tuning often beats larger general models by
10–25% on in-domain retrieval benchmarks. A fine-tuned MiniLM on medical text
frequently outperforms text-embedding-3-large on medical Q&A. Fine-tuning cost:
~$50–200 on a single GPU for a few thousand labeled query-passage pairs.

**Training-serving parity:** the embedding model used at index time *must* match
the model used at query time. If you re-embed documents (e.g., after a model
upgrade), you must also re-index the entire vector store. This is a frequent
production footgun — document the model version as part of the index metadata.

---

## Section 4 — Retrieval Methods

| Method | Algorithm | Speed | Pros | Cons |
|--------|-----------|-------|------|------|
| **Sparse (BM25)** | TF-IDF weighting | < 50 ms | Exact keyword match, transparent, no embeddings | Misses synonyms/semantics |
| **Dense (embedding)** | Cosine similarity via ANN | < 20 ms | Semantic understanding, multilingual | Can miss rare keywords |
| **Hybrid (RRF)** | BM25 + Dense + Reciprocal Rank Fusion | < 100 ms | Best of both worlds | More complex pipeline |

**Reciprocal Rank Fusion (RRF):**
```
score(d) = Σ  1 / (rank_i(d) + k)
```
where `k = 60` is a smoothing constant (prevents high-rank documents from
dominating). Each ranked list (BM25 rank, dense rank) contributes independently.
Documents appearing high in multiple lists receive a compounded score boost.

**When each method wins:**
- BM25 wins for exact-match queries: product IDs, names, version numbers, jargon
- Dense wins for paraphrase/synonym queries: "car" vs "automobile", "ML" vs "machine learning"
- Hybrid wins as the default: BEIR benchmark shows 8–15% recall improvement over
  either method alone across 18 retrieval tasks

---

## Section 5 — Reranking

First-stage retrieval is optimized for **recall** — retrieve top-100 candidates
as fast as possible. Reranking is optimized for **precision** — score each
(query, chunk) pair carefully and return the top-5.

**Bi-encoder (retrieval stage):** encodes query and document independently, computes
cosine similarity. Fast (~1 ms/document). Used at retrieval stage over millions of
documents.

**Cross-encoder (reranking stage):** concatenates [query, document] and runs
the full attention over the pair. Much more accurate — captures query-document
interaction. But ~5–10× slower than bi-encoder. Used only over 20–50 candidates.

**Typical reranking models:**
- `ms-marco-MiniLM-L-6-v2`: fast, strong performance on MS-MARCO benchmark
- `ms-marco-electra-base`: slightly better accuracy, higher latency
- Cohere rerank API: hosted, no GPU required, ~50 ms/20 candidates

**Typical improvement from reranking:** 10–20% precision gain with 50–150 ms
added latency. For a production system with a 2-second total latency budget,
this tradeoff is usually worth it.

---

## Section 6 — RAGAS Evaluation Metrics

RAGAS (Retrieval-Augmented Generation Assessment) is the standard evaluation
framework for RAG pipelines. It decouples retrieval quality from generation quality.

| Metric | What it measures | Score range |
|--------|-----------------|-------------|
| **Faithfulness** | Answer grounded in retrieved context (not hallucinated) | 0–1, higher = better |
| **Answer Relevancy** | Answer actually addresses the question asked | 0–1 |
| **Context Precision** | Retrieved chunks contain relevant information | 0–1 |
| **Context Recall** | All relevant information is present in retrieved chunks | 0–1 |

**Evaluation without a ground-truth dataset:** use BM25 keyword overlap or
cosine similarity between the query and retrieved chunk as a proxy relevance signal.
This enables offline evaluation before deploying a new chunking strategy or
embedding model.

**Online evaluation metrics:**
- User thumbs up/down on responses
- Citation click-through rate (did the user verify the cited source?)
- Follow-up question rate (high follow-ups suggest the first answer was incomplete)
- Task completion rate in agentic use cases

---

## Section 7 — Reference Cloud Architectures

### AWS RAG Architecture
```
Documents → S3 → Lambda (chunking) → Bedrock Embeddings → OpenSearch (vector store)
Query → API Gateway → Lambda → OpenSearch (hybrid retrieval) → Bedrock Claude (generation)
```
**Key AWS components:**
- Amazon Kendra: managed intelligent search with built-in BM25 + ML ranking
- Amazon Bedrock: embedding models (Titan, Cohere) + generation (Claude, Llama)
- Amazon OpenSearch Service: k-NN vector search built on HNSW
- AWS Lambda: serverless chunking and orchestration

### Azure RAG Architecture
```
Documents → Blob Storage → Azure Functions (chunking) → Azure OpenAI (embeddings) → AI Search (vector index)
Query → API Management → Azure Functions → AI Search (hybrid retrieval) → Azure OpenAI GPT-4 (generation)
```
**Key Azure components:**
- Azure AI Search: BM25 + vector hybrid retrieval built-in, semantic ranker add-on
- Azure OpenAI Service: text-embedding-3-large, GPT-4o generation
- Azure Cosmos DB: chat history storage with vector search (preview)
- Azure Prompt Flow: orchestration layer for RAG pipelines

### GCP RAG Architecture
```
Documents → Cloud Storage → Cloud Functions → Vertex AI Embeddings → Vertex AI Vector Search
Query → Cloud Run → Vertex AI Vector Search → Vertex AI Gemini (generation)
```
**Key GCP components:**
- Vertex AI Vector Search: managed HNSW/ScaNN, billions-scale
- Vertex AI Embeddings: text-embedding-004, multilingual support
- Cloud Spanner: structured metadata store with full-text search
- Vertex AI Gemini: generation with long context window (1M tokens)

---

## Section 8 — Production Considerations

**Semantic caching:** before hitting the retriever, compute the query embedding and
check a cache for similar queries (cosine similarity > 0.95). Return cached answer
directly. Reduces LLM costs by 30–40% in production (many users ask the same
question in slightly different words). Redis with vector similarity search works
well as the semantic cache store.

**Multi-tenancy:** namespace the vector store by tenant ID. At retrieval time, apply
a metadata filter to restrict results to the tenant's documents. This prevents
cross-tenant data leakage — a critical security requirement for SaaS RAG products.
Also enforce ACL at the retrieval layer, not just at the UI layer.

**Vector database choices:**
| Option | Best for | Key differentiator |
|--------|----------|-------------------|
| Pinecone | Managed, quick start | Zero infrastructure, serverless |
| Weaviate | Open-source, flexible | GraphQL API, multi-modal |
| Milvus | High-scale, on-prem | 10B+ vector scale, hybrid search built-in |
| FAISS | Local, research | No persistence, no server — just a library |
| pgvector | PostgreSQL users | Single DB for structured + vector data |

**Document versioning:** when source documents update, identify changed chunks via
content fingerprinting (SHA-256 hash of chunk text). Re-embed only changed chunks.
This avoids a full re-index on every document update — critical for large corpora.

**Latency budget (target for < 2s total):**
- Chunking: offline (not on the critical path)
- Embedding (query): 20–50 ms (batch on GPU)
- Retrieval (vector search): < 50 ms
- Reranking (top-20): 50–150 ms
- LLM generation: < 1.5 s (streaming helps perceived latency)

---

## Section 9 — Interview Questions

**"Design a document QA system for 10M internal documents":**
Hybrid retrieval (BM25 + dense) with RRF fusion → cross-encoder reranking to top-5
→ semantic cache for repeated queries → multi-tenant namespace with ACL enforcement.
Index with Milvus or OpenSearch for scale. Embedding model: BGE-M3 for multilingual
support. Reranker: ms-marco-MiniLM for latency. Evaluate with RAGAS monthly.

**"How do you evaluate RAG quality without ground truth?":**
BM25 keyword overlap as a synthetic relevance signal for context precision. Embed
a sample of (query, answer) pairs and measure cosine similarity to retrieved chunks
for faithfulness. Human evaluation on a 200-query sample. Online: thumbs up/down
rates and follow-up question frequency.

**Red flags:**
- "I'd just embed everything and do cosine similarity" — misses BM25 for exact
  keyword matching; no mention of reranking
- Not discussing chunking strategy at all
- Proposing a single embedding model without mentioning fine-tuning potential

**Green flags:**
- Mentioning training-serving parity in embeddings (same model at index and query time)
- RAGAS metrics for offline evaluation + thumbs up/down for online signals
- Semantic cache to reduce LLM costs
- Distinguishing bi-encoders (fast retrieval) from cross-encoders (accurate reranking)
"""

# ─────────────────────────────────────────────────────────────────────────────
# CODE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

CODE_EXAMPLE = '''import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Chunking ───────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Fixed-size chunking with overlap."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ── Dense Retrieval (TF-IDF as embedding proxy) ────────────────
class DenseRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.chunk_vectors = None
        self.chunks = []

    def index(self, chunks: list[str]):
        self.chunks = chunks
        self.chunk_vectors = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.chunk_vectors).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(i, float(scores[i])) for i in top_idx]

# ── BM25 Retrieval ─────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    def bm25_retrieve(chunks: list[str], query: str, top_k: int = 5) -> list[tuple[int, float]]:
        tokenized = [c.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(i, float(scores[i])) for i in top_idx]
except ImportError:
    bm25_retrieve = None

# ── Hybrid Retrieval via RRF ───────────────────────────────────
def reciprocal_rank_fusion(*ranked_lists: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
    """Merge multiple ranked lists using RRF. k=60 is standard."""
    rrf_scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rank + k)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

# ── RAGAS-Style Evaluation ─────────────────────────────────────
def evaluate_context_precision(retrieved_chunks: list[str], relevant_keywords: list[str]) -> float:
    """Proxy: fraction of retrieved chunks containing at least one relevant keyword."""
    hits = sum(1 for c in retrieved_chunks if any(kw.lower() in c.lower() for kw in relevant_keywords))
    return hits / len(retrieved_chunks) if retrieved_chunks else 0.0

# ── Example usage ──────────────────────────────────────────────
corpus = [
    "Apache Spark is a distributed computing engine for large-scale data processing.",
    "Kafka is a distributed event streaming platform used for real-time data pipelines.",
    "The CAP theorem states a distributed system can only guarantee two of: Consistency, Availability, Partition Tolerance.",
    "Kubernetes orchestrates containerised workloads across clusters of nodes.",
    "BM25 is a probabilistic retrieval function that ranks documents by term frequency and inverse document frequency.",
]

retriever = DenseRetriever()
chunks = []
for doc in corpus:
    chunks.extend(chunk_text(doc, chunk_size=20, overlap=5))

retriever.index(chunks)
dense_results = retriever.retrieve("How does BM25 work?", top_k=3)

keywords = ["bm25", "ranking", "retrieval"]
precision = evaluate_context_precision(
    retrieved_chunks=[chunks[i] for i, _ in dense_results],
    relevant_keywords=keywords,
)
print(f"Context Precision (proxy): {precision:.2f}")
# → 1.00  (the BM25 chunk surfaces in top-3)
'''

# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN CORPUS  (60 paragraphs × 6 topics)
# ─────────────────────────────────────────────────────────────────────────────

CORPUS: list[str] = [
    # ── Data Engineering (10) ──────────────────────────────────────────────
    "Apache Spark is a distributed computing engine for large-scale data processing using in-memory computation to speed up ETL pipelines.",
    "Kafka is a distributed event streaming platform that enables real-time data pipelines with high throughput and fault tolerance.",
    "Change Data Capture (CDC) is a technique that tracks row-level changes in a source database and streams them downstream using tools like Debezium.",
    "ETL pipelines extract data from source systems, apply transformations to clean and reshape the data, and load it into a data warehouse.",
    "Apache Flink is a stream processing framework that supports stateful computations over unbounded data streams with exactly-once semantics.",
    "Delta Lake adds ACID transactions, schema enforcement, and time travel on top of Apache Parquet files stored in cloud object storage.",
    "Data partitioning in Spark divides large datasets into smaller chunks distributed across worker nodes to enable parallel processing.",
    "Apache Iceberg is an open table format for large analytical datasets that supports schema evolution, hidden partitioning, and row-level deletes.",
    "A medallion architecture organizes data into Bronze (raw), Silver (cleaned), and Gold (aggregated) layers to progressively refine quality.",
    "dbt (data build tool) enables analytics engineers to transform data using SQL models, with built-in testing and documentation generation.",

    # ── Machine Learning (10) ──────────────────────────────────────────────
    "A feature store is a centralized repository that stores, serves, and manages ML features to prevent training-serving skew.",
    "Model training involves minimizing a loss function over labeled examples using optimization algorithms such as gradient descent.",
    "Feature engineering is the process of selecting and transforming raw variables into numerical inputs that improve model performance.",
    "Overfitting occurs when a model memorizes training data noise, resulting in high accuracy on training data but poor generalization to new data.",
    "Cross-validation splits data into k folds to evaluate model performance without a separate test set, reducing variance in evaluation metrics.",
    "Hyperparameter tuning searches for optimal model configuration values such as learning rate, regularization strength, and tree depth.",
    "Gradient boosting builds an ensemble of weak learners sequentially, where each tree corrects errors made by the previous trees.",
    "Transfer learning fine-tunes a pretrained model on a new task, leveraging representations learned from a much larger dataset.",
    "A confusion matrix summarizes classification model performance by showing true positives, false positives, true negatives, and false negatives.",
    "MLflow is an open-source platform for managing the ML lifecycle including experiment tracking, model registry, and deployment.",

    # ── System Design (10) ─────────────────────────────────────────────────
    "The CAP theorem states that a distributed system can only guarantee two of three properties: Consistency, Availability, and Partition Tolerance.",
    "Consistent hashing places nodes on a virtual ring so that adding or removing a node only remaps a fraction of keys rather than all keys.",
    "A relational database stores data in tables with rows and columns, enforcing schemas and supporting ACID transactions via SQL.",
    "Horizontal sharding partitions a database across multiple nodes by a shard key, distributing both data and query load.",
    "A message queue decouples producers from consumers, enabling asynchronous processing and absorbing traffic spikes without data loss.",
    "Load balancers distribute incoming requests across multiple server instances using algorithms such as round-robin or least-connections.",
    "Eventual consistency guarantees that all replicas will converge to the same value given sufficient time and no new writes.",
    "A content delivery network (CDN) caches static assets at edge locations geographically close to users to reduce latency.",
    "Database indexing creates auxiliary data structures that accelerate read queries at the cost of increased write overhead and storage.",
    "Rate limiting controls the number of requests a client can make in a time window to protect services from abuse and overload.",

    # ── Kubernetes (10) ────────────────────────────────────────────────────
    "Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications.",
    "A Pod is the smallest deployable unit in Kubernetes, typically containing one or more tightly coupled containers sharing network and storage.",
    "A Kubernetes Service provides a stable DNS name and IP address that routes traffic to a dynamic set of Pods using label selectors.",
    "A Deployment manages a ReplicaSet to ensure the specified number of Pod replicas are running, enabling rolling updates and rollbacks.",
    "Kubernetes Ingress defines HTTP and HTTPS routing rules that expose services outside the cluster through an Ingress controller.",
    "ConfigMaps store non-sensitive configuration data as key-value pairs that can be injected into Pods as environment variables or files.",
    "Kubernetes Secrets store sensitive information such as passwords and API keys, with values base64-encoded and access controlled via RBAC.",
    "Horizontal Pod Autoscaler (HPA) automatically adjusts the number of Pod replicas based on observed CPU utilization or custom metrics.",
    "A Kubernetes Namespace provides a virtual cluster within a physical cluster, enabling resource isolation and multi-team environments.",
    "PersistentVolumes provide storage resources in a Kubernetes cluster that survive Pod restarts, with access modes like ReadWriteOnce.",

    # ── Security (10) ──────────────────────────────────────────────────────
    "Authentication verifies the identity of a user or service, commonly implemented via passwords, OAuth tokens, or mutual TLS certificates.",
    "Encryption transforms plaintext data into ciphertext using a key, protecting data at rest and in transit from unauthorized access.",
    "OWASP Top 10 lists the most critical web application security risks including injection attacks, broken authentication, and SSRF.",
    "SQL injection attacks insert malicious SQL code into user inputs to manipulate database queries and exfiltrate or corrupt data.",
    "JWT (JSON Web Tokens) are compact, self-contained tokens used to securely transmit claims between parties using digital signatures.",
    "A firewall monitors and filters incoming and outgoing network traffic based on predefined security rules to block unauthorized access.",
    "Zero-trust security assumes no implicit trust based on network location, requiring authentication and authorization for every request.",
    "TLS (Transport Layer Security) encrypts data in transit between client and server using asymmetric key exchange followed by symmetric encryption.",
    "Cross-Site Request Forgery (CSRF) tricks authenticated users into submitting malicious requests by exploiting session cookies.",
    "Secret rotation automatically changes credentials and API keys on a schedule to limit the blast radius of a compromised secret.",

    # ── Cloud Computing (10) ───────────────────────────────────────────────
    "Amazon S3 is an object storage service offering 99.999999999% durability, versioning, lifecycle policies, and cross-region replication.",
    "Azure Blob Storage is Microsoft's object storage solution for unstructured data, with hot, cool, and archive access tiers.",
    "Google Cloud Storage provides globally distributed object storage with strong consistency and automatic multi-region redundancy.",
    "AWS Lambda is a serverless compute service that runs code in response to events without provisioning or managing servers.",
    "Azure Functions enables event-driven serverless compute that scales automatically and integrates natively with Azure services.",
    "Google Cloud Run executes containerized applications in a fully managed environment that scales to zero when idle.",
    "Amazon RDS manages relational databases including PostgreSQL, MySQL, and Aurora, handling backups, patching, and failover automatically.",
    "Azure SQL Database is a fully managed PaaS relational database with built-in high availability and intelligent performance tuning.",
    "Google Cloud Spanner is a globally distributed relational database that provides horizontal scaling with strong external consistency.",
    "AWS CloudFront is a CDN that delivers content from edge locations worldwide and integrates natively with S3, EC2, and ALB origins.",
]

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE QUERIES
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "How does BM25 work?",
    "What is CAP theorem?",
    "Explain Kubernetes pods",
    "How to secure an API?",
    "What is a feature store?",
    "Compare AWS vs Azure storage",
    "What is CDC in data engineering?",
    "How does Spark process data?",
]

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


def _chunk_corpus(corpus: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Split each corpus document into fixed-size word chunks with overlap."""
    chunks: list[str] = []
    for doc in corpus:
        words = doc.split()
        if len(words) <= chunk_size:
            chunks.append(doc)
        else:
            i = 0
            while i < len(words):
                chunk = " ".join(words[i:i + chunk_size])
                chunks.append(chunk)
                i += chunk_size - max(overlap, 1)
    return chunks


def _dense_retrieve(chunks: list[str], query: str, top_k: int) -> list[tuple[int, float]]:
    """TF-IDF cosine similarity retrieval."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    try:
        mat = vec.fit_transform(chunks)
        q_vec = vec.transform([query])
        scores = cos_sim(q_vec, mat).flatten()
    except ValueError:
        return [(i, 0.0) for i in range(min(top_k, len(chunks)))]

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


def _bm25_retrieve(chunks: list[str], query: str, top_k: int) -> list[tuple[int, float]]:
    """BM25 retrieval — falls back to dense if rank_bm25 is not installed."""
    if not _BM25_AVAILABLE:
        return _dense_retrieve(chunks, query, top_k)

    tokenized = [c.lower().split() for c in chunks]
    bm25 = _BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


def _rrf_fuse(*ranked_lists: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists."""
    rrf: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rank + k)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


def _hybrid_retrieve(chunks: list[str], query: str, top_k: int) -> list[tuple[int, float]]:
    """Hybrid BM25 + Dense with RRF fusion."""
    bm25_results   = _bm25_retrieve(chunks, query, top_k=max(top_k * 3, 20))
    dense_results  = _dense_retrieve(chunks, query, top_k=max(top_k * 3, 20))
    fused          = _rrf_fuse(bm25_results, dense_results)
    return fused[:top_k]


def _context_precision(retrieved_chunks: list[str], query: str) -> float:
    """Proxy precision: fraction of retrieved chunks with cosine sim > threshold to query."""
    if not retrieved_chunks:
        return 0.0
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    try:
        all_text = retrieved_chunks + [query]
        vec = TfidfVectorizer(max_features=1000)
        mat = vec.fit_transform(all_text)
        q_vec = mat[-1]
        chunk_mat = mat[:-1]
        sims = cos_sim(q_vec, chunk_mat).flatten()
        return float(np.mean(sims > 0.05))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_demo(
    query: str,
    chunk_size: int,
    overlap: int,
    method: str,
    top_k: int,
    provider: str,
    model: str,
    api_key: str,
) -> tuple:
    """
    Returns (fig, metrics_md).

    fig: 2-panel Plotly figure
        Left  — Horizontal bar chart: top-k chunks with retrieval scores
        Right — Bar chart comparing BM25 vs Dense vs Hybrid precision on this query

    metrics_md: Markdown summary with RAG pipeline metrics and optional LLM answer
    """
    try:
        palette = COLORS["palette"]
        query = query.strip() or "What is CAP theorem?"

        # ── 1. Chunk corpus ──────────────────────────────────────────────
        t0 = time.perf_counter()
        chunks = _chunk_corpus(CORPUS, chunk_size=int(chunk_size), overlap=int(overlap))
        n_chunks = len(chunks)

        # ── 2. Run selected retrieval method ────────────────────────────
        if method == "BM25":
            results = _bm25_retrieve(chunks, query, top_k=int(top_k))
        elif method == "Dense (TF-IDF)":
            results = _dense_retrieve(chunks, query, top_k=int(top_k))
        else:  # Hybrid (RRF)
            results = _hybrid_retrieve(chunks, query, top_k=int(top_k))

        retrieval_ms = int((time.perf_counter() - t0) * 1000)

        # ── 3. Always compute all 3 precision scores for comparison ──────
        bm25_chunks   = [chunks[i] for i, _ in _bm25_retrieve(chunks, query, top_k=int(top_k))]
        dense_chunks  = [chunks[i] for i, _ in _dense_retrieve(chunks, query, top_k=int(top_k))]
        hybrid_chunks = [chunks[i] for i, _ in _hybrid_retrieve(chunks, query, top_k=int(top_k))]

        bm25_prec   = _context_precision(bm25_chunks, query)
        dense_prec  = _context_precision(dense_chunks, query)
        hybrid_prec = _context_precision(hybrid_chunks, query)

        # ── 4. Build figure ──────────────────────────────────────────────
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"Top-{top_k} Retrieved Chunks ({method})",
                "Method Comparison — Context Precision",
            ],
            column_widths=[0.60, 0.40],
        )

        # Left: horizontal bar of top-k chunks
        retrieved_indices = [i for i, _ in results]
        retrieved_scores  = [s for _, s in results]
        chunk_labels = []
        for idx in retrieved_indices:
            preview = chunks[idx][:60].replace("\n", " ")
            chunk_labels.append(f"[{idx}] {preview}...")

        # Normalize scores for display (handle RRF's small absolute values)
        if retrieved_scores and max(retrieved_scores) > 0:
            max_s = max(retrieved_scores)
            display_scores = [s / max_s for s in retrieved_scores]
        else:
            display_scores = retrieved_scores

        bar_colors = [palette[i % len(palette)] for i in range(len(chunk_labels))]

        fig.add_trace(
            go.Bar(
                x=display_scores[::-1],
                y=chunk_labels[::-1],
                orientation="h",
                marker_color=bar_colors[::-1],
                text=[f"{s:.3f}" for s in display_scores[::-1]],
                textposition="outside",
                hovertemplate="%{y}<br>Score: %{x:.4f}<extra></extra>",
                name="Retrieval Score",
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.update_xaxes(range=[0, 1.25], title_text="Normalized Score", row=1, col=1)

        # Right: method precision comparison bars
        method_names   = ["BM25", "Dense\n(TF-IDF)", "Hybrid\n(RRF)"]
        method_scores  = [bm25_prec, dense_prec, hybrid_prec]
        method_colors  = [palette[0], palette[1], palette[2]]
        highlight      = [method.startswith("BM25"), method.startswith("Dense"), method.startswith("Hybrid")]
        final_colors   = [palette[4] if h else c for h, c in zip(highlight, method_colors)]

        fig.add_trace(
            go.Bar(
                x=method_names,
                y=method_scores,
                marker_color=final_colors,
                text=[f"{s:.2f}" for s in method_scores],
                textposition="outside",
                hovertemplate="%{x}: %{y:.2f}<extra></extra>",
                name="Context Precision",
                showlegend=False,
            ),
            row=1, col=2,
        )
        fig.update_yaxes(range=[0, 1.2], title_text="Context Precision (proxy)", row=1, col=2)

        fig.update_layout(
            template="plotly_white",
            height=500,
            margin=dict(l=20, r=40, t=60, b=20),
            title=dict(
                text=f'RAG Pipeline — Query: "{query[:60]}{"..." if len(query) > 60 else ""}"',
                font=dict(size=14),
                x=0.5,
            ),
        )

        # ── 5. LLM answer (optional) ─────────────────────────────────────
        top_chunk = chunks[retrieved_indices[0]] if retrieved_indices else ""
        context_for_llm = "\n\n".join(
            f"[Chunk {i}]: {chunks[idx]}"
            for i, idx in enumerate(retrieved_indices[:5], 1)
        )

        llm_answer = "*(No API key provided — showing retrieved context only)*"
        if call_llm is not None and api_key and api_key.strip():
            rag_prompt = (
                f"You are a technical assistant. Answer the following question using "
                f"only the context provided. Be concise.\n\n"
                f"Question: {query}\n\n"
                f"Context:\n{context_for_llm}\n\n"
                f"Answer:"
            )
            llm_answer = call_llm(rag_prompt, provider=provider, model=model, api_key=api_key)

        # ── 6. Metrics markdown ──────────────────────────────────────────
        bm25_note = "" if _BM25_AVAILABLE else " *(rank_bm25 not installed — using TF-IDF fallback)*"
        top_preview = top_chunk[:100] if top_chunk else "N/A"

        metrics_md = f"""### RAG Pipeline Results

**Query:** "{query}"
**Method:** {method} | **Chunk size:** {chunk_size} | **Top-k:** {top_k}

| Metric | Value |
|--------|-------|
| Chunks indexed | `{n_chunks}` |
| Retrieval time | `~{retrieval_ms} ms` (estimated) |
| BM25 precision{bm25_note} | `{bm25_prec:.2f}` |
| Dense precision | `{dense_prec:.2f}` |
| Hybrid precision | `{hybrid_prec:.2f}` |

**Top Retrieved Chunk (preview):**
> {top_preview}...

**LLM Answer:** {llm_answer}

**Architecture Note:**
For production with 10M docs:
- Index offline in Pinecone / OpenSearch
- Hybrid retrieval (BM25 + dense) gives best recall
- Reranker (cross-encoder) for final top-5
- Semantic cache for repeated queries
"""

        return fig, metrics_md

    except Exception:
        import traceback
        return go.Figure(), f"**Error:** {traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO TAB BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_tab():
    gr.Markdown(
        "# Retrieval-Augmented Generation — System Design\n"
        "*Module 04 | Level: Advanced*"
    )

    with gr.Accordion("Theory", open=False):
        gr.Markdown(THEORY)

    with gr.Accordion("Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown(
        "---\n"
        "## Interactive Demo — RAG Pipeline Explorer\n\n"
        "Select a query, tune chunking parameters, choose a retrieval method, "
        "then click **Run RAG Demo** to see retrieval scores and method comparison."
    )

    with gr.Row():
        # ── Left column: controls ────────────────────────────────────────
        with gr.Column(scale=1):
            query_dd = gr.Dropdown(
                choices=EXAMPLE_QUERIES,
                value=EXAMPLE_QUERIES[0],
                label="Example Query",
            )
            query_tb = gr.Textbox(
                label="Custom Query (overrides dropdown if non-empty)",
                placeholder="Type your own question here...",
                lines=2,
            )

            chunk_size_dd = gr.Dropdown(
                choices=[128, 256, 512, 1024],
                value=512,
                label="Chunk Size (words)",
            )
            overlap_sl = gr.Slider(
                minimum=0,
                maximum=100,
                step=5,
                value=50,
                label="Overlap (words)",
            )
            method_rd = gr.Radio(
                choices=["BM25", "Dense (TF-IDF)", "Hybrid (RRF)"],
                value="Hybrid (RRF)",
                label="Retrieval Method",
            )
            top_k_sl = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=5,
                label="Top-k Results",
            )

            with gr.Accordion("LLM Provider (optional — for generated answer)", open=False):
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

                def _toggle_gemini(provider: str):
                    show = provider == "Google Gemini"
                    return gr.update(visible=show), gr.update(visible=show)

                provider_dd.change(
                    fn=_toggle_gemini,
                    inputs=[provider_dd],
                    outputs=[model_dd, api_key_tb],
                )

            run_btn = gr.Button("Run RAG Demo", variant="primary")

        # ── Right column: outputs ────────────────────────────────────────
        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Retrieval Visualization")
            metrics_out = gr.Markdown()

    def _resolve_query(example_q: str, custom_q: str) -> str:
        return custom_q.strip() if custom_q.strip() else example_q

    run_btn.click(
        fn=lambda eq, cq, cs, ov, mth, tk, prov, mdl, key: run_rag_demo(
            _resolve_query(eq, cq), cs, ov, mth, tk, prov, mdl, key
        ),
        inputs=[
            query_dd, query_tb,
            chunk_size_dd, overlap_sl, method_rd, top_k_sl,
            provider_dd, model_dd, api_key_tb,
        ],
        outputs=[plot_out, metrics_out],
    )
