"""Module 01 — Data Model & Storage Design
Level: Intermediate"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import COLORS

# ---------------------------------------------------------------------------
# THEORY
# ---------------------------------------------------------------------------

THEORY = """## 🗄️ Data Model & Storage Design — Interview Masterclass

Choosing the right storage technology is one of the **most scrutinised decisions** in a system design interview.
Every senior engineer interviewer will probe whether you understand the *why* behind each choice — not just the tool names.

---

### 1. CAP Theorem

The CAP theorem (Brewer, 2000) states that any distributed data store can guarantee **at most two** of the following three properties simultaneously:

| Property | Meaning |
|----------|---------|
| **Consistency (C)** | Every read receives the most recent write or an error |
| **Availability (A)** | Every request receives a (non-error) response — no guarantee it is the latest data |
| **Partition Tolerance (P)** | The system continues operating despite network partitions (dropped messages between nodes) |

Because **network partitions are unavoidable** in real distributed systems, the practical choice is always between **CP** and **AP**:

- **CP systems** (sacrifice availability during partitions): HBase, Redis (single primary), Zookeeper, Google Spanner
  - Correct choice when stale reads are *never* acceptable — banking, seat reservation, inventory counts
- **AP systems** (sacrifice consistency, stay available): Cassandra, DynamoDB, CouchDB, MongoDB (default settings)
  - Correct choice when temporary stale reads are fine — social feeds, shopping carts, user preferences
- **CA systems** (no partition tolerance): Traditional single-node RDBMS — PostgreSQL, MySQL, Oracle
  - Fine for a single server; cannot scale horizontally without sacrificing either C or A

> **Interview tip:** Start your data model section by asking *"What are the consistency requirements?"* Interviewers reward candidates who clarify this before picking a database.

---

### 2. SQL vs NoSQL — The 5-Criterion Decision Matrix

| Criterion | Choose SQL | Choose NoSQL |
|-----------|-----------|-------------|
| **Schema stability** | Schema is well-defined and unlikely to change | Schema evolves rapidly; flexible documents needed |
| **Scale requirement** | Single node or vertical scale is acceptable | Horizontal sharding across dozens of nodes required |
| **Query patterns** | Complex multi-table JOINs, aggregations, ad-hoc queries | Predictable access patterns: key lookups, range scans |
| **Consistency needs** | ACID transactions required (e.g., money transfers) | Eventual consistency is acceptable (BASE model) |
| **Team / ecosystem** | Analytics-heavy team, existing SQL expertise | High dev velocity, schema churn, microservices |

A common interview mistake is treating this as binary. Many production systems use **polyglot persistence**: PostgreSQL for transactional data, Redis for cache, Cassandra for time-series, S3 for blobs.

---

### 3. ACID vs BASE

**ACID** (SQL / relational guarantees):
- **Atomicity** — A transaction either fully commits or fully rolls back; no partial writes
- **Consistency** — Each transaction brings the DB from one valid state to another; constraints are enforced
- **Isolation** — Concurrent transactions behave as if they run sequentially (serialisability)
- **Durability** — Once committed, data survives crashes (written to persistent storage / WAL)

*Best for:* Banking transactions, payment processing, order placement, anything where partial writes cause corruption.

**BASE** (NoSQL / distributed guarantees):
- **Basically Available** — The system guarantees availability, even if some nodes are down
- **Soft State** — State may change over time even without new writes (due to propagation)
- **Eventually Consistent** — Given enough time with no new writes, all replicas converge

*Best for:* Social media likes/follows, shopping cart, user activity logs, recommendation signals — where showing a slightly stale count is acceptable.

**The core trade-off:** ACID = strong guarantees, higher latency, harder to scale horizontally. BASE = faster writes, lower latency, but applications must tolerate stale reads and handle conflicts.

---

### 4. Indexing Strategies

Indexes are the single biggest lever for read performance — but they have a real write cost.

| Index Type | Data Structure | Lookup | Range Query | Use Case |
|------------|---------------|--------|-------------|----------|
| **B-tree** (default) | Balanced tree | O(log n) | ✅ Yes | Most read queries, sorting, range scans |
| **Hash** | Hash table | O(1) | ❌ No | Exact equality lookups only (e.g., session tokens) |
| **Composite** | B-tree on (col1, col2, …) | O(log n) on prefix | ✅ prefix only | Multi-column WHERE clauses |
| **Partial** | B-tree on filtered rows | O(log n) | ✅ | Indexing only active records (`WHERE status='active'`) |
| **GIN / GiST** | Inverted / generalised | varies | varies | Full-text search, JSON fields, arrays (PostgreSQL) |

**The prefix rule for composite indexes:** An index on `(last_name, first_name, city)` accelerates queries filtering on `last_name` or `(last_name, first_name)` — but *not* on `first_name` alone.

**Write amplification:** Every INSERT/UPDATE/DELETE must update all indexes on that table. A table with 10 indexes pays 10x write overhead. On write-heavy tables (IoT, logs), keep indexes minimal and use append-only designs.

**`EXPLAIN ANALYZE`** (PostgreSQL) — always show the interviewer you know how to verify index usage. Look for `Index Scan` vs `Seq Scan` in the query plan.

---

### 5. Sharding & Partitioning

When a single node can no longer hold the data or handle the write throughput, you shard.

**Range sharding:** Assign rows to shards based on a key range (e.g., user_id 0–1M → shard 1).
- Pro: Easy range queries; simple routing
- Con: **Hot spots** — if all new users land on the latest shard, one node is overwhelmed

**Hash sharding:** Compute `hash(key) % N` to pick a shard.
- Pro: Even key distribution
- Con: Range queries require scatter-gather across all shards; re-sharding (changing N) remaps almost all keys

**Consistent hashing:** Place nodes on a hash ring (0–2³²). Each key maps to the nearest clockwise node.
- Benefit: When a node is added/removed, only `k/N` keys need to be remapped (where k = keys, N = nodes)
- Used by: Cassandra, Redis Cluster, DynamoDB, Riak, Memcached (ketama)

**Virtual nodes (vnodes):** Each physical node is responsible for *multiple* non-contiguous segments of the hash ring.
- Benefit: Better load balancing during node failures and cluster rebalancing
- Cassandra uses 256 vnodes per physical node by default

> **Interview green flag:** Mentioning consistent hashing unprompted signals senior-level distributed systems knowledge.

---

### 6. Real-World Design Examples

| System | Primary DB | Why | Caching | Notes |
|--------|-----------|-----|---------|-------|
| **Twitter user profiles** | PostgreSQL | Low write rate, complex queries, ACID for follows | Redis | user_id PK, handle unique index |
| **Uber trip history** | Cassandra | High write volume, time-range queries, AP fine | — | Partition key = driver_id, clustering key = start_time |
| **Netflix viewing history** | DynamoDB | AP, eventual consistency fine, horizontal scale | ElastiCache | PK = (user_id, content_id) |
| **Banking transactions** | PostgreSQL | ACID mandatory, no partial commits | Redis (read-only balance) | Row-level locking, audit log |
| **IoT sensor readings** | InfluxDB / Cassandra | High write throughput, time-ordered, TTL-based expiry | — | Partition by device_id + time bucket |
| **Real-time chat** | Redis (recent) + Cassandra (history) | Sub-ms reads from Redis, durable long-term in Cassandra | Redis itself | Fan-out on write for active users |

---

### 7. Interview Red Flags vs Green Flags

| 🔴 Red Flag | 🟢 Green Flag |
|------------|--------------|
| "I'll just use PostgreSQL for everything" with no trade-off analysis | "Let me first clarify the read/write ratio and consistency requirements" |
| Jumping to sharding without considering indexing or caching first | Proposing indexing → caching → read replicas → sharding as a progressive scale ladder |
| "NoSQL is always faster than SQL" | Understanding that NoSQL trades flexibility for specific access patterns |
| Not knowing what ACID means | Explaining atomicity with a real two-phase commit scenario |
| Ignoring the write amplification cost of indexes | Noting that write-heavy tables need minimal indexes |
| "We can shard by user_id with modulo hashing" | Immediately noting that modulo hashing requires re-sharding and proposing consistent hashing |
"""

# ---------------------------------------------------------------------------
# CODE EXAMPLE
# ---------------------------------------------------------------------------

CODE_EXAMPLE = '''# Database schema design examples for system design interviews

# --- SQLAlchemy: Relational Schema (Twitter-like) ---
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Index, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id     = Column(Integer, primary_key=True)
    handle      = Column(String(50), unique=True, nullable=False)
    created_at  = Column(DateTime, nullable=False)
    follower_count = Column(Integer, default=0)
    # Composite index: common query pattern is filtering by handle then created_at
    __table_args__ = (Index("ix_handle_created", "handle", "created_at"),)

class Tweet(Base):
    __tablename__ = "tweets"
    tweet_id   = Column(Integer, primary_key=True)
    user_id    = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    content    = Column(String(280))
    created_at = Column(DateTime)
    # Index for timeline queries: all tweets by user, newest first
    __table_args__ = (Index("ix_user_created", "user_id", "created_at"),)


# --- Consistent Hashing simulation ---
import hashlib

def consistent_hash(key: str, n_nodes: int = 10) -> int:
    """Map a key to a node using consistent hashing.

    In production (Cassandra, Redis Cluster) the ring has 2^32 positions
    and vnodes distribute each physical node across multiple ring segments.
    """
    hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return hash_val % n_nodes

# Demonstrate even distribution
keys = [f"user:{i}" for i in range(1000)]
from collections import Counter
distribution = Counter(consistent_hash(k, n_nodes=5) for k in keys)
print("Key distribution across 5 nodes:", dict(sorted(distribution.items())))
# Expected: roughly 200 keys per node  →  {'0': 198, '1': 203, ...}


# --- CAP-aware database recommendation ---
def recommend_database(
    consistency: str,   # "strong" | "eventual"
    scale: str,         # "low" | "high"
    query_type: str,    # "relational" | "key_lookup" | "time_series"
) -> str:
    if consistency == "strong" and scale == "low":
        return "PostgreSQL  — ACID, excellent for complex queries & transactions"
    elif consistency == "strong" and scale == "high":
        return "Google Spanner / CockroachDB  — distributed SQL, CP, global consistency"
    elif consistency == "eventual" and query_type == "key_lookup":
        return "DynamoDB / Cassandra  — AP, horizontal scale, key-based access"
    elif query_type == "time_series":
        return "InfluxDB / Cassandra  — optimised for high-throughput time-ordered writes"
    elif scale == "high" and query_type == "relational":
        return "Vitess (MySQL sharding) or CockroachDB  — relational at scale"
    return "Evaluate: consistency vs availability vs cost — no universal answer"

print(recommend_database("strong", "low", "relational"))
# → PostgreSQL  — ACID, excellent for complex queries & transactions
print(recommend_database("eventual", "high", "key_lookup"))
# → DynamoDB / Cassandra  — AP, horizontal scale, key-based access
'''

# ---------------------------------------------------------------------------
# SCORING LOGIC HELPERS
# ---------------------------------------------------------------------------

# Axes for the radar chart
_RADAR_AXES = ["Consistency", "Scalability", "Query\nFlexibility", "Write\nSpeed", "Operational\nSimplicity"]

# Pre-baked scores per system type
# Format: {system_type: {db_option: [c, s, q, w, o]}}
_SCORES: dict[str, dict[str, list[int]]] = {
    "Social Media Feed": {
        "SQL (PostgreSQL)":   [4, 2, 5, 2, 4],
        "NoSQL (Cassandra)":  [2, 5, 2, 5, 3],
        "NewSQL (Spanner)":   [5, 4, 4, 3, 2],
    },
    "E-Commerce Orders": {
        "SQL (PostgreSQL)":   [5, 3, 5, 3, 5],
        "NoSQL (Cassandra)":  [2, 5, 2, 5, 3],
        "NewSQL (Spanner)":   [5, 5, 4, 3, 2],
    },
    "IoT Sensor Data": {
        "SQL (PostgreSQL)":   [3, 2, 4, 2, 4],
        "NoSQL (Cassandra)":  [2, 5, 2, 5, 3],
        "NewSQL (Spanner)":   [4, 4, 3, 3, 2],
    },
    "Banking Transactions": {
        "SQL (PostgreSQL)":   [5, 3, 5, 3, 5],
        "NoSQL (Cassandra)":  [1, 5, 1, 5, 2],
        "NewSQL (Spanner)":   [5, 5, 4, 3, 2],
    },
    "Ride-Sharing History": {
        "SQL (PostgreSQL)":   [4, 3, 5, 2, 4],
        "NoSQL (Cassandra)":  [2, 5, 2, 5, 3],
        "NewSQL (Spanner)":   [4, 5, 4, 3, 2],
    },
    "Real-Time Chat": {
        "SQL (PostgreSQL)":   [4, 2, 4, 2, 4],
        "NoSQL (Cassandra)":  [2, 5, 2, 5, 3],
        "NewSQL (Spanner)":   [4, 4, 3, 3, 2],
    },
}

# Fit-score adjustments per (system_type, consistency, scale, rw_ratio)
# Returns {db_name: score (0-100)} for the bar chart
def _compute_fit_scores(
    system_type: str,
    scale_level: str,
    rw_ratio: float,
    consistency_req: str,
) -> dict[str, float]:
    """Compute a 0–100 fit score for each candidate database."""
    is_large = "Large" in scale_level
    is_medium = "Medium" in scale_level
    strong = "Strong" in consistency_req
    write_heavy = rw_ratio < 0.4   # more writes than reads
    read_heavy  = rw_ratio > 0.7

    candidates: dict[str, float] = {}

    # ---- PostgreSQL ----
    pg = 50.0
    if strong:          pg += 20
    if not is_large:    pg += 15
    if not write_heavy: pg += 10
    if system_type in ("Banking Transactions", "E-Commerce Orders"): pg += 15
    if is_large and write_heavy: pg -= 20
    candidates["PostgreSQL"] = min(pg, 100)

    # ---- Cassandra ----
    cass = 50.0
    if not strong:      cass += 20
    if is_large:        cass += 15
    if write_heavy:     cass += 15
    if system_type in ("IoT Sensor Data", "Ride-Sharing History", "Real-Time Chat"): cass += 10
    if strong:          cass -= 25
    if system_type == "Banking Transactions": cass -= 30
    candidates["Cassandra"] = max(min(cass, 100), 0)

    # ---- DynamoDB ----
    ddb = 50.0
    if not strong:      ddb += 15
    if is_large:        ddb += 10
    if read_heavy:      ddb += 10
    if system_type in ("Social Media Feed", "Ride-Sharing History"): ddb += 10
    if strong:          ddb -= 20
    candidates["DynamoDB"] = max(min(ddb, 100), 0)

    # ---- Redis (cache layer, not primary) ----
    redis = 40.0
    if read_heavy:      redis += 20
    if not is_large:    redis += 10
    if system_type in ("Social Media Feed", "Real-Time Chat", "E-Commerce Orders"): redis += 15
    candidates["Redis (Cache)"] = min(redis, 100)

    # ---- Google Spanner / CockroachDB ----
    spanner = 40.0
    if strong:          spanner += 20
    if is_large:        spanner += 20
    if system_type in ("Banking Transactions", "E-Commerce Orders"): spanner += 10
    if not is_large:    spanner -= 10   # overkill for small scale
    candidates["Spanner/CockroachDB"] = max(min(spanner, 100), 0)

    # ---- InfluxDB / TimescaleDB ----
    tsdb = 30.0
    if system_type == "IoT Sensor Data": tsdb += 45
    if write_heavy:     tsdb += 15
    if system_type in ("Ride-Sharing History",): tsdb += 10
    candidates["InfluxDB/TimescaleDB"] = min(tsdb, 100)

    # Return top 4 for the bar chart
    sorted_candidates = dict(sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:4])
    return sorted_candidates


def _build_radar_scores(
    system_type: str,
    scale_level: str,
    rw_ratio: float,
    consistency_req: str,
) -> dict[str, list[float]]:
    """Return adjusted radar scores for SQL, NoSQL, NewSQL."""
    base = _SCORES.get(system_type, _SCORES["Social Media Feed"])
    result: dict[str, list[float]] = {}

    strong = "Strong" in consistency_req
    large  = "Large" in scale_level
    write_heavy = rw_ratio < 0.4

    for db_name, scores in base.items():
        adj = list(map(float, scores))
        # Boost consistency axis for SQL when strong required
        if "PostgreSQL" in db_name or "SQL" in db_name:
            adj[0] = min(5.0, adj[0] + (1 if strong else 0))
            adj[1] = max(1.0, adj[1] - (1 if large else 0))
        if "Cassandra" in db_name or "NoSQL" in db_name:
            adj[3] = min(5.0, adj[3] + (0.5 if write_heavy else 0))
            adj[0] = max(1.0, adj[0] - (1 if strong else 0))
        result[db_name] = adj

    return result


# ---------------------------------------------------------------------------
# RECOMMENDATION TEXT LOGIC
# ---------------------------------------------------------------------------

_RECOMMENDATIONS: dict[str, dict] = {
    "Social Media Feed": {
        "primary_db":   "Cassandra (AP, column-family store)",
        "cache":        "Redis — cache hot timelines and user sessions",
        "arch_pattern": "Write fan-out → Cassandra (durable) + Redis sorted sets (hot feeds) + CDN for media",
        "cap":          "AP — eventually consistent. Follower counts and like counts may lag by seconds.",
        "trade_offs":   "Stale counts acceptable; complex JOINs not needed; write amplification on fan-out to followers is the key bottleneck",
        "talking_pts":  "Mention the 'celebrity problem' — users with 100M followers require pull-on-read (not push fan-out) to avoid write storms.",
        "red_flags":    "Don't propose ACID transactions for like counts; don't shard by username (hot spots for popular users).",
    },
    "E-Commerce Orders": {
        "primary_db":   "PostgreSQL (ACID, relational) for orders + DynamoDB / Redis for cart",
        "cache":        "Redis — shopping cart (session-level), product catalogue reads",
        "arch_pattern": "PostgreSQL primary-replica for order records; Redis for cart; Elasticsearch for product search",
        "cap":          "CP for orders (must not lose a payment), AP for cart (stale cart contents acceptable)",
        "trade_offs":   "Cart abandonment is acceptable data loss; order commits are not. Two-phase commit for payment + inventory check.",
        "talking_pts":  "Idempotency keys on order creation prevent duplicate charges. Use optimistic locking on inventory counts.",
        "red_flags":    "Don't use Cassandra as primary for orders — no multi-table transactions. Don't skip idempotency on payment APIs.",
    },
    "IoT Sensor Data": {
        "primary_db":   "Cassandra or InfluxDB (high-throughput time-series writes)",
        "cache":        "Redis — latest reading per device (O(1) lookup)",
        "arch_pattern": "Kafka ingest → Cassandra (durable store) + InfluxDB (real-time aggregation) + S3 (cold archive)",
        "cap":          "AP — sensor readings are append-only; occasional stale aggregates are fine.",
        "trade_offs":   "Schema-less time-series rows; TTL-based expiry for old data; partition key must include time bucket to avoid hot partitions.",
        "talking_pts":  "Time-bucket partitioning: partition key = (device_id, YYYY-MM-DD) to spread writes. Without bucketing, one partition gets all writes.",
        "red_flags":    "Don't use a relational DB with auto-increment PK for sensor writes — it serialises inserts. Don't forget TTL for data expiry.",
    },
    "Banking Transactions": {
        "primary_db":   "PostgreSQL (ACID, row-level locking, WAL durability)",
        "cache":        "Redis — read-only balance cache with short TTL (invalidated on each transaction)",
        "arch_pattern": "PostgreSQL primary (strong consistency) + read replicas for statements + Redis balance cache",
        "cap":          "CP — must never show stale balance. Partition tolerance sacrificed at the expense of availability during split-brain.",
        "trade_offs":   "Two-phase commit for cross-account transfers; serialisable isolation level for concurrent debit/credit; audit log is append-only.",
        "talking_pts":  "Distributed saga pattern for multi-service transactions (e.g., transfer + notification). Compensating transactions for rollback.",
        "red_flags":    "Never propose Cassandra or DynamoDB as primary for financial transactions — no multi-row ACID. Never use optimistic locking on account balances.",
    },
    "Ride-Sharing History": {
        "primary_db":   "Cassandra for completed trip history + PostgreSQL for active ride state",
        "cache":        "Redis — active driver locations (geo-indexed), active ride status",
        "arch_pattern": "PostgreSQL (active rides, ACID for payments) + Cassandra (trip archive) + Redis (driver geolocation)",
        "cap":          "CP for active ride payments; AP for historical trip data.",
        "trade_offs":   "Active ride count is small (fits in PostgreSQL). Historical trips grow unbounded → Cassandra with partition key = (driver_id, month).",
        "talking_pts":  "Separate the hot path (active ride: low volume, high consistency) from cold path (trip history: high volume, AP). Redis GEOADD for driver proximity.",
        "red_flags":    "Don't store all rides in PostgreSQL long-term — will grow to billions of rows. Don't use Cassandra for the active ride payment flow.",
    },
    "Real-Time Chat": {
        "primary_db":   "Cassandra for message history + Redis for active sessions and recent messages",
        "cache":        "Redis — unread counts, online presence, recent 50 messages per conversation",
        "arch_pattern": "WebSocket servers → Redis pub/sub (fan-out to active users) → Cassandra (durable message log)",
        "cap":          "AP — message delivery is eventually consistent; brief duplicates or ordering issues are handled client-side.",
        "trade_offs":   "Cassandra partition key = (conversation_id, bucket) for even write distribution. Redis handles real-time fan-out; Cassandra handles durability.",
        "talking_pts":  "Message ordering via Cassandra clustering key = (created_at, message_id). Client-side de-duplication using idempotency keys. Redis TTL for presence expiry.",
        "red_flags":    "Don't use a single PostgreSQL table for all messages — will become a hot-spot write bottleneck. Don't rely solely on Redis for durability (it's volatile by default).",
    },
}


# ---------------------------------------------------------------------------
# MAIN DEMO FUNCTION
# ---------------------------------------------------------------------------

def run_data_model_demo(
    system_type: str,
    scale_level: str,
    rw_ratio: float,
    consistency_req: str,
) -> tuple:
    """
    Returns (fig, metrics_md).

    fig: 2-panel Plotly figure
        Left  — Radar chart: SQL vs NoSQL vs NewSQL across 5 axes
        Right — Horizontal bar chart: Fit Score for top 4 database options
    metrics_md: Markdown with recommendation, trade-offs, and interview tips
    """
    try:
        palette = COLORS["palette"]

        # --- Scores ---
        radar_scores = _build_radar_scores(system_type, scale_level, rw_ratio, consistency_req)
        fit_scores   = _compute_fit_scores(system_type, scale_level, rw_ratio, consistency_req)

        # --- Figure layout ---
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "polar"}, {"type": "xy"}]],
            subplot_titles=["SQL vs NoSQL vs NewSQL — Capability Radar", "Database Fit Score for Selected System"],
            column_widths=[0.48, 0.52],
        )

        # --- Left panel: Radar chart ---
        categories = _RADAR_AXES + [_RADAR_AXES[0]]   # close the polygon

        for idx, (db_name, scores) in enumerate(radar_scores.items()):
            values = scores + [scores[0]]   # close loop
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill="toself",
                    name=db_name,
                    line=dict(color=palette[idx % len(palette)], width=2),
                    fillcolor=palette[idx % len(palette)],
                    opacity=0.25,
                    hovertemplate="%{theta}: %{r:.1f}<extra>" + db_name + "</extra>",
                ),
                row=1, col=1,
            )

        fig.update_polars(
            radialaxis=dict(visible=True, range=[0, 5], tickvals=[1, 2, 3, 4, 5]),
            bgcolor="rgba(0,0,0,0)",
        )

        # --- Right panel: Horizontal bar chart ---
        db_names   = list(fit_scores.keys())
        scores_arr = [fit_scores[d] for d in db_names]
        bar_colors = [palette[i % len(palette)] for i in range(len(db_names))]

        fig.add_trace(
            go.Bar(
                x=scores_arr,
                y=db_names,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{s:.0f}/100" for s in scores_arr],
                textposition="outside",
                hovertemplate="%{y}: %{x:.0f}/100<extra></extra>",
                name="Fit Score",
                showlegend=False,
            ),
            row=1, col=2,
        )

        fig.update_xaxes(range=[0, 115], title_text="Fit Score (0–100)", row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=1, col=2)

        # --- Global layout ---
        fig.update_layout(
            template="plotly_white",
            height=480,
            margin=dict(l=20, r=30, t=60, b=20),
            legend=dict(orientation="h", y=-0.08, x=0.0),
            title=dict(
                text=f"Storage Design Analysis — {system_type}",
                font=dict(size=16),
                x=0.5,
            ),
        )

        # --- Metrics Markdown ---
        rec = _RECOMMENDATIONS.get(system_type, _RECOMMENDATIONS["Social Media Feed"])

        # RW ratio description
        if rw_ratio >= 0.8:
            rw_label = "very read-heavy (80 %+ reads)"
        elif rw_ratio >= 0.6:
            rw_label = "read-heavy (~" + f"{int(rw_ratio*100)}% reads)"
        elif rw_ratio >= 0.4:
            rw_label = "balanced read/write"
        elif rw_ratio >= 0.2:
            rw_label = "write-heavy (~" + f"{int((1-rw_ratio)*100)}% writes)"
        else:
            rw_label = "very write-heavy (80 %+ writes)"

        top_db  = db_names[0]
        top_score = scores_arr[0]

        metrics_md = f"""### Analysis: **{system_type}**

| Parameter | Value |
|-----------|-------|
| Scale | {scale_level} |
| Read/Write Profile | {rw_label} |
| Consistency Requirement | {consistency_req} |
| Top Recommended DB | **{top_db}** ({top_score:.0f}/100 fit) |

---

#### Recommended Primary Database
**{rec['primary_db']}**

#### Recommended Caching Layer
{rec['cache']}

#### Architecture Pattern
`{rec['arch_pattern']}`

#### CAP Classification
{rec['cap']}

---

#### Key Trade-offs to Mention in the Interview
> {rec['trade_offs']}

#### Interview Talking Points
> {rec['talking_pts']}

#### Red Flags to Avoid
> {rec['red_flags']}

---
*Fit scores are heuristic estimates based on system characteristics — always adapt to your specific scale and team constraints.*
"""

        return fig, metrics_md

    except Exception as e:
        import traceback
        return go.Figure(), f"**Error:** {traceback.format_exc()}"


# ---------------------------------------------------------------------------
# GRADIO TAB BUILDER
# ---------------------------------------------------------------------------

def build_tab():
    gr.Markdown("# 🗄️ Module 01 — Data Model & Storage Design\n*Level: Intermediate*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)

    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown(
        "---\n"
        "## 🎮 Interactive Demo — Database Design Advisor\n\n"
        "Configure your system's characteristics and click **Run Analysis** to receive a tailored "
        "storage recommendation with interview talking points."
    )

    with gr.Row():
        with gr.Column(scale=1):
            system_type_dd = gr.Dropdown(
                choices=[
                    "Social Media Feed",
                    "E-Commerce Orders",
                    "IoT Sensor Data",
                    "Banking Transactions",
                    "Ride-Sharing History",
                    "Real-Time Chat",
                ],
                value="Social Media Feed",
                label="System Type",
            )
            scale_radio = gr.Radio(
                choices=["Small (< 1K users)", "Medium (10K–1M users)", "Large (> 100M users)"],
                value="Large (> 100M users)",
                label="Scale Level",
            )
            rw_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.7,
                label="Read/Write Ratio  (0 = write-heavy → 1 = read-heavy)",
            )
            consistency_radio = gr.Radio(
                choices=["Strong (ACID)", "Eventual (BASE)"],
                value="Eventual (BASE)",
                label="Consistency Requirement",
            )
            run_btn = gr.Button("▶ Run Analysis", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Database Comparison")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_data_model_demo,
        inputs=[system_type_dd, scale_radio, rw_slider, consistency_radio],
        outputs=[plot_out, metrics_out],
    )
