"""Module 02 — Data Architecture & Observability
Level: Advanced"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import COLORS

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
## 🏗️ The Modern Data Stack — A Senior Engineer's Complete Guide

Mastering data architecture is one of the highest-leverage skills an engineer
can develop. Nearly every staff-level system design interview at Google, Meta,
Amazon, Databricks, or Snowflake will involve designing or critiquing a data
platform. This module covers every concept you need — not just *what* each
tool does, but *why* it was built and *when* to choose it.

---

## Section 1 — The Three Storage Paradigms

Before choosing a tool, understand the three fundamental paradigms that
underpin every modern data platform:

### 1a. Data Lake
A data lake stores raw data at minimal cost using object storage (Amazon S3,
Google Cloud Storage, Azure Data Lake Storage Gen2). The defining principle is
**schema-on-read**: data is written in its original form with no upfront
transformation, and the schema is applied at query time.

**Strengths:**
- Extremely cheap storage (S3 ≈ $0.023/GB/month vs Snowflake ≈ $40/TB/month
  for managed storage)
- Stores unstructured data — JSON logs, images, videos, raw CDC streams
- Ideal as the landing zone for all data before transformation
- Native home for ML training data (data scientists need raw signals)

**Weaknesses:**
- No ACID transactions → concurrent writes can corrupt files
- No schema enforcement → "data swamp" risk — quality degrades over time
- Poor BI/SQL performance without careful partitioning and file format choices
- No update/delete capability with plain Parquet/ORC

**When to use:** ML/AI workloads that need raw features, archiving raw events
for replay, storing unstructured data (PDFs, images, logs), cheap long-term
retention.

### 1b. Data Warehouse
A data warehouse enforces **schema-on-write**: data must conform to a defined
schema before it is loaded. It is purpose-built for structured SQL analytics
and BI dashboards.

**Key platforms:** Snowflake, Google BigQuery, Amazon Redshift, Azure Synapse

**Strengths:**
- Exceptional SQL performance — query billions of rows in seconds
- ACID guarantees — no corrupt data
- Governed schema enforcement, data types, primary/foreign keys
- Native connectors for BI tools (Tableau, Looker, Power BI)
- Zero-copy data sharing (Snowflake) — share live data with partners without
  duplication

**Weaknesses:**
- Expensive for storing large raw datasets
- Poor fit for unstructured data or ML pipelines that need raw bytes
- Schema changes require migrations (adding columns to 500-column tables is
  painful)

**When to use:** BI dashboards, financial reporting, ad-hoc SQL analytics,
any workload dominated by structured aggregation queries.

### 1c. Data Lakehouse
The Lakehouse is the modern synthesis: **ACID transactions and SQL
performance directly on top of cheap object storage.** It achieves this via
**open table formats** (Delta Lake, Apache Iceberg, Apache Hudi) that add a
metadata/transaction log layer on top of Parquet files in S3/GCS/ADLS.

**Key platforms:** Databricks (Delta Lake), Apache Iceberg on any engine,
AWS Glue, Azure Fabric

**Strengths:**
- Combine ML workloads (raw data, feature engineering) and BI analytics on
  the same dataset — no copying
- ACID transactions enable concurrent reads and writes safely
- Schema evolution without full table rewrites
- Time travel — query data as it existed at any past point in time
- Significantly cheaper than a pure warehouse for large data volumes

**Decision framework:**
| Need | Best Fit |
|---|---|
| Pure SQL BI, strong governance, no ML | Data Warehouse |
| Raw ML data only, cost-sensitive | Data Lake |
| Both ML + BI, regulatory compliance, want time travel | **Lakehouse** |
| Frequent real-time updates/deletes (CDC) | Lakehouse with Hudi/Iceberg |

---

## Section 2 — Open Table Formats: Delta Lake vs Iceberg vs Hudi

Open table formats are the foundational technology that makes the Lakehouse
possible. Understanding their trade-offs is a senior engineer signal.

| Feature | Delta Lake | Apache Iceberg | Apache Hudi |
|---|---|---|---|
| **Origin** | Databricks (2019) | Netflix (2018) | Uber (2017) |
| **Core innovation** | Transaction log on S3 | Manifest-based metadata | Write-optimized CDC |
| **Best for** | Databricks ecosystem | Multi-cloud portability | High-frequency CDC |
| **Schema evolution** | Basic (add columns) | In-place, no rewrite | Basic |
| **Multi-engine support** | Databricks-native; Spark, Trino | Snowflake, BigQuery, Trino, Athena, Spark, Flink | Spark, Flink, Hive |
| **Vendor lock-in** | Medium (Databricks ecosystem) | Low — most portable format | Medium |
| **ACID compliance** | Full | Full | Full |
| **Time travel** | Yes (transaction log) | Yes (snapshot metadata) | Yes (timeline) |
| **Row-level deletes** | Yes (merge) | Yes (merge-on-read or copy-on-write) | Native (designed for it) |
| **Streaming ingestion** | Excellent (Spark Structured Streaming) | Good | Excellent (designed for it) |

**When to choose Delta Lake:**
You are already on Databricks, you need strong consistency guarantees with
minimal configuration, and your primary workload is ELT pipelines feeding
downstream analytics. Delta Lake's transaction log is the simplest to operate.

**When to choose Apache Iceberg:**
You need maximum portability across cloud providers and query engines. Iceberg
tables can be queried natively from Snowflake, BigQuery, Trino, Athena, and
Spark without any conversion. It has the lowest vendor lock-in of the three,
making it the default recommendation when the company does not standardize on
a single cloud or compute engine. Netflix chose it specifically to avoid
depending on any single vendor.

**When to choose Apache Hudi:**
Your workload involves high-frequency record-level updates and deletes — the
canonical example is CDC (Change Data Capture) from a transactional database
where you're continuously applying inserts/updates/deletes to a data lake copy
of your operational data. Uber built Hudi to handle exactly this: ingesting
billions of trip record updates per day into their data lake. Full table scans
to apply CDC are impossibly expensive; Hudi's record-level indexing makes it
practical.

---

## Section 3 — Medallion Architecture (Bronze → Silver → Gold)

The Medallion Architecture is a data layering pattern that progressively
refines raw data into business-ready assets. It was popularized by Databricks
but is now the standard pattern across Snowflake, BigQuery, and most modern
platforms.

```
Raw Sources → [Bronze] → [Silver] → [Gold] → BI / ML Consumers
```

### Bronze Layer — Raw, High-Fidelity Copy
**Principle:** Write exactly what arrived. Never transform. Never drop. Append-only.

- Ingest from Kafka topics, Salesforce APIs, CDC streams, S3 file drops
- Store as-is: JSON, CSV, raw Avro — whatever the source produces
- Timestamp every record with ingestion time (`_ingested_at`)
- Tag with source system (`_source = "salesforce_crm"`)
- No schema enforcement at write time — sources are unpredictable

**Why:** The Bronze layer is your source of truth for reprocessing. If a bug
corrupts the Silver layer, you replay from Bronze. If business logic changes,
you re-derive from Bronze. Losing Bronze data is permanent; losing Silver/Gold
is recoverable.

### Silver Layer — Validated, Deduplicated, Schema-Enforced
**Principle:** Apply business rules. Quarantine bad records. Enforce contracts.

- Drop or quarantine null primary keys, invalid event types, negative amounts
- Deduplicate on natural keys (event_id, transaction_id)
- Enforce schema and data types (string → timestamp, string → float)
- Apply GDPR right-to-delete here: remove PII on request before it propagates
  to Gold
- Split into a quarantine table for investigation — never silently drop bad data
- Run dbt tests at this layer: unique, not-null, accepted_values

**Why:** The Silver layer is the trust boundary. Downstream consumers (Gold,
ML features, dashboards) should never need to defend against bad data. Silver
is the single layer that absorbs all cleanup debt.

### Gold Layer — Business-Ready Aggregations and ML Features
**Principle:** Optimized for consumption. No transformations needed by consumers.

- Joins across Silver tables (orders + users + products)
- Aggregations: daily revenue by region, user lifetime value, churn features
- Feature engineering for ML: rolling 30-day averages, user activity windows
- Pre-computed for BI performance — dashboards should read Gold, not re-compute

**Real-world guidance:** Databricks recommends this exact pattern in their
reference architectures. Snowflake supports it as "raw → transformed →
presentation" layers. dbt naturally maps to Silver (staging, intermediate)
and Gold (marts) models.

---

## Section 4 — Data Mesh: Decentralized Ownership at Scale

Data Mesh was introduced by Zhamak Dehghani at ThoughtWorks in 2020 as a
response to a fundamental scaling failure: centralized data teams become
bottlenecks as organizations grow. By the time a central team has ingested,
cleaned, and published the orders domain data, the product requirement has
changed.

### The 4 Principles

**1. Domain-Oriented Decentralized Ownership**
Data ownership belongs to the domain team that generates the data. The Orders
team owns `orders_domain.orders_gold`. The Recommendations team owns
`recs_domain.user_signals_gold`. No central "data team" is the bottleneck.
Analogy: microservices gave application ownership to domain teams; Data Mesh
does the same for data.

**2. Data as a Product**
Domain teams treat their data outputs like products, not byproducts. This
means: SLAs on freshness (data updated within 30 minutes of source),
discoverability (documented in a catalog), quality guarantees (>99% row
completeness), versioned interfaces (schema changes require deprecation
period), and support channels.

**3. Self-Serve Data Infrastructure Platform**
A central platform team provides the *tools* (catalog, lineage, pipeline
templates, governance APIs) that domain teams use to build and publish data
products. The platform team does not own the data itself — it owns the
infrastructure layer. Think: AWS gives you S3, EC2, and IAM; it does not
build your application.

**4. Federated Computational Governance**
Global policies (GDPR compliance, PII classification, retention periods,
access control) are set centrally and automatically enforced. But domain teams
make local decisions within those guardrails. This prevents both anarchy
(no governance) and paralysis (central team approves every query).

### Data Mesh vs Centralized: Trade-offs
| Dimension | Centralized | Data Mesh |
|---|---|---|
| Team bottleneck | High (central team saturates) | Low (domain teams self-serve) |
| Data quality ownership | Unclear (who is accountable?) | Clear (domain team owns SLA) |
| Coordination overhead | Low (one team) | High (N domain teams to align) |
| Best for | Small orgs, <20 data engineers | Large orgs, domain teams >5 |
| Risk | Bottleneck at scale | Inconsistency between domains |

**When to recommend Data Mesh in an interview:** When the problem describes
a large organization (500+ engineers), multiple business domains (logistics,
payments, recommendations), and a central data team struggling to keep up.

---

## Section 5 — Cloud Platform Comparison

| Platform | Compute Model | Best For | Key Differentiator |
|---|---|---|---|
| **Snowflake** | Decoupled storage/compute; virtual warehouses | SQL analytics, data sharing, governed BI | Zero-copy cloning, Snowflake Data Marketplace, best SQL governance |
| **Databricks** | Unified analytics on Apache Spark | ML + data engineering together | MLflow, Delta Lake, native notebooks, Unity Catalog |
| **BigQuery** | Serverless SQL; no infrastructure | Ad-hoc analytics at scale, pay per query | No infrastructure management, per-TB query pricing, native ML with BQML |
| **AWS Redshift** | Cluster-based MPP warehouse | AWS-native workloads, Spectrum for lake queries | Deep AWS integration, Redshift Spectrum queries S3 directly |
| **Azure Synapse** | Unified workspace: SQL + Spark | Azure ecosystem, existing Microsoft footprint | Built-in Spark pools + SQL pools, Power BI native integration |

**Interview shorthand:**
- Snowflake: "I need best-in-class SQL analytics and data governance"
- Databricks: "I need ML and data engineering on the same platform"
- BigQuery: "I need serverless scale without managing clusters"
- Redshift: "I'm all-in AWS and need tight ecosystem integration"
- Synapse: "I'm all-in Azure with existing Microsoft investments"

**Common trap:** Saying "just use BigQuery for everything" without trade-off
analysis is a red flag. Each platform has genuine weaknesses — BigQuery is
expensive for frequent small queries; Snowflake requires compute cluster
management; Databricks has a steeper operational learning curve.

---

## Section 6 — Data Observability: Proactive Testing vs Reactive Monitoring

The distinction between data testing and data observability is one of the most
important concepts for senior engineers — and one of the most commonly
conflated.

### Data Testing (Proactive — Known Unknowns)
Testing catches conditions you anticipated and wrote rules for.

**dbt tests** are SQL assertions that run after every transformation:
- `unique`: no duplicate primary keys in `orders_gold.order_id`
- `not_null`: `user_id` is never null in `users_silver`
- `accepted_values`: `payment_status` is always one of [pending, completed,
  failed, refunded]
- `relationships`: every `order.user_id` has a matching record in `users`

**Great Expectations** is a Python framework for data quality assertions:
```python
expect_column_values_to_not_be_null("user_id")
expect_column_values_to_be_between("amount", 0, 1_000_000)
expect_column_proportion_of_unique_values_to_be_between("session_id", 0.95, 1.0)
```

Testing is excellent but has a critical blind spot: it only catches what you
thought to check. It cannot detect distributions shifting over time.

### Data Observability (Reactive — Unknown Unknowns)
Observability monitors the *behavior* of data over time and alerts when
something looks anomalous — even if you didn't write a rule for it.

**Monte Carlo** is the leading data observability platform. It monitors:
- **Freshness**: "Orders table has not updated in 4 hours — SLA breach"
- **Volume**: "Events table received 40% fewer rows than yesterday at this hour"
- **Distribution**: "The `amount` column distribution shifted — mean went from
  $45 to $120 overnight"
- **Schema**: "Column `user_segment` was dropped from `users_gold`"
- **Lineage impact**: "This upstream table failure affects 12 downstream
  dashboards"

**Datadog Observability Pipelines** operates at the infrastructure layer —
monitoring pipeline throughput (events/second), latency (lag between
Kafka produce and Spark consume), drop rate (events lost due to serialization
errors), and cost (bytes processed per pipeline).

**The key insight**: dbt tests catch *known* rules violations. Monte Carlo
catches *statistical anomalies* — the category of problems no one anticipated.
Senior engineers implement both layers.

---

## Section 7 — Data Lineage and Governance

### Why Lineage Matters
When a dashboard shows the wrong revenue number, the first question is: "which
upstream table is wrong?" Without lineage, answering this requires hours of
manual tracing through dbt models, Airflow DAGs, and Spark jobs. With lineage,
it takes seconds.

### OpenLineage
OpenLineage is an **open standard** (vendor-neutral specification) for
capturing metadata about data pipeline runs: which datasets were read, which
were written, job parameters, run timestamps, and facets (statistics). It
integrates natively with Airflow (via the OpenLineage Airflow provider), dbt,
Spark, and Flink. The key advantage: because it is a standard, not a vendor
product, lineage metadata from different tools is interoperable.

### DataHub (LinkedIn → Open Source)
DataHub is a **metadata platform** that ingests OpenLineage events and other
sources to provide:
- **Column-level lineage**: "This `revenue_usd` column in the Gold table traces
  back to `raw_orders.amount` (Bronze) → `orders_silver.amount_usd` (Silver)"
- **Dataset-to-dashboard tracking**: "This Tableau dashboard depends on 4 Gold
  tables — if any of them fail, the dashboard is stale"
- **AI-powered discovery**: Semantic search over datasets, auto-generated
  descriptions
- **Impact analysis**: "If I change the schema of `orders_silver`, which 28
  downstream jobs and 14 dashboards are affected?"

### Apache Atlas
Atlas provides lineage for Hadoop-ecosystem sources: Hive, Spark on YARN,
Sqoop, HBase. It is deeply integrated with the Apache ecosystem but has
limited support for modern cloud-native sources (no native Snowflake or
Salesforce connectors). If your stack is primarily on-premises Hadoop, Atlas
is a natural fit; for cloud-native stacks, DataHub is the better choice.

### Governance Framework
- **RBAC on the data catalog**: not everyone should see PII columns — apply
  column-level masking policies (Snowflake Column Masking, BigQuery authorized
  views)
- **GDPR compliance**: right-to-delete must propagate from Silver → Gold → all
  downstream materialized views; tag PII columns in DataHub; automate retention
  policies
- **Data classification**: label every dataset as PII / Sensitive / Internal /
  Public; automate enforcement via catalog policies
- **Data contracts**: a formal interface agreement between the producer
  (domain team) and consumers — defines schema, freshness SLA, and quality
  rules

---

## Section 8 — Data Integration Patterns

### ETL vs ELT: The Architectural Shift

**ETL (Extract → Transform → Load)** is the traditional pattern: transform data
in a staging server before loading into the warehouse. This made sense when
warehouse compute was expensive and scarce.

**ELT (Extract → Load → Transform)** is the modern cloud-native pattern: load
raw data into the warehouse first, then use the warehouse's own compute for
transformation (dbt models). Warehouse compute is now cheap and elastic —
Snowflake and BigQuery can transform billions of rows in seconds. ELT is simpler
(no separate transform infrastructure), more scalable, and makes re-running
transformations trivial.

**Interview red flag:** Proposing ETL for a cloud-native stack without
mentioning ELT. ELT has been the dominant pattern since approximately 2018.

### CDC (Change Data Capture)
CDC captures only the changes (inserts, updates, deletes) from a transactional
database by reading the database's replication log (binlog in MySQL, WAL in
PostgreSQL). This is far more efficient than full table scans.

**Architecture:** PostgreSQL WAL → Debezium (CDC connector) → Kafka topic →
Kafka Connect → Delta Lake / Iceberg (upsert via merge)

**Why CDC matters at interview:** A database with 100M rows might have only
10K changes per day. A full table scan reads 100M rows; CDC reads 10K. At
scale, this is the difference between a 6-hour pipeline and a 2-minute one.

### Streaming vs Batch
| Dimension | Batch (Fivetran + dbt) | Streaming (Kafka + Spark) |
|---|---|---|
| **Data freshness** | Hours (scheduled) | Seconds/minutes |
| **Complexity** | Low | High |
| **Fault tolerance** | Simple (re-run job) | Requires offset management, watermarking |
| **Cost** | Lower | Higher (always-on compute) |
| **Best for** | Reporting, overnight ETL | Fraud detection, real-time personalization |

---

## Common Interview Questions and Model Answers

**"Design a data pipeline for 1 billion daily events"**
> "I'd use Kafka as the event bus for ingestion at scale. Spark Structured
> Streaming reads from Kafka topics and writes to Bronze (Delta Lake on S3) —
> append-only, exact copy. A dbt pipeline runs every 30 minutes to promote
> clean records to Silver (dedup, null checks, schema enforcement). Gold layer
> aggregations run hourly, materialized into Snowflake for BI tools. The table
> format would be Apache Iceberg for multi-engine flexibility."

**"How do you ensure data quality at scale?"**
> "I implement two complementary layers: dbt tests (unique, not-null,
> referential integrity) run after every transformation for known rules, and
> Monte Carlo for observability — it detects freshness delays, volume anomalies,
> and distribution drift that no one wrote rules for. Bad records in the Silver
> layer are quarantined, not dropped — that quarantine table is the team's
> quality backlog."

**Red flags to avoid:**
- Proposing full table scans when CDC is available
- "Just use ETL" for a cloud-native stack (should be ELT)
- "Just use BigQuery for everything" without trade-off analysis
- Storing all data in a warehouse (expensive; should use lake for raw/ML data)
- No mention of data quality testing OR observability (needs both layers)

**Green flags to demonstrate:**
- Distinguishing CDC vs full scans and knowing when each applies
- Explaining Bronze/Silver/Gold layers with concrete tool choices
- Mentioning data lineage for audit trails and impact analysis
- Recommending Iceberg for multi-cloud or Hudi for high-frequency CDC
- Knowing that dbt = known unknowns, Monte Carlo = unknown unknowns
"""

# ─────────────────────────────────────────────────────────────────────────────
# CODE EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

CODE_EXAMPLE = '''
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ── Medallion Architecture Simulation ────────────────────────────────────────

# Bronze: raw ingestion — no cleanup, exact copy of source
def ingest_bronze(raw_records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_records)
    df["_ingested_at"] = datetime.now()
    df["_source"] = "kafka_events"
    # Never transform, never drop — Bronze is the source of truth for replay
    return df

# Silver: validation + deduplication + schema enforcement
def transform_silver(bronze_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_mask = (
        bronze_df["user_id"].notna() &
        bronze_df["event_type"].isin(["click", "view", "purchase"]) &
        (bronze_df["amount"] >= 0)
    )
    silver = bronze_df[valid_mask].drop_duplicates(subset=["event_id"])
    # Quarantine bad records for investigation — never silently drop
    quarantine = bronze_df[~valid_mask].copy()
    quarantine["_quarantine_reason"] = "failed_validation"
    return silver, quarantine

# Gold: business-ready aggregations and ML features
def build_gold(silver_df: pd.DataFrame) -> pd.DataFrame:
    return (
        silver_df
        .groupby("user_id")
        .agg(
            total_spend    = ("amount",       "sum"),
            event_count    = ("event_id",     "count"),
            purchase_count = ("event_type",   lambda x: (x == "purchase").sum()),
            last_seen      = ("_ingested_at", "max"),
        )
        .reset_index()
        .assign(
            avg_order_value = lambda df: (
                df["total_spend"] / df["purchase_count"].replace(0, np.nan)
            ).round(2)
        )
    )

# ── Data Freshness Check (Observability) ─────────────────────────────────────
def check_freshness(df: pd.DataFrame, max_lag_minutes: int = 30) -> dict:
    latest = df["_ingested_at"].max()
    lag    = (datetime.now() - latest).total_seconds() / 60
    return {
        "latest_record":     latest.isoformat(),
        "lag_minutes":       round(lag, 2),
        "status":            "PASS" if lag <= max_lag_minutes else "FAIL",
        "threshold_minutes": max_lag_minutes,
    }

# ── Schema Validation (Great Expectations style) ─────────────────────────────
def validate_schema(df: pd.DataFrame) -> list[dict]:
    results = []

    # Expectation 1: user_id never null
    null_pct = df["user_id"].isna().mean()
    results.append({
        "expectation":  "expect_column_values_to_not_be_null(user_id)",
        "passed":       null_pct == 0,
        "observed":     f"{null_pct:.1%} null",
    })

    # Expectation 2: amount in valid range
    invalid_amount = ((df["amount"] < 0) | (df["amount"] > 10_000)).mean()
    results.append({
        "expectation":  "expect_column_values_to_be_between(amount, 0, 10000)",
        "passed":       invalid_amount == 0,
        "observed":     f"{invalid_amount:.1%} out of range",
    })

    # Expectation 3: uniqueness of event_id
    dupe_rate = 1 - df["event_id"].nunique() / len(df)
    results.append({
        "expectation":  "expect_column_proportion_of_unique_values_to_be_between(event_id, 0.95, 1.0)",
        "passed":       dupe_rate < 0.05,
        "observed":     f"{dupe_rate:.1%} duplicate rate",
    })

    return results

# ── Run the full pipeline ─────────────────────────────────────────────────────
if __name__ == "__main__":
    raw = [
        {"event_id": f"e{i:04d}", "user_id": f"u{random.randint(1,5):02d}",
         "event_type": random.choice(["click","view","purchase","INVALID"]),
         "amount": random.uniform(-5, 200)}
        for i in range(1000)
    ] + [{"event_id": "e0001", "user_id": None, "event_type": "click", "amount": 10}]  # duplicate + null

    bronze              = ingest_bronze(raw)
    silver, quarantine  = transform_silver(bronze)
    gold                = build_gold(silver)
    freshness           = check_freshness(bronze)
    quality             = validate_schema(bronze)

    print(f"Bronze: {len(bronze):,} rows | Silver: {len(silver):,} rows | Quarantine: {len(quarantine):,} rows")
    print(f"Gold users: {len(gold):,} | Freshness: {freshness[\'status\']} ({freshness[\'lag_minutes\']} min lag)")
    print(f"Quality checks: {sum(r[\'passed\'] for r in quality)}/{len(quality)} passed")
'''

# ─────────────────────────────────────────────────────────────────────────────
# SCORING TABLES
# ─────────────────────────────────────────────────────────────────────────────

# Base platform scores for 6 dimensions: SQL, ML, Real-time, Multi-cloud, Governance, Cost
# Score 1-5; higher = better fit. Cost is inverted (5 = cheapest).
_BASE_SCORES = {
    "Snowflake":     [5, 2, 2, 4, 5, 2],
    "Databricks":    [3, 5, 4, 3, 3, 2],
    "BigQuery":      [5, 3, 3, 4, 4, 4],
    "AWS Redshift":  [4, 2, 2, 2, 3, 3],
    "Azure Synapse": [4, 3, 2, 2, 4, 3],
}

_DIMENSIONS = ["SQL Analytics", "ML Workloads", "Real-time", "Multi-cloud", "Governance", "Cost Efficiency"]

# Use-case base score adjustments  {platform: [SQL, ML, RT, MC, Gov, Cost]}
_USE_CASE_ADJUSTMENTS = {
    "Regulatory Finance":    {"Snowflake": [0,0,0,0,1,0],  "Databricks": [0,0,0,0,-1,0], "BigQuery": [0,0,0,0,0,0], "AWS Redshift": [0,0,0,0,0,0], "Azure Synapse": [0,0,0,0,0,0]},
    "Startup Analytics":     {"Snowflake": [0,0,0,0,0,-1], "Databricks": [0,0,0,0,0,-1], "BigQuery": [0,0,0,1,0,1],  "AWS Redshift": [0,0,0,0,0,0], "Azure Synapse": [0,0,0,0,0,-1]},
    "Real-Time IoT Platform":{"Snowflake": [0,0,-1,0,0,0], "Databricks": [0,1,1,0,0,0],  "BigQuery": [0,0,-1,1,0,0], "AWS Redshift": [0,0,-1,0,0,0], "Azure Synapse": [0,0,-1,0,0,0]},
    "Global E-Commerce":     {"Snowflake": [1,0,0,1,1,0],  "Databricks": [0,1,1,0,0,0],  "BigQuery": [1,0,0,1,0,1],  "AWS Redshift": [0,0,0,-1,0,0], "Azure Synapse": [0,0,0,-1,0,0]},
    "ML Feature Platform":   {"Snowflake": [0,-1,0,1,1,0], "Databricks": [0,1,1,0,0,0],  "BigQuery": [0,1,0,1,0,1],  "AWS Redshift": [0,-1,0,-1,0,0],"Azure Synapse": [0,0,0,-1,0,0]},
    "Media Streaming":       {"Snowflake": [0,0,-1,1,1,0], "Databricks": [0,1,1,0,0,0],  "BigQuery": [1,0,-1,1,0,1], "AWS Redshift": [0,0,-1,-1,0,0],"Azure Synapse": [0,0,-1,-1,0,0]},
}

# Requirements → score deltas per platform
_REQ_ADJUSTMENTS = {
    "Multi-cloud portability":    {"BigQuery": [0,0,0,1,0,0], "Snowflake": [0,0,0,1,0,0], "Databricks": [0,0,0,-1,0,0], "AWS Redshift": [0,0,0,-2,0,0], "Azure Synapse": [0,0,0,-2,0,0]},
    "Real-time CDC":              {"Databricks": [0,0,2,0,0,0], "BigQuery": [0,0,1,0,0,0], "Snowflake": [0,0,-1,0,0,0],  "AWS Redshift": [0,0,-1,0,0,0], "Azure Synapse": [0,0,0,0,0,0]},
    "ML workloads":               {"Databricks": [0,2,0,0,0,0], "BigQuery": [0,1,0,0,0,0],  "Snowflake": [0,-1,0,0,0,0],   "AWS Redshift": [0,-1,0,0,0,0],  "Azure Synapse": [0,0,0,0,0,0]},
    "GDPR compliance":            {"Snowflake": [0,0,0,0,1,0],  "Databricks": [0,0,0,0,0,0], "BigQuery": [0,0,0,0,1,0],    "AWS Redshift": [0,0,0,0,0,0],   "Azure Synapse": [0,0,0,0,1,0]},
    "High write throughput":      {"Databricks": [0,0,1,0,0,0], "BigQuery": [0,0,0,0,0,1],   "Snowflake": [0,0,-1,0,0,-1],  "AWS Redshift": [0,0,-1,0,0,0],  "Azure Synapse": [0,0,-1,0,0,0]},
    "Complex SQL analytics":      {"Snowflake": [1,0,0,0,1,0],  "BigQuery": [1,0,0,0,0,1],   "Databricks": [-1,0,0,0,0,0],  "AWS Redshift": [1,0,0,0,0,0],   "Azure Synapse": [1,0,0,0,0,0]},
}

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE TOOL RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

_ARCH_RECS = {
    "Regulatory Finance": {
        "table_format": "Apache Iceberg",
        "table_format_why": "Column-level encryption, audit history via snapshots, multi-engine for compliance queries",
        "platform": "Snowflake + Databricks",
        "platform_why": "Snowflake for governed SQL reporting; Databricks for ML fraud models and data lineage",
        "ingestion": "Kafka + Debezium CDC",
        "ingestion_why": "Capture every transaction change at the DB replication log level for complete audit trail",
        "observability": "Monte Carlo + dbt tests + DataHub lineage",
        "observability_why": "Monte Carlo for real-time freshness/volume anomalies; DataHub for column-level PII lineage required by regulators",
        "bronze_tools": "Kafka CDC streams → Iceberg tables (S3/ADLS) via Spark Structured Streaming",
        "silver_tools": "dbt + Spark: drop null IDs, enforce amount precision, GDPR right-to-delete applied here",
        "gold_tools": "Snowflake: regulatory reports, risk aggregations, audit-ready materialized views",
        "talking_points": [
            "I'd apply GDPR right-to-delete in the Silver layer before data reaches Gold or any BI report",
            "DataHub column-level lineage proves to auditors exactly where every reported number came from",
            "Iceberg's snapshot history provides a tamper-evident audit log of every table version",
            "Snowflake column masking policies hide PII from analysts while still allowing aggregate queries",
        ],
        "red_flags": [
            "Using full table scans for CDC — at financial transaction volumes, this is untenable",
            "Dropping bad records silently — in finance, every record must be accounted for in a quarantine table",
            "No column-level lineage — auditors will ask 'prove this revenue number is correct'",
        ],
    },
    "Startup Analytics": {
        "table_format": "Delta Lake",
        "table_format_why": "Simplest to operate; minimal DevOps overhead; Databricks Community Edition available",
        "platform": "BigQuery + dbt Cloud",
        "platform_why": "Serverless — no cluster management; pay only for queries run; dbt Cloud for ELT",
        "ingestion": "Fivetran + dbt",
        "ingestion_why": "Managed connectors require zero maintenance; dbt for all transformations in warehouse",
        "observability": "dbt tests + Elementary (open-source observability on dbt)",
        "observability_why": "Lower cost alternative to Monte Carlo; dbt Elementary is free and integrates natively",
        "bronze_tools": "Fivetran syncs Postgres/Salesforce/Stripe → BigQuery raw dataset (ELT pattern)",
        "silver_tools": "dbt staging models: cast types, apply naming conventions, basic null checks",
        "gold_tools": "dbt mart models: revenue by cohort, activation funnel, ARR calculations",
        "talking_points": [
            "For a startup, serverless BigQuery eliminates the cluster management overhead that would slow down a small team",
            "Fivetran handles all connector maintenance — the team ships product, not Salesforce API integrations",
            "Elementary gives Monte Carlo-style freshness and volume alerting at zero additional cost",
            "ELT pattern: load raw data into BigQuery first, transform with dbt — never maintain a separate ETL server",
        ],
        "red_flags": [
            "Building custom ETL pipelines — Fivetran handles this; engineering time is too precious at a startup",
            "Using Databricks before the team has data engineers — too much operational complexity",
            "No dbt tests — even simple unique/not_null checks prevent broken dashboard outages",
        ],
    },
    "Real-Time IoT Platform": {
        "table_format": "Apache Hudi",
        "table_format_why": "Designed for high-frequency record-level updates; native CDC support; built by Uber for exactly this use case",
        "platform": "Databricks + Kafka",
        "platform_why": "Spark Structured Streaming + Delta Lake for real-time pipeline; Kafka for device event bus",
        "ingestion": "Kafka + Spark Structured Streaming",
        "ingestion_why": "Kafka handles millions of device events/second; Spark Structured Streaming provides exactly-once semantics",
        "observability": "Datadog Observability Pipelines + Monte Carlo",
        "observability_why": "Datadog monitors pipeline throughput, latency, and drop rate at the infrastructure layer in real time",
        "bronze_tools": "IoT devices → Kafka topics (partitioned by device_id) → Hudi Bronze table (5-second microbatches)",
        "silver_tools": "Spark Structured Streaming: device ID validation, out-of-order event handling with watermarks, dedup",
        "gold_tools": "Aggregated device telemetry: rolling averages, anomaly scores, alert threshold evaluations",
        "talking_points": [
            "Kafka topics partitioned by device_id ensure events from the same device are ordered; crucial for sensor time series",
            "Spark watermarks handle late-arriving IoT events — sensors may be offline for hours, then send batched events",
            "Hudi's record-level indexing makes device state updates (latitude/longitude/temperature) efficient at scale",
            "I'd use Datadog to monitor Kafka consumer lag — if lag grows, the Spark cluster needs more executors",
        ],
        "red_flags": [
            "Batch ingestion for IoT — devices produce continuous streams; a nightly batch is 12 hours stale",
            "No watermarking strategy — out-of-order IoT events will produce incorrect time-window aggregations",
            "Full table scans to update device state — at 1M devices, a row-level index (Hudi) is orders of magnitude faster",
        ],
    },
    "Global E-Commerce": {
        "table_format": "Apache Iceberg",
        "table_format_why": "Multi-cloud portability; Snowflake + Databricks both read Iceberg natively; zero lock-in",
        "platform": "Snowflake + Databricks",
        "platform_why": "Snowflake for BI and financial reporting; Databricks for recommendation models and fraud detection",
        "ingestion": "Kafka + Debezium CDC + Fivetran",
        "ingestion_why": "CDC for order/inventory database changes; Fivetran for SaaS sources (Salesforce, Zendesk)",
        "observability": "Monte Carlo + dbt tests",
        "observability_why": "Monte Carlo critical for detecting revenue pipeline anomalies before they reach finance reports",
        "bronze_tools": "Order events + CDC → Iceberg Bronze on S3 (multi-region replication for DR)",
        "silver_tools": "Dedup order events, enforce product SKU validation, apply currency conversion",
        "gold_tools": "Snowflake: daily/hourly GMV, category revenue, regional performance; Databricks: recommendation features",
        "talking_points": [
            "I'd separate ML compute (Databricks for recommendations/fraud) from BI analytics (Snowflake) to optimize costs and avoid resource contention",
            "Iceberg as the shared table format means Databricks writes features and Snowflake reads them — no data copying",
            "GDPR right-to-delete in the Silver layer ensures user deletion propagates before reaching any BI dashboard or ML model",
            "Multi-region Bronze tables provide disaster recovery — if us-east-1 fails, eu-west-1 Bronze is an exact copy",
        ],
        "red_flags": [
            "Single platform for both ML and BI — creates resource contention and forces suboptimal choices for each workload",
            "ETL instead of ELT — Snowflake and Databricks both have more compute than any ETL server you'd maintain",
            "No observability on the revenue pipeline — a data quality bug in orders would flow through to financial reporting silently",
        ],
    },
    "ML Feature Platform": {
        "table_format": "Apache Iceberg",
        "table_format_why": "Time-travel queries are essential for point-in-time feature correctness; multi-engine for training vs serving",
        "platform": "Databricks + Feature Store (Tecton or Databricks Feature Store)",
        "platform_why": "Databricks Unity Catalog for governance; Feature Store eliminates training-serving skew",
        "ingestion": "Kafka + Spark Structured Streaming + Batch Spark jobs",
        "ingestion_why": "Dual ingestion: streaming for real-time features (user session signals); batch for historical feature backfills",
        "observability": "Monte Carlo + custom feature drift monitoring (PSI/KL divergence)",
        "observability_why": "Feature distribution drift is the primary cause of silent model degradation in production",
        "bronze_tools": "Event streams + historical data dumps → Bronze Iceberg tables partitioned by date",
        "silver_tools": "Feature computation: rolling aggregations, entity joins, normalization — with point-in-time correctness",
        "gold_tools": "Feature Store: online (Redis, sub-10ms) + offline (Iceberg, for training) — identical logic in both",
        "talking_points": [
            "Training-serving skew is the #1 silent killer of ML systems — a shared Feature Store with identical logic for training and serving eliminates it",
            "Point-in-time correctness: when training a churn model, features must reflect what was known at the label timestamp, not what was computed later",
            "Iceberg time travel lets me query 'what did the user's 30-day purchase history look like on 2024-01-15' — critical for reproducing training datasets",
            "PSI (Population Stability Index) monitoring on features detects distribution drift before it degrades model performance",
        ],
        "red_flags": [
            "Separate feature pipelines for training and serving — this is how training-serving skew is created",
            "No feature versioning — changing a feature definition invalidates all models trained on the old version",
            "Using future data in training features — always validate point-in-time correctness before training",
        ],
    },
    "Media Streaming": {
        "table_format": "Delta Lake",
        "table_format_why": "Databricks-native; excellent for high-volume event streams (play, pause, seek events); MERGE for user profile updates",
        "platform": "Databricks + Snowflake",
        "platform_why": "Databricks for recommendation model training and real-time event processing; Snowflake for content performance analytics",
        "ingestion": "Kafka (high throughput) + Spark Structured Streaming",
        "ingestion_why": "Millions of play/pause/seek events per second; Kafka partitioned by content_id for ordered processing",
        "observability": "Datadog + Monte Carlo",
        "observability_why": "Datadog for pipeline infrastructure (Kafka lag, Spark task failures); Monte Carlo for content analytics freshness",
        "bronze_tools": "Kafka playback events → Delta Lake Bronze (Databricks Auto Loader for schema evolution)",
        "silver_tools": "Session stitching, dedup overlapping play events, calculate true watch time (subtract pauses)",
        "gold_tools": "Content engagement metrics: completion rate, rewatch rate, skip rate; user watch history for recommendations",
        "talking_points": [
            "Session stitching is the core Silver challenge: a user pausing and resuming creates multiple events that must be merged into one continuous session",
            "Watch time is the key metric — not play events, but actual seconds watched; this requires careful Silver logic for pause/seek/buffer events",
            "Databricks Auto Loader handles schema evolution in playback events without manual intervention — critical as the mobile app adds new event fields",
            "I'd separate content analytics (Snowflake) from recommendation feature computation (Databricks) — very different SLA requirements",
        ],
        "red_flags": [
            "Counting play events instead of computing true watch time — pause/buffer/seek events mean 1 play event ≠ watched content",
            "No session stitching — without it, watch time calculations are fundamentally wrong",
            "Batch-only pipeline for recommendation features — users expect personalized recommendations within minutes, not hours",
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DEMO FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_data_architecture_demo(use_case: str, requirements: list, show_lineage: bool) -> tuple:
    """
    Returns (fig_heatmap, fig_architecture, metrics_md)
    fig_heatmap   : Platform fit heatmap across 6 dimensions
    fig_architecture: Data flow / architecture diagram with recommended tools
    metrics_md    : Interview prep markdown
    """
    try:
        if not use_case:
            use_case = "Startup Analytics"
        if requirements is None:
            requirements = []

        # ── Compute platform scores ───────────────────────────────────────────
        platforms = ["Snowflake", "Databricks", "BigQuery", "AWS Redshift", "Azure Synapse"]
        scores = {}
        for p in platforms:
            base = list(_BASE_SCORES[p])
            # Apply use-case adjustment
            uc_adj = _USE_CASE_ADJUSTMENTS.get(use_case, {}).get(p, [0]*6)
            # Apply requirements adjustments
            req_adj = [0]*6
            for req in requirements:
                delta = _REQ_ADJUSTMENTS.get(req, {}).get(p, [0]*6)
                req_adj = [req_adj[i] + delta[i] for i in range(6)]
            # Clamp 1-5
            final = [max(1, min(5, base[i] + uc_adj[i] + req_adj[i])) for i in range(6)]
            scores[p] = final

        score_matrix = np.array([scores[p] for p in platforms], dtype=float)

        # ── Figure 1: Platform Fit Heatmap ────────────────────────────────────
        fig = go.Figure(data=go.Heatmap(
            z=score_matrix,
            x=_DIMENSIONS,
            y=platforms,
            colorscale=[
                [0.0, "#EF4444"],
                [0.25, "#F59E0B"],
                [0.5, "#FBBF24"],
                [0.75, "#34D399"],
                [1.0, "#10B981"],
            ],
            zmin=1,
            zmax=5,
            text=score_matrix.astype(int),
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>%{x}: %{z}/5<extra></extra>",
        ))

        # Highlight best platform per dimension
        best_per_dim = score_matrix.argmax(axis=0)
        for col_idx, row_idx in enumerate(best_per_dim):
            fig.add_shape(
                type="rect",
                x0=col_idx - 0.5, x1=col_idx + 0.5,
                y0=row_idx - 0.5, y1=row_idx + 0.5,
                line=dict(color="white", width=3),
                fillcolor="rgba(0,0,0,0)",
            )

        req_label = ", ".join(requirements) if requirements else "no specific requirements"
        fig.update_layout(
            title=dict(
                text=f"Platform Fit Scores — {use_case}<br><sub>White border = best per dimension | Requirements: {req_label}</sub>",
                font=dict(size=14),
            ),
            height=360,
            margin=dict(l=10, r=10, t=80, b=10),
            xaxis=dict(side="top"),
            font=dict(size=12),
        )

        # ── Figure 2: Architecture Diagram ────────────────────────────────────
        rec = _ARCH_RECS.get(use_case, _ARCH_RECS["Startup Analytics"])

        fig2 = go.Figure()
        fig2.update_xaxes(visible=False, range=[0, 1])
        fig2.update_yaxes(visible=False, range=[0, 1])

        # Node definitions: (x_center, y_center, label_line1, label_line2, color, layer_label)
        nodes = [
            # Row 0: Data Sources (y=0.88)
            (0.12, 0.88, "Operational DB", "(Postgres/MySQL)", "#3B82F6",   "source"),
            (0.38, 0.88, "Event Stream",   "(Kafka Topics)",  "#6366F1",   "source"),
            (0.63, 0.88, "SaaS APIs",      "(Salesforce/Stripe)", "#8B5CF6","source"),
            (0.88, 0.88, "Files/Blobs",    "(S3/GCS)",        "#06B6D4",   "source"),

            # Row 1: Ingestion (y=0.68)
            (0.20, 0.68, "CDC",            "(Debezium)",      "#F59E0B",   "ingest"),
            (0.50, 0.68, "Streaming",      "(Spark SS / Flink)", "#F97316","ingest"),
            (0.80, 0.68, "Batch ELT",      "(Fivetran / dbt)", "#EF4444",  "ingest"),

            # Row 2: Bronze (y=0.50)
            (0.50, 0.50, "Bronze Layer",   rec["bronze_tools"][:38], "#64748B", "bronze"),

            # Row 3: Silver (y=0.33)
            (0.50, 0.33, "Silver Layer",   rec["silver_tools"][:38], "#475569", "silver"),

            # Row 4: Gold (y=0.16)
            (0.25, 0.16, "Gold — BI",      rec["gold_tools"][:30],   "#10B981", "gold"),
            (0.75, 0.16, "Gold — ML",      "Feature Store / MLflow", "#10B981", "gold"),
        ]

        # Color map for borders
        layer_border = {
            "source":  "#93C5FD",
            "ingest":  "#FCD34D",
            "bronze":  "#94A3B8",
            "silver":  "#CBD5E1",
            "gold":    "#6EE7B7",
        }

        for (x, y, l1, l2, fill, layer) in nodes:
            w, h = 0.16, 0.07
            fig2.add_shape(
                type="rect",
                x0=x - w/2, y0=y - h/2, x1=x + w/2, y1=y + h/2,
                fillcolor=fill,
                line=dict(color=layer_border[layer], width=2),
                opacity=0.88,
            )
            fig2.add_annotation(
                x=x, y=y + 0.01,
                text=f"<b>{l1}</b>",
                showarrow=False,
                font=dict(size=9, color="white"),
                align="center",
            )
            fig2.add_annotation(
                x=x, y=y - 0.022,
                text=f"<i>{l2}</i>",
                showarrow=False,
                font=dict(size=7, color="#E2E8F0"),
                align="center",
            )

        # Arrows: (x0, y0, x1, y1)
        arrows = [
            # Sources → Ingestion
            (0.12, 0.84, 0.20, 0.72),
            (0.38, 0.84, 0.50, 0.72),
            (0.63, 0.84, 0.80, 0.72),
            (0.88, 0.84, 0.80, 0.72),
            # Ingestion → Bronze
            (0.20, 0.64, 0.50, 0.54),
            (0.50, 0.64, 0.50, 0.54),
            (0.80, 0.64, 0.50, 0.54),
            # Bronze → Silver
            (0.50, 0.46, 0.50, 0.37),
            # Silver → Gold
            (0.50, 0.29, 0.25, 0.20),
            (0.50, 0.29, 0.75, 0.20),
        ]

        # Figure canvas size for pixel conversion (px, py are axis ranges 0-1)
        _FIG_W, _FIG_H = 800, 520

        streaming_path = use_case in ["Real-Time IoT Platform", "Media Streaming", "ML Feature Platform"]
        for i, (x0, y0, x1, y1) in enumerate(arrows):
            # Highlight streaming path (arrows through the streaming node) in bright color
            is_streaming = (i in [1, 4]) and streaming_path
            color = "#F97316" if is_streaming else "#94A3B8"
            width = 2.5 if is_streaming else 1.5
            # ax/ay must be in pixels relative to the annotation tip (x1,y1).
            # Convert paper-unit source (x0,y0) → pixel offset from tip.
            ax_px = (x0 - x1) * _FIG_W
            ay_px = (y0 - y1) * _FIG_H
            fig2.add_annotation(
                x=x1, y=y1,
                ax=ax_px, ay=ay_px,
                xref="paper", yref="paper",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=width,
                arrowcolor=color,
            )

        # Layer labels on the right
        for y_pos, label, color in [
            (0.88, "DATA SOURCES",  "#93C5FD"),
            (0.68, "INGESTION",     "#FCD34D"),
            (0.50, "BRONZE",        "#94A3B8"),
            (0.33, "SILVER",        "#CBD5E1"),
            (0.16, "GOLD",          "#6EE7B7"),
        ]:
            fig2.add_annotation(
                x=0.99, y=y_pos,
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=8, color=color),
                align="right",
                xanchor="right",
            )

        # Observability badge
        fig2.add_annotation(
            x=0.01, y=0.01,
            text=f"<b>Observability:</b> {rec['observability']}",
            showarrow=False,
            font=dict(size=8, color="#F59E0B"),
            align="left",
            xanchor="left",
        )

        streaming_note = "  |  Orange path = streaming fast lane" if streaming_path else ""
        fig2.update_layout(
            title=dict(
                text=f"Recommended Data Architecture — {use_case}<br>"
                     f"<sub>Table Format: {rec['table_format']}{streaming_note}</sub>",
                font=dict(size=13),
            ),
            height=520,
            margin=dict(l=5, r=5, t=80, b=5),
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
        )

        # ── Show lineage overlay if requested ─────────────────────────────────
        if show_lineage:
            # Add a lineage trace path annotation along the left side
            for y_pos, step in [
                (0.88, "raw_events (Bronze)"),
                (0.68, "validated_events (Silver)"),
                (0.50, "user_metrics (Gold)"),
                (0.33, "revenue_dashboard (BI)"),
            ]:
                fig2.add_annotation(
                    x=0.02, y=y_pos,
                    text=f"↳ {step}",
                    showarrow=False,
                    font=dict(size=7, color="#A78BFA"),
                    align="left",
                    xanchor="left",
                )
            fig2.add_annotation(
                x=0.02, y=0.96,
                text="<b>Lineage trace (DataHub):</b>",
                showarrow=False,
                font=dict(size=8, color="#A78BFA"),
                align="left",
                xanchor="left",
            )

        # ── Metrics Markdown ──────────────────────────────────────────────────
        # Identify recommended platform (highest average score)
        avg_scores = {p: np.mean(scores[p]) for p in platforms}
        top_platform = max(avg_scores, key=avg_scores.get)
        top2 = sorted(platforms, key=lambda p: -avg_scores[p])[:2]

        # Build requirements impact notes
        req_notes = []
        if "Multi-cloud portability" in requirements:
            req_notes.append("- **Multi-cloud portability** detected → Apache Iceberg recommended (lowest vendor lock-in; reads natively in Snowflake, BigQuery, Trino, Athena)")
        if "Real-time CDC" in requirements:
            req_notes.append("- **Real-time CDC** detected → Apache Hudi or Kafka + Debezium; Hudi's record-level index makes high-frequency upserts efficient")
        if "ML workloads" in requirements:
            req_notes.append("- **ML workloads** detected → Databricks scored up; consider Databricks Feature Store to eliminate training-serving skew")
        if "GDPR compliance" in requirements:
            req_notes.append("- **GDPR compliance** detected → Apply right-to-delete in Silver layer; use DataHub for column-level PII lineage; Snowflake column masking policies")
        if "High write throughput" in requirements:
            req_notes.append("- **High write throughput** detected → Apache Hudi or Delta Lake with Z-ORDER clustering for efficient high-volume writes")
        if "Complex SQL analytics" in requirements:
            req_notes.append("- **Complex SQL analytics** detected → Snowflake or BigQuery scored up; both excel at multi-TB analytical queries")

        req_section = "\n".join(req_notes) if req_notes else "No specific requirements selected — showing base platform scores."

        talking_points_md = "\n".join(f'- "{tp}"' for tp in rec["talking_points"])
        red_flags_md = "\n".join(f"- {rf}" for rf in rec["red_flags"])

        # Score table rows
        score_rows = ""
        for p in sorted(platforms, key=lambda x: -avg_scores[x]):
            row_scores = " | ".join(str(s) for s in scores[p])
            avg = f"{avg_scores[p]:.1f}"
            mark = " ← **top pick**" if p == top_platform else ""
            score_rows += f"| {p} | {row_scores} | **{avg}**{mark} |\n"

        metrics_md = f"""
### Recommended Architecture for {use_case}

| Component | Recommendation | Why |
|---|---|---|
| **Table Format** | {rec["table_format"]} | {rec["table_format_why"]} |
| **Platform** | {rec["platform"]} | {rec["platform_why"]} |
| **Ingestion** | {rec["ingestion"]} | {rec["ingestion_why"]} |
| **Observability** | {rec["observability"]} | {rec["observability_why"]} |

---

### Medallion Layer Tools

| Layer | Tools & Responsibilities |
|---|---|
| **Bronze** | {rec["bronze_tools"]} |
| **Silver** | {rec["silver_tools"]} |
| **Gold** | {rec["gold_tools"]} |

---

### Requirements Impact Analysis

{req_section}

---

### Platform Score Matrix (1–5; higher = better fit)

| Platform | SQL | ML | Real-time | Multi-cloud | Governance | Cost | Avg |
|---|---|---|---|---|---|---|---|
{score_rows}
*Scores adjusted for use case and selected requirements. White border on heatmap = best per dimension.*

---

### Interview Talking Points

{talking_points_md}

---

### Red Flags to Avoid

{red_flags_md}

---

### Key Architectural Principles (always mention these)

1. **Medallion Architecture**: Bronze (raw, append-only) → Silver (validated, deduped) → Gold (business-ready). Never skip layers.
2. **ELT over ETL**: Load raw into the warehouse/lakehouse first; transform with dbt using warehouse compute.
3. **CDC over full scans**: Always use Debezium/Kafka for database changes at scale — full table scans are prohibitively expensive.
4. **Two-layer observability**: dbt tests for known rules (known unknowns) + Monte Carlo for anomaly detection (unknown unknowns).
5. **Data lineage from day one**: OpenLineage → DataHub. Debugging a wrong dashboard number without lineage takes days; with lineage, minutes.
"""

        return fig, fig2, metrics_md

    except Exception as e:
        import traceback
        return go.Figure(), go.Figure(), f"**Error:** {traceback.format_exc()}"


# ─────────────────────────────────────────────────────────────────────────────
# BUILD TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_tab():
    gr.Markdown("# 🏗️ Module 02 — Data Architecture & Observability\n*Level: Advanced*")

    with gr.Accordion("📖 Theory", open=False):
        gr.Markdown(THEORY)

    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(CODE_EXAMPLE, language="python")

    gr.Markdown(
        "---\n"
        "## 🎮 Interactive Demo\n\n"
        "Select a use case and check off your requirements. The platform fit heatmap "
        "scores each cloud warehouse/lakehouse across 6 dimensions, adjusted for your "
        "specific context. The architecture diagram shows the recommended tool stack "
        "with Bronze → Silver → Gold layers labelled with concrete tool choices. "
        "The analysis panel provides interview-ready talking points and red flags."
    )

    with gr.Row():
        with gr.Column(scale=1):
            use_case_dd = gr.Dropdown(
                label="Use Case",
                choices=[
                    "Regulatory Finance",
                    "Startup Analytics",
                    "Real-Time IoT Platform",
                    "Global E-Commerce",
                    "ML Feature Platform",
                    "Media Streaming",
                ],
                value="Global E-Commerce",
            )
            requirements_cb = gr.CheckboxGroup(
                label="Requirements (select all that apply)",
                choices=[
                    "Multi-cloud portability",
                    "Real-time CDC",
                    "ML workloads",
                    "GDPR compliance",
                    "High write throughput",
                    "Complex SQL analytics",
                ],
                value=[],
            )
            lineage_cb = gr.Checkbox(
                label="Show data lineage diagram overlay",
                value=True,
            )
            run_btn = gr.Button("▶ Analyze Architecture", variant="primary")

        with gr.Column(scale=2):
            plot_out  = gr.Plot(label="Platform Fit Heatmap")
            plot2_out = gr.Plot(label="Recommended Data Architecture")
            metrics_out = gr.Markdown()

    run_btn.click(
        fn=run_data_architecture_demo,
        inputs=[use_case_dd, requirements_cb, lineage_cb],
        outputs=[plot_out, plot2_out, metrics_out],
    )
