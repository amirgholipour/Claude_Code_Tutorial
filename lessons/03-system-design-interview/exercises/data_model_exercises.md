# Data Model Design Exercises

Practice exercises for Modules 01 (Data Model Design) and 02 (Data Architecture).
Target companies: Google, Meta, Amazon, Microsoft, Nvidia.

---

## Exercise 1 — Design a Social Feed (Meta/Twitter Level)

**Prompt:**
Design the data model for a social media feed that supports:
- 500M daily active users
- Users can follow other users (up to 10,000 followers)
- Posts include text (max 280 chars), images, and videos
- Feed shows posts from followed users, sorted by recency and relevance
- Like, comment, and share counts must be queryable in real-time

**Your task:**
1. Identify the core entities and their relationships
2. Choose a primary database and justify with CAP theorem trade-offs
3. Design the schema for at least 3 entities
4. Explain how you'd handle the "hot user" problem (celebrity with 50M followers)
5. Describe your caching strategy

**Key concepts to demonstrate:**
- Fan-out on write vs. fan-out on read
- Denormalization for read performance
- Partition key selection for distributed systems

---

## Exercise 2 — E-Commerce Inventory System (Amazon Level)

**Prompt:**
Design the data model for Amazon's inventory system:
- 350M+ product SKUs across 200+ categories
- Real-time inventory counts (must not oversell)
- Supports flash sales (100K concurrent buyers for 1 item)
- Multi-warehouse fulfillment (pick closest warehouse)
- Historical stock level tracking for analytics

**Your task:**
1. Design the entity-relationship model
2. Choose between strong vs. eventual consistency for inventory counts — justify
3. Design a schema that prevents overselling under concurrent writes
4. Explain how you'd model the multi-warehouse dimension
5. Describe your strategy for the analytics (OLAP) use case

**Key concepts to demonstrate:**
- Optimistic vs. pessimistic locking
- ACID transactions for inventory reservation
- CQRS (Command Query Responsibility Segregation)

---

## Exercise 3 — Ride-Sharing Location System (Uber/Lyft Level)

**Prompt:**
Design the data model for a ride-sharing driver location system:
- 5M active drivers worldwide
- Location updates every 4 seconds per driver
- Queries: "find all drivers within 2km of passenger"
- Historical trip data for routing ML models
- Real-time ETA computation

**Your task:**
1. Choose a storage engine suitable for geospatial queries — justify
2. Design the schema for driver location, trip, and user entities
3. Explain how you'd partition/shard driver location data globally
4. Describe the time-series challenges of 5M × 15 updates/min write throughput
5. How would you handle a driver going offline mid-trip?

**Key concepts to demonstrate:**
- Geospatial indexing (R-tree, Geohash, S2 cells)
- Write-heavy workload optimization
- TTL-based data expiry for stale location records

---

## Exercise 4 — ML Feature Store (Nvidia/Google AI Level)

**Prompt:**
Design a feature store for a large-scale ML platform:
- Serves 200 ML models in production
- 10,000 features across 50 feature groups
- Online serving: p99 latency < 10ms
- Offline training: consistent features with point-in-time correctness
- Feature sharing across teams to avoid duplication

**Your task:**
1. Design the metadata schema (feature registry)
2. Design the online store schema for low-latency retrieval
3. Explain point-in-time correctness and design a schema that guarantees it
4. How would you detect and prevent training-serving skew at the data layer?
5. Design a feature versioning strategy

**Key concepts to demonstrate:**
- Online vs. offline store separation
- Point-in-time joins
- Feature lineage tracking

---

## Exercise 5 — Multi-Tenant SaaS Database (Microsoft Azure Level)

**Prompt:**
Design the data model for a multi-tenant SaaS analytics platform:
- 10,000 enterprise tenants
- Each tenant has 1–500 users
- Tenant data must be strictly isolated (compliance requirement)
- Some tenants have 100GB+ of data; most have < 1GB
- Aggregate analytics across tenants for the platform team

**Your task:**
1. Compare the three multi-tenancy models: shared schema, separate schema, separate database
2. Recommend an approach for this scenario and justify
3. Design the tenant isolation mechanism at the query layer
4. How would you handle "noisy neighbor" resource contention?
5. Design the cross-tenant aggregation pipeline without violating tenant isolation

**Key concepts to demonstrate:**
- Row-level security (RLS)
- Tenant sharding strategies
- Separation of OLTP (per-tenant) and OLAP (cross-tenant aggregate)

---

## Scoring Rubric

| Criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Requirements clarification | None | Partial | Good | Proactively asked + scope bounded |
| Entity modeling | Missing | 1-2 entities | Core entities | Full ERD with cardinality |
| Storage choice justification | None | Named a DB | Named + feature list | CAP/PACELC trade-off analysis |
| Scale handling | None | Mentioned | Designed for | Specific numbers + calculations |
| Failure mode analysis | None | Mentioned | 1 scenario | Multiple scenarios + mitigations |

**Target score for FAANG L5+:** 12/15 or higher

---

## Further Reading

- [Designing Data-Intensive Applications](https://dataintensive.net/) — Chapters 2, 3, 5, 6
- [Amazon DynamoDB Design Patterns](https://aws.amazon.com/blogs/database/amazon-dynamodb-single-table-design/)
- [Google Spanner whitepaper](https://research.google/pubs/spanner-googles-globally-distributed-database/)
- [Meta TAO: Distributed Data Store](https://engineering.fb.com/2013/06/25/core-data/tao-the-power-of-the-graph/)
