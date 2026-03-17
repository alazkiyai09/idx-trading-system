# IMSS Update — OpenViking Integration Plan

## Status: REVIEW — DO NOT APPLY UNTIL PHASE 1 IS COMPLETE
**Applies to:** IDX_Market_Swarm_Simulator_Spec.md, IDX_Market_Swarm_Simulator_Implementation.md
**When to apply:** Beginning of Phase 2 (after Phase 1 is validated and working)
**Repository:** https://github.com/volcengine/OpenViking

---

## 1. What OpenViking Replaces

OpenViking is a context database for AI agents that uses a filesystem paradigm (viking:// URIs) with hierarchical storage and tiered context loading (L0 abstract → L1 overview → L2 full detail).

### Components it replaces in Phase 2:

| Current Component | Replaced By | Benefit |
|---|---|---|
| ChromaDB event_embeddings collection | `viking://resources/events/` hierarchy | Directory-recursive retrieval > flat vector search |
| ChromaDB agent_episodic_{id} collections | `viking://agent/{name}/memories/` | Automatic session memory extraction |
| Manual episodic memory store/retrieve logic | OpenViking's built-in session management | Less custom code to maintain |
| Flat semantic search for causal parallels | OpenViking's directory-recursive retrieval | Structured + semantic retrieval combined |

### Components it does NOT replace:

| Component | Why It Stays |
|---|---|
| SQLite / PostgreSQL | Time-series price data, simulation run logs, causal link table — relational data |
| LLM Router + Batcher | Compute layer, not context layer |
| Simulation Engine | Core orchestration logic |
| Tier 3 rule-based agents | No LLM or context needed |
| Order book / price impact model | Pure computation |

---

## 2. Proposed viking:// Structure for IMSS

```
viking://
├── resources/
│   ├── events/
│   │   ├── regulatory/
│   │   │   ├── ojk_2024_01_npl_provisioning
│   │   │   ├── ojk_2024_03_digital_lending
│   │   │   └── ...
│   │   ├── earnings/
│   │   │   ├── bbri_q1_2024
│   │   │   ├── bbri_q2_2024
│   │   │   └── ...
│   │   ├── macro/
│   │   │   ├── bi_rate_2024_01
│   │   │   ├── usd_idr_spike_2024_03
│   │   │   └── ...
│   │   ├── news/
│   │   │   └── ...
│   │   └── political/
│   │       └── ...
│   │
│   ├── market_data/
│   │   ├── bbri/
│   │   │   ├── price_history      # summary of price patterns
│   │   │   └── sector_context     # banking sector overview
│   │   ├── bmri/
│   │   └── ...
│   │
│   └── causal_knowledge/
│       ├── regulatory_impacts/     # learned: OJK announcements → price patterns
│       ├── earnings_reactions/     # learned: earnings surprise → price patterns
│       └── macro_effects/          # learned: BI rate → banking sector patterns
│
├── agent/
│   ├── pak_budi/
│   │   ├── instructions/          # persona prompt + behavioral rules
│   │   ├── memories/              # episodic memory from simulations
│   │   │   ├── profitable_decisions/
│   │   │   └── loss_decisions/
│   │   └── skills/                # investment strategy rules
│   │       ├── regulatory_response
│   │       └── earnings_analysis
│   │
│   ├── sarah/
│   │   ├── instructions/
│   │   ├── memories/
│   │   └── skills/
│   │
│   ├── andi/
│   ├── dr_lim/
│   └── marketbot/
│
└── user/
    └── simulation_preferences/    # user's preferred configs, stock focus, etc.
```

---

## 3. How Tiered Loading Maps to Agent Tiers

OpenViking's L0/L1/L2 layers naturally align with our agent tier system:

```
Tier 1 (Named Agents — full context):
  Event retrieval: L0 to identify relevant events → L1 for decision context → L2 only if event is critical
  Episodic memory: L1 summaries of past experiences
  Causal knowledge: L1 summaries of learned patterns
  
  Token budget per step: ~2000-3000 tokens of context (mostly L1)

Tier 2 (Typed Agents — lightweight context):
  Event retrieval: L0 abstracts only (one-sentence summaries)
  Episodic memory: Not stored in OpenViking (keep sliding window in-memory)
  Causal knowledge: L0 only
  
  Token budget per step: ~500-800 tokens of context (all L0)

Tier 3 (Statistical Agents):
  No OpenViking access (rule-based, no context needed)
```

This is a significant improvement over the current design where both Tier 1 and Tier 2 hit the same ChromaDB with the same retrieval depth. OpenViking's tiered loading gives us natural cost control.

---

## 4. Key Integration Points

### 4.1 Event Storage (replaces ChromaDB event_embeddings)

```
Current (Phase 1):
  chromadb_collection.add(
      documents=[event_summary],
      embeddings=[embedding_vector],
      metadatas=[{category, timestamp, sentiment, ...}],
      ids=[event_id]
  )

With OpenViking (Phase 2):
  # Add event as a resource
  client.add_resource(
      path=event_file_path,  # or URL, or text content
      # OpenViking auto-generates L0 abstract and L1 overview
  )
  
  # Event gets placed in the hierarchy:
  # viking://resources/events/regulatory/ojk_2024_npl_provisioning
  #   ├── .abstract    ← L0: "OJK tightened NPL provisioning for digital lending in Q3 2024"
  #   ├── .overview     ← L1: Key details, affected entities, sentiment
  #   └── full_content  ← L2: Complete announcement text
```

### 4.2 Agent Episodic Memory (replaces ChromaDB agent_episodic_{id})

```
Current (Phase 1):
  # Store experience
  episodic_collection.add(
      documents=[experience_summary],
      embeddings=[embedding],
      metadatas=[{step, action, outcome, ...}]
  )
  
  # Retrieve
  results = episodic_collection.query(query_texts=[current_context], n_results=5)

With OpenViking (Phase 2):
  # After simulation run, trigger session memory extraction
  # OpenViking automatically analyzes the run and extracts key memories
  client.extract_memory(session_id=simulation_run_id)
  
  # Memories get organized:
  # viking://agent/pak_budi/memories/
  #   ├── profitable_decisions/
  #   │   ├── held_through_ojk_announcement_2024_08  ← auto-extracted
  #   │   └── bought_bbri_dip_after_earnings_miss
  #   └── loss_decisions/
  #       └── sold_too_early_before_recovery
  
  # Retrieve relevant memories
  results = client.find(
      "OJK regulatory announcement impact on banking stocks",
      target_uri="viking://agent/pak_budi/memories/"
  )
```

### 4.3 Causal Knowledge Retrieval (replaces custom causal memory)

```
Current (Phase 1):
  # Semantic search + DB lookup
  similar_events = chromadb.query(current_event_summary)
  causal_links = db.query(CausalLink).filter(event_id=...).all()

With OpenViking (Phase 2):
  # Directory-recursive retrieval within causal knowledge
  results = client.find(
      "OJK digital lending regulation impact on BBRI",
      target_uri="viking://resources/causal_knowledge/regulatory_impacts/"
  )
  
  # OpenViking first navigates to regulatory_impacts/ directory,
  # then recursively searches subdirectories for best matches.
  # Returns results with L0/L1/L2 available — load detail as needed.
```

---

## 5. OpenViking Configuration for IMSS

### 5.1 Installation

```bash
pip install openviking
```

### 5.2 Configuration File (ov.conf)

```json
{
  "embedding": {
    "dense": {
      "api_base": "https://api.z.ai/api/paas/v4/",
      "api_key": "<your-zhipu-api-key>",
      "provider": "openai",
      "dimension": 1024,
      "model": "embedding-3"
    }
  },
  "vlm": {
    "api_base": "https://api.z.ai/api/paas/v4/",
    "api_key": "<your-zhipu-api-key>",
    "provider": "openai",
    "model": "glm-5"
  }
}
```

Notes:
- Provider is "openai" because Zhipu uses OpenAI-compatible format
- Same API key for both embedding and VLM
- VLM (GLM-5) is used by OpenViking for content understanding and L0/L1 generation
- Verify that OpenViking accepts the z.ai endpoint — may need to test with bigmodel.cn as fallback

### 5.3 Environment Variable

```bash
export OPENVIKING_CONFIG_FILE=./config/ov.conf
```

---

## 6. Migration Plan (Phase 1 → Phase 2)

### Step 1: Install and Test OpenViking (Day 1)

```python
"""test_openviking.py — Verify OpenViking works with your Zhipu API"""
import openviking as ov

client = ov.SyncOpenViking(path="./data/openviking_test")
client.initialize()

# Test: add a sample event as a resource
# Create a temp file with event content
with open("/tmp/test_event.md", "w") as f:
    f.write("""# OJK Tightens NPL Provisioning
    
Date: 2024-03-15
Category: Regulatory
Affected: BBRI, BMRI, BBCA

OJK issued new regulation requiring banks to increase provisioning 
reserves for digital lending products, effective Q3 2024. 
Expected to impact net interest margins for banks with significant 
digital lending exposure.

Sentiment: Moderately negative for banking sector
Magnitude: Medium — affects margins but not core business
""")

result = client.add_resource(path="/tmp/test_event.md")
print(f"Added resource: {result['root_uri']}")

# Wait for processing
client.wait_processed()

# Test L0/L1 generation
abstract = client.abstract(result['root_uri'])
overview = client.overview(result['root_uri'])
print(f"L0 Abstract: {abstract}")
print(f"L1 Overview: {overview}")

# Test search
results = client.find("regulatory impact on banking", target_uri=result['root_uri'])
print(f"Search results: {results}")

client.close()
print("OpenViking test complete!")
```

### Step 2: Build the Viking Filesystem Structure (Day 2)

Create a migration script that:
1. Reads all events from SQLite Event table
2. Creates markdown files organized by category
3. Adds each to OpenViking under `viking://resources/events/{category}/`
4. Verifies L0/L1 generation works for financial events

### Step 3: Update Agent Memory Interface (Day 3-4)

Create an abstraction layer so the agent code doesn't need to know whether it's talking to ChromaDB or OpenViking:

```python
# imss/memory/context_store.py

class ContextStore(Protocol):
    """Abstract interface for context storage"""
    
    def store_event(self, event: Event) -> str: ...
    def find_similar_events(self, query: str, top_k: int) -> list[EventMatch]: ...
    def store_agent_experience(self, agent_id: str, experience: Experience) -> None: ...
    def retrieve_agent_memories(self, agent_id: str, query: str, top_k: int) -> list[Memory]: ...

class ChromaContextStore(ContextStore):
    """Phase 1 implementation — ChromaDB backend"""
    ...

class VikingContextStore(ContextStore):
    """Phase 2 implementation — OpenViking backend"""
    ...
```

This way you can swap backends without changing agent code.

### Step 4: Run Comparative Backtest (Day 5)

Run the same backtest scenario with both ChromaDB and OpenViking backends. Compare:
- Retrieval relevance (are agents getting better context?)
- Token usage (is L0/L1 tiering actually reducing tokens?)
- Latency (is OpenViking faster or slower than ChromaDB?)
- Cost (embedding API calls for OpenViking vs local for ChromaDB)

Only fully switch to OpenViking if it shows clear benefits.

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| OpenViking is new (v0.1.11) — may have bugs | Simulation reliability | Keep ChromaDB as fallback; abstraction layer allows quick swap |
| L0/L1 generation uses GLM-5 calls — adds cost | Higher per-event processing cost | Events are processed once, not per-simulation; amortized cost is low |
| OpenViking's VLM dependency may not work with z.ai endpoint | Can't initialize | Test with bigmodel.cn endpoint; or configure VLM separately |
| Directory structure design may not match retrieval patterns well | Poor retrieval relevance | Start with simple flat structure, add hierarchy incrementally |
| OpenViking session memory extraction may not suit financial simulation context | Memories not useful | Override with custom memory extraction logic if needed |

---

## 8. Decision Criteria

Apply this update ONLY if ALL of these are true:

- [ ] Phase 1 is complete and passing all smoke tests
- [ ] test_openviking.py runs successfully with your Zhipu API key
- [ ] L0/L1 generation produces useful abstracts for financial events
- [ ] You're ready to invest 3-5 days in Phase 2 memory upgrade
- [ ] ChromaDB has shown limitations that OpenViking would address

If any of these fail, continue with ChromaDB through Phase 2. OpenViking is an optimization, not a requirement.
