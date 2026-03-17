# IMSS Update — LLM Configuration Corrections

## Status: REVIEW BEFORE APPLYING
**Applies to:** IDX_Market_Swarm_Simulator_Implementation.md
**When to apply:** Before starting Phase 1 implementation

---

## 1. Base URL Update

The implementation guide currently references the older Zhipu China-mainland endpoint. Update to the international Z.AI endpoint.

### Change in Section 1.2 (.env)

```
# OLD
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# NEW
GLM_BASE_URL=https://api.z.ai/api/paas/v4/
```

### Change in Section 2.2 (GLM-5 API Specifics)

```
# OLD
Endpoint: https://open.bigmodel.cn/api/paas/v4/chat/completions
Client initialization:
  openai.AsyncOpenAI(
      api_key=GLM_API_KEY,
      base_url="https://open.bigmodel.cn/api/paas/v4"
  )

# NEW
Endpoint: https://api.z.ai/api/paas/v4/chat/completions
Client initialization:
  openai.AsyncOpenAI(
      api_key=GLM_API_KEY,
      base_url="https://api.z.ai/api/paas/v4/"
  )
```

### Change in Appendix B

```
# OLD
Base URL: https://open.bigmodel.cn/api/paas/v4

# NEW
Base URL: https://api.z.ai/api/paas/v4/
```

Note: Both URLs work (z.ai is the international domain, bigmodel.cn is the China-mainland domain). Use whichever gives you better latency from Indonesia. Test both:

```python
import time, httpx

for url in ["https://api.z.ai/api/paas/v4/", "https://open.bigmodel.cn/api/paas/v4/"]:
    start = time.time()
    r = httpx.get(url)
    print(f"{url}: {(time.time()-start)*1000:.0f}ms")
```

---

## 2. Embedding Model Update

The implementation guide uses local sentence-transformers (`all-MiniLM-L6-v2`). Zhipu offers `embedding-3` on the same API key, which is likely better for financial text and avoids the local model download.

### Option A: Use Zhipu embedding-3 (Recommended)

```
# .env changes
EMBEDDING_PROVIDER=zhipu
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIMENSION=1024
EMBEDDING_API_KEY=${GLM_API_KEY}  # same key
EMBEDDING_BASE_URL=https://api.z.ai/api/paas/v4/
```

Implementation in `imss/data/embedder.py`:

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GLM_API_KEY"),
    base_url="https://api.z.ai/api/paas/v4/"
)

def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model="embedding-3",
        input=text,
        dimensions=1024  # embedding-3 supports configurable dimensions
    )
    return response.data[0].embedding

def embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="embedding-3",
        input=texts,
        dimensions=1024
    )
    return [d.embedding for d in response.data]
```

Advantages:
- No local model download (saves ~90MB)
- Better multilingual quality (important for Indonesian + English financial text)
- Same API key as GLM-5
- Configurable dimensions (256, 512, 1024, 2048)

Disadvantages:
- Adds API cost per embedding call
- Requires network for every embedding
- Slight latency vs local model

### Option B: Keep local sentence-transformers (Fallback)

If Zhipu embedding costs are a concern or you want offline capability, keep the original local approach. No changes needed.

### Recommendation

Start with Option A (Zhipu embedding-3). If costs become an issue, switch to local sentence-transformers later — the ChromaDB interface stays the same either way.

---

## 3. Dependencies Update

### pyproject.toml changes if using Zhipu embeddings

```toml
# REMOVE (if not using local embeddings):
# "sentence-transformers>=2.5",

# KEEP (still needed for ChromaDB custom embedding function):
# If using Zhipu embeddings, you'll pass embeddings directly to ChromaDB
# rather than using ChromaDB's built-in embedding function.
```

### ChromaDB integration change

When using Zhipu embeddings instead of local models, ChromaDB collection initialization changes:

```python
# OLD: ChromaDB uses its own embedding function
collection = client.get_or_create_collection(
    name="event_embeddings",
    metadata={"hnsw:space": "cosine"}
)
# Then: collection.add(documents=[text])  ← ChromaDB embeds internally

# NEW: You provide pre-computed embeddings
collection = client.get_or_create_collection(
    name="event_embeddings",
    metadata={"hnsw:space": "cosine"}
)
# Then: collection.add(
#     documents=[text],
#     embeddings=[embed_text(text)]  ← you embed externally via Zhipu
# )
```

This gives you full control over the embedding model and avoids ChromaDB's default embedding dependency.

---

## 4. Verification Steps

Before starting Phase 1, run this quick test to verify your API access:

```python
"""test_api_access.py — Run this before starting implementation"""
import os
from openai import OpenAI

api_key = os.getenv("GLM_API_KEY", "your-key-here")
base_url = "https://api.z.ai/api/paas/v4/"

client = OpenAI(api_key=api_key, base_url=base_url)

# Test 1: GLM-5 chat
print("Testing GLM-5 chat...")
response = client.chat.completions.create(
    model="glm-5",
    messages=[{"role": "user", "content": "Say 'hello' in JSON: {\"greeting\": \"...\"}"}],
    temperature=0.3,
    max_tokens=50
)
print(f"  Response: {response.choices[0].message.content}")
print(f"  Tokens: {response.usage.total_tokens}")

# Test 2: Embedding
print("\nTesting embedding-3...")
try:
    emb_response = client.embeddings.create(
        model="embedding-3",
        input="OJK mengumumkan peraturan baru tentang perbankan digital",
        dimensions=1024
    )
    print(f"  Dimension: {len(emb_response.data[0].embedding)}")
    print(f"  First 5 values: {emb_response.data[0].embedding[:5]}")
    print("  Embedding API: AVAILABLE")
except Exception as e:
    print(f"  Embedding API: NOT AVAILABLE — {e}")
    print("  Fallback: Use local sentence-transformers instead")

print("\nAll tests complete.")
```

Run this and share the output — it'll confirm exactly what's available on your API key.
