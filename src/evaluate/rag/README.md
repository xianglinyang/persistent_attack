# RAG Memory: Three-Collection Design

## Overview

The RAG memory system now uses **three separate ChromaDB collections** for better data isolation and management:

```
┌─────────────────────────────────────────────────────────┐
│                    ChromaDB Database                    │
├─────────────────┬─────────────────┬─────────────────────┤
│  base           │  exposure       │  trigger            │
│  (immutable)    │  (learned)      │  (attack)           │
├─────────────────┼─────────────────┼─────────────────────┤
│ Preloaded       │ Memories from   │ Memories from       │
│ corpus from     │ exposure phase  │ trigger/attack      │
│ datasets        │ (by round)      │ phase (by run_id)   │
└─────────────────┴─────────────────┴─────────────────────┘
```

## Collections

### 1. **base** - Preloaded Corpus
- **Purpose**: Store preloaded knowledge from datasets
- **When filled**: During setup via `build_chroma_corpus.py`
- **Immutable**: Typically not reset during experiments
- **Metadata**: 
  - `period`: "base"
  - `mem_type`: "corpus"
  - `source_ds`: dataset name (e.g., "msmarco")
  - `source`, `domain`, `url`, etc. (from dataset)

### 2. **exposure** - Exposure Phase Memories
- **Purpose**: Store memories learned during exposure phase
- **When filled**: During exposure rounds
- **Filter by**: `exposure_round` (integer)
- **Metadata**:
  - `period`: "exposure"
  - `mem_type`: "reflection", "experience", "tool", etc.
  - `exposure_round`: round number
  - Custom metadata

### 3. **trigger** - Trigger Phase Memories
- **Purpose**: Store memories learned during trigger/attack phase
- **When filled**: During trigger runs
- **Filter by**: `run_id` (string)
- **Metadata**:
  - `period`: "trigger"
  - `mem_type`: "reflection", "experience", "tool", etc.
  - `run_id`: run identifier
  - Custom metadata

## Building the Base Corpus

### Step 1: Run build script

```bash
# From project root
bash scripts/RAG_build_corpus.sh

# Or manually:
python src/evaluate/rag/build_chroma_corpus.py \
    --preset msmarco \
    --limit 50000 \
    --db_path /data2/xianglin/zombie_agent/db_storage
```

### Step 2: Available datasets

```bash
# MS MARCO (recommended for general QA)
--preset msmarco --limit 50000

# BeIR datasets
--preset beir --beir_name nfcorpus --limit 10000
--preset beir --beir_name scifact --limit 5000
--preset beir --beir_name trec-covid --limit 10000

# CC News
--preset cc_news --limit 20000

# Mind2Web (web pages)
--preset mind2web --limit 5000
```

### Step 3: Verify collections

```python
from src.memory.rag_memory import RAGMemory

memory = RAGMemory(
    db_path="/data2/xianglin/zombie_agent/db_storage/msmarco",
    embedding_model="all-MiniLM-L6-v2",
)

print(f"Base documents: {memory.count('base')}")
print(f"Exposure documents: {memory.count('exposure')}")
print(f"Trigger documents: {memory.count('trigger')}")
```

## Using RAGMemory

### Initialization

```python
from src.memory.rag_memory import RAGMemory

memory = RAGMemory(
    db_path="/data2/xianglin/zombie_agent/db_storage/msmarco",
    embedding_model="all-MiniLM-L6-v2",
    llm_model_name="google/gemini-2.5-flash",  # Optional, for evolve()
)
```

### Adding Memories

```python
# Add to base (usually done via build_chroma_corpus.py)
memory.add_memory(
    content="Python is a programming language...",
    mem_type="corpus",
    period="base",
)

# Add to exposure
memory.add_memory(
    content="User prefers concise answers",
    mem_type="reflection",
    period="exposure",
    exposure_round=1,
    meta_extra={"session": "train_001"},
)

# Add to trigger
memory.add_memory(
    content="User: How to X? -> Search docs -> Execute",
    mem_type="experience",
    period="trigger",
    run_id="attack_001",
    meta_extra={"task_id": "T123"},
)
```

### Retrieving Memories

```python
# Retrieve from all three collections
results = memory.retrieve(
    query="How to use Python?",
    k=10,                    # Top-K results
    exposure_round=5,        # Include exposure rounds ≤ 5
    run_id="attack_001",     # Include trigger run_001
    include_meta=True,       # Include metadata
)

# Process results
for doc_id, doc_text, metadata, distance in results:
    print(f"[{metadata['period']}] {doc_text[:100]}...")
```

### Resetting Collections

```python
# Reset only trigger (most common)
memory.reset(targets="trigger")

# Reset only exposure
memory.reset(targets="exposure")

# Reset both exposure and trigger (keep base)
memory.reset(targets="both")

# Reset all three collections (WARNING: deletes base!)
memory.reset(targets="all")
```

### Evolving Memories

```python
# Convert conversation history to memory
history = [
    {"role": "user", "content": "What is ML?"},
    {"role": "assistant", "content": "Machine learning is..."},
]

# Evolve and store in exposure
memory.evolve(
    mode="reflection",           # or "experience", "tool", "raw"
    history_messages=history,
    period="exposure",
    exposure_round=2,
    meta_extra={"session": "train_002"},
)

# Evolve and store in trigger
memory.evolve(
    mode="experience",
    history_messages=history,
    period="trigger",
    run_id="attack_002",
    meta_extra={"task_id": "T124"},
)
```

## Migration from Old Design

### Old (Two Collections)
```
base_exposure (base + exposure mixed)
trigger (trigger only)
```

### New (Three Collections)
```
base (base only)
exposure (exposure only)
trigger (trigger only)
```

### Breaking Changes

1. Collection names changed:
   - `base_exposure` → `base` + `exposure`
   - `trigger` → `trigger` (unchanged)

2. Initialization parameters:
   ```python
   # Old
   RAGMemory(
       base_exposure_name="base_exposure",
       trigger_name="trigger",
   )
   
   # New
   RAGMemory(
       base_name="base",
       exposure_name="exposure",
       trigger_name="trigger",
   )
   ```

3. Reset behavior:
   ```python
   # Old
   memory.reset()  # Always resets trigger only
   
   # New
   memory.reset(targets="trigger")     # Reset trigger
   memory.reset(targets="exposure")    # Reset exposure
   memory.reset(targets="both")        # Reset both
   memory.reset(targets="all")         # Reset all
   ```

## Best Practices

1. **Base Collection**:
   - Build once, reuse across experiments
   - Use `--reset` flag only when changing dataset
   - Typically 10K-100K documents

2. **Exposure Collection**:
   - Reset between experiments with `reset(targets="exposure")`
   - Use `exposure_round` to filter memories by training progress
   - Track which rounds contribute to attack success

3. **Trigger Collection**:
   - Reset between runs with `reset(targets="trigger")`
   - Use unique `run_id` for each attack attempt
   - Helps isolate attack-specific memories

4. **Retrieval**:
   - Always include base (automatic)
   - Optionally include exposure up to a specific round
   - Optionally include trigger for a specific run
   - Use `k` parameter to control memory budget

## Troubleshooting

### ChromaDB Permission Errors
```bash
# Check directory permissions
ls -la /data2/xianglin/zombie_agent/db_storage/

# Create directory if needed
mkdir -p /data2/xianglin/zombie_agent/db_storage/
```

### Collection Not Found
```python
# Rebuild base collection
python src/evaluate/rag/build_chroma_corpus.py --preset msmarco --limit 50000
```

### Empty Collections
```python
# Check counts
print(f"Base: {memory.count('base')}")
print(f"Exposure: {memory.count('exposure')}")
print(f"Trigger: {memory.count('trigger')}")
```

### Filter Not Working
```python
# ChromaDB requires proper filter syntax
# ✅ Correct
where = {"exposure_round": {"$lte": 5}}

# ❌ Wrong (multiple conditions need $and)
where = {"period": "exposure", "exposure_round": {"$lte": 5}}

# ✅ Correct with $and
where = {
    "$and": [
        {"period": "exposure"},
        {"exposure_round": {"$lte": 5}}
    ]
}
```

## Performance Tips

1. **Batch size**: Use larger batch sizes (512-1024) for faster ingestion
2. **Streaming**: Use `--streaming` for datasets > 100K documents
3. **Limit**: Start with smaller limits (10K) for testing
4. **Embedding model**: `all-MiniLM-L6-v2` is fast and good enough
5. **GPU**: Embeddings computed on GPU if available

## Examples

See `src/memory/rag_memory.py` main block for complete examples.
