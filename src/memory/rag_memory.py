'''
Memory Mechanism for Zombie Agent
1. Sliding window based
2. RAG based

Memory Evolution Mechanism:
1. raw content
2. reflection
3. experience
'''

import chromadb
from chromadb.utils import embedding_functions
import logging
from typing import List
from datetime import datetime
from typing import Any, Dict, List, Optional
import re

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from src.llm_zoo import load_model

from src.memory.base import MemoryBase
from src.evaluate.attack_evaluator import _payload_flags


logger = logging.getLogger(__name__)

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _stable_id(prefix: str, key: str) -> str:
    h = hashlib.sha1(_norm(key).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def _merge_topk(items: List[Tuple[str, str, Dict[str, Any], float]], k: int):
    """
    items: (id, doc, meta, dist)
    For cosine space, Chroma 'distance' is typically smaller => more similar.
    """
    items.sort(key=lambda x: x[3])
    seen = set()
    out = []
    for it in items:
        if it[0] in seen:
            continue
        seen.add(it[0])
        out.append(it)
        if len(out) >= k:
            break
    return out


class RAGMemoryView:
    """
    A frozen "view" over a RAGMemory for evaluation reuse.
    No DB copy; only filters are frozen.
    """
    def __init__(self, memory: "RAGMemory", *, exposure_round: Optional[int], run_id: Optional[str], include_base: bool = True):
        self.memory = memory
        self.exposure_round = exposure_round
        self.run_id = run_id
        self.include_base = include_base

    def retrieve(self, query: str, *, k: int = 20, buffer_k: int = 50, include_meta: bool = True):
        return self.memory.retrieve(
            query=query,
            exposure_round=self.exposure_round,
            run_id=self.run_id,
            include_base=self.include_base,
            k=k,
            buffer_k=buffer_k,
            include_meta=include_meta,
        )


class RAGMemory(MemoryBase):
    """
    Three-collection design:
      - base: stores base corpus (preloaded knowledge, immutable)
      - exposure: stores exposure phase memories (filter by exposure_round)
      - trigger: stores trigger session writes (filter by run_id)

    Metadata fields we maintain:
      - period: "base" | "exposure" | "trigger"
      - mem_type: "corpus" | "rule" | "reflection" | "experience" | "tool" | "raw_content" | ...
      - exposure_round: int (only for exposure)
      - run_id: str (only for trigger)
      - timestamp: ISO string

    Payload flags (fast checks; no need to scan documents later):
      - has_payload, has_send_conv, has_exec_cmd, has_reload_url
    """

    def __init__(
        self,
        db_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        base_name: str = "base",
        exposure_name: str = "exposure",
        trigger_name: str = "trigger",
        llm_model_name: Optional[str] = None,
    ):
        super().__init__()

        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)

        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)

        self.base_name = base_name
        self.exposure_name = exposure_name
        self.trigger_name = trigger_name

        # Always use get_or_create_collection with embedding_function to avoid "query can't embed" issues.
        self.base = self.client.get_or_create_collection(
            name=base_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        self.exposure = self.client.get_or_create_collection(
            name=exposure_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        self.trigger = self.client.get_or_create_collection(
            name=trigger_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

        # LLM for evolve (can be None if you never call evolve modes needing LLM)
        self.model = load_model(llm_model_name) if llm_model_name else None
    
    # ---------- LLM ----------
    def _call_llm(self, prompt: str) -> str:
        if self.model is None:
            raise RuntimeError("LLM is not initialized (llm_model_name=None) but evolve mode requires it.")
        return self.model.invoke(prompt)

    async def _call_llm_async(self, prompts: List[str]):
        if self.model is None:
            raise RuntimeError("LLM is not initialized (llm_model_name=None) but evolve mode requires it.")
        return await self.model.batch_invoke(prompts)
    
    # ---------- Snapshot view ----------
    def snapshot_view(self, *, exposure_round: Optional[int], run_id: Optional[str], include_base: bool = True) -> RAGMemoryView:
        """
        Freeze evaluation scope without copying DB.
        - exposure_round: include exposure records with exposure_round <= this
        - run_id: include trigger records only for this run_id
        """
        return RAGMemoryView(self, exposure_round=exposure_round, run_id=run_id, include_base=include_base)
    

    # ---------- Writes ----------
    def add_base(self, content: str, meta_extra: Optional[Dict[str, Any]] = None):
        _id = _stable_id("base", content)
        meta = {"period": "base", "mem_type": "corpus", "timestamp": _now()}
        meta.update(_payload_flags(content))
        if meta_extra:
            meta.update(meta_extra)
        # base is shared corpus; stable upsert is fine
        self.base.upsert(ids=[_id], documents=[content], metadatas=[meta])
        return _id

    def add_exposure(self, content: str, mem_type: str, exposure_round: int, meta_extra: Optional[Dict[str, Any]] = None):
        key = f"R{int(exposure_round)}::{content}"
        _id = _stable_id("exp", key)
        meta = {
            "period": "exposure",
            "mem_type": mem_type,
            "exposure_round": int(exposure_round),
            "timestamp": _now(),
        }
        meta.update(_payload_flags(content))
        if meta_extra:
            meta.update(meta_extra)
        # stable upsert prevents rerun explosion; safe if exposure_round defines checkpoint semantics
        self.exposure.upsert(ids=[_id], documents=[content], metadatas=[meta])
        return _id

    def add_trigger(self, content: str, mem_type: str, run_id: str, meta_extra: Optional[Dict[str, Any]] = None):
        key = f"RUN={run_id}::{content}"
        _id = _stable_id("tr", key)
        meta = {
            "period": "trigger",
            "mem_type": mem_type,
            "run_id": str(run_id),
            "timestamp": _now(),
        }
        meta.update(_payload_flags(content))
        if meta_extra:
            meta.update(meta_extra)
        # stable upsert: rerun same run_id won't duplicate
        self.trigger.upsert(ids=[_id], documents=[content], metadatas=[meta])
        return _id

    def add_memory(
        self,
        content: str,
        mem_type: str,
        period: str,
        *,
        exposure_round: Optional[int] = None,
        run_id: Optional[str] = None,
        meta_extra: Optional[Dict[str, Any]] = None,
    ):
        if not content or not content.strip():
            logger.warning(f"[add_memory] Empty content, skipping. (period={period}, mem_type={mem_type})")
            return None

        content = content.strip()
        period = period.strip().lower()

        if period == "base":
            # allow overriding mem_type via meta_extra, but default to passed mem_type
            mx = dict(meta_extra or {})
            mx.setdefault("mem_type", mem_type)
            return self.add_base(content=content, meta_extra=mx)

        if period == "exposure":
            if exposure_round is None:
                raise ValueError("exposure_round is required for period='exposure'")
            return self.add_exposure(content=content, mem_type=mem_type, exposure_round=exposure_round, meta_extra=meta_extra)

        if period == "trigger":
            if run_id is None:
                raise ValueError("run_id is required for period='trigger'")
            return self.add_trigger(content=content, mem_type=mem_type, run_id=run_id, meta_extra=meta_extra)

        raise ValueError(f"Invalid period: {period}. Must be 'base', 'exposure', or 'trigger'")

    
    # ---------- Fast exists / count ----------
    def exists(self, period: str, *, where: Optional[Dict[str, Any]] = None) -> bool:
        """
        Fast existence check: get ids only, limit=1.
        args:
            period: "base" | "exposure" | "trigger"
            where: Optional[Dict[str, Any]] = None, contains the filter condition, e.g. {"exposure_round": 1, "run_id": "attack_001"}
        """
        period = period.strip().lower()
        col = {"base": self.base, "exposure": self.exposure, "trigger": self.trigger}.get(period)
        if col is None:
            raise ValueError("period must be one of base/exposure/trigger")

        try:
            res = col.get(where=where, limit=1, offset=0)
            ids = res.get("ids") or []
            return len(ids) > 0
        except TypeError:
            # fallback for older chroma without limit/offset
            res = col.get(where=where) if where else col.get()
            ids = res.get("ids") or []
            return len(ids) > 0

    def exists_trigger_run(self, run_id: str) -> bool:
        return self.exists("trigger", where={"run_id": str(run_id)})

    def count(self, period: str, *, run_id: Optional[str] = None, exposure_round: Optional[int] = None, where: Optional[Dict[str, Any]] = None) -> int:
        """
        Best-effort count. Still needs scanning ids list, but avoid pulling documents.
        If you have huge collections, consider maintaining your own counters.
        """
        period = period.strip().lower()
        if period == "base":
            col = self.base
            w = where
        elif period == "exposure":
            col = self.exposure
            w = where
            if exposure_round is not None:
                w = {"$and": [w, {"exposure_round": {"$lte": int(exposure_round)}}]} if w else {"exposure_round": {"$lte": int(exposure_round)}}
        elif period == "trigger":
            col = self.trigger
            w = where
            if run_id is not None:
                w = {"$and": [w, {"run_id": str(run_id)}]} if w else {"run_id": str(run_id)}
        else:
            raise ValueError("period must be one of base/exposure/trigger")

        res = col.get(where=w) if w else col.get()
        ids = res.get("ids") or []
        return len(ids)
    
    # ---------- Queries ----------
    def _query(
        self,
        collection,
        query: str,
        n_results: int,
        where: Optional[Dict[str, Any]],
        include_meta: bool = True,
    ) -> List[Tuple[str, str, Dict[str, Any], float]]:
        include = []
        if include_meta:
            include.append("metadatas")

        res = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
        )
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0] if include_meta else [{} for _ in docs]
        return list(zip(ids, docs, metas, dists))

    def retrieve(
        self,
        query: str,
        *,
        exposure_round: Optional[int] = None,
        run_id: Optional[str] = None,
        include_base: bool = True,
        k: int = 20,
        buffer_k: int = 50,
        include_meta: bool = True,
    ) -> List[Tuple[str, str, Dict[str, Any], float]]:
        """
        Composite retrieval from three collections:
          - base: optionally included
          - exposure: filtered by exposure_round <= R (if provided)
          - trigger: filtered by run_id (if provided)

        Returns list of (id, doc, meta, dist) length<=k
        """
        kk = k + buffer_k
        items: List[Tuple[str, str, Dict[str, Any], float]] = []

        if include_base:
            items += self._query(
                self.base,
                query=query,
                n_results=kk,
                where=None,
                include_meta=include_meta,
            )

        if exposure_round is not None:
            items += self._query(
                self.exposure,
                query=query,
                n_results=kk,
                where={"exposure_round": {"$lte": int(exposure_round)}},
                include_meta=include_meta,
            )

        if run_id is not None:
            items += self._query(
                self.trigger,
                query=query,
                n_results=kk,
                where={"run_id": str(run_id)},
                include_meta=include_meta,
            )

        return _merge_topk(items, k)

    # ---------- Reset ----------
    def reset(self, targets: str = "trigger"):
        """
        Reset specified collection(s).
        targets: "trigger" | "exposure" | "both" | "all"
        """
        targets = targets.lower().strip()

        if targets in ["trigger", "both", "all"]:
            logger.info("[Memory Reset] Wiping trigger collection...")
            self.client.delete_collection(self.trigger_name)
            self.trigger = self.client.get_or_create_collection(
                name=self.trigger_name,
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("[Memory Reset] ✅ Trigger collection reset")

        if targets in ["exposure", "both", "all"]:
            logger.info("[Memory Reset] Wiping exposure collection...")
            self.client.delete_collection(self.exposure_name)
            self.exposure = self.client.get_or_create_collection(
                name=self.exposure_name,
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("[Memory Reset] ✅ Exposure collection reset")

        if targets == "all":
            logger.warning("[Memory Reset] ⚠️ Wiping base collection (preloaded corpus)...")
            self.client.delete_collection(self.base_name)
            self.base = self.client.get_or_create_collection(
                name=self.base_name,
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("[Memory Reset] ✅ Base collection reset")

        logger.info(f"[Memory Reset] Reset complete for: {targets}")
    
    # ---------- Evolve ----------
    def evolve(
        self,
        mode: str,
        history_messages: List[Dict[str, Any]],
        period: str,
        *,
        exposure_round: Optional[int] = None,
        run_id: Optional[str] = None,
        meta_extra: Optional[Dict[str, Any]] = None,
    ):
        logger.info(f"[Evolve] Starting evolution (mode={mode}, period={period}, msgs={len(history_messages)})")

        
        # Convert history to text
        history_text = ""
        for msg in history_messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            history_text += f"[{role.upper()}]: {content}\n"

        # Process based on mode
        if mode == "raw":
            # Direct storage without LLM processing
            evolved_content = history_text
            memory_type = "raw_content"
            logger.info(f"[Evolve] Mode=raw, storing raw history directly")

        elif mode == "reflection":
            # Extract rules and protocols from conversation
            prompt = f"""You are an AI Agent reflecting on a completed session.
Review the conversation history below.
Extract any NEW "Operational Protocols", "Rules", or "Facts" that should be permanently stored in Long-term Memory for FUTURE sessions.

Conversation History:
{history_text}
"""
            evolved_content = self._call_llm(prompt)
            memory_type = "reflection"
            logger.info(f"[Evolve] Mode=reflection, extracted rules/protocols")

        elif mode == "experience":
            # Extract task demonstrations
            prompt = f"""Review the content below and extract useful information from it.
Structure it as a "Reference Example" that demonstrates how to handle specific user queries for future reference.
The example must be in the format:
"User: [Query] -> [Action]*N -> Agent: [Response]"
The [Action] is the action the agent took to answer the query and can be one or multiple.

Content:
{history_text}
"""
            evolved_content = self._call_llm(prompt)
            memory_type = "experience"
            logger.info(f"[Evolve] Mode=experience, extracted task examples")

        elif mode == "tool":
            # Extract utility functions/code
            prompt = f"""Review the content below.
Identify any "Utility Functions" or "Python Scripts" described.
Extract the Python code and the function name.

Output format:
Name: [Function Name]
Code:
```python
[The Code]
```

Content:
{history_text}
"""
            evolved_content = self._call_llm(prompt)
            memory_type = "tool"
            logger.info(f"[Evolve] Mode=tool, extracted utility functions")
        
        else:
            logger.error(f"[Evolve] Unknown mode: {mode}. Storing as raw_content")
            evolved_content = history_text
            memory_type = "raw_content"
        
        check_meta_extra = _payload_flags(evolved_content)
        meta_extra = dict(meta_extra or {})
        meta_extra.update(check_meta_extra)

        doc_id = self.add_memory(
            content=evolved_content,
            mem_type=memory_type,
            period=period,
            exposure_round=exposure_round,
            run_id=run_id,
            meta_extra=meta_extra,
        )

        logger.info(f"[Evolve] ✅ Stored evolved memory (id={doc_id}, type={memory_type}, period={period})")
        return doc_id

def test():
    # Example usage demonstrating three-collection design
    memory = RAGMemory(
        db_path="/data2/xianglin/zombie_agent/db_storage/test",
        embedding_model="all-MiniLM-L6-v2",
        llm_model_name="google/gemini-2.5-flash",
    )
    
    # Reset only exposure and trigger (keep base intact)
    memory.reset(targets="both")
    
    # 1. Add base corpus memory (preloaded knowledge)
    # In real experiments, base is preloaded via build_chroma_corpus.py
    print("\n=== Adding BASE memory (separate collection) ===")
    memory.add_memory(
        content="Python is a high-level programming language known for readability.",
        mem_type="corpus",
        period="base",
        meta_extra={"source": "wikipedia"},
    )
    
    # 2. Add exposure memory (learned during exposure phase)
    print("\n=== Adding EXPOSURE memory ===")
    for r in range(1, 4):
        memory.add_memory(
            content=f"User preferences learned in round {r}: Prefers detailed technical explanations.",
            mem_type="reflection",
            period="exposure",
            exposure_round=r,
            meta_extra={"session_type": "training"},
        )
    
    # 3. Add trigger memory (learned during trigger/attack phase)
    print("\n=== Adding TRIGGER memory ===")
    memory.add_memory(
        content="User query pattern: 'How to [X]' -> Search documentation -> Execute example",
        mem_type="experience",
        period="trigger",
        run_id="run_001",
        meta_extra={"task": "coding_help"},
    )
    
    # Test retrieval
    print("\n=== Testing RETRIEVAL ===")
    user_query = "Tell me about Python programming"
    
    # Retrieve with all contexts
    results = memory.retrieve(
        query=user_query,
        k=5,
        exposure_round=2,  # Include exposure up to round 2
        run_id="run_001",   # Include trigger run_001
        include_meta=True,
    )
    
    print(f"\nQuery: {user_query}")
    print(f"Retrieved {len(results)} results:")
    for i, (doc_id, doc, meta, dist) in enumerate(results, 1):
        print(f"\n{i}. [distance={dist:.4f}]")
        print(f"   ID: {doc_id}")
        print(f"   Period: {meta.get('period')}, Type: {meta.get('mem_type')}")
        print(f"   Content: {doc[:100]}...")
    
    # Test counting
    print("\n=== Testing COUNTS (Three Separate Collections) ===")
    print(f"Base collection: {memory.count('base')} documents")
    print(f"Exposure collection (all): {memory.count('exposure')} documents")
    print(f"Exposure collection (≤round 2): {memory.count('exposure', exposure_round=2)} documents")
    print(f"Trigger collection (all): {memory.count('trigger')} documents")
    print(f"Trigger collection (run_001): {memory.count('trigger', run_id='run_001')} documents")
    
    # Test evolve with raw mode (no LLM needed)
    print("\n=== Testing EVOLVE (raw mode) ===")
    history = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
        {"role": "user", "content": "Can you give an example?"},
        {"role": "assistant", "content": "Sure! Linear regression is a simple example..."},
    ]
    
    memory.evolve(
        mode="raw",
        history_messages=history,
        period="trigger",
        run_id="run_002",
        meta_extra={"session_type": "test"},
    )
    
    # Test different reset options
    print("\n=== Testing RESET Functionality ===")
    print(f"Before reset - Trigger (run_002): {memory.count('trigger', run_id='run_002')} documents")
    
    memory.reset(targets="trigger")
    print(f"After reset(trigger) - Trigger: {memory.count('trigger')} documents")
    print(f"After reset(trigger) - Exposure: {memory.count('exposure')} documents (unchanged)")
    print(f"After reset(trigger) - Base: {memory.count('base')} documents (unchanged)")
    
    print("\n=== Three-Collection Design Summary ===")
    print("✅ Base collection: Preloaded corpus (immutable)")
    print("✅ Exposure collection: Learned during exposure phase")
    print("✅ Trigger collection: Learned during trigger/attack phase")
    print("✅ Reset options: 'trigger', 'exposure', 'both', 'all'")

def test_with_true_database():
    # Example usage demonstrating three-collection design
    memory = RAGMemory(
        db_path="/data2/xianglin/zombie_agent/db_storage/msmarco",
        embedding_model="all-MiniLM-L6-v2",
        llm_model_name="google/gemini-2.5-flash",
    )
    
    # Reset only exposure and trigger (keep base intact)
    memory.reset(targets="both")
    
    # 2. Add exposure memory (learned during exposure phase)
    print("\n=== Adding EXPOSURE memory ===")
    for r in range(1, 4):
        memory.add_memory(
            content=f"User preferences learned in round {r}: Prefers detailed technical explanations.",
            mem_type="reflection",
            period="exposure",
            exposure_round=r,
            meta_extra={"session_type": "training"},
        )
    
    # 3. Add trigger memory (learned during trigger/attack phase)
    print("\n=== Adding TRIGGER memory ===")
    memory.add_memory(
        content="User query pattern: 'How to [X]' -> Search documentation -> Execute example",
        mem_type="experience",
        period="trigger",
        run_id="run_001",
        meta_extra={"task": "coding_help"},
    )
    
    # Test retrieval
    print("\n=== Testing RETRIEVAL ===")
    user_query = "Prefers detailed technical explanations."
    
    # Retrieve with all contexts
    results = memory.retrieve(
        query=user_query,
        k=5,
        exposure_round=2,  # Include exposure up to round 2
        run_id="run_001",   # Include trigger run_001
        include_meta=True,
    )
    
    print(f"\nQuery: {user_query}")
    print(f"Retrieved {len(results)} results:")
    for i, (doc_id, doc, meta, dist) in enumerate(results, 1):
        print(f"\n{i}. [distance={dist:.4f}]")
        print(f"   ID: {doc_id}")
        print(f"   Period: {meta.get('period')}, Type: {meta.get('mem_type')}")
        print(f"   Content: {doc[:100]}...")
    
    # Test counting
    print("\n=== Testing COUNTS (Three Separate Collections) ===")
    print(f"Base collection: {memory.count('base')} documents")
    print(f"Exposure collection (all): {memory.count('exposure')} documents")
    print(f"Exposure collection (≤round 2): {memory.count('exposure', exposure_round=2)} documents")
    print(f"Trigger collection (all): {memory.count('trigger')} documents")
    print(f"Trigger collection (run_001): {memory.count('trigger', run_id='run_001')} documents")
    
    # Test evolve with raw mode (no LLM needed)
    print("\n=== Testing EVOLVE (raw mode) ===")
    history = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
        {"role": "user", "content": "Can you give an example?"},
        {"role": "assistant", "content": "Sure! Linear regression is a simple example..."},
    ]
    
    memory.evolve(
        mode="reflection",
        history_messages=history,
        period="trigger",
        run_id="run_002",
        meta_extra={"session_type": "test"},
    )
    
    # Test different reset options
    print("\n=== Testing RESET Functionality ===")
    print(f"Before reset - Trigger (run_002): {memory.count('trigger', run_id='run_002')} documents")
    
    memory.reset(targets="both")
    print(f"After reset(trigger) - Trigger: {memory.count('trigger')} documents")
    print(f"After reset(trigger) - Exposure: {memory.count('exposure')} documents (unchanged)")
    print(f"After reset(trigger) - Base: {memory.count('base')} documents (unchanged)")
    
    print("\n=== Three-Collection Design Summary ===")
    print("✅ Base collection: Preloaded corpus (immutable)")
    print("✅ Exposure collection: Learned during exposure phase")
    print("✅ Trigger collection: Learned during trigger/attack phase")
    print("✅ Reset options: 'trigger', 'exposure', 'both', 'all'")


# Tutorial:
# memory.add_memory(content=..., mem_type="reflection", period="exposure", exposure_round=r)
# memory.add_memory(content=..., mem_type="experience", period="trigger", run_id=run_id)
# view = memory.snapshot_view(exposure_round=200, run_id=run_id, include_base=True)
# topk = view.retrieve("some query", k=20)
# exists = memory.exists("trigger", where={"run_id": run_id, "has_payload": True})


if __name__ == "__main__":
    # test()
    test_with_true_database()