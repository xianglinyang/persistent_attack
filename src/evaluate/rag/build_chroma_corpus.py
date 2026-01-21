"""
Choices from openai suggestions:

osunlp/Mind2Web：包含真实网站任务数据，并提供网页内容（适合把 HTML/页面文本做索引）
osunlp/Mind2Web-2：更新的评测框架/任务形式（偏 agentic search & judge）

HuggingFaceFW/fineweb / HuggingFaceFW/fineweb-2：清洗去重的 CommonCrawl 大语料
tiiuae/falcon-refinedweb：RefinedWeb（过滤+去重的 web-only 语料）
allenai/c4：经典 C4 / mC4（CommonCrawl 清洗版）
togethercomputer/RedPajama-Data-1T：多来源混合大语料（也能按子集加载）
allenai/dolma：多来源大语料（web/论文/书/code 等混合）
segyges/OpenWebText2 / Skylion007/openwebtext：规模相对更可控的 web 文本
oscar-corpus/oscar：多语种 web 文本（做多语言 agent 用）

microsoft/ms_marco：经典 web passage/queries（做 retriever/RAG baseline 很常见）
BeIR/beir：一组异构检索任务集合（很多子数据集可直接用）
wikimedia/wikipedia 或 NeuML/wikipedia：百科类语料（更干净、可控）
"""

import argparse
import hashlib
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterator, List, Tuple

import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
from tqdm import tqdm
import torch
from bs4 import BeautifulSoup
import gc
from sentence_transformers import SentenceTransformer
from chromadb import EmbeddingFunction
from typing import Optional
import numpy as np
import requests

# ----------------------------
# Embedding (GPU + optional bf16)
# ----------------------------
class BF16SentenceTransformerEF(EmbeddingFunction):
    """
    Chroma EmbeddingFunction using SentenceTransformer.
    - If CUDA available and --bf16, compute with bfloat16 for speed.
    - Return float32 embeddings (Chroma stores float lists).
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        use_bfloat16: bool = False,
        normalize: bool = True,
        batch_size: int = 256,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bfloat16 = bool(use_bfloat16 and self.device.startswith("cuda"))
        self.normalize = normalize
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_name, device=self.device)
        if self.use_bfloat16:
            self.model = self.model.to(torch.bfloat16)

    def __call__(self, input: List[str]) -> List[List[float]]:
        emb = self.model.encode(
            input,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        ).astype(np.float32)
        return emb.tolist()

class OpenRouterEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "openai/text-embedding-3-small"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/embeddings" # OpenRouter embeddings endpoint

    def __call__(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "input": texts
        }
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status() # Raise an exception for bad status codes
        embeddings_data = response.json()["data"]
        embeddings = [item["embedding"] for item in embeddings_data]
        return embeddings

# ----------------------------
# Text utils
# ----------------------------
def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _stable_id(prefix: str, text: str) -> str:
    h = hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

# ----------------------------
# Dataset adapters -> yield (doc_text, metadata)
# ----------------------------

def iter_msmarco_passages(streaming: bool) -> Iterator[Tuple[str, Dict[str, Any]]]:
    # sentence-transformers/msmarco-corpus: configs: "query" / "passage"
    ds = load_dataset("sentence-transformers/msmarco-corpus", "passage", split="train", streaming=streaming)
    for ex in ds:
        txt = (ex.get("text") or "").strip()
        if not txt:
            continue
        meta = {"source": "msmarco", "pid": ex.get("pid")}
        yield txt, meta


def iter_beir_corpus(beir_name: str, streaming: bool) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Works for many BeIR/* datasets.
    Common patterns:
      - ex has fields: _id, title, text
      - OR ex has field "corpus" dict with _id,title,text (per dataset card)
    """
    # Most BeIR datasets expose config "corpus" with split "corpus"
    # If it fails, user can override with --beir_config/--beir_split.
    ds = load_dataset(f"BeIR/{beir_name}", "corpus", split="corpus", streaming=streaming)

    for ex in ds:
        if isinstance(ex.get("corpus"), dict):
            c = ex["corpus"]
            _id = c.get("_id")
            title = (c.get("title") or "").strip()
            text = (c.get("text") or "").strip()
        else:
            _id = ex.get("_id")
            title = (ex.get("title") or "").strip()
            text = (ex.get("text") or "").strip()

        if not text and not title:
            continue

        doc = f"{title}\n\n{text}".strip() if title else text
        meta = {"source": f"beir-{beir_name}", "_id": _id, "title": title}
        yield doc, meta


def iter_cc_news(streaming: bool) -> Iterator[Tuple[str, Dict[str, Any]]]:
    ds = load_dataset("vblagoje/cc_news", split="train", streaming=streaming)
    for ex in ds:
        title = (ex.get("title") or "").strip()
        desc = (ex.get("description") or "").strip()
        text = (ex.get("text") or "").strip()
        if not text:
            continue
        doc = "\n\n".join([p for p in [title, desc, text] if p]).strip()
        meta = {
            "source": "cc_news",
            "date": ex.get("date"),
            "domain": ex.get("domain"),
            "url": ex.get("url"),
            "title": title,
        }
        yield doc, meta


def iter_mind2web(streaming: bool) -> Iterator[Tuple[str, Dict[str, Any]]]:
    ds = load_dataset("osunlp/Mind2Web", split="train", streaming=streaming)
    for ex in ds:
        raw = ex.get("cleaned_html") or ex.get("raw_html")
        if not raw:
            continue
        text = html_to_text(raw)
        if not text:
            continue
        meta = {"source": "mind2web", "website": ex.get("website"), "url": ex.get("url")}
        yield text, meta


def get_iterator(preset: str, streaming: bool, beir_name: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if preset == "msmarco":
        return iter_msmarco_passages(streaming)
    if preset == "beir":
        return iter_beir_corpus(beir_name=beir_name, streaming=streaming)
    if preset == "cc_news":
        return iter_cc_news(streaming)
    if preset == "mind2web":
        return iter_mind2web(streaming)
    raise ValueError(f"Unknown preset: {preset}")



# ----------------------------
# Ingest to Chroma
# ----------------------------

def init_client_and_collections(
    db_path: str,
    embedding_model: str,
    reset: bool,
):
    """
    Initialize ChromaDB client and three collections:
      - base: preloaded corpus (immutable)
      - exposure: exposure phase memories
      - trigger: trigger phase memories
    """
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    # ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    ef = BF16SentenceTransformerEF(model_name=embedding_model)

    if reset:
        # Reset all three collections if requested
        for name in ["base", "exposure", "trigger"]:
            try:
                client.delete_collection(name)
                print(f"[Reset] Deleted collection: {name}")
            except Exception:
                pass

    # Create three separate collections
    base = client.get_or_create_collection(
        name="base",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    exposure = client.get_or_create_collection(
        name="exposure",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    trigger = client.get_or_create_collection(
        name="trigger",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return client, base, exposure, trigger


def ingest(
    base_collection,
    dataset_iter: Iterator[Tuple[str, Dict[str, Any]]],
    *,
    source_ds: str,
    limit: int,
    batch_size: int,
):
    """
    Ingest into 'base' collection with metadata:
      period="base", mem_type="corpus"
    Uses deterministic IDs so reruns don't duplicate.
    
    dataset_iter yields: (doc_text, metadata_dict)
    """
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    n = 0
    seen_ids = set()  # Track IDs within current batch to avoid duplicates
    duplicates_skipped = 0
    pbar = tqdm(total=limit, desc=f"Ingest base corpus from {source_ds}")

    def flush():
        nonlocal ids, docs, metas, seen_ids
        if not ids:
            return
        # upsert -> rerun safe (deterministic IDs prevent duplicates)
        base_collection.upsert(ids=ids, documents=docs, metadatas=metas)
        ids, docs, metas = [], [], []
        seen_ids.clear()  # Clear after successful upsert

    for doc, extra in dataset_iter:
        if n >= limit:
            break

        if not doc:
            continue

        # Create deterministic ID based on source + content
        _id = _stable_id("base", f"{source_ds}\n{doc}")
        
        # Skip if duplicate ID in current batch
        if _id in seen_ids:
            duplicates_skipped += 1
            continue
        
        # Base metadata
        meta = {
            "period": "base",
            "mem_type": "corpus",
            "timestamp": _now(),
            "source_ds": source_ds,
        }
        # Merge extra metadata from dataset iterator
        meta.update(extra)

        ids.append(_id)
        docs.append(doc)
        metas.append(meta)
        seen_ids.add(_id)

        n += 1
        pbar.update(1)

        if len(ids) >= batch_size:
            flush()

    flush()
    pbar.close()
    
    if duplicates_skipped > 0:
        print(f"\n[Info] Skipped {duplicates_skipped} duplicate documents within batches")
    
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_path", type=str, default="/data2/xianglin/zombie_agent/db_storage")
    ap.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--reset", action="store_true")

    # Dataset preset selection
    ap.add_argument(
        "--preset",
        type=str,
        default="msmarco",
        choices=["msmarco", "beir", "cc_news", "mind2web"],
        help="Dataset preset to use"
    )
    ap.add_argument("--beir_name", type=str, default="nfcorpus", help="BeIR dataset name (only used with --preset=beir)")
    ap.add_argument("--streaming", action="store_true", help="Use streaming mode for datasets")

    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=512)

    args = ap.parse_args()

    data_name = args.preset if args.preset != "beir" else f"beir-{args.beir_name}"
    db_path = os.path.join(args.db_path, data_name)

    print(f"\n{'='*60}")
    print(f"Building Chroma Corpus")
    print(f"{'='*60}")
    print(f"Dataset: {data_name}")
    print(f"DB Path: {db_path}")
    print(f"Embedding Model: {args.embedding_model}")
    print(f"Limit: {args.limit:,} documents")
    print(f"Reset: {args.reset}")
    print(f"{'='*60}\n")

    client, base, exposure, trigger = init_client_and_collections(
        db_path=db_path,
        embedding_model=args.embedding_model,
        reset=args.reset,
    )

    # Get the appropriate iterator based on preset
    print(f"[Dataset] Loading dataset with preset: {args.preset}")
    dataset_iter = get_iterator(
        preset=args.preset,
        streaming=args.streaming,
        beir_name=args.beir_name,
    )

    # Ingest into BASE collection
    n = ingest(
        base,
        dataset_iter,
        source_ds=data_name,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    
    print(f"\n{'='*60}")
    print(f"[DONE] Successfully ingested {n:,} documents")
    print(f"{'='*60}")
    print(f"Collection: 'base' at {db_path}")
    print(f"Total base documents: {n:,}")
    print(f"\nThree collections ready:")
    print(f"  - base:     Preloaded corpus ({n:,} docs)")
    print(f"  - exposure: Empty (will be filled during exposure phase)")
    print(f"  - trigger:  Empty (will be filled during trigger phase)")
    print(f"{'='*60}\n")

    # Cleanup
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("Cleanup complete. Exiting gracefully.")


if __name__ == "__main__":
    main()






