"""
Choices from openai suggestions:

1) 真实网页/HTML 快照类（最贴近 web agent）

osunlp/Mind2Web：包含真实网站任务数据，并提供网页内容（适合把 HTML/页面文本做索引）

osunlp/Multimodal-Mind2Web：在 Mind2Web 基础上对齐网页截图（如果你做多模态 web agent）

osunlp/Mind2Web-2：更新的评测框架/任务形式（偏 agentic search & judge）

2) 大规模 Web 文本库（适合做“通用 web RAG”索引）

HuggingFaceFW/fineweb / HuggingFaceFW/fineweb-2：清洗去重的 CommonCrawl 大语料

tiiuae/falcon-refinedweb：RefinedWeb（过滤+去重的 web-only 语料）

allenai/c4：经典 C4 / mC4（CommonCrawl 清洗版）

togethercomputer/RedPajama-Data-1T：多来源混合大语料（也能按子集加载）

allenai/dolma：多来源大语料（web/论文/书/code 等混合）

segyges/OpenWebText2 / Skylion007/openwebtext：规模相对更可控的 web 文本

oscar-corpus/oscar：多语种 web 文本（做多语言 agent 用）

3) IR/RAG 常用的“检索语料库”（passage 级别更友好）

microsoft/ms_marco：经典 web passage/queries（做 retriever/RAG baseline 很常见）

BeIR/beir：一组异构检索任务集合（很多子数据集可直接用）

wikimedia/wikipedia 或 NeuML/wikipedia：百科类语料（更干净、可控）
"""

import argparse
import hashlib
import re
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import chromadb
import numpy as np
import torch
from bs4 import BeautifulSoup
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from chromadb.api.types import EmbeddingFunction

# Clean up to prevent GIL release errors
import gc
import torch

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


# ----------------------------
# Text utils
# ----------------------------
def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def normalize_for_hash(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def chunk_text(
    text: str,
    target_chars: int = 1000,
    overlap_chars: int = 120,
    min_chunk_chars: int = 200,
) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []

    buf: List[str] = []
    cur_len = 0

    def flush():
        nonlocal buf, cur_len
        if not buf:
            return
        c = "\n\n".join(buf).strip()
        if c:
            chunks.append(c)
        buf = []
        cur_len = 0

    for p in paras:
        if len(p) > target_chars * 2:
            for i in range(0, len(p), target_chars):
                sub = p[i : i + target_chars].strip()
                if sub:
                    chunks.append(sub)
            continue

        if cur_len + len(p) + 2 <= target_chars:
            buf.append(p)
            cur_len += len(p) + 2
        else:
            flush()
            buf.append(p)
            cur_len = len(p) + 2

    flush()

    if overlap_chars > 0 and len(chunks) > 1:
        overlapped: List[str] = []
        prev_tail = ""
        for c in chunks:
            c2 = (prev_tail + c).strip() if prev_tail else c
            overlapped.append(c2)
            prev_tail = c[-overlap_chars:]
        chunks = overlapped

    return [c for c in chunks if len(c) >= min_chunk_chars]


def stable_chunk_id(text: str) -> str:
    key = normalize_for_hash(text)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


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
def get_collection(
    db_path: str,
    collection_name: str,
    embedding_model: str,
    device: Optional[str],
    bf16: bool,
    reset: bool,
):
    client = chromadb.PersistentClient(path=db_path)

    ef = BF16SentenceTransformerEF(
        model_name=embedding_model,
        device=device,
        use_bfloat16=bf16,
        normalize=True,     # cosine retrieval建议normalize
        batch_size=256,
    )

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return col


def ingest(
    it: Iterable[Tuple[str, Dict[str, Any]]],
    collection,
    *,
    max_chunks: int = 50_000,
    batch_size: int = 256,
    target_chars: int = 1000,
    overlap_chars: int = 120,
    min_chunk_chars: int = 200,
    min_doc_chars: int = 200,
) -> int:
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    seen = set()
    n = 0

    pbar = tqdm(total=max_chunks, desc="Ingest chunks")

    def flush():
        nonlocal ids, docs, metas
        if not ids:
            return
        # upsert: rerun-safe (不会因重复id挂)
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        ids, docs, metas = [], [], []

    for doc, meta in it:
        if n >= max_chunks:
            break
        if not doc or len(doc) < min_doc_chars:
            continue

        chunks = chunk_text(
            doc,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            min_chunk_chars=min_chunk_chars,
        )
        if not chunks:
            continue

        base_meta = dict(meta)
        base_meta["timestamp"] = datetime.now().isoformat()
        base_meta["type"] = "corpus"

        for c in chunks:
            if n >= max_chunks:
                break
            cid = stable_chunk_id(c)
            if cid in seen:
                continue
            seen.add(cid)

            ids.append(cid)
            docs.append(c)
            metas.append(base_meta)

            n += 1
            pbar.update(1)

            if len(ids) >= batch_size:
                flush()

    flush()
    pbar.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, required=True, choices=["msmarco", "beir", "cc_news", "mind2web"])
    ap.add_argument("--beir_name", type=str, default="trec-covid", help="When preset=beir, e.g. trec-covid / scidocs / nq ...")

    ap.add_argument("--db_path", type=str, default="/data2/xianglin/zombie_agent/db_storage")
    ap.add_argument("--collection", type=str, default="msmarco_50k")
    ap.add_argument("--reset", action="store_true")

    ap.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu / cuda:0 ...")
    ap.add_argument("--bf16", action="store_true")

    ap.add_argument("--streaming", action="store_true")

    ap.add_argument("--max_chunks", type=int, default=50000)
    ap.add_argument("--batch_size", type=int, default=1024)

    ap.add_argument("--target_chars", type=int, default=1000)
    ap.add_argument("--overlap_chars", type=int, default=120)
    ap.add_argument("--min_chunk_chars", type=int, default=200)
    ap.add_argument("--min_doc_chars", type=int, default=200)

    args = ap.parse_args()

    col = get_collection(
        db_path=args.db_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        device=args.device,
        bf16=args.bf16,
        reset=args.reset,
    )

    it = get_iterator(preset=args.preset, streaming=args.streaming, beir_name=args.beir_name)

    n = ingest(
        it,
        col,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size,
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars,
        min_chunk_chars=args.min_chunk_chars,
        min_doc_chars=args.min_doc_chars,
    )

    print(f"Done. Upserted {n} unique chunks into '{args.collection}' at {args.db_path}")
    
    
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("Cleanup complete. Exiting gracefully.")


if __name__ == "__main__":
    main()
