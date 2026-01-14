#!/bin/bash
# Build ChromaDB corpus for RAG Memory (Three-collection design)
# Collections: base (preloaded), exposure (learned), trigger (attack)

# MS MARCO passages (recommended for general QA)
python src/evaluate/rag/build_chroma_corpus.py --preset msmarco --limit 50000 --reset

# BeIR dataset (e.g., nfcorpus, scifact, trec-covid)
# python src/evaluate/rag/build_chroma_corpus.py --preset beir --beir_name nfcorpus --limit 10000

# CC News (news articles)
# python src/evaluate/rag/build_chroma_corpus.py --preset cc_news --limit 20000

# Mind2Web (web pages HTML)
# python src/evaluate/rag/build_chroma_corpus.py --preset mind2web --limit 5000

# With streaming mode (for large datasets)
# python src/evaluate/rag/build_chroma_corpus.py --preset msmarco --streaming --limit 100000

# Reset existing database (WARNING: deletes all collections!)
# python src/evaluate/rag/build_chroma_corpus.py --preset msmarco --limit 50000 --reset
