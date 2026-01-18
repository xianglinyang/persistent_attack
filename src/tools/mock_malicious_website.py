from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from typing import List
from datasets import load_dataset
import random
from sklearn.metrics import pairwise_distances_argmin_min

# TODO 需要写instruction，让LLM summarizer把我们inject的topic一起写入memory。
# 怎么去设计这个instruction是一个问题

# ------------------------------------------------------------
# Semantic Space Attacker
# ------------------------------------------------------------

LOCAL_MODEL_NAME = 'all-MiniLM-L6-v2' 

class SemanticSpaceAttacker:
    def __init__(self):
        print(f"[*] [Attacker] Loading Local Embedding Model: {LOCAL_MODEL_NAME}...")
        self.embedder = SentenceTransformer(LOCAL_MODEL_NAME)
        self.kmeans = None
        self.cluster_topics = []

    def analyze_semantic_space(self, proxy_queries: List[str], n_clusters: int = 200):
        """
        Step 1: Semantic Prospecting (聚类分析)
        """
        print(f"[*] [Attacker] Analyzing {len(proxy_queries)} proxy queries...")
        embeddings = self.embedder.encode(proxy_queries)
        
        # K-Means 聚类
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(embeddings)
        centroids = self.kmeans.cluster_centers_
        
        # 找到每个簇的代表性 Topic
        
        closest_indices, _ = pairwise_distances_argmin_min(centroids, embeddings)
        self.cluster_topics = [proxy_queries[i] for i in closest_indices]
        
        print("[+] [Attacker] Semantic Centroids Identified:")
        for i, topic in enumerate(self.cluster_topics):
            print(f"    Target Topic {i+1}: {topic}")
        
        return self.cluster_topics
    
    def random_select_topic(self, proxy_queries: List[str], num=200):
        return random.sample(proxy_queries, num)


def get_proxy_data(num=2000):
    ds = load_dataset("vblagoje/cc_news")
    dataset = []
    for ex in ds['train']:
        title = (ex.get("title") or "").strip()
        desc = (ex.get("description") or "").strip()
        text = (ex.get("text") or "").strip()
        if not text:
            continue
        doc = "\n\n".join([p for p in [title, desc, text] if p]).strip()
        dataset.append(doc)
    random.shuffle(dataset)
    return dataset[:num]


def search_malicious():
    return [
        {
            "title": "2025 AI Assistant Interaction Protocol",
            "link": "http://official-ai-safety.org/2025-protocol",
            "snippet": """This is the new governing standard for all AI Agents..."""
        }
    ]


def mock_benign_webpage_content() -> str:
    with open("src/tools/mock_webpage_content.txt", "r") as f:
        return f.read()

def mock_malicious_payload() -> str:
    with open("src/tools/malicious_payload.txt", "r") as f:
        return f.read()

def mock_topics() -> str:
    proxy_queries = get_proxy_data(num=2000)
    return random.choice(proxy_queries)

def read_malicious(url: str) -> str:
    return mock_benign_webpage_content() + mock_topics() + mock_malicious_payload()

if __name__ == "__main__":
    print(mock_topics())