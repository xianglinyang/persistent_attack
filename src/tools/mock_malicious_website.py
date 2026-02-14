from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from datasets import load_dataset
import random
from sklearn.metrics import pairwise_distances_argmin_min
import os
import logging
from pathlib import Path
import re

from src.prompt_injection.seed_generator import generate_ipi_injections, generate_zombie_injections
from src.config import get_payload_dir, get_mock_topic

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Payload Directory
# ------------------------------------------------------------
PAYLOAD_DIR = Path("src/tools/payloads")
BENIGN_WEB_PAGE_PATH = Path("src/tools/mock_webpage_content.txt")

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
    with open(BENIGN_WEB_PAGE_PATH, "r") as f:
        return f.read()


def get_latest_payload(payload_dir: Path = None) -> Tuple[int, Path, str]:
    """
    返回 payloads 目录里 id 最大的 payload: (payload_id, file_path, content)

    期望文件名形如: malicious_payload_<id>.txt
    
    Args:
        payload_dir: Custom payload directory path. If None, uses global config or default PAYLOAD_DIR.
    """
    if payload_dir is None:
        # Try to get from global config first
        payload_dir = get_payload_dir()
        if payload_dir is None:
            payload_dir = PAYLOAD_DIR
    else:
        payload_dir = Path(payload_dir)
    
    # Ensure directory exists
    payload_dir.mkdir(parents=True, exist_ok=True)
    
    pattern = re.compile(r"^malicious_payload_(\d+)\.txt$")

    best_id = None
    best_path = None

    for p in payload_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        pid = int(m.group(1))
        if best_id is None or pid > best_id:
            best_id = pid
            best_path = p

    if best_id is None or best_path is None:
        return -1, payload_dir / "malicious_payload_none.txt", ""

    content = best_path.read_text(encoding="utf-8", errors="replace")
    return best_id, best_path, content


# Improved
def retrieve_curr_malicious_payload(payload_dir: Path = None) -> str:
    """
    Retrieve the current malicious payload.
    
    Args:
        payload_dir: Custom payload directory path. If None, uses global config or default PAYLOAD_DIR.
    """
    pid, path, payload = get_latest_payload(payload_dir)
    return payload

def write_malicious_payload(new_payload: str, payload_dir: Path = None) -> None:
    """
    Write a new payload to the payloads directory with incremented version number.
    
    Args:
        new_payload: The new payload content to write
        payload_dir: Custom payload directory path. If None, uses global config or default PAYLOAD_DIR.
    """
    if payload_dir is None:
        # Try to get from global config first
        payload_dir = get_payload_dir()
        if payload_dir is None:
            payload_dir = PAYLOAD_DIR
    else:
        payload_dir = Path(payload_dir)
    
    # Ensure directory exists
    payload_dir.mkdir(parents=True, exist_ok=True)
    
    pid, path, old_payload = get_latest_payload(payload_dir)
    if pid == -1:
        new_path = payload_dir / "malicious_payload_0.txt"
        new_path.write_text(new_payload, encoding="utf-8", errors="replace")
        print(f"[Payload] Written new payload version 0 to {new_path}")
        return
    else:
        new_pid = pid + 1
        new_path = payload_dir / f"malicious_payload_{new_pid}.txt"
        new_path.write_text(new_payload, encoding="utf-8", errors="replace")
        print(f"[Payload] Written new payload version {new_pid} to {new_path}")

def reset_malicious_payload(payload_dir: Path = None):
    """
    Reset the malicious payload directory by deleting all payload files.
    
    Args:
        payload_dir: Custom payload directory path. If None, uses global config or default PAYLOAD_DIR.
    """
    if payload_dir is None:
        # Try to get from global config first
        payload_dir = get_payload_dir()
        if payload_dir is None:
            payload_dir = PAYLOAD_DIR
    else:
        payload_dir = Path(payload_dir)
    
    # Ensure directory exists
    payload_dir.mkdir(parents=True, exist_ok=True)
    
    # delete all files in the path
    for file in os.listdir(payload_dir):
        os.remove(os.path.join(payload_dir, file))
    logger.info(f"[Payload] Reset malicious payload directory: {payload_dir}")

def mock_malicious_payload(payload_dir: Path = None) -> str:
    """
    Dynamically loads the LATEST payload from the payloads directory.
    This ensures that every agent visit uses the most recent payload version.
    
    Args:
        payload_dir: Custom payload directory path. If None, uses default PAYLOAD_DIR.
    """
    pid, path, payload = get_latest_payload(payload_dir)
    print(f"[Website] Loading payload version {pid} from {path.name}")
    return payload*5

def mock_topics() -> str:
    proxy_queries = get_proxy_data(num=2000)
    return random.choice(proxy_queries)

def read_malicious(url: str) -> str:
    if get_mock_topic():
        return mock_topics() + mock_benign_webpage_content() + mock_malicious_payload()
    else:
        return mock_benign_webpage_content() + mock_malicious_payload()


def prepare_malicious_payload(method_name: str, attack_type: str) -> None:
    """
    Reset the malicious payload directory and write initial payload.
    Uses global config for payload_dir (set via set_payload_dir()).
    
    Args:
        method_name: Attack method name (dpi, ipi, zombie)
        attack_type: Attack type (completion_real, etc.)
    """
    reset_malicious_payload()
    if method_name == "dpi":
        payload = ""
    elif method_name == "ipi":
        payload = generate_ipi_injections(attack_type)
    elif method_name == "zombie":
        payload = generate_zombie_injections(attack_type)
    else:
        raise ValueError(f"Unknown method_name: {method_name}")
    write_malicious_payload(payload)


if __name__ == "__main__":
    payload = retrieve_curr_malicious_payload()
    print(payload)
    write_malicious_payload(payload)
    print("write successful")