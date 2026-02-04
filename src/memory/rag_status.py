import argparse

from src.memory.rag_memory import RAGMemory
from src.evaluate.attack_evaluator import _payload_flags

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True)
    args = parser.parse_args()
    
    db_path = f"/data2/xianglin/zombie_agent/{args.name}_db/msmarco"
    memory = RAGMemory(db_path, embedding_model="all-MiniLM-L6-v2", llm_model_name="z-ai/glm-4.7-flash")
    print(f"Base: {memory.count('base')}")

    trigger_docs = memory.trigger.get(where=None).get("documents")
    trigger_payload_count = 0
    for doc in trigger_docs:
        if _payload_flags(doc):
            trigger_payload_count += 1
    exposure_docs = memory.exposure.get(where=None).get("documents")
    exposure_payload_count = 0
    for doc in exposure_docs:
        if _payload_flags(doc):
            exposure_payload_count += 1
    print(f"Trigger payload count: {trigger_payload_count}/{len(trigger_docs)}")
    print(f"Exposure payload count: {exposure_payload_count}/{len(exposure_docs)}")


# Example:
# python -m src.memory.rag_status -n kimi
if __name__ == "__main__":
    main()