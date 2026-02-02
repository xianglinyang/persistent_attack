from src.memory.rag_memory import RAGMemory
from src.evaluate.attack_evaluator import _payload_flags


if __name__ == "__main__":
    db_path = "/data2/xianglin/zombie_agent/glm_db/msmarco"
    memory = RAGMemory(db_path, embedding_model="all-MiniLM-L6-v2", llm_model_name="z-ai/glm-4.7-flash")
    print(f"Base: {memory.count('base')}")
    print(f"Exposure: {memory.count('exposure')}")
    print(f"Trigger: {memory.count('trigger')}")

    # count how many docs contain payload
    # iterate over all docs in trigger collection
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