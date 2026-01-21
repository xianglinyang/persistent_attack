from src.memory.rag_memory import RAGMemory

def clean():
    pass

def count():
    pass

def reset(period: str):
    memory = RAGMemory(db_path, embedding_model="all-MiniLM-L6-v2", llm_model_name="google/gemini-2.5-flash")
    memory.reset(period)
    print(f"Reset {period} successfully")

if __name__ == "__main__":
    db_path = "/data2/xianglin/zombie_agent/gemini_db"
    memory = RAGMemory(db_path, embedding_model="all-MiniLM-L6-v2", llm_model_name="google/gemini-2.5-flash")
    print(f"Base: {memory.count('base')}")
    print(f"Exposure: {memory.count('exposure')}")
    print(f"Trigger: {memory.count('trigger')}")