'''
Memory Mechanism for Zombie Agent
1. Sliding window based
2. RAG based

Memory Evolution Mechanism:
1. raw content
2. reflection
3. experience
'''
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class MemoryBase(ABC):
    def __init__(self):
        pass
        
    @abstractmethod
    def add_memory(self):
        pass

    @abstractmethod
    def retrieve(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def evolve(self):
        pass
