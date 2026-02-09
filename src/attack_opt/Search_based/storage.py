import random
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field

@dataclass
class Candidate:
    payload: str
    query: str
    stats: Dict[str, Any] = field(default_factory=dict)
    lineage_history: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    features: tuple = (0, 0)

    @property
    def text(self):
        """Backwards compatibility helper"""
        return f"{self.query} | {self.payload}"


class MapElitesStorage:
    """
    Implements the Grid-like storage described in the text.
    It partitions the search space based on characteristics (Length, Diversity).
    """
    def __init__(self):
        # The grid: Key=(feature_x, feature_y), Value=Candidate (Elite)
        self.grid: Dict[Tuple[int, int], Candidate] = {}
        self.history: List[Candidate] = [] # Keep all for random sampling

    def get_feature_descriptor(self, text: str) -> Tuple[int, int]:
        """
        Quantizes properties of the candidate.
        Dimension 1: Length (quantized by 10 chars)
        Dimension 2: Diversity (Mocked here using hash/random for demo)
        """
        # Feature 1: Length bucket
        f1 = len(text) // 10
        
        # Feature 2: Diversity/Complexity (In real impl, use embedding distance or entropy)
        # Here we mock it simply for the code to run
        f2 = sum(ord(c) for c in text) % 10 
        
        return (f1, f2)

    def add(self, candidate: Candidate):
        """
        Tries to add a candidate to the grid.
        Replaces existing occupant only if the new candidate has a higher score.
        """
        # Calculate features if not already set
        if candidate.features == (0, 0):
            candidate.features = self.get_feature_descriptor(candidate.text)
            
        key = candidate.features
        
        # MAP-Elites Logic:
        # If cell is empty, populate it.
        # If cell is occupied, replace only if new score is better.
        if key not in self.grid:
            self.grid[key] = candidate
        else:
            if candidate.score > self.grid[key].score:
                self.grid[key] = candidate
        
        self.history.append(candidate)

    def sample_parents(self, k: int = 8) -> List[Candidate]:
        """
        Samples a subset of triggers to provide to the Mutator.
        The text implies a mix of elites and random history.
        """
        elites = list(self.grid.values())
        
        # Strategy: Mix of current elites and random historical attempts
        selection_pool = elites + self.history[-20:] # Bias towards elites + recent
        
        if not selection_pool:
            return []
            
        # Sample k candidates (with replacement allowed if pool is small)
        return [random.choice(selection_pool) for _ in range(k)]

    def get_best(self) -> Optional[Candidate]:
        if not self.grid:
            return None
        return max(self.grid.values(), key=lambda x: x.score)
