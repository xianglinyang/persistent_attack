import random
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod


@dataclass
class Candidate:
    payload: str
    query: str
    stats: Dict[str, Any] = field(default_factory=dict)
    lineage_history: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    features: tuple = (0, 0)


class SearchStorage:
    def __init__(self):
        pass

    @abstractmethod
    def get_best(self) -> Optional[Candidate]:
        pass

    @abstractmethod
    def add(self, candidate: Candidate):
        pass

    @abstractmethod
    def get_current_candidates(self) -> List[Candidate]:
        pass
    
    @abstractmethod
    def get_best(self) -> Optional[Candidate]:
        pass



class AutoDanStorage(SearchStorage):
    """
    Implements a standard population container for Genetic Algorithms.
    """
    def __init__(self, max_population_size: int = 20):
        self.max_pop_size = max_population_size
        self.population: List[Candidate] = []
        self.archive: List[Candidate] = [] # Stores all historical high-scorers
        self.global_best: Optional[Candidate] = None

    def add_population(self, candidates: List[Candidate]):
        """
        Adds new offspring.
        Note: In GA, this usually happens in a batch at the end of a generation.
        """
        for c in candidates:
            self._update_best(c)
        
        self.population.extend(candidates)

    def selection(self, top_k: int = 5) -> List[Candidate]:
        """
        Returns the Elites (highest scores).
        Used to be parents for the next generation.
        """
        # Sort descending by score
        sorted_pop = sorted(self.population, key=lambda x: x.score, reverse=True)
        return sorted_pop[:top_k]

    def prune_population(self):
        """
        Keeps population within max_pop_size (Survival of the fittest).
        """
        # Sort descending
        sorted_pop = sorted(self.population, key=lambda x: x.score, reverse=True)
        # Keep top N
        self.population = sorted_pop[:self.max_pop_size]

    def _update_best(self, candidate):
        if self.global_best is None or candidate.score > self.global_best.score:
            self.global_best = candidate
        
        # Optional: Save to archive if score is decent
        if candidate.score > 5.0:
            self.archive.append(candidate)

    def get_best(self) -> Optional[Candidate]:
        return self.global_best


class TapStorage(SearchStorage):
    """
    Implements a Breadth-First Search (BFS) Queue for the TAP attack.
    It holds candidates that need to be expanded (branched).
    """
    def __init__(self):
        self.queue = deque()  # The frontier of the tree
        self.history = []     # Archive of all attempts (for logging/deduplication)
        self.best_candidate: Optional[Candidate] = None

    def add(self, candidate: Candidate):
        """
        Adds a candidate to the queue to be expanded later.
        TAP typically only adds candidates that pass the 'off-topic' filter.
        """
        # Update global best
        if self.best_candidate is None or candidate.score > self.best_candidate.score:
            self.best_candidate = candidate
            
        self.queue.append(candidate)
        self.history.append(candidate)

    def get_next(self) -> Optional[Candidate]:
        """
        Pop the next candidate to branch from (BFS order).
        """
        if not self.queue:
            return None
        return self.queue.popleft()

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def get_best(self) -> Optional[Candidate]:
        return self.best_candidate


class PairStorage(SearchStorage):
    """
    Implements parallel storage for PAIR.
    Maintains 'num_streams' independent lines of evolution.
    """
    def __init__(self):
        self.curr_best: Optional[Candidate] = None

    def add(self, candidate: Candidate):
        """
        PAIR Logic: 
        Compare candidate with the current one.
        If candidate is better, replace old. If not, keep old (or logic defined by controller).
        """
        if self.curr_best is None or candidate.score >= self.curr_best.score:
            self.curr_best = candidate

    def get_best(self) -> Optional[Candidate]:
        return self.curr_best
    
    def get_current_candidates(self) -> List[Candidate]:
        return [self.curr_best]


class MapElitesStorage(SearchStorage):
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
            candidate.features = self.get_feature_descriptor(candidate.payload)
            
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
