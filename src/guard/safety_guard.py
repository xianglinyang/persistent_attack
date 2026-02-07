from dataclasses import dataclass

@dataclass
class GuardDecision:
    blocked: bool
    valid: bool
    category: str
    confidence: float
    reason: str