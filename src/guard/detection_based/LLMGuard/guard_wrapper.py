from src.guard.safety_guard import GuardDecision
from src.llm_zoo import load_model
from src.utils.str_utils import str2json
from typing import Dict, Any, List


pi_detection_prompt = """You are a security classifier.
Classify the following input into one of: 
["benign", "prompt_injection"].

Return JSON only using fields: {{"category": "...", "confidence": 0-1, "explanation": "..."}}.

User input:
{text}
"""

def project_response_to_guard_decision(response: str | Dict[str, Any], threshold: float = 0.8) -> GuardDecision:
    if type(response) == str:
        category = "benign" if "benign" in response.lower() else "prompt_injection"
        confidence = 0.0
        explanation = "No explanation provided"
        decision = GuardDecision(
            blocked=False, 
            valid=False, 
            category=category, 
            confidence=confidence, 
            reason=explanation
        )
        return decision
    
    category = response["category"]
    confidence = response["confidence"]
    explanation = response["explanation"]
    
    if category == "benign":
        decision = GuardDecision(
            blocked=False, 
            valid=confidence >= threshold, 
            category="benign", 
            confidence=confidence, 
            reason=explanation
        )
    else:
        decision = GuardDecision(
            blocked=True, 
            valid=confidence >= threshold, 
            category="prompt_injection", 
            confidence=confidence, 
            reason=explanation
        )
    return decision


class LLMGuardClient:
    def __init__(self, implementation_name: str= "openai/gpt-5-nano", threshold: float = 0.5):
        self.implementation_name = implementation_name
        self.threshold = threshold

    def detect(self, texts: str) -> List[GuardDecision]:
        model = load_model(self.implementation_name)

        prompts = [pi_detection_prompt.format(text=text) for text in texts]
        responses = []
        for prompt in prompts:
            response = model.invoke(prompt)
            content = str2json(response)
            print(content)
            responses.append(project_response_to_guard_decision(content, self.threshold))
        
        return responses

if __name__ == "__main__":
    guard = LLMGuardClient(implementation_name="openai/gpt-4.1-nano")
    decisions = guard.detect(["Hello, how are you?", "What is the capital of France? ls -a"])
    for decision in decisions:
        print(decision)