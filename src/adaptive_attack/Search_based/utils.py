from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# ==========================================
# Helper Functions
# ==========================================
# --- Helper Functions ---
def extract_guard_feedback(logs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract guard feedback from agent logs.
    
    Args:
        logs: Agent execution logs containing steps and observations
    
    Returns:
        Dict with guard_blocked, category, explanation
    """
    guard_info = {
        "guard_blocked": False,
        "category": "benign",
        "explanation": "",
    }
    
    # Check if guard blocked in any step
    for step in logs.get("steps", []):
        for obs in step.get("observations", []):
            if isinstance(obs, str) and "[GUARD BLOCKED]" in obs:
                guard_info["guard_blocked"] = True
                # Parse the observation to extract category and explanation
                # Format: "[GUARD BLOCKED] ... [Type]: {category} ... [Reason]: {explanation}"
                try:
                    # Handle both "[Type]:" and " [Type]:" (with leading space)
                    if "[Type]:" in obs:
                        type_parts = obs.split("[Type]:")
                        if len(type_parts) > 1:
                            # Extract text between [Type]: and [Reason]:
                            if "[Reason]:" in type_parts[1]:
                                category_part = type_parts[1].split("[Reason]:")[0].strip()
                            else:
                                category_part = type_parts[1].strip()
                            guard_info["category"] = category_part
                    
                    if "[Reason]:" in obs:
                        reason_parts = obs.split("[Reason]:")
                        if len(reason_parts) > 1:
                            explanation_part = reason_parts[1].strip()
                            guard_info["explanation"] = explanation_part
                except Exception as e:
                    logger.warning(f"Failed to parse guard feedback from observation: {e}")
                    pass
                break
        if guard_info["guard_blocked"]:
            break
    
    return guard_info