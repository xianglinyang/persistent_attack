"""
Progent integration for tool-level privilege control.

Progent (https://github.com/sunblaze-ucb/progent) enforces security policies
on tool calls at runtime, blocking tool invocations that violate the policy.

Two modes:
  - "static": uses a fixed policy (forbids execute_command, allows search/read)
  - "dynamic": generates a per-task policy from the user query via OpenRouter.
               Uses dynamic_policy.py (adapted from secagent) so the model is
               fully configurable without patching the secagent library.

Install: pip install git+https://github.com/sunblaze-ucb/progent.git
"""

import logging
from typing import Optional, Tuple

from src.guard.system_level.progent.dynamic_policy import generate_and_apply_dynamic_policy

logger = logging.getLogger(__name__)

# Tool definitions matching our agent's tools (in Progent's Tool TypedDict format)
TOOL_DEFINITIONS = [
    {
        "name": "search",
        "description": "Search the web with a query and return snippets and URLs.",
        "args": {
            "query": {"type": "string", "description": "The search query"},
        },
    },
    {
        "name": "read",
        "description": "Read the content of a web page at the given URL.",
        "args": {
            "url": {"type": "string", "description": "The URL to read"},
        },
    },
    {
        "name": "execute_command",
        "description": "Execute a shell command on the system.",
        "args": {
            "command": {"type": "string", "description": "The shell command to execute"},
        },
    },
]

# Static policy: a web agent answering user questions has no need to execute commands.
# Policy tuple format: (priority, effect, condition_dict, fallback_mode)
#   effect 0 = ALLOW, 1 = FORBID
#   fallback_mode 0 = return error message
STATIC_POLICY = {
    "search": [(1, 0, {}, 0)],           # ALLOW all search calls
    "read": [(1, 0, {}, 0)],             # ALLOW all read calls
    "execute_command": [(1, 1, {}, 0)],  # FORBID all execute_command calls
}


class ProgentGuard:
    """
    Wraps Progent's policy enforcement for use in our agent.

    Args:
        mode:       "static" for fixed policy, "dynamic" for LLM-generated per-task policy.
        model_name: OpenRouter model used in dynamic mode (e.g. "openai/gpt-4o").
                    Ignored in static mode.
    """

    def __init__(self, mode: str = "static", model_name: Optional[str] = None):
        assert mode in ("static", "dynamic"), f"Invalid progent mode: {mode}"
        self.mode = mode
        self.model_name = model_name or "openai/gpt-4o"
        self._initialized = False
        self._load_secagent()

    def _load_secagent(self):
        try:
            from secagent import (
                update_available_tools,
                update_security_policy,
                reset_security_policy,
                generate_security_policy,
                check_tool_call,
            )
            from secagent.tool import ValidationError

            self._update_available_tools = update_available_tools
            self._update_security_policy = update_security_policy
            self._reset_security_policy = reset_security_policy
            self._generate_security_policy = generate_security_policy
            self._check_tool_call = check_tool_call
            self._ValidationError = ValidationError

            self._update_available_tools(TOOL_DEFINITIONS)
            self._initialized = True
            logger.info("[ProgentGuard] secagent loaded successfully.")
        except ImportError as e:
            logger.error(
                f"[ProgentGuard] Failed to import secagent: {e}. "
                "Install with: pip install git+https://github.com/sunblaze-ucb/progent.git"
            )
            self._initialized = False

    def init_for_task(self, user_query: str):
        """
        Call at the start of each task to reset and configure the policy.
        In static mode: apply fixed STATIC_POLICY.
        In dynamic mode: use Progent's LLM to generate policy from user_query.
        """
        if not self._initialized:
            return

        self._reset_security_policy()

        if self.mode == "static":
            self._update_security_policy(STATIC_POLICY)
            logger.info("[ProgentGuard] Static policy applied.")
        elif self.mode == "dynamic":
            logger.info(
                f"[ProgentGuard] Generating dynamic policy via {self.model_name} "
                f"for query: {user_query[:80]}..."
            )
            try:
                generate_and_apply_dynamic_policy(
                    user_query=user_query,
                    tool_definitions=TOOL_DEFINITIONS,
                    model_name=self.model_name,
                    update_security_policy_fn=self._update_security_policy,
                    reset_security_policy_fn=self._reset_security_policy,
                )
                logger.info("[ProgentGuard] Dynamic policy applied.")
            except Exception as e:
                logger.warning(
                    f"[ProgentGuard] Dynamic policy generation failed: {e}. Falling back to static."
                )
                self._reset_security_policy()
                self._update_security_policy(STATIC_POLICY)

    def check(self, tool_name: str, action: dict) -> Tuple[bool, str]:
        """
        Validate a tool call against the current policy.

        Args:
            tool_name: Name of the tool being called.
            action: The full action dict from the agent (includes "action" key).

        Returns:
            (allowed, reason): True if the call is permitted, False with reason if blocked.
        """
        if not self._initialized:
            return True, ""

        # Strip the "action" key — Progent only needs the tool arguments
        kwargs = {k: v for k, v in action.items() if k != "action"}

        try:
            self._check_tool_call(tool_name, kwargs)
            logger.info(f"[ProgentGuard] Tool '{tool_name}' ALLOWED by policy.")
            return True, ""
        except self._ValidationError as e:
            reason = str(e)
            logger.info(f"[ProgentGuard] Tool '{tool_name}' BLOCKED by policy. Reason: {reason}")
            return False, reason
        except Exception as e:
            # On unexpected errors, allow the call (fail open) to avoid false positives
            logger.warning(f"[ProgentGuard] Unexpected error during check for '{tool_name}': {e}. Allowing.")
            return True, ""
