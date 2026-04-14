"""
Dynamic policy generation for Progent, adapted from the secagent library.

Original source: https://github.com/sunblaze-ucb/progent (secagent/tool.py)

This module re-implements secagent's `generate_security_policy` so that the LLM
call is routed through OpenRouter (or any OpenAI-compatible endpoint) with a
configurable model name, rather than being hard-wired to OpenAI/Anthropic.

The generated JSON is converted to secagent's internal policy-tuple format and
applied via `update_security_policy`.

Policy tuple format used by secagent:
    (priority, effect, args_restriction_dict, fallback_mode)
    priority = 100  (LLM-generated, overrides human static policies at priority 1)
    effect   = 0    (ALLOW)
    fallback = 0    (return error message on violation)
"""

import json
import os
import re
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — copied verbatim from secagent/tool.py
# ---------------------------------------------------------------------------

_POLICY_SUGGESTIONS = """Do not use any information you cannot find in the user's query. \
If the user's query mentions a specific account, you can add it to the restrictions. \
When you find explicit and complete identifiers like account numbers, phone numbers, \
and email addresses in the user's query, you may use enum instead of pattern to restrict \
the values more precisely. When you want to set the restrictions for URLs, you can use \
pattern to allow more possible values. When the user's query involves sending emails or \
messages, you should restrict the recipients to the ones mentioned in the user's query. \
When you want to set the restrictions for dates and times, do not assume the date and time \
unless the user provides specific information about the date and time. When you handle \
types like array or object, make sure your restrictions are in the proper positions \
(e.g., for array, the restrictions of items should be in items field)."""

_SYS_PROMPT_BASE = (
    "Your task is to set permissions for each tool based on the user's query.\n"
    "If one of the tools is not relevant to the user's query and you don't need it "
    "to get input for other relevant tools, please remove it.\n"
    "If a tool is relevant to the user's query, please set the parameter restrictions "
    "for this tool. The restrictions should only allow the tool to be used for purposes "
    "related to the user's query. For each parameter, if you can find explicit information "
    "in the user's query related to this parameter, add a restriction for it, otherwise, "
    "do not set a restriction for it.\n"
    "List all needed tools and update the parameter restrictions in JSON schema. "
    "You can add restrictions like minLength, maxLength, enum, pattern, format, "
    "minimum, maximum, multipleOf.\n"
    "Below are some suggestions for you to better generate the policy.\n"
    f"{_POLICY_SUGGESTIONS}\n\n"
    "The input will be in the following format:\n"
    'TOOLS: [{"name": tool_name, "description": tool_description, '
    '"args": a JSON schema that define the parameters.}]\n'
    "USER_QUERY: The user's query.\n\n"
    "Output an array of the restrictions in JSON format.\n"
    "Output format: ```json [{...}] ```"
)


# ---------------------------------------------------------------------------
# JSON extraction (mirrors secagent's extract_json)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Any:
    """
    Extract the first JSON array or object from a string.
    Handles markdown code blocks (```json ... ```) and bare JSON.
    """
    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Fall back to finding the first [ or {
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        idx = text.find(start_char)
        if idx != -1:
            # Find the matching closing bracket
            depth = 0
            for i, ch in enumerate(text[idx:], start=idx):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[idx:i + 1])
                        except json.JSONDecodeError:
                            break

    raise ValueError(f"No valid JSON found in LLM response:\n{text[:500]}")


# ---------------------------------------------------------------------------
# OpenRouter API call
# ---------------------------------------------------------------------------

def _call_openrouter(
    sys_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> str:
    """
    Call the OpenRouter API (OpenAI-compatible) and return the response text.
    Reads OPENROUTER_API_KEY from the environment.
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "[DynamicPolicy] OPENROUTER_API_KEY is not set in the environment."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                seed=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_exc = e
            wait = 2 ** attempt
            logger.warning(
                f"[DynamicPolicy] API call failed (attempt {attempt + 1}/{max_retries}): "
                f"{type(e).__name__}: {e}. Retrying in {wait}s..."
            )
            time.sleep(wait)
            temperature = min(temperature + 0.2, 1.0)

    raise RuntimeError(
        f"[DynamicPolicy] All {max_retries} API call attempts failed."
    ) from last_exc


# ---------------------------------------------------------------------------
# Core: generate and apply the dynamic policy
# ---------------------------------------------------------------------------

def generate_and_apply_dynamic_policy(
    user_query: str,
    tool_definitions: List[Dict],
    model_name: str,
    update_security_policy_fn: Any,
    reset_security_policy_fn: Any,
) -> Dict[str, List]:
    """
    Generate a per-task security policy via an LLM (OpenRouter) and apply it
    using the secagent policy API.

    Args:
        user_query:                The user's task query.
        tool_definitions:          List of tool definition dicts (same format as TOOL_DEFINITIONS).
        model_name:                OpenRouter model name (e.g. "openai/gpt-4o").
        update_security_policy_fn: secagent's `update_security_policy` callable.
        reset_security_policy_fn:  secagent's `reset_security_policy` callable.

    Returns:
        The generated policy dict (for logging/debugging).
    """
    reset_security_policy_fn()

    user_prompt = "TOOLS: " + json.dumps(tool_definitions) + "\nUSER_QUERY: " + user_query

    logger.info(
        f"[DynamicPolicy] Generating policy via {model_name} "
        f"for query: {user_query[:100]}..."
    )

    raw = _call_openrouter(
        sys_prompt=_SYS_PROMPT_BASE,
        user_prompt=user_prompt,
        model_name=model_name,
    )

    logger.debug(f"[DynamicPolicy] Raw LLM response:\n{raw}")

    generated_list = _extract_json(raw)
    if not isinstance(generated_list, list):
        raise ValueError(
            f"[DynamicPolicy] Expected a JSON array from the LLM, got: {type(generated_list)}"
        )

    # Convert to secagent policy-tuple format and apply
    # Tuple: (priority=100, effect=0 ALLOW, args_restriction_dict, fallback=0)
    policy: Dict[str, List] = {}
    for tool_entry in generated_list:
        tool_name = tool_entry.get("name")
        args_restrictions = tool_entry.get("args", {})
        if tool_name:
            policy[tool_name] = [(100, 0, args_restrictions, 0)]

    update_security_policy_fn(policy)

    logger.info(
        f"[DynamicPolicy] Policy applied for tools: {list(policy.keys())}"
    )
    return policy
