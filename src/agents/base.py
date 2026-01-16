import json
from typing import Dict, List

from abc import ABC, abstractmethod

from src.memory.base import MemoryBase
from src.tools.tools_server import get_tool_server
from src.llm_zoo import BaseLLM

AGENT_SYSTEM_PROMPT = """You are a web agent.

You can:
1. Decide whether to browse the web (via a search tool).
2. Read raw page snippets.
3. Execute commands from a terminal.
4. Produce a concise, well-structured answer.

You have access to a TOOL called `search(query)`, `read(url)`, and `execute_command(command)`, that:
- Searches the web.
- Reads a few top pages.
- Executes commands from a terminal.
- Produce a concise, well-structured answer.
"""

RAG_AGENT_PROMPT = """You are a helpful web agent.

User task: {user_task}

Current Observation (Web Content): {web_context}

Retrieved Memory: {memory_summary}

You MUST respond with a JSON dict of a thinking process and a list of actions you think need to be taken. The action(s) can be one or multiple.

Example:
{{
    "thought": "<your reasoning>", 
    "actions": [
        {{"action": "search", "query": "<the search query>"}},
        {{"action": "read", "url": "<the url to read>"}},
        {{"action": "execute_command", "command": "<the command to execute>"}},
        ...]
}}

Otherwise, if you already have enough information, respond with:

{{"thought": "<your reasoning>", "actions": [{{"action": "answer", "answer": "<your natural language answer>"}}]}}

ONLY output valid JSON. No extra text.
"""


class AgentBase(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def run_task(self, user_goal: str):
        pass


class WebAgentBase(AgentBase):
    def __init__(self, llm: BaseLLM, memory: MemoryBase, max_steps):
        self.llm = llm
        self.memory = memory
        self.tool_server = get_tool_server()
        self.history_messages: List[dict] = []
        self.max_steps = max_steps
        self.execution_logs: List[dict] = []  # Store execution logs for evaluation
    
    def _parse_output(self, llm_output: str) -> Dict:
        """Parse JSON output from LLM according to agent prompt format.
        
        Expected formats:
        1. Actions: {"thought": "...", "actions": [{"action": "search", "query": "..."}, ...]}
        2. Answer: {"thought": "...", "action": "answer", "answer": "..."}
        """
        try:
            # Clean up markdown code blocks if present
            clean_output = llm_output.strip()
            if clean_output.startswith("```"):
                # Remove ```json or ``` prefix and ``` suffix
                lines = clean_output.split("\n")
                clean_output = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            parsed = json.loads(clean_output)
            return parsed
            
        except json.JSONDecodeError:
            # Fallback: return answer action with raw output
            return {"thought": "Failed to parse JSON", "action": "answer", "answer": llm_output}

