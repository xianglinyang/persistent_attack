"""Tool server for handling tool calls from agents."""
import logging
from typing import Dict, Callable, Any, Optional

from src.tools.tools import search, read, execute_command

logger = logging.getLogger(__name__)


class ToolServer:
    """
    A server that manages and executes tools for agents.
    
    Supports registering custom tools and executing them based on action names.
    """
    
    def __init__(self):
        """Initialize the tool server with default tools."""
        self.tools: Dict[str, Callable] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register the default set of tools."""
        self.register_tool("search", search, required_params=["query"])
        self.register_tool("read", read, required_params=["url"])
        self.register_tool("execute_command", execute_command, required_params=["command"])
    
    def register_tool(
        self, 
        name: str, 
        func: Callable, 
        required_params: Optional[list] = None
    ):
        """
        Register a new tool.
        
        Args:
            name: Name of the tool/action
            func: The function to execute
            required_params: List of required parameter names
        """
        self.tools[name] = {
            "func": func,
            "required_params": required_params or []
        }
        logger.info(f"Registered tool: {name}")
    
    def execute(self, action: str, params: Dict[str, Any]) -> str:
        """
        Execute a tool based on action name and parameters.
        
        Args:
            action: The action/tool name to execute
            params: Dictionary of parameters for the tool
            
        Returns:
            Result string from the tool execution
        """
        if action not in self.tools:
            error_msg = f"Unknown action '{action}'. Available tools: {list(self.tools.keys())}"
            logger.error(error_msg)
            return f"[ERROR] {error_msg}"
        
        tool_info = self.tools[action]
        func = tool_info["func"]
        required_params = tool_info["required_params"]
        
        # Validate required parameters
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            error_msg = f"Missing required parameters for '{action}': {missing_params}"
            logger.error(error_msg)
            return f"[ERROR] {error_msg}"
        
        try:
            # Extract only the parameters the function needs
            func_params = {k: params[k] for k in required_params if k in params}
            
            logger.info(f"Executing tool '{action}' with params: {func_params}")
            result = func(**func_params)
            
            # Convert result to string if it's not already
            if not isinstance(result, str):
                result = str(result)
            
            logger.info(f"Tool '{action}' executed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Error executing tool '{action}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"[ERROR] {error_msg}"
    
    def get_available_tools(self) -> list:
        """Return a list of available tool names."""
        return list(self.tools.keys())
    
    def get_tool_info(self, action: str) -> Optional[Dict]:
        """Get information about a specific tool."""
        return self.tools.get(action)


# Singleton instance for easy access
_tool_server_instance = None

def get_tool_server() -> ToolServer:
    """Get or create the singleton ToolServer instance."""
    global _tool_server_instance
    if _tool_server_instance is None:
        _tool_server_instance = ToolServer()
    return _tool_server_instance


if __name__ == "__main__":
    # Example usage
    server = ToolServer()
    
    print("Available tools:", server.get_available_tools())
    print("\n" + "="*50 + "\n")
    
    # Test search
    result = server.execute("search", {"query": "Python programming"})
    print(f"Search result: {result[:200]}...")
    print("\n" + "="*50 + "\n")
    
    # Test read
    result = server.execute("read", {"url": "https://www.python.org"})
    print(f"Read result: {result[:200]}...")
    print("\n" + "="*50 + "\n")
    
    # Test execute_command
    result = server.execute("execute_command", {"command": "ls -la"})
    print(f"Command result: {result}")
    print("\n" + "="*50 + "\n")
    
    # Test error handling
    result = server.execute("unknown_tool", {})
    print(f"Error result: {result}")
