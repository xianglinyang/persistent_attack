"""
Global configuration for persistent attack experiments.

This module provides a centralized way to manage experiment-level hyperparameters
that need to be accessed across different modules, such as custom payload directories.

Thread-safe design:
- Each thread gets its own configuration context
- Multiple experiments can run in parallel without conflicts
- Process-level isolation is automatic (different processes have separate memory)
"""
from pathlib import Path
from typing import Optional
import threading
import os

# Thread-local storage for per-thread configuration
_thread_local = threading.local()

# Process-level default configuration (fallback)
_config_lock = threading.Lock()
_process_config = {
    "payload_dir": None,  # Process-level default
}


def set_payload_dir(payload_dir: Optional[str]) -> None:
    """
    Set the payload directory for the current thread/experiment.
    
    Thread-safe: Each thread can have its own payload directory.
    If running multiple experiments in parallel threads, each will use its own directory.
    
    Args:
        payload_dir: Path to the payload directory. If None, uses default.
    """
    if not hasattr(_thread_local, 'config'):
        _thread_local.config = {}
    
    _thread_local.config["payload_dir"] = Path(payload_dir) if payload_dir else None
    
    if payload_dir:
        thread_id = threading.get_ident()
        process_id = os.getpid()
        print(f"[Config] Payload directory set to: {payload_dir}")
        print(f"         (Process: {process_id}, Thread: {thread_id})")


def get_payload_dir() -> Optional[Path]:
    """
    Get the payload directory for the current thread/experiment.
    
    Returns the thread-local value if set, otherwise returns the process-level default.
    
    Returns:
        Path to payload directory, or None if using default.
    """
    # Try thread-local config first
    if hasattr(_thread_local, 'config') and "payload_dir" in _thread_local.config:
        return _thread_local.config["payload_dir"]
    
    # Fall back to process-level config
    with _config_lock:
        return _process_config["payload_dir"]


def set_process_default_payload_dir(payload_dir: Optional[str]) -> None:
    """
    Set the process-level default payload directory.
    
    This is used as a fallback when thread-local config is not set.
    Useful for single-threaded experiments.
    
    Args:
        payload_dir: Path to the payload directory. If None, uses default.
    """
    with _config_lock:
        _process_config["payload_dir"] = Path(payload_dir) if payload_dir else None
        if payload_dir:
            print(f"[Config] Process-level default payload directory set to: {payload_dir}")


def reset_config() -> None:
    """Reset the current thread's configuration to default values."""
    if hasattr(_thread_local, 'config'):
        _thread_local.config.clear()
    print("[Config] Thread-local configuration reset to defaults")


def reset_process_config() -> None:
    """Reset the process-level configuration to default values."""
    with _config_lock:
        _process_config["payload_dir"] = None
    print("[Config] Process-level configuration reset to defaults")


def get_config_summary() -> dict:
    """Get a summary of current configuration (thread-local + process-level)."""
    thread_local_dir = None
    if hasattr(_thread_local, 'config') and "payload_dir" in _thread_local.config:
        thread_local_dir = _thread_local.config["payload_dir"]
    
    with _config_lock:
        process_dir = _process_config["payload_dir"]
    
    return {
        "thread_local_payload_dir": str(thread_local_dir) if thread_local_dir else "not set",
        "process_default_payload_dir": str(process_dir) if process_dir else "not set",
        "effective_payload_dir": str(get_payload_dir()) if get_payload_dir() else "default",
        "process_id": os.getpid(),
        "thread_id": threading.get_ident(),
    }
