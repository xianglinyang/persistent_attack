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

from src.config.page_registry import PageConfig, PAGES  # noqa: F401 – re-exported

# Thread-local storage for per-thread configuration
_thread_local = threading.local()

# Process-level default configuration (fallback)
_config_lock = threading.Lock()
_process_config = {
    "payload_dir": None,  # Process-level default
    "mock_topic": True,   # If True, include mock_topics() in read_malicious(); if False, only benign + payload
    "page_type": "telemedicine",  # Which attack page this process uses by default
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


def set_page_type(page_type: str) -> None:
    """
    Set the attack page type for the current thread/experiment.

    Args:
        page_type: One of the keys in PAGES (e.g. "telemedicine", "fintech", ...).
    """
    if page_type not in PAGES:
        raise ValueError(f"Unknown page_type '{page_type}'. Available: {list(PAGES.keys())}")
    if not hasattr(_thread_local, 'config'):
        _thread_local.config = {}
    _thread_local.config["page_type"] = page_type
    print(f"[Config] Page type set to: {page_type}")


def get_page_type() -> str:
    """
    Get the current page type.
    Returns thread-local value if set, otherwise process-level default ("telemedicine").
    """
    if hasattr(_thread_local, 'config') and "page_type" in _thread_local.config:
        return _thread_local.config["page_type"]
    with _config_lock:
        return _process_config["page_type"]


def get_current_page() -> PageConfig:
    """Return the PageConfig for the currently active page type."""
    return PAGES[get_page_type()]


def set_mock_topic(mock_topic: bool) -> None:
    """
    Set whether to include mock_topics() in read_malicious() for the current thread/experiment.
    If True: mock_topics() + mock_benign_webpage_content() + mock_malicious_payload()
    If False: mock_benign_webpage_content() + mock_malicious_payload()
    """
    if not hasattr(_thread_local, 'config'):
        _thread_local.config = {}
    _thread_local.config["mock_topic"] = mock_topic


def get_mock_topic() -> bool:
    """
    Get whether to include mock_topics() in read_malicious().
    Returns thread-local value if set, otherwise process-level default (True).
    """
    if hasattr(_thread_local, 'config') and "mock_topic" in _thread_local.config:
        return _thread_local.config["mock_topic"]
    with _config_lock:
        return _process_config["mock_topic"]


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
        _process_config["mock_topic"] = True
        _process_config["page_type"] = "telemedicine"
    print("[Config] Process-level configuration reset to defaults")


def get_config_summary() -> dict:
    """Get a summary of current configuration (thread-local + process-level)."""
    thread_local_dir = None
    if hasattr(_thread_local, 'config') and "payload_dir" in _thread_local.config:
        thread_local_dir = _thread_local.config["payload_dir"]
    mock_topic = get_mock_topic()
    page_type = get_page_type()
    with _config_lock:
        process_dir = _process_config["payload_dir"]
    return {
        "thread_local_payload_dir": str(thread_local_dir) if thread_local_dir else "not set",
        "process_default_payload_dir": str(process_dir) if process_dir else "not set",
        "effective_payload_dir": str(get_payload_dir()) if get_payload_dir() else "default",
        "mock_topic": mock_topic,
        "page_type": page_type,
        "process_id": os.getpid(),
        "thread_id": threading.get_ident(),
    }
