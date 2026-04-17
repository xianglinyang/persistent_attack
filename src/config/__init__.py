"""Configuration package for persistent attack experiments."""
from src.config.global_config import (
    set_payload_dir,
    get_payload_dir,
    set_process_default_payload_dir,
    set_mock_topic,
    get_mock_topic,
    set_page_type,
    get_page_type,
    get_current_page,
    reset_config,
    reset_process_config,
    get_config_summary,
)
from src.config.page_registry import PageConfig, PAGES

__all__ = [
    "set_payload_dir",
    "get_payload_dir",
    "set_process_default_payload_dir",
    "set_mock_topic",
    "get_mock_topic",
    "set_page_type",
    "get_page_type",
    "get_current_page",
    "PageConfig",
    "PAGES",
    "reset_config",
    "reset_process_config",
    "get_config_summary",
]
