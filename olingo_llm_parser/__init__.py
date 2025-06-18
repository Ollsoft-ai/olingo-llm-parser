"""
Olingo LLM Parser - A elegant Jinja2-based chat template parser for AI applications.

This package provides functionality to parse Jinja2 templates containing custom
{% chat role="..." %}...{% endchat %} blocks and convert them to message format
suitable for AI chat APIs like OpenAI, Anthropic, etc.
"""

from .parser import (
    parse_chat_template,
    load_json_schema,
    parse_chat_and_schema,
    ChatExtension
)

__version__ = "0.1.0"
__author__ = "Ollsoft Team"
__email__ = "info@ollsoft.ai"
__description__ = "Elegant Jinja2-based chat template parser for AI applications"

__all__ = [
    "parse_chat_template",
    "load_json_schema", 
    "parse_chat_and_schema",
    "ChatExtension"
] 