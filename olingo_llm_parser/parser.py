"""
Core parsing functionality for Olingo LLM Parser.

This module contains the main chat template parsing functionality using 
custom Jinja2 extensions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from jinja2 import Environment, Template, nodes
from jinja2.exceptions import TemplateError
from jinja2.ext import Extension


class ChatExtension(Extension):
    """
    Custom Jinja2 extension to handle {% chat role="..." %} blocks.

    Usage:
    {% chat role="user" %}
    Your message content here
    {% endchat %}
    """

    tags = {"chat"}

    def __init__(self, environment):
        super().__init__(environment)
        if not hasattr(environment, "_chat_messages"):
            environment._chat_messages = []

    def parse(self, parser):
        """Parse the {% chat role="..." %} tag."""
        lineno = next(parser.stream).lineno

        # Parse role attribute
        role_value = None
        while parser.stream.current.test_any("name:role"):
            key = parser.stream.expect("name").value
            parser.stream.expect("assign")
            value = parser.parse_expression()
            if key == "role":
                role_value = value

        if role_value is None:
            self.fail("chat tag requires role attribute", lineno)

        # Parse content until {% endchat %}
        content_nodes = parser.parse_statements(["name:endchat"], drop_needle=True)
        call_node = self.call_method("_handle_chat", [role_value])

        return nodes.CallBlock(call_node, [], [], content_nodes).set_lineno(lineno)

    def _handle_chat(self, role: str, caller) -> str:
        """Handle the chat block content and store the message."""
        content = caller().strip()
        self.environment._chat_messages.append({"role": role, "content": content})
        return ""


def _load_template_content(template: Union[str, Path]) -> str:
    """
    Load template content from either a file path or direct string.

    Args:
        template: Either a file path or template string content

    Returns:
        Template content as string

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    template_str = str(template)

    # Check if it's a file path (has extension or exists as file)
    if (
        template_str.endswith(".jinja")
        or template_str.endswith(".jinja2")
        or Path(template_str).exists()
    ):
        if not Path(template_str).exists():
            raise FileNotFoundError(f"Template file not found: {template_str}")

        with open(template_str, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # Treat as template content string
        return template_str


def parse_chat_template(
    template: Union[str, Path], variables: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """
    Parse a Jinja2 template containing {% chat %} blocks and convert to message format.

    Args:
        template: Either path to the Jinja template file or template string content
        variables: Dictionary of variables to pass to the template

    Returns:
        List of dictionaries with 'role' and 'content' keys

    Raises:
        FileNotFoundError: If template file doesn't exist
        TemplateError: If there are Jinja2 syntax or rendering errors
    """
    if variables is None:
        variables = {}

    try:
        # Create Jinja environment with custom extension
        env = Environment(extensions=[ChatExtension])

        # Load template content
        template_content = _load_template_content(template)

        template_obj = env.from_string(template_content)
        env._chat_messages = []
        template_obj.render(**variables)

        return env._chat_messages.copy()

    except FileNotFoundError:
        # Re-raise FileNotFoundError directly for backward compatibility
        raise
    except Exception as e:
        raise TemplateError(f"Failed to parse template: {e}") from e


def _load_schema(schema: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load schema from either a file path, JSON string, or dictionary.

    Args:
        schema: Either a file path, JSON string content, or schema dictionary

    Returns:
        Schema dictionary

    Raises:
        FileNotFoundError: If schema file doesn't exist
        json.JSONDecodeError: If schema string/file is not valid JSON
    """
    if isinstance(schema, dict):
        return schema

    schema_str = str(schema)

    # Check if it's a file path (has .json extension or exists as file)
    if schema_str.endswith(".json") or Path(schema_str).exists():
        if not Path(schema_str).exists():
            raise FileNotFoundError(f"Schema file not found: {schema_str}")

        with open(schema_str, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Treat as JSON string
        try:
            return json.loads(schema_str)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in schema string: {e}", e.doc, e.pos)


def load_json_schema(schema_path: str) -> Dict[str, Any]:
    """
    Load a JSON schema file and format it for OpenAI response_format.

    Args:
        schema_path: Path to the JSON schema file

    Returns:
        Dictionary in OpenAI response_format structure

    Raises:
        FileNotFoundError: If schema file doesn't exist
        json.JSONDecodeError: If schema file is not valid JSON
    """
    if not Path(schema_path).exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        return {
            "type": "json_schema",
            "json_schema": {"schema": schema, "strict": True},
        }

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in schema file '{schema_path}': {e}", e.doc, e.pos
        )


def parse_template_and_schema(
    template: Union[str, Path],
    schema: Optional[Union[str, Path, Dict[str, Any]]] = None,
    variables: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, str]], Optional[Dict[str, Any]]]:
    """
    Parse both chat template and JSON schema, returning both for AI API calls.
    The JSON schema is also made available in the template variables as 'json_schema'.

    Args:
        template: Either path to the Jinja chat template file or template string content
        schema: Either path to JSON schema file, JSON string content, or schema dictionary (optional)
        variables: Dictionary of variables to pass to the template

    Returns:
        Tuple of (messages, response_format) ready for AI API calls

    Raises:
        FileNotFoundError: If template or schema file doesn't exist
        TemplateError: If there are Jinja2 syntax or rendering errors
        json.JSONDecodeError: If schema file/string is not valid JSON
    """
    if variables is None:
        variables = {}

    # First load the schema if provided
    response_format = None
    if schema is not None:
        schema_dict = _load_schema(schema)
        response_format = {"type": "json_schema", "json_schema": schema_dict, "strict": True}

    # Add the schema to variables so it's accessible in the template
    enhanced_variables = variables.copy()
    if response_format:
        enhanced_variables["json_schema"] = response_format["json_schema"]

    # Then parse the template with the schema available as a variable
    messages = parse_chat_template(template, enhanced_variables)

    return messages, response_format
