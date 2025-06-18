"""
Core parsing functionality for Olingo LLM Parser.

This module contains the main chat template parsing functionality using 
custom Jinja2 extensions.
"""

import json
from pathlib import Path
from jinja2 import Environment, Template
from jinja2.ext import Extension
from jinja2 import nodes
from jinja2.exceptions import TemplateError
from typing import List, Dict, Any, Optional, Tuple


class ChatExtension(Extension):
    """
    Custom Jinja2 extension to handle {% chat role="..." %} blocks.
    
    Usage:
    {% chat role="user" %}
    Your message content here
    {% endchat %}
    """
    
    tags = {'chat'}
    
    def __init__(self, environment):
        super().__init__(environment)
        if not hasattr(environment, '_chat_messages'):
            environment._chat_messages = []
    
    def parse(self, parser):
        """Parse the {% chat role="..." %} tag."""
        lineno = next(parser.stream).lineno
        
        # Parse role attribute
        role_value = None
        while parser.stream.current.test_any('name:role'):
            key = parser.stream.expect('name').value
            parser.stream.expect('assign')
            value = parser.parse_expression()
            if key == 'role':
                role_value = value
        
        if role_value is None:
            self.fail('chat tag requires role attribute', lineno)
        
        # Parse content until {% endchat %}
        content_nodes = parser.parse_statements(['name:endchat'], drop_needle=True)
        call_node = self.call_method('_handle_chat', [role_value])
        
        return nodes.CallBlock(call_node, [], [], content_nodes).set_lineno(lineno)
    
    def _handle_chat(self, role: str, caller) -> str:
        """Handle the chat block content and store the message."""
        content = caller().strip()
        self.environment._chat_messages.append({
            'role': role,
            'content': content
        })
        return ''


def parse_chat_template(template_path: str, variables: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """
    Parse a Jinja2 template file containing {% chat %} blocks and convert to message format.
    
    Args:
        template_path: Path to the Jinja template file
        variables: Dictionary of variables to pass to the template
        
    Returns:
        List of dictionaries with 'role' and 'content' keys
        
    Raises:
        FileNotFoundError: If template file doesn't exist
        TemplateError: If there are Jinja2 syntax or rendering errors
    """
    if variables is None:
        variables = {}
    
    if not Path(template_path).exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    try:
        # Create Jinja environment with custom extension
        env = Environment(extensions=[ChatExtension])
        
        # Read and render template
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        template = env.from_string(template_content)
        env._chat_messages = []
        template.render(**variables)
        
        return env._chat_messages.copy()
        
    except Exception as e:
        raise TemplateError(f"Failed to parse template '{template_path}': {e}") from e


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
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        return {
            "type": "json_schema",
            "json_schema": schema,
            "strict": True
        }
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in schema file '{schema_path}': {e}", e.doc, e.pos)


def parse_chat_and_schema(template_path: str, schema_path: Optional[str] = None, variables: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, str]], Optional[Dict[str, Any]]]:
    """
    Parse both chat template and JSON schema, returning both for AI API calls.
    The JSON schema is also made available in the template variables as 'json_schema'.
    
    Args:
        template_path: Path to the Jinja chat template file
        schema_path: Path to the JSON schema file (optional)
        variables: Dictionary of variables to pass to the template
        
    Returns:
        Tuple of (messages, response_format) ready for AI API calls
        
    Raises:
        FileNotFoundError: If template or schema file doesn't exist
        TemplateError: If there are Jinja2 syntax or rendering errors
        json.JSONDecodeError: If schema file is not valid JSON
    """
    if variables is None:
        variables = {}
    
    # First load the schema if provided
    response_format = None
    if schema_path:
        response_format = load_json_schema(schema_path)
    
    # Add the schema to variables so it's accessible in the template
    enhanced_variables = variables.copy()
    if response_format:
        enhanced_variables['json_schema'] = response_format['json_schema']
    
    # Then parse the template with the schema available as a variable
    messages = parse_chat_template(template_path, enhanced_variables)
    
    return messages, response_format 