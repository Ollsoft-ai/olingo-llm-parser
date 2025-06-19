"""
Unit tests for Olingo LLM Parser.
"""

import json
import tempfile
from pathlib import Path

import pytest

from olingo_llm_parser import load_json_schema, parse_chat_template, parse_template_and_schema


class TestChatTemplateParser:
    """Test cases for chat template parsing."""

    def test_basic_parsing(self):
        """Test basic template parsing."""
        template_content = """{% chat role="system" %}
You are helpful.
{% endchat %}

{% chat role="user" %}
Hello!
{% endchat %}"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write(template_content)
            temp_path = f.name

        try:
            messages = parse_chat_template(temp_path)
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are helpful."
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "Hello!"
        finally:
            Path(temp_path).unlink()

    def test_variable_substitution(self):
        """Test template variable substitution."""
        template_content = """{% chat role="user" %}
Hello {{ name }}!
{% endchat %}"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write(template_content)
            temp_path = f.name

        try:
            messages = parse_chat_template(temp_path, {"name": "Alice"})
            assert len(messages) == 1
            assert messages[0]["content"] == "Hello Alice!"
        finally:
            Path(temp_path).unlink()

    def test_loops(self):
        """Test template loops."""
        template_content = """{% for item in items %}
{% chat role="user" %}
Item: {{ item }}
{% endchat %}
{% endfor %}"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write(template_content)
            temp_path = f.name

        try:
            messages = parse_chat_template(temp_path, {"items": ["A", "B"]})
            assert len(messages) == 2
            assert messages[0]["content"] == "Item: A"
            assert messages[1]["content"] == "Item: B"
        finally:
            Path(temp_path).unlink()

    def test_file_not_found(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            parse_chat_template("nonexistent.jinja")


class TestSchemaLoader:
    """Test cases for JSON schema loading."""

    def test_basic_schema_loading(self):
        """Test basic schema loading."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(schema, f)
            temp_path = f.name

        try:
            response_format = load_json_schema(temp_path)
            assert response_format["type"] == "json_schema"
            assert response_format["json_schema"]["schema"] == schema
            assert response_format["json_schema"]["strict"] == True
        finally:
            Path(temp_path).unlink()

    def test_schema_file_not_found(self):
        """Test error handling for missing schema files."""
        with pytest.raises(FileNotFoundError):
            load_json_schema("nonexistent.json")


class TestCombinedParsing:
    """Test cases for combined template and schema parsing."""

    def test_combined_parsing(self):
        """Test parsing template and schema together."""
        template_content = """{% chat role="system" %}
Schema: {{ json_schema }}
{% endchat %}"""

        schema = {"type": "string"}

        # Create temporary files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as tf:
            tf.write(template_content)
            template_path = tf.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as sf:
            json.dump(schema, sf)
            schema_path = sf.name

        try:
            messages, response_format = parse_template_and_schema(template_path, schema_path)

            assert len(messages) == 1
            assert "{'type': 'string'}" in str(messages[0]["content"])
            assert response_format["type"] == "json_schema"
            assert response_format["json_schema"] == schema
        finally:
            Path(template_path).unlink()
            Path(schema_path).unlink()

    def test_combined_parsing_no_schema(self):
        """Test parsing template without schema."""
        template_content = """{% chat role="user" %}
Hello!
{% endchat %}"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write(template_content)
            temp_path = f.name

        try:
            messages, response_format = parse_template_and_schema(temp_path)
            assert len(messages) == 1
            assert response_format is None
        finally:
            Path(temp_path).unlink()
