#!/usr/bin/env python3
"""
Basic usage example for Olingo LLM Parser.
"""

from olingo_llm_parser import parse_chat_template, parse_chat_and_schema
import json


def basic_example():
    """Basic template parsing example."""
    print("=== Basic Template Parsing ===")
    
    # Create a simple template
    template_content = """{% chat role="system" %}
You are a helpful AI assistant specializing in {{ domain }}.
{% endchat %}

{% chat role="user" %}
{{ user_question }}
{% endchat %}"""
    
    # Write template to file
    with open("basic_template.jinja", "w") as f:
        f.write(template_content)
    
    # Parse with variables
    messages = parse_chat_template("basic_template.jinja", {
        "domain": "data science",
        "user_question": "What is machine learning?"
    })
    
    print("Generated messages:")
    print(json.dumps(messages, indent=2))


def advanced_example():
    """Advanced example with loops and conditionals."""
    print("\n=== Advanced Template with Loops ===")
    
    template_content = """{% chat role="system" %}
You are an expert {{ role }}.
{% endchat %}

{% for example in examples %}
{% chat role="user" %}
Example: {{ example.question }}
{% endchat %}

{% chat role="assistant" %}
{{ example.answer }}
{% endchat %}
{% endfor %}

{% chat role="user" %}
Now answer: {{ final_question }}
{% endchat %}"""
    
    with open("advanced_template.jinja", "w") as f:
        f.write(template_content)
    
    messages = parse_chat_template("advanced_template.jinja", {
        "role": "mathematician",
        "examples": [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3*3?", "answer": "9"}
        ],
        "final_question": "What is the square root of 16?"
    })
    
    print("Generated messages:")
    for i, msg in enumerate(messages):
        print(f"{i+1}. [{msg['role']}]: {msg['content']}")


def schema_example():
    """Example with JSON schema integration."""
    print("\n=== Schema Integration Example ===")
    
    # Create a schema
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["answer", "confidence"]
    }
    
    with open("response_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    
    # Create schema-aware template
    template_content = """{% chat role="system" %}
You are a helpful assistant. Respond in JSON format following this schema:
{{ json_schema }}
{% endchat %}

{% chat role="user" %}
{{ question }}
{% endchat %}"""
    
    with open("schema_template.jinja", "w") as f:
        f.write(template_content)
    
    # Parse with schema
    messages, response_format = parse_chat_and_schema(
        "schema_template.jinja",
        "response_schema.json",
        {"question": "What is the capital of France?"}
    )
    
    print("Generated messages:")
    print(json.dumps(messages, indent=2))
    print(f"\nResponse format: {response_format['type']}")


if __name__ == "__main__":
    basic_example()
    advanced_example()
    schema_example()
    
    print("\n✨ All examples completed!")
    print("Check the generated .jinja and .json files to see the templates and schemas.") 