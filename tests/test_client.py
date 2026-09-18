"""Tests for DeepSeekClient's response parsing.

DeepSeekClient._extract_code is a @staticmethod with no network or
filesystem dependency, so these run fully offline -- no API key or
sandbox required.
"""

from harness.client import DeepSeekClient


def test_extract_code_from_simple_fence():
    text = "```\nprint('hi')\n```"
    assert DeepSeekClient._extract_code(text) == "print('hi')"


def test_extract_code_strips_language_tag():
    text = "```python\ndef f():\n    return 1\n```"
    assert DeepSeekClient._extract_code(text) == "def f():\n    return 1"


def test_extract_code_falls_back_to_stripped_text_when_no_fence():
    text = "  just plain code, no fence  "
    assert DeepSeekClient._extract_code(text) == "just plain code, no fence"


def test_extract_code_keeps_a_fenced_example_nested_inside_the_file():
    """Regression test: the fence regex used to be non-greedy and would
    stop at the FIRST inner ``` it found, silently truncating a
    translated file that contains its own fenced example (e.g. in a
    docstring). It must now capture through to the final fence instead."""
    text = (
        "```python\n"
        "def foo():\n"
        '    """\n'
        "    Example usage:\n"
        "    ```\n"
        "    foo()\n"
        "    ```\n"
        '    """\n'
        "    return 1\n"
        "```"
    )
    code = DeepSeekClient._extract_code(text)
    assert "return 1" in code
    assert code.count("```") == 2  # the nested example fence pair, not truncated away
