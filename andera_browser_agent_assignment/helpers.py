import re
import ast
import json
from typing import Any


def parse_str_to_list(s: str) -> list[Any] | None:
    """Extract a Python list from a string using regex and ast.literal_eval."""
    match = re.search(r"\[.*\]", s, re.DOTALL)
    if not match:
        return None
    
    return ast.literal_eval(match.group())


def parse_str_to_json(text: str) -> dict[Any, Any]:
    """Extract json from response string."""
    # Clean the string: Remove the ```json and ``` markers using regex
    clean_json = re.sub(r"```json|```", "", text).strip()

    return json.loads(clean_json)
