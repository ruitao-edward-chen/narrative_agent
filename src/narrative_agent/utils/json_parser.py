"""
JSON parsing utilities for handling potentially malformed JSON strings.

This module provides helper functions to parse JSON strings that may be malformed,
including removing code fences, balancing quotes/brackets, and trimming extra commas.
"""

import re
import json
from typing import Any

try:
    import json5

    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False


def clean_json_string(s: str) -> str:
    """
    Remove markdown code fences, extra quotes, and trailing commas from a JSON string.

    Args:
        s: Input string

    Returns:
        Cleaned string
    """
    if s is None:
        return ""
    s = s.strip().strip("'\"")
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s, flags=re.IGNORECASE)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def balance_json_string(s: str) -> str:
    """
    Append missing closing braces/brackets to balance the JSON.
    This scans the string (ignoring text inside quotes) and counts
    unmatched '{' and '['.

    Args:
        s: Input string

    Returns:
        Balanced string
    """
    open_curly = 0
    open_square = 0
    in_string = False
    string_char = None
    escape = False

    for char in s:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if in_string:
            if char == string_char:
                in_string = False
            continue
        if char in ('"', "'"):
            in_string = True
            string_char = char
            continue
        if char == "{":
            open_curly += 1
        elif char == "}":
            if open_curly > 0:
                open_curly -= 1
        elif char == "[":
            open_square += 1
        elif char == "]":
            if open_square > 0:
                open_square -= 1

    s += "}" * open_curly + "]" * open_square
    return s


def complete_quotes(s: str) -> str:
    """
    Heuristic: if the number of unescaped double quotes is odd,
    assume one is missing and append one.

    Args:
        s: Input string

    Returns:
        String with completed quotes
    """
    count = 0
    escape = False
    for char in s:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            count += 1
    if count % 2 != 0:
        s += '"'
    return s


def complete_json(s: str) -> str:
    """
    Apply quote completion and brace/bracket balancing.

    Args:
        s: Input string

    Returns:
        Completed JSON string
    """
    return balance_json_string(complete_quotes(s))


def parse_json_string(s: str) -> Any:
    """
    Attempt to parse a possibly incomplete JSON string by:
      1. Cleaning the input.
      2. Completing missing quotes and closing braces/brackets.
      3. Trying to parse the completed string.
      4. If that fails, progressively trimming characters from the end
         until a valid JSON fragment is found.

    Args:
        s: Input JSON string (possibly malformed)

    Returns:
        The parsed JSON (which may be incomplete) or an empty dict if no valid parse is found.
    """
    cleaned = clean_json_string(s)

    # First, try to complete the JSON and parse it.
    candidate = complete_json(cleaned)

    # Try json first, then json5 if available
    loaders = [json.loads]
    if HAS_JSON5:
        loaders.append(json5.loads)

    for loader in loaders:
        try:
            return loader(candidate)
        except Exception:
            continue

    # If that fails, progressively trim the string from the end.
    for i in range(len(cleaned), 0, -1):
        test_str = cleaned[:i].strip()
        test_str = complete_json(test_str)
        for loader in loaders:
            try:
                return loader(test_str)
            except Exception:
                continue

    # If nothing parses, return an empty dict.
    return {}
