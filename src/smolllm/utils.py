def strip_backticks(text: str) -> str:
    """Strip backticks from the beginning and end of a string"""
    # must be starts with ``` and ends with ```
    if not text.startswith("```") or not text.endswith("```"):
        return text

    lines = text.split("\n")

    # Remove first line if it contains backticks
    if lines[0].startswith("```"):
        lines = lines[1:]

    # Remove last line if it contains backticks
    if lines[-1].endswith("```"):
        if lines[-1] == "```":
            lines = lines[:-1]
        else:
            lines[-1] = lines[-1][:-3]

    return "\n".join(lines)
