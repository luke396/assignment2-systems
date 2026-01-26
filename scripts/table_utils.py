"""Utilities for formatting markdown tables."""


def format_table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    """Format a markdown table with title.

    Args:
        title: Table title
        headers: Column headers
        rows: Table rows

    Returns:
        Formatted markdown table as string

    """
    if not rows:
        return f"{title}\n\nNo data available."

    lines = [title, ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
