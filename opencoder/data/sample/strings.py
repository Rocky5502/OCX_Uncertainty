def slugify(text: str) -> str:
    """Lowercase and replace whitespace with hyphens."""
    return "-".join(text.lower().split())
