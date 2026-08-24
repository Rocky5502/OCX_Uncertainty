"""Tiny sample repo used for smoke tests."""

def mean(values):
    """Return the arithmetic mean of a list of numbers."""
    return sum(values) / len(values) if values else 0.0


def variance(values):
    """Return the population variance of a list of numbers."""
    if not values:
        return 0.0
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)
