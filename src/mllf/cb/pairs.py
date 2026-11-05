"""Directed within-site substituent pairs: the edges every policy predicts on."""
from typing import List, Tuple


def build_directed_pairs(nsubs_per_site: List[int]) -> List[Tuple[int, int]]:
    """Build list of directed pairs for all substituents within each site.

    Generates BOTH directions (i→j and j→i) within each site to allow
    independent predictions for skew and end biases.
    No cross-site pairs are generated.

    This is useful for both graph-based and pairwise approaches when you need
    all directed edges within sites for training or prediction.

    Args:
        nsubs_per_site: List of substituent counts per site

    Returns:
        List of (i, j) pairs including both directions within the same site

    Example:
        >>> build_directed_pairs([2, 3])  # Site 1: 2 subs, Site 2: 3 subs
        [(0, 1), (1, 0), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3)]
    """
    pairs = []
    offset = 0

    for nsubs in nsubs_per_site:
        # Generate both directions for all pairs within site
        for i in range(nsubs):
            for j in range(nsubs):
                if i != j:  # Skip diagonal (self-pairs)
                    pairs.append((offset + i, offset + j))
        offset += nsubs

    return pairs
