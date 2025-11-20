"""Parse CHARMM topology files to extract atom type vocabulary.

This module reads MASS entries from CHARMM toppar files to build a complete
vocabulary of atom types for node feature encoding.
"""
import os
import re
from typing import Dict, Set, List


# Regex to match MASS lines: MASS  -1  ATOMTYPE  MASS  ELEMENT ! comment
MASS_RE = re.compile(r'^\s*MASS\s+-?\d+\s+(\S+)\s+[\d.]+')


def parse_toppar_file(filepath: str) -> Set[str]:
    """Parse a single toppar file and extract atom types from MASS entries.
    
    Args:
        filepath: Path to a .rtf or .str file
        
    Returns:
        Set of atom type strings found in the file
    """
    atom_types = set()
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = MASS_RE.match(line)
                if match:
                    atom_type = match.group(1)
                    atom_types.add(atom_type)
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to parse {filepath}: {e}", UserWarning)
    
    return atom_types


def build_atom_type_vocab_from_toppar(toppar_dir: str = None) -> Dict[str, int]:
    """Build a complete atom type vocabulary from CHARMM toppar files.
    
    Args:
        toppar_dir: Path to directory containing toppar files (.rtf, .str).
                   If None, uses the package's bundled toppar directory.
                   
    Returns:
        Dictionary mapping atom type strings to indices (sorted alphabetically)
    """
    if toppar_dir is None:
        # Default to package toppar directory
        module_dir = os.path.dirname(os.path.abspath(__file__))
        toppar_dir = os.path.join(module_dir, '..', '..', '..', 'toppar')
        toppar_dir = os.path.abspath(toppar_dir)
    
    if not os.path.isdir(toppar_dir):
        import warnings
        warnings.warn(
            f"Toppar directory not found: {toppar_dir}. "
            "Vocabulary will be built dynamically from graph data.",
            UserWarning
        )
        return {}
    
    all_atom_types = set()
    
    # Parse all .rtf and .str files
    for filename in os.listdir(toppar_dir):
        if filename.endswith('.rtf') or filename.endswith('.str'):
            filepath = os.path.join(toppar_dir, filename)
            atom_types = parse_toppar_file(filepath)
            all_atom_types.update(atom_types)
    
    # Create sorted vocabulary
    sorted_types = sorted(all_atom_types)
    vocab = {atom_type: idx for idx, atom_type in enumerate(sorted_types)}
    
    return vocab


# Cache the vocabulary to avoid re-parsing files
_CACHED_VOCAB = None


def get_atom_type_vocab(toppar_dir: str = None, force_rebuild: bool = False) -> Dict[str, int]:
    """Get the atom type vocabulary, using cached version if available.
    
    Args:
        toppar_dir: Path to toppar directory (None for default)
        force_rebuild: If True, rebuild vocabulary even if cached
        
    Returns:
        Dictionary mapping atom type strings to indices
    """
    global _CACHED_VOCAB
    
    if _CACHED_VOCAB is None or force_rebuild:
        _CACHED_VOCAB = build_atom_type_vocab_from_toppar(toppar_dir)
    
    return _CACHED_VOCAB


if __name__ == '__main__':
    # Test the vocabulary builder
    vocab = get_atom_type_vocab()
    print(f"Built vocabulary with {len(vocab)} atom types")
    print(f"First 10: {list(vocab.keys())[:10]}")
    print(f"Last 10: {list(vocab.keys())[-10:]}")
