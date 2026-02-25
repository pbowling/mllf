"""Parse CHARMM topology files to extract atom type vocabulary.

This module reads MASS entries from CHARMM toppar files to build a complete
vocabulary of atom types and elements for node feature encoding.
"""
import os
import re
from typing import Dict, Set, List, Tuple


# Regex to match MASS lines: MASS  -1  ATOMTYPE  MASS  [ELEMENT] ! comment
# Note: Some lines may have the element in the comment instead of as a field
# We match both: element field (if present) or extract from comment
MASS_RE = re.compile(r'^\s*MASS\s+-?\d+\s+(\S+)\s+[\d.]+(?:\s+([A-Za-z]+))?\s*!?\s*(.*)$')


def infer_element_from_cgenff_type(atom_type: str) -> str:
    """Infer element from CGenFF atom type name.
    
    CGenFF atom types typically start with element symbol:
    - CG2R61, CG321, etc. -> C (carbon)
    - HGR61, HGA2, etc. -> H (hydrogen)
    - BRGR1 -> Br (bromine)
    - CLGR1 -> Cl (chlorine)
    - FGR1, FGA3 -> F (fluorine)
    - etc.
    
    Args:
        atom_type: CGenFF atom type string (e.g., 'BRGR1', 'CG2R61')
        
    Returns:
        Element symbol (e.g., 'Br', 'C')
    """
    atom_type_upper = atom_type.upper()
    
    # Check two-letter elements first (must come before single-letter)
    two_letter_elements = ['BR', 'CL', 'AL', 'SE']
    for elem in two_letter_elements:
        if atom_type_upper.startswith(elem):
            # Return proper capitalization
            return elem.capitalize()
    
    # Check single-letter elements
    first_char = atom_type[0].upper()
    if first_char in ['H', 'C', 'N', 'O', 'F', 'S', 'P', 'B', 'I', 'K']:
        return first_char
    
    # Unknown - return 'X'
    return 'X'


def parse_toppar_file(filepath: str) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """Parse a single toppar file and extract atom types and elements from MASS entries.
    
    Args:
        filepath: Path to a .rtf or .str file
        
    Returns:
        Tuple of (atom_types, elements, atom_to_element):
        - atom_types: Set of atom type strings
        - elements: Set of element symbols
        - atom_to_element: Dict mapping atom type to element (e.g., 'CG2R61' -> 'C')
    
    Notes:
        - Skips lines starting with '!' (CHARMM comments)
        - Handles MASS lines with or without explicit element field
        - If element field is missing, tries to extract from comment
    """
    atom_types = set()
    elements = set()
    atom_to_element = {}
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Skip comment lines
                if line.strip().startswith('!'):
                    continue
                    
                match = MASS_RE.match(line)
                if match:
                    atom_type = match.group(1)
                    element_field = match.group(2)  # May be None
                    comment = match.group(3)  # Rest of line after element/mass
                    
                    # Determine element: use field if present, otherwise extract from comment
                    if element_field:
                        element = element_field
                    else:
                        # Try to extract element from comment (e.g., "! N for neutral...")
                        # Look for first alphabetic word in comment
                        comment_words = comment.split()
                        element = None
                        for word in comment_words:
                            # Check if word is a single element symbol (1-2 letters, capitalized)
                            if word and len(word) <= 2 and word[0].isupper():
                                # Check if it looks like an element (not a number or other text)
                                if word.isalpha():
                                    element = word
                                    break
                        
                        if not element:
                            # Fallback: try to infer from atom type prefix
                            # H* -> H, C* -> C, N* -> N, O* -> O, S* -> S, etc.
                            first_char = atom_type[0].upper()
                            if first_char in 'HCNOSFPBLI':  # Common elements
                                element = first_char
                            else:
                                # Use 'X' as unknown
                                element = 'X'
                    
                    atom_types.add(atom_type)
                    elements.add(element)
                    atom_to_element[atom_type] = element
                    
    except Exception as e:
        import warnings
        warnings.warn(f"Failed to parse {filepath}: {e}", UserWarning)
    
    return atom_types, elements, atom_to_element


def build_atom_type_vocab_from_toppar(toppar_dir: str = None, toppar_files: List[str] = None) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
    """Build atom type and element vocabularies from CHARMM toppar files.
    
    Args:
        toppar_dir: Path to directory containing toppar files (.rtf, .str).
                   If None, uses the package's bundled toppar directory.
        toppar_files: List of specific filenames to include (e.g., ['top_all36_cgenff.rtf']).
                     If None, includes all .rtf and .str files in toppar_dir.
                   
    Returns:
        Tuple of (atom_type_vocab, element_vocab, atom_to_element):
        - atom_type_vocab: Dictionary mapping atom type strings to indices (sorted alphabetically)
        - element_vocab: Dictionary mapping element symbols to indices (sorted alphabetically)
        - atom_to_element: Dictionary mapping atom type to element symbol (e.g., 'CG2R61' -> 'C')
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
        return {}, {}, {}
    
    all_atom_types = set()
    all_elements = set()
    all_atom_to_element = {}
    
    # Determine which files to parse
    if toppar_files is not None:
        # Use specified files only
        files_to_parse = toppar_files
    else:
        # Parse all .rtf and .str files
        files_to_parse = [f for f in os.listdir(toppar_dir) 
                         if f.endswith('.rtf') or f.endswith('.str')]
    
    # Parse files
    for filename in files_to_parse:
        filepath = os.path.join(toppar_dir, filename)
        if not os.path.exists(filepath):
            import warnings
            warnings.warn(
                f"Toppar file not found: {filepath}. Skipping.",
                UserWarning
            )
            continue
        atom_types, elements, atom_to_element = parse_toppar_file(filepath)
        all_atom_types.update(atom_types)
        all_elements.update(elements)
        all_atom_to_element.update(atom_to_element)
    
    # Create sorted vocabularies
    sorted_types = sorted(all_atom_types)
    sorted_elements = sorted(all_elements)
    
    atom_type_vocab = {atom_type: idx for idx, atom_type in enumerate(sorted_types)}
    element_vocab = {element: idx for idx, element in enumerate(sorted_elements)}
    
    return atom_type_vocab, element_vocab, all_atom_to_element


# Cache the vocabularies to avoid re-parsing files
_CACHED_ATOM_TYPE_VOCAB = None
_CACHED_ELEMENT_VOCAB = None
_CACHED_ATOM_TO_ELEMENT = None
_CACHED_CONFIG = None


def get_atom_type_vocab(toppar_dir: str = None, toppar_files: List[str] = None, force_rebuild: bool = False) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str]]:
    """Get the atom type and element vocabularies, using cached version if available.
    
    Args:
        toppar_dir: Path to toppar directory (None for default)
        toppar_files: List of specific filenames to include (e.g., ['top_all36_cgenff.rtf'])
        force_rebuild: If True, rebuild vocabularies even if cached
        
    Returns:
        Tuple of (atom_type_vocab, element_vocab, atom_to_element):
        - atom_type_vocab: Dictionary mapping atom type strings to indices
        - element_vocab: Dictionary mapping element symbols to indices
        - atom_to_element: Dictionary mapping atom type to element symbol
    """
    global _CACHED_ATOM_TYPE_VOCAB, _CACHED_ELEMENT_VOCAB, _CACHED_ATOM_TO_ELEMENT, _CACHED_CONFIG
    
    # Check if we need to rebuild (config changed or forced)
    current_config = (toppar_dir, tuple(toppar_files) if toppar_files else None)
    if _CACHED_ATOM_TYPE_VOCAB is None or force_rebuild or current_config != _CACHED_CONFIG:
        _CACHED_ATOM_TYPE_VOCAB, _CACHED_ELEMENT_VOCAB, _CACHED_ATOM_TO_ELEMENT = build_atom_type_vocab_from_toppar(toppar_dir, toppar_files)
        _CACHED_CONFIG = current_config
    
    return _CACHED_ATOM_TYPE_VOCAB, _CACHED_ELEMENT_VOCAB, _CACHED_ATOM_TO_ELEMENT


if __name__ == '__main__':
    # Test the vocabulary builder
    atom_type_vocab, element_vocab, atom_to_element = get_atom_type_vocab()
    print(f"Built atom type vocabulary with {len(atom_type_vocab)} types")
    print(f"First 10 atom types: {list(atom_type_vocab.keys())[:10]}")
    print(f"Last 10 atom types: {list(atom_type_vocab.keys())[-10:]}")
    print(f"\nBuilt element vocabulary with {len(element_vocab)} elements")
    print(f"Elements: {sorted(element_vocab.keys())}")
    print(f"\nBuilt atom_to_element mapping with {len(atom_to_element)} entries")
    print(f"Sample mappings: {list(atom_to_element.items())[:10]}")
