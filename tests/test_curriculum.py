"""Test curriculum learning functionality."""
from pathlib import Path
import sys


# Import the filter function
import re
from typing import List


def filter_combos_by_curriculum(combos: List[Path], 
                                 min_subs: int, max_subs: int,
                                 min_sites: int, max_sites: int) -> List[Path]:
    """Filter combinations based on curriculum stage criteria."""
    filtered = []
    
    for combo_path in combos:
        # Handle both string paths and Path objects
        if isinstance(combo_path, str):
            combo_name = Path(combo_path).name
        else:
            combo_name = combo_path.name
        
        # Parse combination name: comb_NNNN_site1_1__site1_2__site2_3...
        parts = combo_name.split('__')
        
        if len(parts) < 2:
            continue
        
        sites_seen = set()
        num_subs = 0
        
        for part in parts:
            if 'site' in part:
                site_match = re.search(r'site(\d+)_(\d+)', part)
                if site_match:
                    site_id = int(site_match.group(1))
                    sites_seen.add(site_id)
                    num_subs += 1
        
        num_sites = len(sites_seen)
        
        if (min_subs <= num_subs <= max_subs and 
            min_sites <= num_sites <= max_sites):
            filtered.append(combo_path)
    
    return filtered


def test_curriculum_filtering():
    """Test that curriculum filtering correctly identifies combinations."""
    
    # Create mock combination paths (mix of Path objects and strings)
    test_combos = [
        Path('comb_0001_site1_1__site1_2'),                      # 2 subs, 1 site
        'comb_0002_site1_1__site1_2__site1_3',                   # 3 subs, 1 site (string)
        Path('comb_0003_site1_1__site1_2__site1_3__site1_4'),    # 4 subs, 1 site
        'comb_0004_site1_1__site1_2__site2_1__site2_2',          # 4 subs, 2 sites (string)
        Path('comb_0005_site1_1__site1_2__site2_1'),             # 3 subs, 2 sites
        'comb_0006_site2_3__site2_4',                            # 2 subs, 1 site (string)
        Path('comb_0007_site1_1__site1_2__site1_3__site1_4__site1_5'),  # 5 subs, 1 site
    ]
    
    print("Test Combinations (mix of Path and string):")
    for combo in test_combos:
        name = Path(combo).name if isinstance(combo, str) else combo.name
        print(f"  {name} ({type(combo).__name__})")
    
    # Test Stage 1: pairs only (2 subs, 1 site)
    print("\n=== Stage 1: Pairs Only (2 subs, 1 site) ===")
    stage1 = filter_combos_by_curriculum(test_combos, 
                                          min_subs=2, max_subs=2, 
                                          min_sites=1, max_sites=1)
    print(f"Found {len(stage1)} combinations:")
    for combo in stage1:
        name = Path(combo).name if isinstance(combo, str) else combo.name
        print(f"  ✓ {name}")
    assert len(stage1) == 2, f"Expected 2 pairs, got {len(stage1)}"
    
    # Test Stage 2: triplets only (3 subs, 1 site)
    print("\n=== Stage 2: Triplets Only (3 subs, 1 site) ===")
    stage2 = filter_combos_by_curriculum(test_combos,
                                          min_subs=3, max_subs=3,
                                          min_sites=1, max_sites=1)
    print(f"Found {len(stage2)} combinations:")
    for combo in stage2:
        name = Path(combo).name if isinstance(combo, str) else combo.name
        print(f"  ✓ {name}")
    assert len(stage2) == 1, f"Expected 1 triplet, got {len(stage2)}"
    
    # Test Stage 3: two-site pairs (4 subs, 2 sites)
    print("\n=== Stage 3: Two-Site Pairs (4 subs, 2 sites) ===")
    stage3 = filter_combos_by_curriculum(test_combos,
                                          min_subs=4, max_subs=4,
                                          min_sites=2, max_sites=2)
    print(f"Found {len(stage3)} combinations:")
    for combo in stage3:
        name = Path(combo).name if isinstance(combo, str) else combo.name
        print(f"  ✓ {name}")
    assert len(stage3) == 1, f"Expected 1 two-site pair, got {len(stage3)}"
    
    # Test Stage 4: all single-site (2-5 subs, 1 site)
    print("\n=== Stage 4: All Single-Site (2-5 subs, 1 site) ===")
    stage4 = filter_combos_by_curriculum(test_combos,
                                          min_subs=2, max_subs=5,
                                          min_sites=1, max_sites=1)
    print(f"Found {len(stage4)} combinations:")
    for combo in stage4:
        name = Path(combo).name if isinstance(combo, str) else combo.name
        print(f"  ✓ {name}")
    assert len(stage4) == 5, f"Expected 5 single-site combos, got {len(stage4)}"
    
    # Test Stage 5: full complexity
    print("\n=== Stage 5: Full Complexity (2-10 subs, 1-2 sites) ===")
    stage5 = filter_combos_by_curriculum(test_combos,
                                          min_subs=2, max_subs=10,
                                          min_sites=1, max_sites=2)
    print(f"Found {len(stage5)} combinations:")
    for combo in stage5:
        name = Path(combo).name if isinstance(combo, str) else combo.name
        print(f"  ✓ {name}")
    assert len(stage5) == 7, f"Expected all 7 combos, got {len(stage5)}"
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_curriculum_filtering()
