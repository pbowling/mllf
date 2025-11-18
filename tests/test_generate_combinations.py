from pathlib import Path

from mllf.file_handling.generate_combinations import (
    find_site_sub_files,
    all_site_sub_combinations,
    make_combo_dir_name,
)


def test_combination_uniqueness():
    """Test that combinations with the same first element and different
    permutations of remaining elements are NOT both generated.
    
    For example, [2,3,4] should be generated but [2,4,3] should not,
    since they differ only in the order of elements after the first.
    """
    # Create a mock found dict with a single site having subs 1,2,3,4,5
    mock_found = {1: {i: {} for i in [1, 2, 3, 4, 5]}}
    
    combos = all_site_sub_combinations(mock_found)
    
    # Extract just the subs lists for easier checking
    subs_lists = [tuple(subs) for sites, subs in combos]
    
    # Check that [2,3,4] is present
    assert (2, 3, 4) in subs_lists, "Expected [2,3,4] to be generated"
    
    # Check that [2,4,3] is NOT present (would be duplicate with different tail order)
    assert (2, 4, 3) not in subs_lists, "Expected [2,4,3] NOT to be generated (duplicate)"
    
    # Check that [1,2,3] is different from [2,1,3] (different first element)
    assert (1, 2, 3) in subs_lists, "Expected [1,2,3] to be generated"
    assert (2, 1, 3) in subs_lists, "Expected [2,1,3] to be generated (different first element)"
    
    # Verify no duplicates exist when we normalize by sorting tail
    normalized = set()
    for subs in subs_lists:
        if len(subs) >= 2:
            # Normalize: keep first, sort rest
            normalized_form = (subs[0], tuple(sorted(subs[1:])))
            assert normalized_form not in normalized, f"Duplicate found: {subs} normalized to {normalized_form}"
            normalized.add(normalized_form)


def test_print_combinations_for_example():
    repo_root = Path(__file__).resolve().parents[1]
    example_dir = repo_root / 'examples' / 'cb' / '14benz_solv_5.5'
    assert example_dir.exists(), f"Example directory not found: {example_dir}"

    found = find_site_sub_files(example_dir)
    # follow script policy: only consider sites with >= 2 subs
    eligible = {s: subs for s, subs in found.items() if len(subs) >= 2}

    combos = all_site_sub_combinations(eligible)

    # Print combinations directly to the controlling terminal (bypass pytest capture)
    import sys
    out = sys.__stdout__
    out.write(f"Found {len(eligible)} eligible sites and {len(combos)} combinations\n")
    for idx, (sites, subs) in enumerate(combos, start=1):
        name = make_combo_dir_name(idx, sites, subs)
        out.write(name + '\n')
    out.flush()
    # Also write combos to a file for easy inspection
    out_path = repo_root / 'combos_14benz_solv_5.5.txt'
    with out_path.open('w') as fh:
        fh.write(f"Found {len(eligible)} eligible sites and {len(combos)} combinations\n")
        for idx, (sites, subs) in enumerate(combos, start=1):
            name = make_combo_dir_name(idx, sites, subs)
            fh.write(name + '\n')

    # sanity check so CI treats this as a test: ensure some combos exist
    assert len(combos) > 0
