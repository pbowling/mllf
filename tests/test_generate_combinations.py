from pathlib import Path

from mllf.file_handling.generate_combinations import (
    find_site_sub_files,
    all_site_sub_combinations,
    make_combo_dir_name,
)


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
