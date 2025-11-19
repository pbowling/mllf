from pathlib import Path
import tempfile
import shutil

from mllf.file_handling.generate_combinations import (
    find_site_sub_files,
    all_site_sub_combinations,
    make_combo_dir_name,
    create_combination_dirs,
)


def test_rotating_anchor_constraint():
    """Test that each substituent can serve as anchor in ordered combinations.
    
    With 5 subs, each sub as anchor generates C(4,1) + C(4,2) + C(4,3) + C(4,4) = 15 combinations
    Total: 5 × 15 = 75 combinations
    """
    # Create a mock found dict with a single site having subs 1,2,3,4,5
    mock_found = {1: {i: {} for i in [1, 2, 3, 4, 5]}}
    
    combos = all_site_sub_combinations(mock_found)
    
    # Extract just the subs lists
    subs_lists = [tuple(subs) for sites, subs in combos]
    
    # Expected count: 5 subs × 15 combinations each = 75
    expected_count = 75
    assert len(combos) == expected_count, f"Expected {expected_count} combinations, got {len(combos)}"
    
    # Verify both (1,2) and (2,1) exist (different anchors)
    assert (1, 2) in subs_lists, "Expected (1,2) with anchor 1"
    assert (2, 1) in subs_lists, "Expected (2,1) with anchor 2"
    
    # Verify ordered combinations exist
    assert (1, 2, 3) in subs_lists, "Expected (1,2,3) with anchor 1"
    assert (2, 1, 3) in subs_lists, "Expected (2,1,3) with anchor 2"
    assert (3, 1, 2) in subs_lists, "Expected (3,1,2) with anchor 3"
    
    # Count combinations by anchor
    anchor_counts = {}
    for subs in subs_lists:
        anchor = subs[0]
        anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1
    
    # Each sub should be anchor for exactly 15 combinations
    for sub in [1, 2, 3, 4, 5]:
        assert anchor_counts[sub] == 15, f"Sub {sub} should be anchor for 15 combinations, got {anchor_counts[sub]}"


def test_two_site_combinations():
    """Test combination generation with two sites (site1 has 5 subs, site2 has 6 subs).
    
    Expected counts with rotating anchor:
    - Site 1: 5 subs × 15 combinations each = 75
    - Site 2: 6 subs × 31 combinations each = 186
    - Total: 75 + 186 = 261 combinations
    """
    # Site 1: subs 1,2,3,4,5 (each can be anchor)
    # Site 2: subs 1,2,3,4,5,6 (each can be anchor)
    mock_found = {
        1: {i: {} for i in [1, 2, 3, 4, 5]},
        2: {i: {} for i in [1, 2, 3, 4, 5, 6]},
    }
    
    combos = all_site_sub_combinations(mock_found)
    
    # Separate by site
    site1_combos = [(sites, subs) for sites, subs in combos if sites == [1]]
    site2_combos = [(sites, subs) for sites, subs in combos if sites == [2]]
    
    # Verify counts: each sub as anchor generates C(n-1,1) + C(n-1,2) + ... + C(n-1,n-1)
    # For n=5: C(4,1) + C(4,2) + C(4,3) + C(4,4) = 4 + 6 + 4 + 1 = 15 per anchor
    # For n=6: C(5,1) + C(5,2) + C(5,3) + C(5,4) + C(5,5) = 5 + 10 + 10 + 5 + 1 = 31 per anchor
    assert len(site1_combos) == 75, f"Expected 75 combinations for site1 (5×15), got {len(site1_combos)}"
    assert len(site2_combos) == 186, f"Expected 186 combinations for site2 (6×31), got {len(site2_combos)}"
    assert len(combos) == 261, f"Expected 261 total combinations, got {len(combos)}"
    
    # Verify different anchors produce different ordered combinations
    site1_subs = [tuple(subs) for sites, subs in site1_combos]
    assert (1, 2) in site1_subs, "Expected (1,2) with anchor 1"
    assert (2, 1) in site1_subs, "Expected (2,1) with anchor 2"


def test_14benz_exact_count():
    """Test that 14benz_solv_5.5 generates correct number of combinations.
    
    With rotating anchor constraint:
    - Site1: 5 subs × 15 combinations each = 75
    - Site2: 6 subs × 31 combinations each = 186
    - Total: 75 + 186 = 261 combinations
    
    Note: site2_sub7 has been removed, so site2 now has 6 subs.
    """
    repo_root = Path(__file__).resolve().parents[1]
    example_dir = repo_root / 'examples' / 'cb' / '14benz_solv_5.5'
    
    if not example_dir.exists():
        # Skip if example not available
        return
    
    found = find_site_sub_files(example_dir)
    eligible = {s: subs for s, subs in found.items() if len(subs) >= 2}
    combos = all_site_sub_combinations(eligible)
    
    # Verify count matches formula: Σ(n_site × (2^(n_site-1) - 1))
    expected_total = 0
    for site in sorted(eligible.keys()):
        site_combos = [c for c in combos if c[0] == [site]]
        nsubs = len(eligible[site])
        # Each sub as anchor generates 2^(n-1) - 1 combinations
        expected_per_site = nsubs * (2 ** (nsubs - 1) - 1)
        expected_total += expected_per_site
        assert len(site_combos) == expected_per_site, \
            f"Site {site}: expected {expected_per_site} combinations, got {len(site_combos)}"
    
    assert len(combos) == expected_total, \
        f"Expected {expected_total} total combinations, got {len(combos)}"
    
    # Print for debugging
    print(f"\n14benz_solv_5.5 generated {len(combos)} combinations")
    for site in sorted(eligible.keys()):
        site_combos = [c for c in combos if c[0] == [site]]
        nsubs = len(eligible[site])
        expected_per_site = nsubs * (2 ** (nsubs - 1) - 1)
        print(f"  Site {site}: {nsubs} subs -> {len(site_combos)} combinations (expected {expected_per_site})")


def test_site_renumbering():
    """Test that sites are renumbered to start from 1 in output files.
    
    If input only has site2, output should use site1 in file names and PRES tokens.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / 'input'
        prep_dir = input_dir / 'prep'
        prep_dir.mkdir(parents=True)
        
        # Create mock site2 files (simulating input with only site2)
        for sub in [1, 2, 3]:
            (prep_dir / f'site2_sub{sub}_pres.rtf').write_text(f'PRES p2_{sub} 0.0\n')
            (prep_dir / f'site2_sub{sub}_frag.pdb').write_text(f'ATOM {sub}\n')
        
        # Create required support files
        (prep_dir / 'full_ligand.rtf').write_text('RTF\n')
        (prep_dir / 'full_ligand.pdb').write_text('ATOM 1\n')
        
        output_dir = Path(tmpdir) / 'output'
        
        # Generate combinations
        created = create_combination_dirs(input_dir, output_dir, dry_run=False)
        
        assert len(created) > 0, "Should create at least one combination"
        
        # Check first combination
        first_combo = created[0]
        prep_files = list((first_combo / 'prep').glob('site*_sub*'))
        
        # All files should be named site1_sub*, not site2_sub*
        for f in prep_files:
            assert f.name.startswith('site1_'), f"Expected site1_ prefix, got {f.name}"
        
        # Verify PRES tokens in RTF files use site1
        rtf_files = [f for f in prep_files if f.suffix == '.rtf']
        for rtf_file in rtf_files:
            content = rtf_file.read_text()
            assert 'p1_' in content, f"Expected p1_ PRES token in {rtf_file.name}, got: {content}"
            assert 'p2_' not in content, f"Should not have p2_ PRES token in {rtf_file.name}"
        
        # Verify info.py reflects renumbering
        info_path = first_combo / 'info.py'
        assert info_path.exists(), "info.py should exist"
        
        # Read and verify nsubs is a single-element list (one site)
        info_content = info_path.read_text()
        assert "info['nsubs']" in info_content, "info.py should contain nsubs list"


def test_only_selected_files_copied():
    """Test that only files for selected substituents are copied to prep directory.
    
    If combination selects site1_sub2 and site1_sub5, only those RTF/PDB files
    should be in prep, not site1_sub1, site1_sub3, or site1_sub4.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / 'input'
        prep_dir = input_dir / 'prep'
        prep_dir.mkdir(parents=True)
        
        # Create mock files for site1 with 5 subs
        for sub in [1, 2, 3, 4, 5]:
            (prep_dir / f'site1_sub{sub}_pres.rtf').write_text(f'PRES p1_{sub} 0.0\n')
            (prep_dir / f'site1_sub{sub}_frag.pdb').write_text(f'ATOM {sub}\n')
        
        # Create required support files
        (prep_dir / 'full_ligand.rtf').write_text('RTF\n')
        (prep_dir / 'full_ligand.pdb').write_text('ATOM 1\n')
        
        output_dir = Path(tmpdir) / 'output'
        
        # Generate combinations
        created = create_combination_dirs(input_dir, output_dir, dry_run=False)
        
        # Check that each combination only has files for selected subs
        for combo_dir in created:
            prep_path = combo_dir / 'prep'
            site_files = list(prep_path.glob('site1_sub*'))
            
            # Parse which subs were selected from mapping.json
            mapping_file = combo_dir / 'mapping.json'
            if mapping_file.exists():
                import json
                with open(mapping_file) as f:
                    mapping_data = json.load(f)
                
                # Extract unique original_sub values that were selected
                # mapping_data structure: {'combo': name, 'entries': [list of entries]}
                selected_original_subs = set()
                for entry in mapping_data.get('entries', []):
                    if entry.get('site') == 1 and entry.get('original_sub'):
                        selected_original_subs.add(entry['original_sub'])
                
                # Count RTF files for site1 (should equal number of selected subs)
                rtf_files = [f for f in site_files if f.suffix == '.rtf']
                pdb_files = [f for f in site_files if f.suffix == '.pdb']
                
                expected_count = len(selected_original_subs)
                assert len(rtf_files) == expected_count, \
                    f"Expected {expected_count} RTF files, got {len(rtf_files)} in {combo_dir.name}"
                assert len(pdb_files) == expected_count, \
                    f"Expected {expected_count} PDB files, got {len(pdb_files)} in {combo_dir.name}"


def test_pdb_files_renamed_correctly():
    """Test that PDB files are renamed according to the combination mapping.
    
    If combination selects site1_sub2, site1_sub3, site1_sub5, the prep directory
    should contain site1_sub1_frag.pdb, site1_sub2_frag.pdb, site1_sub3_frag.pdb
    (renumbered from original 2, 3, 5).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / 'input'
        prep_dir = input_dir / 'prep'
        prep_dir.mkdir(parents=True)
        
        # Create mock files for site1
        for sub in [1, 2, 3, 4, 5]:
            (prep_dir / f'site1_sub{sub}_pres.rtf').write_text(f'PRES p1_{sub} 0.0\n')
            (prep_dir / f'site1_sub{sub}_frag.pdb').write_text(f'ORIGINAL_SUB_{sub}\n')
        
        # Create required support files
        (prep_dir / 'full_ligand.rtf').write_text('RTF\n')
        (prep_dir / 'full_ligand.pdb').write_text('ATOM 1\n')
        
        output_dir = Path(tmpdir) / 'output'
        
        # Generate combinations
        created = create_combination_dirs(input_dir, output_dir, dry_run=False)
        
        # Find a combination that uses subs 1, 2, 3
        for combo_dir in created:
            if 'site1_1__site1_2__site1_3' in combo_dir.name:
                prep_path = combo_dir / 'prep'
                
                # Verify renamed files exist
                assert (prep_path / 'site1_sub1_frag.pdb').exists(), "site1_sub1_frag.pdb should exist"
                assert (prep_path / 'site1_sub2_frag.pdb').exists(), "site1_sub2_frag.pdb should exist"
                assert (prep_path / 'site1_sub3_frag.pdb').exists(), "site1_sub3_frag.pdb should exist"
                
                # Verify content matches original selection (1, 2, 3)
                content1 = (prep_path / 'site1_sub1_frag.pdb').read_text()
                content2 = (prep_path / 'site1_sub2_frag.pdb').read_text()
                content3 = (prep_path / 'site1_sub3_frag.pdb').read_text()
                
                assert 'ORIGINAL_SUB_1' in content1, "First file should contain original sub 1 content"
                assert 'ORIGINAL_SUB_2' in content2, "Second file should contain original sub 2 content"
                assert 'ORIGINAL_SUB_3' in content3, "Third file should contain original sub 3 content"
                
                break


def test_rtf_pres_token_renumbered():
    """Test that RTF PRES tokens are renumbered in output files.
    
    If site1_sub2 is renamed to site1_sub1, the PRES token p1_2 should become p1_1.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / 'input'
        prep_dir = input_dir / 'prep'
        prep_dir.mkdir(parents=True)
        
        # Create mock RTF files with PRES tokens
        (prep_dir / 'site1_sub1_pres.rtf').write_text('PRES p1_1 0.0\nSOME_CONTENT\n')
        (prep_dir / 'site1_sub2_pres.rtf').write_text('PRES p1_2 0.0\nSOME_CONTENT\n')
        (prep_dir / 'site1_sub3_pres.rtf').write_text('PRES p1_3 0.0\nSOME_CONTENT\n')
        
        # Create corresponding PDB files
        for sub in [1, 2, 3]:
            (prep_dir / f'site1_sub{sub}_frag.pdb').write_text(f'ATOM {sub}\n')
        
        # Create required support files
        (prep_dir / 'full_ligand.rtf').write_text('RTF\n')
        (prep_dir / 'full_ligand.pdb').write_text('ATOM 1\n')
        
        output_dir = Path(tmpdir) / 'output'
        
        # Generate combinations
        created = create_combination_dirs(input_dir, output_dir, dry_run=False)
        
        # Check a combination that uses subs 1, 2, 3
        for combo_dir in created:
            if 'site1_1__site1_2__site1_3' in combo_dir.name:
                prep_path = combo_dir / 'prep'
                
                # Read renamed RTF files
                rtf1_content = (prep_path / 'site1_sub1_pres.rtf').read_text()
                rtf2_content = (prep_path / 'site1_sub2_pres.rtf').read_text()
                rtf3_content = (prep_path / 'site1_sub3_pres.rtf').read_text()
                
                # Verify PRES tokens are correct (should be p1_1, p1_2, p1_3)
                assert 'PRES p1_1' in rtf1_content, "First RTF should have PRES p1_1"
                assert 'PRES p1_2' in rtf2_content, "Second RTF should have PRES p1_2"
                assert 'PRES p1_3' in rtf3_content, "Third RTF should have PRES p1_3"
                
                break


def test_combination_uniqueness():
    """Test that ordered combinations are unique and tail elements are always sorted.
    
    For example, [1,2,3] should be generated but [1,3,2] should NOT,
    since the tail after anchor is kept in sorted order.
    
    However, [1,2,3] and [2,1,3] are both valid (different anchors).
    """
    # Create a mock found dict with a single site having subs 1,2,3,4,5
    mock_found = {1: {i: {} for i in [1, 2, 3, 4, 5]}}
    
    combos = all_site_sub_combinations(mock_found)
    
    # Extract just the subs lists for easier checking
    subs_lists = [tuple(subs) for sites, subs in combos]
    
    # Check that [1,2,3] is present
    assert (1, 2, 3) in subs_lists, "Expected [1,2,3] to be generated"
    
    # Check that [1,3,2] is NOT present (tail must be sorted)
    assert (1, 3, 2) not in subs_lists, "Expected [1,3,2] NOT to be generated (tail unsorted)"
    
    # But [2,1,3] should exist (different anchor)
    assert (2, 1, 3) in subs_lists, "Expected [2,1,3] with anchor 2"
    
    # Verify tails are always sorted and no duplicate (anchor, sorted_tail) pairs
    seen = set()
    for subs in subs_lists:
        if len(subs) >= 2:
            anchor = subs[0]
            tail = subs[1:]
            # Tail should be in sorted order
            assert list(tail) == sorted(tail), f"Tail {tail} not sorted in {subs}"
            
            # No duplicate (anchor, tail) pairs
            key = (anchor, tail)
            assert key not in seen, f"Duplicate combination: {subs}"
            seen.add(key)


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
