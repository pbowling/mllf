from pathlib import Path
import tempfile
import shutil
import warnings
import pytest

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
    
    # Extract just the subs lists (handle 3-tuple format)
    subs_lists = [tuple(subs) for sites, subs, _ in combos]
    
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
    
    # Separate by site (handle 3-tuple with None for within-site)
    site1_combos = [(sites, subs) for sites, subs, _ in combos if sites == [1]]
    site2_combos = [(sites, subs) for sites, subs, _ in combos if sites == [2]]
    cross_site_combos = [(sites, subs) for sites, subs, counts in combos if len(sites) > 1]
    
    # Verify counts: each sub as anchor generates C(n-1,1) + C(n-1,2) + ... + C(n-1,n-1)
    # For n=5: C(4,1) + C(4,2) + C(4,3) + C(4,4) = 4 + 6 + 4 + 1 = 15 per anchor
    # For n=6: C(5,1) + C(5,2) + C(5,3) + C(5,4) + C(5,5) = 5 + 10 + 10 + 5 + 1 = 31 per anchor
    assert len(site1_combos) == 75, f"Expected 75 combinations for site1 (5×15), got {len(site1_combos)}"
    assert len(site2_combos) == 186, f"Expected 186 combinations for site2 (6×31), got {len(site2_combos)}"
    
    # Cross-site: 75 × 186 = 13,950
    assert len(cross_site_combos) == 13950, f"Expected 13,950 cross-site combos, got {len(cross_site_combos)}"
    
    # Total: within-site + cross-site = 75 + 186 + 13,950 = 14,211
    assert len(combos) == 14211, f"Expected 14,211 total combinations, got {len(combos)}"
    
    # Verify different anchors produce different ordered combinations
    site1_subs = [tuple(subs) for sites, subs in site1_combos]
    assert (1, 2) in site1_subs, "Expected (1,2) with anchor 1"
    assert (2, 1) in site1_subs, "Expected (2,1) with anchor 2"


def test_14benz_exact_count():
    """Test that 14benz_solv_5.5 generates correct number of combinations.
    
    With rotating anchor constraint:
    - Site1: 5 subs × 15 combinations each = 75 within-site
    - Site2: 6 subs × 31 combinations each = 186 within-site
    - Cross-site: 75 × 186 = 13,950 combinations
    - Total: 75 + 186 + 13,950 = 14,211 combinations
    
    Note: site2_sub7 has been removed, so site2 now has 6 subs.
    """
    repo_root = Path(__file__).resolve().parents[0]
    example_dir = repo_root / 'samples' / '14benz_solv_5.5'
    
    if not example_dir.exists():
        # Skip if example not available
        return
    
    found = find_site_sub_files(example_dir)
    eligible = {s: subs for s, subs in found.items() if len(subs) >= 2}
    combos = all_site_sub_combinations(eligible)
    
    # Separate within-site and cross-site
    within_site = [(sites, subs) for sites, subs, _ in combos if len(sites) == 1]
    cross_site = [(sites, subs) for sites, subs, _ in combos if len(sites) > 1]
    
    # Verify within-site count matches formula: Σ(n_site × (2^(n_site-1) - 1))
    expected_within = 0
    for site in sorted(eligible.keys()):
        site_combos = [c for c in within_site if c[0] == [site]]
        nsubs = len(eligible[site])
        # Each sub as anchor generates 2^(n-1) - 1 combinations
        expected_per_site = nsubs * (2 ** (nsubs - 1) - 1)
        expected_within += expected_per_site
        assert len(site_combos) == expected_per_site, \
            f"Site {site}: expected {expected_per_site} within-site combinations, got {len(site_combos)}"
    
    assert len(within_site) == expected_within, \
        f"Expected {expected_within} within-site combinations, got {len(within_site)}"
    
    # Verify cross-site count
    # For 2 sites: site1 has 75 selections, site2 has 186 selections
    # Cross-site = 75 × 186 = 13,950
    assert len(cross_site) == 13950, f"Expected 13,950 cross-site combinations, got {len(cross_site)}"
    
    # Total
    assert len(combos) == 14211, f"Expected 14,211 total combinations, got {len(combos)}"
    
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
    
    # Extract just the subs lists for easier checking (handle 3-tuple format)
    subs_lists = [tuple(subs) for sites, subs, _ in combos]
    
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


def test_cross_site_combinations():
    """Test cross-site combination generation.
    
    With 2 sites (site1 has 3 subs, site2 has 3 subs):
    - Within-site combos: site1 has 9, site2 has 9 (total 18)
    - Cross-site combos: 9 × 9 = 81
    - Total: 18 + 81 = 99 combinations
    """
    mock_found = {
        1: {i: {} for i in [1, 2, 3]},
        2: {i: {} for i in [1, 2, 3]},
    }
    
    combos = all_site_sub_combinations(mock_found)
    
    # Separate within-site and cross-site
    within_site = [(sites, subs) for sites, subs, _ in combos if len(sites) == 1]
    cross_site = [(sites, subs, counts) for sites, subs, counts in combos if len(sites) > 1]
    
    # For n=3 subs: 3 anchors × (C(2,1) + C(2,2)) = 3 × 3 = 9 per site
    assert len(within_site) == 18, f"Expected 18 within-site combos, got {len(within_site)}"
    
    # Cross-site: 9 selections for site1 × 9 selections for site2 = 81
    assert len(cross_site) == 81, f"Expected 81 cross-site combos, got {len(cross_site)}"
    
    assert len(combos) == 99, f"Expected 99 total combinations, got {len(combos)}"
    
    # Verify structure of cross-site combos
    sample_sites, sample_subs, sample_counts = cross_site[0]
    assert len(sample_sites) == 2, "Cross-site should have 2 sites"
    assert sample_sites == [1, 2], "Sites should be [1, 2]"
    assert len(sample_subs) >= 4, "Cross-site should have at least 4 subs (2 from each site)"
    assert sample_counts is not None, "Cross-site should have counts"
    assert len(sample_counts) == 2, "Counts should match number of sites"


def test_14benz_with_cross_site():
    """Test that 14benz_solv_5.5 generates correct number including cross-site.
    
    With rotating anchor constraint:
    - Site1: 5 subs × 15 combinations each = 75 within-site
    - Site2: 6 subs × 31 combinations each = 186 within-site
    - Cross-site: 75 × 186 = 13,950 combinations
    - Total: 75 + 186 + 13,950 = 14,211 combinations
    """
    repo_root = Path(__file__).resolve().parents[0]
    example_dir = repo_root / 'samples' / '14benz_solv_5.5'
    
    if not example_dir.exists():
        # Skip if example not available
        return
    
    found = find_site_sub_files(example_dir)
    eligible = {s: subs for s, subs in found.items() if len(subs) >= 2}
    combos = all_site_sub_combinations(eligible)
    
    # Separate within-site and cross-site (handle 3-tuple format)
    within_site = [(sites, subs) for sites, subs, _ in combos if len(sites) == 1]
    cross_site = [(sites, subs, counts) for sites, subs, counts in combos if len(sites) > 1]
    
    # Verify within-site counts
    site1_within = [c for c in within_site if c[0] == [1]]
    site2_within = [c for c in within_site if c[0] == [2]]
    
    print(f"\n14benz_solv_5.5 with cross-site:")
    print(f"  Site 1 within-site: {len(site1_within)} combinations")
    print(f"  Site 2 within-site: {len(site2_within)} combinations")
    print(f"  Cross-site: {len(cross_site)} combinations")
    print(f"  Total: {len(combos)} combinations")
    
    # Expected: site1 has 5 subs → 75, site2 has 6 subs → 186
    assert len(site1_within) == 75, f"Expected 75 site1 within-site, got {len(site1_within)}"
    assert len(site2_within) == 186, f"Expected 186 site2 within-site, got {len(site2_within)}"
    
    # Cross-site: 75 × 186 = 13,950
    assert len(cross_site) == 13950, f"Expected 13,950 cross-site, got {len(cross_site)}"
    
    # Total
    assert len(combos) == 14211, f"Expected 14,211 total, got {len(combos)}"


def test_cross_site_file_structure():
    """Test that cross-site combinations create correct file structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / 'input'
        prep_dir = input_dir / 'prep'
        prep_dir.mkdir(parents=True)
        
        # Create files for site1 (2 subs) and site2 (2 subs)
        for site in [1, 2]:
            for sub in [1, 2]:
                (prep_dir / f'site{site}_sub{sub}_pres.rtf').write_text(f'PRES p{site}_{sub} 0.0\n')
                (prep_dir / f'site{site}_sub{sub}_frag.pdb').write_text(f'SITE{site}_SUB{sub}\n')
        
        # Create required support files
        (prep_dir / 'full_ligand.rtf').write_text('RTF\n')
        (prep_dir / 'full_ligand.pdb').write_text('ATOM 1\n')
        
        output_dir = Path(tmpdir) / 'output'
        
        # Generate combinations
        created = create_combination_dirs(input_dir, output_dir, dry_run=False)
        
        # Should have: 2 site1-only + 2 site2-only + 4 cross-site = 8 total
        # Actually with rotating anchor: site1: 1 combo, site2: 1 combo, cross: 1×1 = 1
        # Wait, with 2 subs each: anchor 1 with [2], anchor 2 with [1] = 2 per site
        # So: 2 + 2 + (2×2) = 8 combinations
        
        within_site = [d for d in created if len([s for s in d.name.split('__') if 'site1' in s]) + 
                                            len([s for s in d.name.split('__') if 'site2' in s]) < 4]
        
        # Find a cross-site combination directory
        cross_site_dirs = [d for d in created if 'site1' in d.name and 'site2' in d.name]
        
        assert len(cross_site_dirs) > 0, "Should have at least one cross-site combination"
        
        # Check first cross-site combo
        cross_combo = cross_site_dirs[0]
        prep_path = cross_combo / 'prep'
        
        # Should have files from both sites
        site1_files = list(prep_path.glob('site1_sub*'))
        site2_files = list(prep_path.glob('site2_sub*'))
        
        assert len(site1_files) >= 4, f"Should have at least 4 site1 files (2 RTF + 2 PDB), got {len(site1_files)}"
        assert len(site2_files) >= 4, f"Should have at least 4 site2 files (2 RTF + 2 PDB), got {len(site2_files)}"
        
        # Verify info.py has nsubs as a list with 2 elements (one per site)
        info_path = cross_combo / 'info.py'
        info_content = info_path.read_text()
        
        # Should have nsubs = [2, 2] or similar
        assert "info['nsubs']" in info_content
        # Parse to verify it's a list with 2 elements
        import re
        match = re.search(r"info\['nsubs'\]\s*=\s*\[([^\]]+)\]", info_content)
        assert match, "Should find nsubs list in info.py"
        nsubs_values = [int(x.strip()) for x in match.group(1).split(',')]
        assert len(nsubs_values) == 2, f"Should have 2 entries in nsubs for cross-site, got {len(nsubs_values)}"
        assert all(n >= 2 for n in nsubs_values), "Each site should contribute at least 2 subs"


def test_max_subs_per_site_limit():
    """Test that max_subs_per_site correctly limits combination size while allowing all subs to participate.
    
    With 5 subs and max_subs_per_site=3:
    - All 5 subs can participate in combinations
    - But no single combination has more than 3 subs
    - Sub 5 can still be in combinations like [5,1,2]
    """
    import warnings
    
    # Use small example for test speed: 5 subs with max_subs_per_site=3
    small_found = {1: {i: {} for i in range(1, 6)}}  # subs 1-5
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        combos = all_site_sub_combinations(small_found, max_subs_per_site=3)
        
        # Should warn about 5 subs exceeding max of 3
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert "has 5 substituents" in str(w[0].message)
        assert "limited to at most 3" in str(w[0].message)
    
    subs_lists = [subs for sites, subs, _ in combos]
    
    # All combinations should have at most 3 subs
    for subs in subs_lists:
        assert len(subs) <= 3, f"Combination {subs} exceeds max_subs_per_site=3"
    
    # All 5 subs should still participate
    all_participating = set()
    for subs in subs_lists:
        all_participating.update(subs)
    assert all_participating == {1, 2, 3, 4, 5}, \
        f"All 5 subs should participate, got {sorted(all_participating)}"
    
    # Verify sub 5 can be an anchor (first element in some combinations)
    sub5_as_anchor = [subs for subs in subs_lists if subs[0] == 5]
    assert len(sub5_as_anchor) > 0, "Sub 5 should be able to serve as anchor"
    
    # Should have combinations of size 2 and 3
    sizes = sorted(set(len(subs) for subs in subs_lists))
    assert sizes == [2, 3], f"Should have combinations of size 2 and 3, got {sizes}"
    
    # Verify a few specific expected combinations exist
    assert [1, 2] in subs_lists, "Should have [1,2]"
    assert [5, 1, 2] in subs_lists, "Should have [5,1,2] (sub 5 as anchor)"
    assert [1, 2, 3] in subs_lists, "Should have [1,2,3] at max size"
    
    # But should NOT have any size-4 combinations
    size_4_combos = [subs for subs in subs_lists if len(subs) == 4]
    assert len(size_4_combos) == 0, f"Should not have size-4 combinations, found {len(size_4_combos)}"


def test_max_subs_cross_site():
    """Test that max_subs_per_site is applied per-site in cross-site combinations.
    
    With site1 having 4 subs and site2 having 3 subs, with max_subs_per_site=3:
    - Site1 contributions are limited to 3 subs per combination
    - Site2 can contribute all 3 subs
    - Cross-site combo could have 3 from site1 + 3 from site2 = 6 total
    """
    import warnings
    
    # Use very small sites for test speed
    mock_found = {
        1: {i: {} for i in range(1, 5)},  # 4 subs (exceeds limit of 3)
        2: {i: {} for i in range(1, 4)},  # 3 subs (at limit)
    }
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        combos = all_site_sub_combinations(mock_found, max_subs_per_site=3)
        
        # Should warn about site1 exceeding limit
        warning_messages = [str(warning.message) for warning in w]
        site1_warned = any("Site 1" in msg and "4 substituents" in msg for msg in warning_messages)
        
        assert site1_warned, f"Should warn about site1 exceeding limit, warnings: {warning_messages}"
    
    # Get cross-site combinations (limit check to avoid long iteration)
    cross_site_combos = []
    for sites, subs, counts in combos:
        if len(sites) > 1:
            cross_site_combos.append((sites, subs, counts))
            # Check first 10 cross-site combinations for efficiency
            if len(cross_site_combos) >= 10:
                break
    
    assert len(cross_site_combos) > 0, "Should have cross-site combinations"
    
    # Verify per-site limits in sampled cross-site combinations
    for sites, subs, counts in cross_site_combos:
        if counts is not None:
            # Extract per-site sub counts
            for i, site in enumerate(sites):
                site_sub_count = counts[i]
                assert site_sub_count <= 3, \
                    f"Site{site} should contribute at most 3 subs, got {site_sub_count} in combo {subs}"
                assert site_sub_count >= 2, \
                    f"Site{site} should contribute at least 2 subs, got {site_sub_count}"


def test_single_sub_error():
    """Test that an error is raised when a site has only 1 substituent.
    
    MSLD simulations require at least 2 substituents per site to run correctly.
    Sites with only 1 sub should cause execution to stop with a helpful error message.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        output_dir = Path(tmpdir) / "output"
        input_dir.mkdir()
        
        # Create site1 with 2 subs (valid)
        (input_dir / "site1_sub1_pres.rtf").write_text("PRES p1_1")
        (input_dir / "site1_sub1_frag.pdb").write_text("ATOM 1")
        (input_dir / "site1_sub2_pres.rtf").write_text("PRES p1_2")
        (input_dir / "site1_sub2_frag.pdb").write_text("ATOM 2")
        
        # Create site2 with only 1 sub (should trigger error)
        (input_dir / "site2_sub1_pres.rtf").write_text("PRES p2_1")
        (input_dir / "site2_sub1_frag.pdb").write_text("ATOM 3")
        
        # Create site3 with 2 subs (valid)
        (input_dir / "site3_sub1_pres.rtf").write_text("PRES p3_1")
        (input_dir / "site3_sub1_frag.pdb").write_text("ATOM 4")
        (input_dir / "site3_sub2_pres.rtf").write_text("PRES p3_2")
        (input_dir / "site3_sub2_frag.pdb").write_text("ATOM 5")
        
        # Should raise RuntimeError for site2 having only 1 sub
        with pytest.raises(RuntimeError) as exc_info:
            create_combination_dirs(input_dir, output_dir, dry_run=False)
        
        error_msg = str(exc_info.value)
        assert "site2" in error_msg, f"Error should mention site2: {error_msg}"
        assert "only 1 substituent" in error_msg, f"Error should mention single substituent: {error_msg}"
        assert "core" in error_msg.lower(), f"Error should mention core files: {error_msg}"
        assert "MSLD simulations require at least 2 substituents" in error_msg, f"Error should explain requirement: {error_msg}"


