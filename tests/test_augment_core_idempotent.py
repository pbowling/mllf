"""augment_core_with_excluded_sub1 must be idempotent: a combo directory reused
by a later epoch keeps its already-augmented core, and augmenting it again
duplicated atom names (CHARMM stops reading core.rtf)."""
from mllf.file_handling.generate_combinations import augment_core_with_excluded_sub1

CORE_RTF = """* core
*
RESI LIG 0.000000
GROUP
ATOM C1 CG2R61 -0.115
ATOM H1 HGR61 0.115
BOND C1 H1
END
"""
CORE_PDB = """ATOM      1  C1  LIG     1       0.000   0.000   0.000  1.00  0.00      LIG
ATOM      2  H1  LIG     1       1.000   0.000   0.000  1.00  0.00      LIG
END
"""
SUB1_RTF = """* sub
*
PRES p2_1 0.000
GROUP
ATOM C21 CG331 -0.27
ATOM H21 HGA3 0.09
BOND C21 H21
END
"""
SUB1_PDB = """ATOM      3  C21 LIG     1       2.000   0.000   0.000  1.00  0.00      LIG
ATOM      4  H21 LIG     1       3.000   0.000   0.000  1.00  0.00      LIG
END
"""


def _setup(tmp_path):
    paths = {n: tmp_path / n for n in ('core.rtf', 'core.pdb', 'sub1.rtf', 'sub1.pdb')}
    for n, txt in zip(paths, (CORE_RTF, CORE_PDB, SUB1_RTF, SUB1_PDB)):
        paths[n].write_text(txt)
    return paths


def test_augments_once(tmp_path):
    p = _setup(tmp_path)
    augment_core_with_excluded_sub1(p['core.rtf'], p['core.pdb'], p['sub1.rtf'], p['sub1.pdb'])
    rtf, pdb = p['core.rtf'].read_text(), p['core.pdb'].read_text()
    assert 'ATOM C21' in rtf and 'BOND C21 H21' in rtf
    assert ' C21 ' in pdb


def test_second_augmentation_is_a_noop(tmp_path):
    p = _setup(tmp_path)
    augment_core_with_excluded_sub1(p['core.rtf'], p['core.pdb'], p['sub1.rtf'], p['sub1.pdb'])
    once = (p['core.rtf'].read_text(), p['core.pdb'].read_text())
    augment_core_with_excluded_sub1(p['core.rtf'], p['core.pdb'], p['sub1.rtf'], p['sub1.pdb'])
    assert (p['core.rtf'].read_text(), p['core.pdb'].read_text()) == once
    assert once[0].count('ATOM C21') == 1
