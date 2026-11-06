"""Figures and reports for pretraining and online training (``mllf analyze``).

``mllf.analysis.pretrain`` analyzes a pretrained model against the runs it
was trained on, ``mllf.analysis.online`` an online-training run against ALF.
Each writes figures (PNG by default, or vector PDF with editable text), a
CSV of each figure's data and a plain-text report to one output directory.
Figures use ``mllf.analysis.style``.

Needs matplotlib, pandas and pillow (``pip install mllf[analysis]``).
"""
