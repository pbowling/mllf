"""Behavior-cloning pretraining of the Uni-Mol bias policy on collected runs.

``runs`` loads collected runs, ``filters`` selects them, ``features`` builds
each run's embeddings, edges and targets, ``rewards`` the advantage weights,
``trainer`` runs the epochs and ``bayesian_init`` builds the Bayesian-head
sibling. ``config`` holds the settings. The command line is
``mllf.cli.pretrain``.
"""
