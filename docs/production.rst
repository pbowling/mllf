Production Runs
===============

Once a system's biases flatten its alchemical landscape, longer simulations with those
biases fixed give the free energy of each end state. ``mllf production`` runs these
simulations as independent replicas and estimates the free energies from them, so that
the biases of different origins (a trained model, an online run or iterative
flattening) can be compared on the free energies they yield.

.. code-block:: bash

   mllf production run production.yaml          # replicas, then the analysis job
   mllf production analyze prod/12benz_vac_g1   # again, with other settings

Running
-------

A production needs three inputs: a prep directory (``core.pdb``, ``site<i>_sub<j>_pres.rtf``
and ``_frag.pdb``, ``toppar.str``, ...), a variables file holding the biases and an
MSLD script such as ``msld_flat.py``. ``combo:`` names a combination directory of an
online run, which supplies the prep directory, the script and the temperature; only the
variables file is then needed, for example the ``variables.py`` of a run that passed.

.. code-block:: yaml

   production:
     combo: training/simulations_refine_v2_r1/12benz_vac/group01_site1_subs_1_2_3_4_5
     variables: training/simulations_refine_v2_r1/12benz_vac/group01_site1_subs_1_2_3_4_5/run_012/variables.py
     # or, instead of combo:
     # prep: training/systems/12benz_vac/prep
     # script: training/systems/12benz_vac/msld_flat.py
     out_dir: production/12benz_vac_group01
     replicas: 5          # independent replicas (default 5)
     ns: 5.0              # production of each replica (default 5 ns)
     # equil_ns: 0.25     # equilibration before it (default: the script's esteps)
   slurm:                 # the simulations (any sbatch option; see mllf.slurm)
     partition: ada6000
     gres: gpu:1
     time: '12:00:00'
     module: charmm/charmm/c51a1
   analysis:
     # run: true          # submit the analysis to start when the replicas end
     # discard_ns: null   # production dropped from each replica (default: a quarter)
     # cutoff: 0.99       # lambda at or above which a substituent is at its end state
     # bootstrap: 50
     # stride: 1          # use every Nth frame
     # potts: true
     slurm:
       partition: cpu
       mem: 16G
       time: '04:00:00'

Paths in the config are relative to it. ``--combo``, ``--prep``, ``--script``,
``--variables``, ``--out``, ``--replicas`` and ``--ns`` override the config, so one
config with the ``slurm:`` block serves every system:

.. code-block:: bash

   mllf production run production.yaml --combo SIMS/12benz_vac/group01_site1_subs_1_2_3_4_5 \
       --variables SIMS/12benz_vac/group01_site1_subs_1_2_3_4_5/run_012/variables.py \
       --out production/12benz_vac_group01_mllf

``out_dir`` gets one directory per replica, each a complete simulation directory: a
copy of the prep directory and of the script, an ``info.py`` (the substituents per
site, counted from the prep files), and ``variables.py``, the given file with the
production's ``esteps`` and ``nsteps`` appended. The script equilibrates, then writes
the production's lambda trajectory to ``res/<name>_flat.lmd``. Every replica starts
from the same minimized structure and diverges through the random forces of the
Langevin thermostat, which CHARMM seeds from the clock; the analysis checks that no
two replicas are identical.

Running ``mllf production run`` again submits only the replicas that neither finished
nor are still queued, so it also resubmits failed replicas. ``--dry-run`` checks the
inputs (the variables file must have one linear bias per substituent of the prep
directory) and lists what would be submitted.

The analysis job runs ``mllf production analyze`` with the same Python and copy of
``mllf`` that submitted it, after every replica ends (successfully or not).

Free Energies
-------------

A frame's state at a site is the substituent with :math:`\lambda \geq 0.99`, or none
when no substituent is that close to its end state. A combination is one substituent at
every site. Its biased population :math:`p(\sigma)` gives its free energy,

.. math::

   G(\sigma) = E(\sigma) - k_BT \ln p(\sigma)

where :math:`E(\sigma)` is the bias energy at the end state, the linear biases of its
substituents plus the coupling terms between substituents at different sites
(:doc:`background`; the simulation applies :math:`-E`). Free energies are reported
relative to the first combination (substituent 1 at every site), or to the first
combination with an estimate when that one was never sampled. Two estimators are used,
the weighted histogram analysis method (WHAM) and a Potts model.

**WHAM.** Frames of every replica are pooled. With one bias set this is the histogram
above. When replicas ran with different biases, the normalization of each bias set
comes from binless WHAM over every frame :cite:`KumRosBou92, ShiCho08`, using the full
bias energy of each frame under each set, and the counts are combined as

.. math::

   G(\sigma) = -k_BT \ln \sum_k n_k(\sigma) + k_BT \ln \sum_k N_k
   \exp\left[f_k + E_k(\sigma) / k_BT\right]

with :math:`n_k(\sigma)` the frames of bias set :math:`k` in :math:`\sigma`,
:math:`N_k` all frames of set :math:`k`, and :math:`f_k` its dimensionless
normalization. A combination that was never sampled has no WHAM value.

**Potts model.** With several sites the number of combinations grows as the product of
the substituents per site, and most are sampled rarely. A Potts model,

.. math::

   \ln P(\sigma) = \sum_i h_i(\sigma_i) + \sum_{i<j} J_{ij}(\sigma_i, \sigma_j) + \text{const},

with a field for each state of each site (including no substituent at the cutoff) and a
coupling for each pair of states at two sites, is fitted to the frames by
pseudo-likelihood maximization :cite:`EkeHarAur14` with an L2 penalty of
:math:`10^{-5}` on the parameters. Then
:math:`G(\sigma) = E(\sigma) - k_BT \ln P(\sigma)`. Each combination's value draws on
every frame that shares one of its substituents or pairs of substituents, so it is less
noisy than the histogram and exists for combinations seen rarely or not at all. A
combination is left out when one of its substituents, or one of its pairs between
sites, was never sampled. For a single site the model and the histogram are the same
estimate. The model is fitted only when every replica ran with the same biases.

The uncertainty of both is the standard deviation over 50 bootstrap samples of the
replicas, so it reflects the spread between independent trajectories. With one replica
there is no uncertainty.

Output
------

``mllf production analyze`` writes ``<out_dir>/analysis`` (``--out`` to change it):

* ``tables/free_energies.csv``: each combination's substituents (and their numbers in
  the full system, from the combination's ``mapping.json``), frames, G by WHAM and by
  the Potts model with their SDs, and each replica's own histogram value
* ``tables/sampling.csv`` and ``tables/replicas.csv``: per replica and substituent the
  fraction of frames at the end state and entries per ns; per replica the ns analyzed,
  the fraction of frames with every site at an end state, and transitions per ns at
  each site
* ``tables/potts_h.csv`` and ``tables/potts_J.csv``: the fitted model in :math:`k_BT`
  (state 0: no substituent at the cutoff), from which any combination can be rebuilt
* ``figures/populations``: frames at each substituent's end state, one bar per
  substituent with each replica's frames stacked, one panel per site. The dashed line
  is the height every bar would reach with the site's end-state frames split evenly,
  so a short bar marks an undersampled substituent and a bar of mostly one color a
  substituent reached by only one replica
* ``figures/lambda_trajectories``: :math:`\lambda` of every substituent over the
  analyzed production, one row per replica (the first eight) and one column per site.
  Frequent crossings mean a well-mixed replica; a substituent absent for long stretches
  or a replica held at one end state shows here before it shows in the averages
* ``figures/profiles``: the biased free-energy profile of each substituent's
  :math:`\lambda`, all replicas pooled, one panel per site; flat profiles with similar
  end states mean the biases sampled every substituent evenly
* ``figures/estimator_parity``: the Potts model against WHAM, for two or more sites
* ``report.txt``: the inputs, the sampling of each replica (flagging substituents below
  1% of frames or 2 entries per ns), and the free energies

The analysis also takes ``--discard-ns`` (production dropped from the start of each replica),
``--cutoff``, ``--bootstrap``, ``--stride``, ``--no-potts``, ``--format pdf`` and
``--only FIGURE ...``.
