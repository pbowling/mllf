Background
==========

Multisite λ-Dynamics
--------------------

Alchemical free-energy calculations are a standard way to rank chemical modifications
before synthesis :cite:`DuaLupWan20, GapMicSee16`. Free Energy Perturbation :cite:`Zwa54`
and Thermodynamic Integration :cite:`Kir35` evaluate one perturbation at a time, which
becomes the bottleneck when ligands must be assessed across many modifications or many
protein variants.

Multisite λ-dynamics (:math:`\mathrm{MS}\lambda\mathrm{D}`) instead gives each
substituent at each site a dynamic λ coordinate, so a single simulation samples many
substitutions together, with precision comparable to single-perturbation methods
:cite:`SeedeGr10b, HayVilBro18`. The free energy of substituent :math:`i` follows from the
time it spends in a physical state:

.. math::

   \Delta G(i) = -kT \ln \left( \prod_{s=1}^{M} \int dt\, \Theta (\lambda_{si,t} - \lambda_c) \right) - U_{bias}(\lambda_i)

where :math:`\Theta` counts frames in which :math:`\lambda_{si}` exceeds the cutoff
:math:`\lambda_c` (usually 0.99), and the integral is a sum over saved frames. The estimate
is only as good as the sampling. A substituent that λ rarely visits, or a pair that λ
rarely crosses between, gives an uncertain or missing ΔΔG.

Bias Potentials
---------------

Sampling is made uniform with a bias potential :math:`U_{bias}` that flattens the
alchemical landscape. It has a linear term and three nonlinear terms
:cite:`HayArmVil17, HayVilBro18`, each correcting a different feature of that landscape:

* **Linear (b)** offsets each substituent's free energy so that every substituent at a
  site is equally populated:

  .. math::

     U_{\text{linear}} = \sum_{s}^{M} \sum_{i}^{N_s} \phi_{si} \lambda_{si}

* **Quadratic (c)** lowers the barrier between two substituents, which is largest at the
  midpoint of λ:

  .. math::

     U_{\text{quad}} = \sum_{s}^{M} \sum_{i}^{N_s} \sum_{t}^{M} \sum_{j}^{N_t} \Psi_{si,tj} \lambda_{si} \lambda_{tj}

* **Skew (x)** corrects an asymmetric barrier, one tilted toward one substituent:

  .. math::

     U_{\text{skew}} = \sum_{s}^{M} \sum_{i}^{N_s} \sum_{t}^{M} \sum_{j}^{N_t} \chi_{si,tj} \lambda_{tj} \left(1 - \exp\left(-\frac{\lambda_{si}}{\sigma}\right)\right)

* **End (s)** acts just short of an endpoint, where a substituent appearing or vanishing
  pays the cost of displacing its surroundings:

  .. math::

     U_{\text{end}} = \sum_{s}^{M} \sum_{i}^{N_s} \sum_{t}^{M} \sum_{j}^{N_t} \frac{ \omega_{si,tj} \lambda_{si} \lambda_{tj} } {\alpha +  \lambda_{si}}

with :math:`\sigma = 0.18` and :math:`\alpha = 0.017`; self-terms (:math:`si = tj`) are
omitted. Terms with :math:`s \ne t` couple sites and can matter for sampling
:cite:`HayVilBro18`; ``mllf`` predicts the within-site terms only.

Finding the Biases
------------------

The biases are system-specific and must be found before production. Adaptive Landscape
Flattening (ALF) does this iteratively :cite:`HayArmVil17, HayVilBro18`. Many short runs
correct gross errors, typically tens to a few hundred. A smaller number of longer runs
then refine the biases, often 10–20. Only after that do production runs, from
nanoseconds to hundreds of nanoseconds, give the ΔΔG estimates. Each correction is
computed from the previous run of the same system, so the cost is paid again for every
system. ``mllf`` aims to reduce that cost by predicting the biases from a model trained
across systems and correcting them with fewer runs (:doc:`cb_setup`, :doc:`workflow`).
