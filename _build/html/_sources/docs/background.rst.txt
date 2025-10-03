Background
==========

Introduction
------------

The vast combinatorial space of possible chemical changes, combined with the
need to rapidly locate potent candidates that remain active against multiple
resistant variants, makes purely experimental screening impractical. That
constraint is a primary reason alchemical free-energy calculations are now a
core tool in lead optimization: they enable in silico triage that focuses
synthetic effort on the most promising modifications :cite:`DuaLupWan20, GapMicSee16`.

This project focuses on methods that expand throughput by evaluating many
alchemical changes within a single simulation. Multisite :math:`\lambda`-dynamics
(:math:`\mathrm{MS}\lambda\mathrm{D}`) parameterizes alternative chemical states with :math:`\lambda` coordinates
so multiple substitutions can be sampled together. When combined with
automated biasing strategies, this approach scales combinatorial exploration
while maintaining precision comparable to many single-perturbation
calculations :cite:`SeedeGr10b, HayVilBro18`.

Motivation and workflow
-----------------------

Classic single-perturbation alchemical methods (for example Free Energy
Perturbation (FEP) :cite:`Zwa54` or Thermodynamic Integration (TI) :cite:`Kir35`)
have proven predictive for many systems, but they are conventionally applied
one mutation at a time. For projects that must assess ligands across many
protein variants, that serial strategy becomes a throughput bottleneck. Using
multisite sampling with tuned bias potentials reduces that bottleneck and
supports broader scanning of chemical space :cite:`GapMicSee16`.

To make multisite sampling efficient we tune bias potentials using Adaptive
Landscape Flattening (ALF). ALF is typically staged: many short flattening
runs rapidly reduce gross bias errors, a refinement stage with longer cycles
fine-tunes bias amplitudes, and final production runs collect the data used
for the ultimate :math:`\Delta\Delta G` estimates. This staged approach concentrates
computational effort where it yields the most value :cite:`HayArmVil17, HayVilBro18`.

Typical ALF workflows use many short Phase I cycles (tens to a few hundred
short runs) for rapid adjustment, a smaller number of longer Phase II
refinement cycles (e.g., 10–20), and then multiple production trajectories
(from nanoseconds to hundreds of nanoseconds) for final estimates. Good
equilibration practice (for example discarding the initial portion of each
cycle) and clear convergence criteria are important to obtain stable bias
estimates and reproducible :math:`\Delta\Delta G` values.

Free energy estimate
--------------------

.. math::

   \Delta G(i) = -kT \ln \left( \prod_{s=1}^{M} \int dt \Theta (\lambda_{si,t} - \lambda_c) \right) - U_{bias}(\lambda_i)


In this expression, :math:`\lambda_{si,t}` is the instantaneous :math:`\lambda`
value for substituent :math:`i` at site :math:`s` and time :math:`t`. The cutoff
:math:`\lambda_c` (commonly near 0.99) and the Heaviside indicator :math:`\Theta`
are used to count frames where the substituent is effectively in a physical
state; the time integral is approximated by a discrete sum over simulation
frames.

Bias decomposition
------------------

ALF decomposes the total bias into a linear term and three nonlinear
components (quadratic, skew, and end). These are typically estimated in the
ALF stages and combined to form :math:`U_{bias}` :cite:`HayArmVil17, HayVilBro18`.

Linear term

.. math::

      U_{\text{linear}} = \sum_{s}^{M} \sum_{i}^{N_s} \phi_{si} \lambda_{si}


Nonlinear terms

Quadratic coupling:

.. math::

      U_{\text{quad}} = \sum_{s}^{M} \sum_{i}^{N_s} \sum_{t}^{M} \sum_{j}^{N_t} \Psi_{si,tj} \lambda_{si} \lambda_{tj}


Skew term:

.. math::

      U_{\text{skew}} = \sum_{s}^{M} \sum_{i}^{N_s} \sum_{t}^{M} \sum_{j}^{N_t} \chi_{si,tj} \lambda_{tj} \left(1 - \exp\left(-\frac{\lambda_{si}}{\sigma}\right)\right)


End term:

.. math::

      U_{\text{end}} = \sum_{s}^{M} \sum_{i}^{N_s} \sum_{t}^{M} \sum_{j}^{N_t} \frac{ \omega_{si,tj} \lambda_{si} \lambda_{tj} } {\alpha +  \lambda_{si}}


We set :math:`\sigma = 0.18` and :math:`\alpha = 0.017` in practice, and self-terms
where :math:`si = tj` are neglected. The coefficients :math:`\Psi, \chi, \omega`
capture inter-site coupling and nonlinear bias effects; such coupling can
substantially affect sampling and therefore free-energy estimates :cite:`HayVilBro18`.

References
----------

.. bibliography:: references.bib
   :style: unsrt
