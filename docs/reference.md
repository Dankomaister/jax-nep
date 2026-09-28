# Scientific references and attribution

JAX-NEP implements neuroevolution potential (NEP) semantics for use with JAX.
The following projects and publications provide the scientific definitions,
reference data, and computational interfaces:

- [GPUMD / NEP documentation](https://gpumd.org/potentials/nep.html) and
  [GPUMD source](https://github.com/brucefan1983/GPUMD).
- Z. Fan et al., *Neuroevolution machine learning potentials: Combining high
  accuracy and low cost in atomistic simulations and application to heat
  transport*, Physical Review B **104**, 104309 (2021),
  [doi:10.1103/PhysRevB.104.104309](https://doi.org/10.1103/PhysRevB.104.104309).
- [NEP_CPU](https://github.com/brucefan1983/NEP_CPU), with stored numerical
  references generated using commit
  [`af615ce0e81bbb860e77923762c505567affa8d1`](https://github.com/brucefan1983/NEP_CPU/tree/af615ce0e81bbb860e77923762c505567affa8d1).
- [JAX](https://github.com/jax-ml/jax) for array computation, JIT, and automatic
  differentiation; see its [documentation](https://docs.jax.dev/).
- [JAX-MD](https://github.com/jax-md/jax-md) for native neighbor lists and
  molecular-dynamics interfaces; see its [documentation](https://jax-md.readthedocs.io/).

This project is independently maintained. These references do not imply
upstream authorship, endorsement, or maintenance of JAX-NEP.

## Source attribution and license

JAX-NEP is distributed under **GPL-3.0-or-later**; see [LICENSE](../LICENSE).
Angular normalization constants, polynomial coefficients, higher-body invariant
polynomials, ZBL constants, covalent radii, and model packing conventions are
adapted from NEP_CPU, Copyright 2022 Zheyong Fan, Junjie Wang, Eric Lindgren,
also GPL-3.0-or-later. Source attribution is retained in the mathematical tables
and formulas. `tools/extract_tables.py` and `tools/extract_invariants.py` provide
maintenance utilities for these mathematical definitions using the pinned
NEP_CPU `src/nep_utilities.h`; NEP_CPU is not a runtime dependency.

Reference NPZ files contain model text, input geometry, species IDs, expected
observables, the reference commit, and a header-normalization flag. Where needed,
fixture generation temporarily normalized `q222=2` to `q222=1` to select the
reference implementation's current descriptor path. Fitted parameters and the
original model text stored in fixtures were not changed. Tests read stored arrays
and never run the reference implementation.

## Mathematical conventions

For a radial or angular domain, the pair cutoff is the arithmetic mean of the
two species cutoffs. With `u = r / cutoff`, the basis inside the cutoff is
`f_k = (T_k(2*(u-1)^2-1) + 1) * (1 + cos(pi*u)) / 4`, and zero outside.
`T_k` is a Chebyshev polynomial. Species-pair coefficients project this basis
to radial channels `g_n = sum_k c[n,k,ti,tj] * f_k`.

Radial descriptors sum `g_n` over directed neighbors. Angular moments sum
`g_n * Y_lm(direction)` using angular cutoffs and coefficients. Spherical
harmonics are real Cartesian polynomials ordered by degree, then `m=0`, then
real and imaginary components for `m=1..l`. Quadratic invariants weight `m>0`
components twice. Higher-body labels are 222, 1111, 112, 123, 233, and 134.
Descriptors concatenate radial orders first, followed by each angular
invariant's radial orders, and multiply by the fitted descriptor scaler.

The local neural-network energy is
`sum_h w_out[h] * tanh(sum_d w_in[d,h]*q[d] - b_hidden[h]) - b_output`.
Weights and hidden biases belong to the central species; the subtracted final
output bias is shared by the model. Total energy sums local energies.

Universal ZBL uses four screened exponential terms, Coulomb factor 14.399645,
and screening inverse length `2.134563*(Zi**0.23+Zj**0.23)`. Directed edges
contribute half the pair energy to local energy. Flexible ZBL reads ten values
per unordered species pair; typewise ZBL uses zero inner radius and limits the
outer radius by the scaled sum of covalent radii. Forces and strain derivatives
follow the [public conventions](virial_stress.md).
