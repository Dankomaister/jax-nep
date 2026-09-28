# Public API

The stable top-level exports are `NEPModel`, `load_model`, `nep_neighbor_list`,
`virial`, `stress`, `ModelDeviation`, `nep_model_deviation`, and `smooth_max`.
Import these from `jax_nep`. There is no `load_nep` alias.
Internal implementation classes and modules are not part of the stable API.

## `load_model(path, *, dtype=numpy.float32) -> NEPModel`

`path` is a filesystem path or path-like object to a NEP4 text model.
`dtype` accepts float32 or float64. Float64 use requires the caller to enable
JAX x64 before constructing the potential. The return value contains parsed,
validated, read-only model parameters. See [supported formats](models.md).

## `NEPModel`

Normally obtained from `load_model`, rather than constructed by hand.

| Member | Meaning |
|---|---|
| `species` | Tuple of model chemical symbols in type-ID order |
| `map_species(symbols)` | Map an iterable of symbols to an int32 array, shape `(N,)` |
| `validate_types(species)` | Validate and return integer type IDs, shape `(N,)` |
| `cutoff` | Maximum physical cutoff in Angstrom |
| `cutoffs` | Per-species radial/angular cutoffs, shape `(S, 2)`, in Angstrom |
| `dimension` | Number of scaled descriptor components per atom |
| `orders`, `basis_orders` | Radial/angular expansion and basis orders |
| `l_max`, `invariants` | Harmonic degree and enabled higher-body labels |

The other dataclass fields hold fitted coefficients, scalers, network and ZBL
parameters. Ordinary use does not require manipulating them.

## `nep_neighbor_list(model, box, species, *, skin=1.0, capacity_multiplier=1.25, disable_cell_list=False)`

Returns `(neighbor_fn, energy_fn)` bound to the given model, box, and species.

- `model`: a `NEPModel`.
- `box`: positive scalar cubic length, three orthorhombic lengths, or a finite
  right-handed `(3, 3)` cell with lattice vectors **as rows**, in Angstrom.
- `species`: nonempty integer model type IDs, shape `(N,)`.
- `skin`: finite nonnegative neighbor buffer in Angstrom.
- `capacity_multiplier`: finite allocation headroom, at least 1.
- `disable_cell_list`: boolean controlling cell-list acceleration where supported
  by the native neighbor backend.

The box and species are fixed for the lifetime of the returned factory.
Positions are Cartesian arrays of shape `(N, 3)`; use the model's floating dtype.

### Neighbor lifecycle

`neighbor_fn.allocate(R, extra_capacity=0)` allocates a neighbor state outside
JIT. `extra_capacity` adds native allocator headroom beyond the capacity
multiplier. `neighbor_fn.update(R, neighbor)` returns an updated state and supports
JIT. Retain this return value. States belong to their factory: raw JAX-MD lists
and states from another factory raise `ValueError`.

Inspect `neighbor.did_buffer_overflow` after allocation and updates, before
accepting computed results. `neighbor.error` exposes the native partition error
status. Overflow does not automatically raise a Python exception within JIT.
Reallocate outside JIT with enough headroom and retry from a valid state.

### Energy callable

| Call | Return shape and units |
|---|---|
| `energy_fn.apply(model, R, *, neighbor)` | Scalar energy with explicit fitted parameters, eV |
| `energy_fn(R, *, neighbor)` | Scalar total potential energy, eV |
| `energy_fn.local_energy(R, *, neighbor)` | `(N,)` local energies including ZBL, eV |
| `energy_fn.descriptors(R, *, neighbor)` | `(N, model.dimension)` scaled descriptors |

All four support JIT. Use `-jax.grad` of total energy for `(N, 3)` forces in
eV/Angstrom; neighbor connectivity is supplied explicitly. Descriptors are
scaled by the fitted scaler and ordered radial first, then angular invariants.
An optional `perturbation` keyword accepts a `(3, 3)` deformation matrix acting
on pair vectors; the virial helper uses this for infinitesimal differentiation.
It is not a way to update the box for a finite-strain trajectory.

## `virial(energy_fn, positions, *, neighbor)`

Returns the total configurational virial, a `(3, 3)` Cartesian tensor in eV,
by differentiating the same energy under affine strain. Pass the matching
`energy_fn` and neighbor state. Supports JIT.

## `stress(energy_fn, positions, *, neighbor, volume)`

Returns tensile-positive configurational stress, shape `(3, 3)`, in
eV/Angstrom³: `-virial / volume`. Supply the positive physical cell volume in
Angstrom³, normally `abs(det(cell))`; volume is not inferred or validated by
this helper. No kinetic contribution is included. See
[virial and stress conventions](virial_stress.md).

## Active learning and smooth math

See [ensemble model deviation](active.md) for the committee factory, minimal
`ModelDeviation` result, force evaluation, and generic `smooth_max` utility.
See [parameter PyTrees](models.md#parameter-pytree-and-explicit-model-energy)
for differentiation with respect to fitted parameters.
