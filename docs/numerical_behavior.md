# Numerical behavior

## Precision and shapes

Float32 is the default for parameters, positions, energies, and forces. Virial
uses scoped float64 for the affine map, strain cotangent accumulation, and ANN
projection reduction. This does not globally enable x64 or turn ordinary force
evaluation into a float64 calculation. Optional float64 models require JAX x64
to be enabled by the caller before potential construction.

Keep atom count, species, box, and array shapes fixed within a compiled simulation.
Allocation is a host operation. Neighbor updates and energy/force/virial
calculations support `jax.jit`. Allocation chooses fixed connectivity storage
and domain structure; ordinary updates do not switch representations. Allocating
a new capacity may require recompilation. Policy uses geometry and connectivity
capacities, not model filenames, system names, or benchmark identities.

## Neighbor validity

Check `neighbor.did_buffer_overflow` after allocation and every update before
accepting results. Overflow means stored connectivity may be incomplete; values
computed with that state must be discarded. Reallocate outside JIT with greater
capacity and retry from the last valid simulation state. The library rejects
raw or foreign neighbor states. Use the factory that produced your energy
function, and always use the updated state returned by `update`.

## Periodic positions and degenerate pairs

Positions may be unwrapped across periodic cells. The potential handles image
geometry internally, including explicit image shifts when required. Very large
coordinates still lose low-order spatial information at float32 precision;
wrap or recenter coordinates to retain that information in long trajectories.
Finite masking prevents invalid padded entries from entering singular distance
or high-order angular calculations.

Exactly coincident pairs have no interaction. Nonzero distances whose squared
norm underflows use finite guarded arithmetic; angular directions are degenerate
in this limit, and this is not a physically meaningful close-contact regime.
ZBL guards its inverse distance as well. These safeguards are not a substitute
for a sensible initial structure and integration time step.

## Validation

```bash
python -m pytest
```

The suite includes parser, species, neighbor, periodic geometry, autodiff,
policy, numerical-safety, and documentation-example tests. It compares scaled
descriptors, local/total energies, forces, and virial against 31 stored NEP_CPU
fixtures. Stress is checked against the stated virial/volume convention. Neither
NEP_CPU nor other external source checkouts are needed to run tests. Original
reference model text and provenance accompany each fixture.

`tests/conftest.py` disables persistent JAX compilation caching for canonical
validation; this applies even if a cache directory is configured in the environment.
The full GPU validation command, after installing an appropriate GPU JAX build, is:

```bash
JAX_PLATFORMS=cuda python -m pytest
```

For a shorter CPU check suitable for basic CI:

```bash
JAX_PLATFORMS=cpu python -m pytest tests/test_parser.py tests/test_public_contract.py
```

The full suite also runs on CPU but compilation can take longer. Passing a short
CPU check does not replace full numerical validation. Tests use the fixed budgets
in `tests/tolerances.json`; reference fixtures are not regenerated during testing.
