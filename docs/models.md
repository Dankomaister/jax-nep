# Models and species

`load_model(path)` reads a scalar `nep4` or `nep4_zbl` text file, commonly named
`nep.txt`. Parameters are float32 by default. Use a potential fitted and validated
for the chemistry and configurations of your application; the small test models
are numerical fixtures, not recommendations for physical simulations.

## Species IDs

`model.species` is the tuple of chemical symbols in the model header's order.
Use `model.map_species(symbols)` to obtain an int32 array of type IDs, one per
atom, in the same order as the positions. IDs are zero-based model indices,
**not atomic numbers**. A model may contain more species than are present in a
particular configuration. Unknown symbols raise `ValueError` listing the expected
symbols. The factory rejects empty, noninteger, negative, and out-of-range IDs.

Changing atom count or species requires a new neighbor/energy factory.

## Supported NEP4 features

- Shared or species-dependent radial and angular cutoffs; each pair uses their
  arithmetic mean for its domain.
- Expansion and basis orders 0–16, harmonic degrees 1–8.
- Higher-body invariants 222, 1111, 112, 123, 233, and 134, with compatible
  harmonic degrees.
- A species-dependent single hidden-layer scalar energy network and descriptor
  scalers.
- Universal, typewise, and flexible ZBL repulsion. Flexible ZBL parameters are
  read for each unordered species pair. ZBL outer cutoffs must fit within the
  corresponding angular cutoff.

Only the scalar `nep4` and `nep4_zbl` model families are accepted. NEP1/2/3/5,
charge/qNEP, dipole, polarizability, and temperature-dependent families are not
supported. Unsupported family names raise `ValueError`; malformed headers,
invalid orders, nonfinite parameters, and incorrect parameter counts also fail
explicitly. Missing files raise the usual filesystem error.

The parsed `NEPModel` is frozen and its parameter arrays are read-only. Treat it
as immutable. It contains model data, independent of a particular simulation.
For mathematical conventions and attribution, see [references](reference.md).

## Parameter PyTree and explicit-model energy

`NEPModel` is a registered JAX PyTree. Its fitted leaves are `input_weights`,
`hidden_bias`, `output_weights`, `output_bias`, the two `coefficients` arrays,
`scale`, and (only in flexible ZBL mode) `zbl`. Species, atomic numbers, cutoffs,
orders, invariant definitions, ZBL mode and fixed ZBL tables remain static.
Loaded parameter arrays retain their dtype and read-only NumPy representation;
JAX transformations produce device-array leaves without changing static metadata.

The existing energy callable remains bound to its original model.
`energy_fn.apply(model, R, neighbor=neighbor)` instead accepts fitted parameters
explicitly, using the same physical kernels and fixed neighbor architecture:

```python
import jax
import jax.numpy as jnp
from jax_nep import load_model, nep_neighbor_list

model = load_model("nep.txt")
R = jnp.array([[0., 0., 0.], [1.4, .2, .1]], dtype=jnp.float32)
provider, energy_fn = nep_neighbor_list(model, 12., model.map_species(["C", "C"]))
neighbor = provider.allocate(R)
parameter_gradient = jax.jit(jax.grad(
    lambda m: energy_fn.apply(m, R, neighbor=neighbor)
))(model)
```

The gradient has the same fitted-leaf tree. Use `dataclasses.replace` to update
selected fields, or named paths from `jax.tree_util.tree_flatten_with_path` for
parameter masks. Parameter updates must preserve architecture and leaf shapes;
new structure requires a new factory. Future fitting code must enforce physical
constraints such as positive scales and valid flexible-ZBL radii. No optimizer or
training pipeline is provided.
