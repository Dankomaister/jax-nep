# JAX-NEP

JAX implementation of NEP4 interatomic potentials for use with JAX and JAX-MD.
Load an existing `nep.txt` model and evaluate periodic systems with autodiff
energies, forces, and configurational virial/stress.

## Features

- Scalar NEP4 and NEP4 + ZBL models.
- Local and total energies; forces through ordinary JAX autodiff.
- Virial and stress through affine-strain autodiff.
- Native JAX-MD neighbor lists with automatic STANDARD/MULTI_IMAGE periodic handling.
- JIT and GPU execution, with float32 as the default.
- Vectorized ensemble force deviation and differentiable active-learning scores.
- Fitted model parameters as JAX PyTree leaves for parameter autodiff.

## Installation

From a cloned or downloaded source tree, using Python 3.12 or later:

```bash
python -m pip install .
```

For editable installation with tests:

```bash
python -m pip install -e ".[test]"
```

Runtime requirements are JAX >=0.11.1, JAX-MD >=0.2.29, and NumPy >=2.

Install a JAX build compatible with your accelerator using the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

See [installation details](docs/installation.md).

## Basic usage

Save a suitable carbon-containing NEP4 model as `nep.txt` in the working
directory. This two-atom example demonstrates the API; choose a model and
structure appropriate to your scientific application.

```python
import jax
import jax.numpy as jnp

from jax_nep import load_model, nep_neighbor_list, stress, virial


# Use a NEP4 model whose species include carbon.
model = load_model("nep.txt")
species = model.map_species(["C", "C"])

R = jnp.array(
    [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]],
    dtype=jnp.float32,
)
box = 12.0  # cubic cell, Angstrom

neighbor_fn, energy_fn = nep_neighbor_list(model, box, species)

# Allocation runs outside JIT.
neighbor = neighbor_fn.allocate(R)
neighbor = jax.jit(neighbor_fn.update)(R, neighbor)

if bool(neighbor.did_buffer_overflow):
    raise RuntimeError(
        "Neighbor capacity exceeded; reallocate before evaluating."
    )

energy = jax.jit(energy_fn)(R, neighbor=neighbor)
forces = -jax.grad(lambda x: energy_fn(x, neighbor=neighbor))(R)
local_energies = energy_fn.local_energy(R, neighbor=neighbor)

W = virial(energy_fn, R, neighbor=neighbor)
sigma = stress(
    energy_fn,
    R,
    neighbor=neighbor,
    volume=box**3,
)

print("Energy (eV):", energy)
print("Forces (eV/Angstrom):", forces)
print("Local energies (eV):", local_energies)
print("Virial (eV):", W)
print("Stress (eV/Angstrom^3):", sigma)
```

Positions and box lengths are in Angstrom; energies are in eV. Species IDs
follow `model.species`, the model header order. A box may be a scalar, three
lengths, or a right-handed `(3, 3)` matrix with lattice vectors **as rows**.
The box and species are fixed after factory creation.

The virial `W` is `-dE/d eps` for row-vector deformation
`r @ (I + eps)`. Stress is tensile-positive `-W / volume`, with no kinetic
contribution.

See [virial/stress conventions](docs/virial_stress.md).

## JAX-MD simulation

Continue the example above with a short NVE trajectory:

```python
from jax_md import simulate, space


# Continue with model, R, neighbor_fn, energy_fn, and neighbor from above.
# Refresh connectivity at the positions where the integrator evaluates forces.
def md_energy(positions, *, neighbor, **unused):
    current = neighbor_fn.update(positions, neighbor)
    return energy_fn(positions, neighbor=current)


# Unwrapped Cartesian motion; the potential handles periodic images.
_, shift = space.free()

initialize, advance = simulate.nve(md_energy, shift, dt=0.01)

state = initialize(
    jax.random.key(0),
    R,
    kT=0.025,
    mass=jnp.array([12.011, 12.011]),
    neighbor=neighbor,
)


@jax.jit
def step(state, neighbor):
    candidate = advance(state, neighbor=neighbor)
    updated = neighbor_fn.update(candidate.position, neighbor)
    return candidate, updated


for _ in range(10):
    candidate, updated = step(state, neighbor)

    if bool(updated.did_buffer_overflow):
        # Keep the last valid state; do not accept this trial step.
        raise RuntimeError(
            "Neighbor overflow: increase capacity and restart the step."
        )

    state, neighbor = candidate, updated

print("Final positions (Angstrom):", state.position)
```

JAX-MD uses consistent units: with eV, Angstrom, and masses in atomic mass
units, one time unit is approximately 10.18 fs. Here `dt=0.01` is about
0.102 fs and `kT=0.025` is in eV. Select a stable time step for your model.

The example keeps Cartesian positions unwrapped and checks neighbor capacity
before accepting each step.

See [JAX-MD integration](docs/jax_md.md) for lifecycle details.

## Supported models

The scalar `nep4` and `nep4_zbl` formats support species-dependent cutoffs,
harmonic degrees 1–8, higher-body invariants, and
universal/typewise/flexible ZBL.

Unsupported NEP versions and charge, dipole, polarizability, and
temperature-dependent families are rejected explicitly.

See [models and species](docs/models.md).

## Public API

The top-level package exports:

```python
from jax_nep import (
    NEPModel,
    ModelDeviation,
    nep_model_deviation,
    smooth_max,
    load_model,
    nep_neighbor_list,
    stress,
    virial,
)
```

See the [public API documentation](docs/api.md) for details and
[ensemble model deviation](docs/active.md) for committee scoring and its equations.

## Testing

Install the test dependencies and run:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

Tests use stored NEP_CPU reference fixtures and do not require NEP_CPU.
The suite disables persistent compilation caching for reference validation.

See [numerical behavior and GPU validation](docs/numerical_behavior.md).

## Documentation

Start with the:

- [documentation index](docs/index.md)
- [quickstart](docs/quickstart.md)
- [public API](docs/api.md)
- [installation guide](docs/installation.md)
- [model and species documentation](docs/models.md)
- [JAX-MD integration guide](docs/jax_md.md)
- [virial/stress conventions](docs/virial_stress.md)
- [scientific references and source attribution](docs/reference.md)

The Python examples in this README and the documentation are exercised by
tests.

## License

Copyright (C) 2026 Daniel Hedman.

JAX-NEP is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

JAX-NEP is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU General Public License for more details.

See the full [GPL-3.0-or-later license](LICENSE).

## References and attribution

NEP definitions and reference data follow GPUMD/NEP and NEP_CPU; computation
and simulation interfaces use JAX and JAX-MD.

See [scientific references and source attribution](docs/reference.md).

This project is not endorsed or maintained by those upstream projects.
