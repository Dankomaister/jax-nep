# Quickstart

Install the package as described in [installation](installation.md). Put your
carbon-containing `nep.txt` in the working directory and run:

```python
import jax
import jax.numpy as jnp
from jax_nep import load_model, nep_neighbor_list, virial, stress

# Use a NEP4 model whose species include carbon.
model = load_model("nep.txt")
species = model.map_species(["C", "C"])
R = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=jnp.float32)
box = 12.0  # cubic cell, Angstrom
neighbor_fn, energy_fn = nep_neighbor_list(model, box, species)
neighbor = neighbor_fn.allocate(R)  # allocation runs outside JIT
neighbor = jax.jit(neighbor_fn.update)(R, neighbor)
if bool(neighbor.did_buffer_overflow):
    raise RuntimeError("Neighbor capacity exceeded; reallocate before evaluating.")

energy = jax.jit(energy_fn)(R, neighbor=neighbor)
forces = -jax.grad(lambda x: energy_fn(x, neighbor=neighbor))(R)
local_energies = energy_fn.local_energy(R, neighbor=neighbor)
W = virial(energy_fn, R, neighbor=neighbor)
sigma = stress(energy_fn, R, neighbor=neighbor, volume=box**3)
print("Energy (eV):", energy)
print("Forces (eV/Angstrom):", forces)
print("Virial (eV):", W)
print("Stress (eV/Angstrom^3):", sigma)
```

For a downloadable source tree, this code is also `examples/basic.py`. It requires
your model file; the reference tests supply their own models automatically.

Replace the symbols and positions together for your system. `map_species`
converts chemical symbols to integer indices into `model.species`; they are not
atomic numbers. Positions have shape `(N, 3)` and the species array has shape
`(N,)`. Model parameters and positions use float32 by default.

The example uses a cubic box. A length vector specifies an orthorhombic box; a
`(3, 3)` matrix specifies a cell with lattice vectors as rows. The potential
handles periodicity internally. Do not pass fractional positions.

Allocation happens outside JIT. After moving atoms, call `neighbor_fn.update`
and check overflow before using the returned state for energies or forces.
A fixed neighbor state must cover the evaluated positions. For changing positions
inside an integrator, follow the [complete JAX-MD example](jax_md.md).

`energy_fn.local_energy` includes ZBL when present; summing it gives the total
energy. `energy_fn.descriptors` returns the scaled descriptor matrix. See
[API](api.md) for all signatures and [virial/stress](virial_stress.md) for tensor
signs and units.
