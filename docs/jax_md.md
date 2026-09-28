# JAX-MD integration

JAX-NEP follows the `(neighbor_fn, energy_fn)` factory pattern. Allocate outside
JIT, then update and evaluate with JAX arrays. The following complete example
uses a carbon-containing `nep.txt` in the working directory:

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
state = initialize(jax.random.key(0), R, kT=0.025,
                   mass=jnp.array([12.011, 12.011]), neighbor=neighbor)

@jax.jit
def step(state, neighbor):
    candidate = advance(state, neighbor=neighbor)
    updated = neighbor_fn.update(candidate.position, neighbor)
    return candidate, updated

for _ in range(10):
    candidate, updated = step(state, neighbor)
    if bool(updated.did_buffer_overflow):
        # Keep the last valid state; do not accept this trial step.
        raise RuntimeError("Neighbor overflow: increase capacity and restart the step.")
    state, neighbor = candidate, updated
print("Final positions (Angstrom):", state.position)
```

The same example is available as `examples/nve.py`. `md_energy` refreshes
connectivity at the force-evaluation positions, including the new positions
inside velocity Verlet. The outer loop checks the corresponding returned
neighbor state before accepting a trial step. If capacity overflows, discard the
candidate; reallocate with greater capacity outside JIT and retry from the last
valid state. A saved force computed with incomplete connectivity is invalid too.

This uses a fixed box and unwrapped Cartesian positions via `space.free()`.
Periodicity belongs to the potential; no JAX-MD fractional-coordinate convention
is imposed on callers. Do not pass a changing box through `simulate.nve` keywords:
create a new potential and neighbor factory for a finite change in cell.

With eV, Angstrom, and atomic mass units, JAX-MD's consistent time unit is about
10.18 fs. Thus the example's `dt=0.01` is about 0.102 fs; `kT=0.025` is in eV.
The library does not attach units or choose a physically stable time step.

## Neighbor settings

`skin` is a nonnegative distance buffer, default 1 Angstrom. It provides
connectivity headroom between rebuilds. Updates normally rebuild when displacement
exceeds half the skin; periodic image shifts can also require rebuilding at a
boundary crossing. Skin zero requests rebuilding on each update.

`capacity_multiplier`, default 1.25, reserves storage above the observed neighbor
count. Increase it if a trajectory produces overflow. Extra headroom can also be
requested through `allocate(R, extra_capacity=...)`. Updates keep capacity fixed;
allocation can change shapes and trigger recompilation.

`disable_cell_list=False` permits native cell-list acceleration where the backend
supports it. Set it to `True` to disable that acceleration. It does not change
which physical interactions are represented.

The library automatically uses multi-image periodic handling when the physical
NEP cutoff requires more than the minimum periodic image. The criterion compares
the physical cutoff with half the shortest nonzero lattice vector, including
skew cells; equality uses STANDARD. The neighbor skin does not change this
physical criterion. No supercell construction is required from the caller.

Neighbor states belong to their factory. Do not substitute a raw JAX-MD list or
a state created for another model, box, or species array. See
[numerical behavior](numerical_behavior.md) for safety and static-shape details.
