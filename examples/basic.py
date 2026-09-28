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
