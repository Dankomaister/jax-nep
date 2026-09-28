"""Public energy callable and differentiation of the same physical expression."""

import numpy as np
import jax
import jax.numpy as jnp
from .policy import PlannedNeighbors
from .descriptors import geometry, descriptors
from .network import local_network
from .zbl import local_zbl


class Energy:
    def __init__(self, model, provider):
        self.model, self.provider = model, provider

    def _pairs(self, positions, neighbor, perturbation):
        def deform(pairs):
            if pairs is None or perturbation is None:
                return pairs
            with jax.enable_x64():
                v = jnp.sum(
                    pairs.vectors.astype(jnp.float64)[..., :, None]
                    * perturbation.astype(jnp.float64),
                    axis=-2,
                ).astype(positions.dtype)
            return pairs._replace(vectors=v)

        pairs, angular = self.provider.pair_domains(positions, neighbor)
        return deform(pairs), deform(angular)

    def descriptors(self, positions, *, neighbor, perturbation=None):
        pairs, angular = self._pairs(positions, neighbor, perturbation)
        return descriptors(self.model, pairs, angular_pairs=angular)

    def local_energy(self, positions, *, neighbor, perturbation=None):
        pairs, angular = self._pairs(positions, neighbor, perturbation)
        g = geometry(pairs)
        q = descriptors(self.model, pairs, g, angular)
        local = local_network(
            self.model,
            q,
            self.provider.types,
            accumulation_dtype=jnp.float64 if perturbation is not None else None,
        )
        if self.model.zbl is not None:
            local += local_zbl(self.model, pairs, g)
        return local

    def __call__(self, positions, *, neighbor, perturbation=None):
        return jnp.sum(
            self.local_energy(positions, neighbor=neighbor, perturbation=perturbation)
        )


def nep_neighbor_list(
    model, box, species, *, skin=1.0, capacity_multiplier=1.25, disable_cell_list=False
):
    """Return a matched native neighbor provider and fixed-box energy callable.

    Allocate outside JIT; update, energy, local_energy, descriptors, and ordinary
    jax.grad support JIT. Check did_buffer_overflow after every update. Positions
    are Cartesian Angstroms; box is a scalar, three lengths, or 3x3 row lattice.
    """
    if model.scale.dtype == np.float64 and not jax.config.x64_enabled:
        raise ValueError("float64 models require JAX x64 enabled by the caller")
    provider = PlannedNeighbors(
        model,
        box,
        model.validate_types(species),
        skin=skin,
        capacity_multiplier=capacity_multiplier,
        disable_cell_list=disable_cell_list,
        dtype=model.scale.dtype,
    )
    return provider, Energy(model, provider)


def virial(energy_fn, positions, *, neighbor):
    """Total configurational W_ab = -dE/d eps_ab for row vectors r @ (I+eps)."""
    zero = jnp.zeros((3, 3), positions.dtype)
    with jax.enable_x64():
        return -jax.grad(
            lambda eps: energy_fn(
                positions,
                neighbor=neighbor,
                perturbation=jnp.eye(3, dtype=positions.dtype) + eps,
            )
        )(zero)


def stress(energy_fn, positions, *, neighbor, volume):
    """Tensile-positive configurational stress in eV/Angstrom^3."""
    return -virial(energy_fn, positions, neighbor=neighbor) / volume
