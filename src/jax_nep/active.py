"""Vectorized NEP committees and dimensionless force-disagreement scores."""

from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp

from .model import NEPModel
from .potential import nep_neighbor_list
from .smooth import smooth_max


class ModelDeviation(NamedTuple):
    """Dimensionless per-atom, smooth maximum, and exact maximum scores."""

    atomic: object
    smooth_max: object
    max: object


def stack_models(models):
    """Validate M >= 2 compatible models and stack their fitted PyTree leaves.

    Structure (including fixed ZBL), parameter shapes, and dtypes must match.
    No precision conversion is performed. The returned model has a leading
    committee axis on every fitted leaf and is intended for JAX vmap.
    """
    models = tuple(models)
    if len(models) < 2:
        raise ValueError('a committee requires at least two models')
    if not all(isinstance(m, NEPModel) for m in models):
        raise TypeError('committee members must be NEPModel instances')
    first, tree = jax.tree_util.tree_flatten(models[0])
    for i, model in enumerate(models[1:], 1):
        leaves, other = jax.tree_util.tree_flatten(model)
        if other != tree:
            raise ValueError(f'committee model {i} has incompatible static structure')
        if any(np.shape(a) != np.shape(b) for a, b in zip(first, leaves)):
            raise ValueError(f'committee model {i} has incompatible parameter shapes')
        if any(np.dtype(a.dtype) != np.dtype(b.dtype) for a, b in zip(first, leaves)):
            raise ValueError(f'committee model {i} has incompatible parameter dtypes')
    if any(np.dtype(a.dtype) == np.float64 for a in first) and not jax.config.x64_enabled:
        raise ValueError('float64 models require JAX x64 enabled by the caller')
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *models)


def _force_statistics(forces):
    forces = jnp.asarray(forces)
    if forces.ndim != 3 or forces.shape[0] < 2 or forces.shape[1] < 1 or forces.shape[2] != 3:
        raise ValueError('forces must have shape (M >= 2, N >= 1, 3)')
    mean = jnp.mean(forces, axis=0)
    sigma = jnp.sqrt(jnp.sum((forces - mean) ** 2, axis=(0, 2)) / (forces.shape[0] - 1))
    return mean, sigma


def force_deviation(forces):
    """GPUMD sample force deviation sigma, shape (N,), in eV/Angstrom.

    Input forces have shape (M, N, 3). Uses M-1 sample normalization.
    No epsilon is added at perfect agreement, where the norm is nondifferentiable.
    """
    return _force_statistics(forces)[1]


def _scaled_deviation(mean, sigma, delta_abs, delta_rel, exponent):
    squared = jnp.sum(mean * mean, axis=-1)
    # Zero mean force is reachable even for a disagreeing committee. Guard the
    # norm's argument to give a zero subgradient there (no epsilon in sigma).
    norm = jnp.sqrt(jnp.where(squared > 0, squared, 1)) * (squared > 0)
    relative = delta_rel * norm
    # Rescale the same generalized p-norm to avoid overflow for large p and
    # ensure delta_rel=0 gives exactly delta_abs for every exponent.
    largest = jnp.maximum(delta_abs, relative)
    scale = largest * ((delta_abs / largest) ** exponent
                       + (relative / largest) ** exponent) ** (1 / exponent)
    return sigma / scale


def _score(forces, delta_abs, delta_rel, exponent, beta):
    mean, sigma = _force_statistics(forces)
    atomic = _scaled_deviation(mean, sigma, delta_abs, delta_rel, exponent)
    return ModelDeviation(atomic, smooth_max(atomic, beta=beta), jnp.max(atomic))


def _shared_forces(energy, models, positions, neighbor):
    # Factor dE_m/dR = dE_m/d(features) @ d(features)/dR. The shared primal
    # computes geometry, basis, and harmonics only once. Both VJPs remain
    # ordinary differentiable JAX operations, including for coordinate Hessians.
    features, pullback, topology = jax.vjp(
        lambda r: energy._prepare(r, neighbor), positions, has_aux=True
    )

    def total(model, features):
        return jnp.sum(energy._local_from_features(model, features, topology))

    cotangents = jax.vmap(jax.grad(total, argnums=1), in_axes=(0, None))(models, features)
    return -jax.vmap(lambda cotangent: pullback(cotangent)[0])(cotangents)


class _ModelDeviationFn:
    def __init__(self, energy, models, delta_abs, delta_rel, exponent, beta):
        self.energy, self.models = energy, models
        self.settings = delta_abs, delta_rel, exponent, beta

    def forces(self, positions, *, neighbor):
        """Committee forces in eV/Angstrom with shape (M, N, 3)."""
        return _shared_forces(self.energy, self.models, positions, neighbor)

    def __call__(self, positions, *, neighbor):
        return _score(self.forces(positions, neighbor=neighbor), *self.settings)


def nep_model_deviation(
    models, box, species, *, delta_abs, beta, delta_rel=0.0, exponent=2.0,
    skin=1.0, capacity_multiplier=1.25, disable_cell_list=False,
):
    """Return one neighbor provider and a JIT/grad-compatible committee score.

    All models must share static architecture, leaf shapes, and dtypes.
    delta_abs > 0 is in eV/Angstrom; delta_rel >= 0 is dimensionless;
    exponent >= 1 (default 2) combines their scales as a generalized p-norm.
    delta_rel=0 gives absolute-only scaling. beta > 0 is explicit and controls
    the unnormalized log-sum-exp smooth maximum. All scalars must be finite.

    The result contains only (atomic, smooth_max, max), all dimensionless.
    max > 1 means an atom exceeds the configured disagreement tolerance.
    Allocate outside JIT, update neighbors and check overflow as for
    nep_neighbor_list. The callable's forces method exposes (M, N, 3) forces.
    """
    for name, value, lower, strict in (
        ('delta_abs', delta_abs, 0, True), ('delta_rel', delta_rel, 0, False),
        ('exponent', exponent, 1, False), ('beta', beta, 0, True),
    ):
        if (np.ndim(value) != 0 or not np.isfinite(value)
                or (value <= lower if strict else value < lower)):
            raise ValueError(f'{name} must be finite and {">" if strict else ">="} {lower}')
    models = tuple(models)
    stacked = stack_models(models)
    provider, energy = nep_neighbor_list(
        models[0], box, species, skin=skin, capacity_multiplier=capacity_multiplier,
        disable_cell_list=disable_cell_list,
    )
    return provider, _ModelDeviationFn(energy, stacked, delta_abs, delta_rel, exponent, beta)
