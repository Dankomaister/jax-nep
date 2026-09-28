"""Basis expansion, neighbor moments, and scaled Dense NEP descriptors."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from .angular import harmonics, invariants


class PairGeometry(NamedTuple):
    distance: object
    direction: object
    positive: object


def geometry(pairs):
    # Guard arguments BEFORE singular primitives, including float32 underflow.
    safe = jnp.where(
        pairs.valid[..., None], pairs.vectors, jnp.array([1, 0, 0], pairs.vectors.dtype)
    )
    squared = jnp.sum(safe * safe, axis=-1)
    positive = squared > 0
    argument = jnp.where(positive, squared, 1)
    distance = jnp.sqrt(argument) * positive
    direction = safe * (jnp.reciprocal(jnp.sqrt(argument)) * positive)[..., None]
    return PairGeometry(distance, direction, positive & pairs.valid)


def basis(distance, cutoff, order, valid):
    u = jnp.where(valid & (distance < cutoff), distance / cutoff, 1)
    envelope = 0.5 * (1 + jnp.cos(jnp.pi * u))
    x = 2 * (u - 1) ** 2 - 1
    polynomials = [jnp.ones_like(x)]
    if order:
        polynomials.append(x)
    for _ in range(2, order + 1):
        polynomials.append(2 * x * polynomials[-1] - polynomials[-2])
    return 0.5 * (jnp.stack(polynomials, axis=0) + 1) * envelope[None]


def descriptors(model, pairs, pair_geometry=None, angular_pairs=None):
    g = geometry(pairs) if pair_geometry is None else pair_geometry
    ti, tj = pairs.central_types, pairs.neighbor_types
    for domain in range(2):
        if domain == 1 and angular_pairs is not None:
            pairs = angular_pairs
            g = geometry(pairs)
            ti, tj = pairs.central_types, pairs.neighbor_types
        radii = jnp.asarray(model.cutoffs[:, domain])
        cutoff = 0.5 * (radii[ti] + radii[tj])
        fn = basis(g.distance, cutoff, model.basis_orders[domain], pairs.valid)
        table = model.coefficients[domain]
        table = table.reshape(-1, *table.shape[2:])
        parts = [
            fn[k, ..., None] * jnp.asarray(table[:, k, :])[ti * len(model.species) + tj]
            for k in range(model.basis_orders[domain] + 1)
        ]
        while len(parts) > 1:
            parts = [
                parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i]
                for i in range(0, len(parts), 2)
            ]
        if domain == 0:
            radial = reduce_neighbors(parts[0], pairs)
        else:
            angular = parts[0]

    y = harmonics(g.direction, model.l_max)
    if pairs.centers is None:
        moments = jnp.einsum("ijc,hij->ich", angular, y, precision="highest")
    else:
        moments = reduce_neighbors(angular[..., None] * y.T[:, None, :], pairs)
    q = invariants(moments, model.l_max, model.invariants).reshape(len(radial), -1)
    return jnp.concatenate((radial, q), axis=-1) * jnp.asarray(model.scale)


def reduce_neighbors(values, pairs):
    """Dense row reduction or Sparse central-index reduction of the same values."""
    if pairs.centers is None:
        return jnp.sum(values, axis=1)
    return jax.ops.segment_sum(values, pairs.centers, pairs.n_atoms)
