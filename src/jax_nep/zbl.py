"""Screened nuclear repulsion, reusing the descriptor pair distances."""

import jax.numpy as jnp
from .descriptors import reduce_neighbors


def local_zbl(model, pairs, geometry):
    radius = jnp.where(geometry.positive, geometry.distance, 1)
    ti, tj = pairs.central_types, pairs.neighbor_types
    p = jnp.asarray(model.zbl)[ti, tj]
    z = jnp.asarray(model.atomic_numbers, dtype=radius.dtype)
    zi, zj = z[ti], z[tj]
    screening = radius * 2.134563 * (zi**0.23 + zj**0.23)
    phi = jnp.sum(p[..., 2::2] * jnp.exp(-p[..., 3::2] * screening[..., None]), axis=-1)
    fraction = jnp.clip((radius - p[..., 0]) / (p[..., 1] - p[..., 0]), 0, 1)
    switch = 0.5 * (1 + jnp.cos(jnp.pi * fraction))
    edge = 0.5 * 14.399645 * zi * zj * phi * switch / radius
    return reduce_neighbors(jnp.where(geometry.positive, edge, 0), pairs)
