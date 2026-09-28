import itertools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_nep.neighbors import NeighborProvider
from jax_md import partition


def edges(provider, r, n):
    pairs = provider.pairs(jnp.asarray(r), n)
    return [
        sorted(map(tuple, np.round(np.asarray(v)[mask], 5)))
        for v, mask in zip(pairs.vectors, np.asarray(pairs.valid))
    ]


@pytest.mark.parametrize("representation", [partition.Dense, partition.Sparse])
@pytest.mark.parametrize(
    "cell",
    [
        np.eye(3) * 20,
        np.array([[12.0, 0, 0], [19.0, 9, 0], [1, 2, 14.0]]),
        np.eye(3) * 5,
        np.array([[5.0, 0, 0], [7.0, 4.0, 0], [1.0, 0.5, 6.0]]),
    ],
)
def test_native_edges(cell, representation):
    rng = np.random.default_rng(6)
    r = (rng.uniform(0.1, 0.9, (4, 3)) @ cell).astype("float32")
    cutoff = 3.1
    provider = NeighborProvider(
        cell,
        np.zeros(4, dtype=np.int32),
        cutoff,
        skin=0.0,
        capacity_multiplier=1.25,
        disable_cell_list=True,
        representation=representation,
    )
    n = provider.allocate(jnp.asarray(r))
    pairs = provider.pairs(jnp.asarray(r), n)
    for i in range(len(r)):
        if representation == partition.Dense:
            actual = np.asarray(pairs.vectors[i])[np.asarray(pairs.valid[i])]
        else:
            actual = np.asarray(pairs.vectors)[
                np.asarray(pairs.valid) & (np.asarray(pairs.centers) == i)
            ]
        actual = actual[np.linalg.norm(actual, axis=-1) < cutoff - 1e-5]
        expected = []
        for j, shift in itertools.product(
            range(len(r)), itertools.product(range(-4, 5), repeat=3)
        ):
            if i == j and shift == (0, 0, 0):
                continue
            v = r[j] - r[i] + np.array(shift) @ cell
            if np.linalg.norm(v) < cutoff:
                expected.append(v)
        assert len(actual) == len(expected)
        if len(actual):
            distances = np.linalg.norm(
                actual[:, None] - np.array(expected)[None], axis=-1
            )
            np.testing.assert_allclose(distances.min(1), 0, atol=5e-6)
    updated = jax.jit(provider.update)(jnp.asarray(r) + 0.01, n)
    assert not updated.did_buffer_overflow


def test_backend_independent_of_skin():
    for skin in (0.0, 1.0, 4.0):
        provider = NeighborProvider(
            8.0, np.array([0]), 4.0, skin=skin, disable_cell_list=True
        )
        assert provider.backend == "STANDARD"
        provider = NeighborProvider(
            7.9, np.array([0]), 4.0, skin=skin, disable_cell_list=True
        )
        assert provider.backend == "MULTI_IMAGE"


@pytest.mark.parametrize("representation", [partition.Dense, partition.Sparse])
def test_multi_image_wrap_update(representation):
    provider = NeighborProvider(
        5.0,
        np.zeros(3, dtype=np.int32),
        3.1,
        skin=1.0,
        disable_cell_list=True,
        representation=representation,
    )
    r = jnp.array(
        [[4.98, 1.0, 1.0], [1.0, 1.3, 1.2], [3.0, 2.0, 1.0]], dtype=jnp.float32
    )
    old = provider.allocate(r)
    moved = r.at[0, 0].add(0.05)
    updated = jax.jit(provider.update)(moved, old)
    fresh = provider.allocate(moved)

    def physical_edges(state):
        p = provider.pairs(moved, state)
        v = np.asarray(p.vectors).reshape(-1, 3)
        valid = np.asarray(p.valid).reshape(-1) & (np.linalg.norm(v, axis=1) < 3.1)
        return sorted(map(tuple, np.round(v[valid], 5)))

    assert physical_edges(updated) == physical_edges(fresh)
    assert not updated.did_buffer_overflow


@pytest.mark.parametrize(
    "cell", [np.eye(3) * 20, np.array([[20.0, 0, 0], [9.0, 19.0, 0], [5.0, 2.0, 22.0]])]
)
def test_joint_domain_geometry_and_cotangents(cell):
    rng = np.random.default_rng(73)
    r = rng.uniform(0, 6, (8, 3)).astype("float32")
    r += (rng.integers(-3, 4, r.shape) @ cell).astype("float32")
    settings = dict(skin=1.0, disable_cell_list=True, representation=partition.Sparse)
    provider = NeighborProvider(cell, np.zeros(8, dtype=np.int32), 6.0, **settings)
    provider.angular_provider = NeighborProvider(
        cell, np.zeros(8, dtype=np.int32), 3.0, **settings
    )
    neighbor = provider.allocate(jnp.asarray(r))
    joint = provider.pair_domains(jnp.asarray(r), neighbor)
    separate = (
        provider.pairs(jnp.asarray(r), neighbor),
        provider.angular_pairs(jnp.asarray(r), neighbor),
    )
    for actual, expected in zip(joint, separate):
        np.testing.assert_allclose(
            actual.vectors, expected.vectors, atol=2e-5, rtol=1e-6
        )
        np.testing.assert_array_equal(actual.valid, expected.valid)

    def value(x, combined):
        domains = (
            provider.pair_domains(x, neighbor)
            if combined
            else (provider.pairs(x, neighbor), provider.angular_pairs(x, neighbor))
        )
        return sum(
            (i + 1) * jnp.sum(jnp.where(p.valid, jnp.sum(p.vectors**2, axis=-1), 0))
            for i, p in enumerate(domains)
        )

    actual = jax.jit(jax.value_and_grad(lambda x: value(x, True)))(jnp.asarray(r))
    expected = jax.jit(jax.value_and_grad(lambda x: value(x, False)))(jnp.asarray(r))
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-4, rtol=2e-6)
