from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_nep import load_model, nep_neighbor_list, virial
from jax_nep.neighbors import NeighborProvider
from jax_nep.potential import Energy
from jax_md import partition


@pytest.fixture
def model(tmp_path):
    p = tmp_path / "nep.txt"
    p.write_text(
        str(np.load(Path(__file__).parent / "fixtures/synthetic_l4.npz")["model_text"])
    )
    return load_model(p)


@pytest.mark.parametrize("representation", [partition.Dense, partition.Sparse])
@pytest.mark.parametrize("separation", [0.0, 1e-23, 1e-19, 1.2])
def test_safe_distance(model, separation, representation):
    r = jnp.array(
        [[0.0, 0.0, 0.0], [separation, 0.0, 0.0], [9.0, 9.0, 9.0]], dtype=jnp.float32
    )
    nf = NeighborProvider(
        30.0,
        np.array([0, 1, 0]),
        model.cutoff,
        disable_cell_list=True,
        representation=representation,
    )
    energy = Energy(model, nf)
    n = nf.allocate(r)
    e, f, w = jax.jit(
        lambda x: (
            energy(x, neighbor=n),
            jax.grad(lambda y: energy(y, neighbor=n))(x),
            virial(energy, x, neighbor=n),
        )
    )(r)
    for value in (e, f, w):
        assert np.isfinite(value).all()


@pytest.mark.parametrize("box", [30.0, 5.0])
def test_large_unwrapped_and_foreign(model, box):
    r = jnp.array(
        [[0.3, 0.4, 0.5], [1.5, 0.7, 0.9], [2.2, 1.8, 0.3]], dtype=jnp.float32
    )
    types = np.array([0, 1, 0])
    nf, energy = nep_neighbor_list(model, box, types, disable_cell_list=True)
    n = nf.allocate(r)
    other, _ = nep_neighbor_list(model, box, types, disable_cell_list=True)
    with pytest.raises(ValueError):
        energy(r, neighbor=other.allocate(r))
    with pytest.raises(ValueError):
        energy(r, neighbor=n.native)
    huge = r + jnp.float32(1e12)
    for x in (r, huge):
        updated = jax.jit(nf.update)(x, n)
        f = jax.jit(jax.grad(lambda y: energy(y, neighbor=updated)))(x)
        assert np.isfinite(f).all()


@pytest.mark.parametrize("box", [20.0, 5.0])
def test_translation_and_force_sum(model, box):
    r = jnp.array(
        [[0.3, 0.4, 0.5], [1.5, 0.7, 0.9], [2.2, 1.8, 0.3]], dtype=jnp.float32
    )
    nf, energy = nep_neighbor_list(
        model, box, np.array([0, 1, 0]), disable_cell_list=True
    )
    evaluate = jax.jit(
        lambda x, n: (
            energy(x, neighbor=n),
            jax.grad(lambda y: energy(y, neighbor=n))(x),
        )
    )
    e, f = evaluate(r, nf.allocate(r))
    moved = r + jnp.array([[2, -1, 0], [-1, 1, 1], [0, 0, -2]]) * box
    e2, f2 = evaluate(moved, nf.allocate(moved))
    np.testing.assert_allclose(e, e2, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(f, f2, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(f.sum(0), 0, atol=1e-6)


def test_overflow(model):
    r = jnp.array([[i * 10.0, 0.0, 0.0] for i in range(8)])
    nf, _ = nep_neighbor_list(
        model,
        100.0,
        np.zeros(8, dtype=int),
        skin=0,
        capacity_multiplier=1,
        disable_cell_list=True,
    )
    n = nf.allocate(r)
    packed = jnp.arange(24, dtype=jnp.float32).reshape(8, 3) * 0.05
    updated = jax.jit(nf.update)(packed, n)
    assert updated.did_buffer_overflow
    assert int(updated.error.code) != 0
    assert not nf.allocate(packed).did_buffer_overflow


@pytest.mark.parametrize("species", [[-1, 0], [0, 2], [0.0, 1.0], [[0, 1]], []])
def test_bad_species(model, species):
    with pytest.raises(ValueError):
        nep_neighbor_list(model, 10.0, species)


@pytest.mark.parametrize(
    "box", [-1.0, [1.0, 2.0], np.zeros((3, 3)), np.eye(3) * np.nan]
)
def test_bad_box(model, box):
    with pytest.raises(ValueError):
        nep_neighbor_list(model, box, [0, 1])


@pytest.mark.parametrize("representation", [partition.Dense, partition.Sparse])
@pytest.mark.parametrize("separation", [0.0, 1e-23])
def test_zbl_underflow(tmp_path, representation, separation):
    path = tmp_path / "nep.txt"
    path.write_text(
        str(
            np.load(Path(__file__).parent / "fixtures/synthetic_universal.npz")[
                "model_text"
            ]
        )
    )
    model = load_model(path)
    r = jnp.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], dtype=jnp.float32)
    nf = NeighborProvider(
        30.0,
        np.array([0, 1]),
        model.cutoff,
        disable_cell_list=True,
        representation=representation,
    )
    energy = Energy(model, nf)
    n = nf.allocate(r)
    values = jax.jit(
        lambda x: (
            energy(x, neighbor=n),
            jax.grad(lambda y: energy(y, neighbor=n))(x),
            virial(energy, x, neighbor=n),
        )
    )(r)
    assert all(np.isfinite(value).all() for value in values)
