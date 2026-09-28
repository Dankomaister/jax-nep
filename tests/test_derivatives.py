from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_nep import load_model, nep_neighbor_list, virial


@pytest.mark.parametrize(
    "fixture",
    ["synthetic_l8", "synthetic_multi", "synthetic_flexible", "synthetic_skew"],
)
def test_directional_derivatives(fixture, tmp_path):
    d = np.load(Path(__file__).parent / "fixtures" / (fixture + ".npz"))
    p = tmp_path / "nep.txt"
    p.write_text(str(d["model_text"]))
    m = load_model(p)
    nf, energy = nep_neighbor_list(m, d["box"], d["species"], disable_cell_list=True)
    r = jnp.asarray(d["positions"])
    n = nf.allocate(r)
    rng = np.random.default_rng(29)
    direction = jnp.asarray(rng.normal(size=r.shape), dtype=r.dtype)
    e = jax.jit(lambda x: energy(x, neighbor=n))
    analytic = jnp.sum(jax.grad(e)(r) * direction)
    h = 0.001
    finite = (e(r + h * direction) - e(r - h * direction)) / (2 * h)
    np.testing.assert_allclose(analytic, finite, rtol=2e-3, atol=1e-4)
    strain = jnp.asarray(rng.normal(size=(3, 3)), dtype=r.dtype)
    deformed = jax.jit(lambda eps: energy(r, neighbor=n, perturbation=jnp.eye(3) + eps))
    analytic = -jnp.sum(virial(energy, r, neighbor=n) * strain)
    finite = (deformed(h * strain) - deformed(-h * strain)) / (2 * h)
    np.testing.assert_allclose(analytic, finite, rtol=2e-3, atol=1e-4)
    jax.clear_caches()
