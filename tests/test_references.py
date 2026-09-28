from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_nep import load_model, nep_neighbor_list, virial, stress

FIXTURES = sorted((Path(__file__).parent / "fixtures").glob("*.npz"))
# Fixed before implementation. Absolute budgets reflect float32 arithmetic;
# extensive quantities are checked per atom to avoid a size-dependent rule.
TOLERANCES = json.loads((Path(__file__).parent / "tolerances.json").read_text())


@pytest.mark.reference
@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda p: p.stem)
def test_cpu_reference(fixture, tmp_path, record_property):
    data = np.load(fixture)
    path = tmp_path / "nep.txt"
    path.write_text(str(data["model_text"]))
    model = load_model(path)
    nf, energy = nep_neighbor_list(
        model, data["box"], data["species"], disable_cell_list=True
    )
    r = jnp.asarray(data["positions"])
    neighbor = nf.allocate(r)
    assert not neighbor.did_buffer_overflow
    values = jax.jit(
        lambda x, n: (
            energy.descriptors(x, neighbor=n),
            energy.local_energy(x, neighbor=n),
            energy(x, neighbor=n),
            -jax.grad(lambda y: energy(y, neighbor=n))(x),
            virial(energy, x, neighbor=n),
        )
    )(r, neighbor)
    for key, value in zip(TOLERANCES, values):
        divisor = len(r) if key in ("energy", "virial") else 1
        record_property(
            key + "_max_abs", float(np.max(np.abs(np.asarray(value) - data[key])))
        )
        record_property(
            key + "_max_abs_per_atom",
            float(np.max(np.abs(np.asarray(value) - data[key]))) / len(r),
        )
        rtol, atol = TOLERANCES[key]
        np.testing.assert_allclose(
            np.asarray(value) / divisor,
            data[key] / divisor,
            rtol=rtol,
            atol=atol,
            err_msg=key,
        )
    volume = np.linalg.det(data["box"])
    got = jax.jit(lambda x: stress(energy, x, neighbor=neighbor, volume=volume))(r)
    np.testing.assert_allclose(got, -values[-1] / volume, rtol=2e-6, atol=1e-7)
    jax.clear_caches()
