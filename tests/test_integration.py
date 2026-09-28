from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import simulate, space
from jax_nep import load_model, nep_neighbor_list


def test_jax_md_nve(tmp_path):
    data = np.load(Path(__file__).parent / "fixtures/synthetic_l4.npz")
    path = tmp_path / "nep.txt"
    path.write_text(str(data["model_text"]))
    model = load_model(path)
    positions = jnp.asarray(data["positions"])
    box = data["box"]
    provider, energy = nep_neighbor_list(model, box, data["species"])
    neighbor = provider.allocate(positions)
    _, shift = space.free()
    initialize, advance = simulate.nve(energy, shift, dt=0.001)
    state = initialize(jax.random.PRNGKey(7), positions, kT=0.025, neighbor=neighbor)

    @jax.jit
    def step(state, neighbor):
        neighbor = provider.update(state.position, neighbor)
        return advance(state, neighbor=neighbor), neighbor

    for _ in range(3):
        state, neighbor = step(state, neighbor)
    assert not neighbor.did_buffer_overflow
    assert np.isfinite(state.position).all()
    np.testing.assert_allclose(
        state.force,
        -jax.grad(lambda x: energy(x, neighbor=neighbor))(state.position),
        atol=2e-6,
    )
