"""User-facing API and actionable input errors."""
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pytest
import jax_nep


@pytest.fixture
def model(tmp_path):
    data = np.load(Path(__file__).parent / 'fixtures/synthetic_l4.npz')
    path = tmp_path / 'nep.txt'
    path.write_text(str(data['model_text']))
    return jax_nep.load_model(path)


def test_exports():
    assert set(jax_nep.__all__) == {
        'NEPModel', 'load_model', 'nep_neighbor_list', 'virial', 'stress',
        'ModelDeviation', 'nep_model_deviation', 'smooth_max'}


@pytest.mark.parametrize('settings', [dict(skin=-1), dict(skin=np.nan),
    dict(capacity_multiplier=.5), dict(disable_cell_list='yes')])
def test_neighbor_settings_errors(model, settings):
    with pytest.raises(ValueError, match='neighbor settings'):
        jax_nep.nep_neighbor_list(model, 20., [0, 1], **settings)


@pytest.mark.parametrize('shape', [(2, 2), (3, 3), (6,)])
def test_position_shape_error(model, shape):
    provider, _ = jax_nep.nep_neighbor_list(model, 20., [0, 1])
    with pytest.raises(ValueError, match='positions must have shape'):
        provider.allocate(jnp.zeros(shape))
