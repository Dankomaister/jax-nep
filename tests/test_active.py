"""Sample deviation, strict stacking, and differentiated physical committees."""
from dataclasses import replace
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nep import load_model, nep_neighbor_list
from jax_nep.active import (
    ModelDeviation, force_deviation, nep_model_deviation, stack_models,
    _score, _scaled_deviation,
)
from jax_nep.potential import Energy
from jax_nep.smooth import smooth_max

FIXTURES = Path(__file__).parent / 'fixtures'
TOLERANCES = json.loads((Path(__file__).parent / 'tolerances.json').read_text())


def load_case(tmp_path, name='synthetic_l4'):
    data = np.load(FIXTURES / (name + '.npz'))
    path = tmp_path / 'nep.txt'
    path.write_text(str(data['model_text']))
    return load_model(path), data


def test_sample_statistics():
    forces = np.random.default_rng(7).normal(size=(5, 11, 3)).astype(np.float32)
    expected = np.sqrt(np.var(forces.astype(float), axis=0, ddof=1).sum(axis=-1))
    np.testing.assert_allclose(jax.jit(force_deviation)(forces), expected, rtol=2e-6)
    # Two members at +/-1 in each direction: variance sum = 6, not 3.
    np.testing.assert_allclose(force_deviation(jnp.array([[[-1., -1., -1.]], [[1., 1., 1.]]])), np.sqrt(6), rtol=1e-6)
    np.testing.assert_array_equal(force_deviation(jnp.ones((3, 2, 3))), 0.)


@pytest.mark.parametrize('shape', [(1, 2, 3), (2, 0, 3), (2, 3), (2, 3, 2)])
def test_force_shape_errors(shape):
    with pytest.raises(ValueError, match='forces must have shape'):
        force_deviation(jnp.zeros(shape))


@pytest.mark.parametrize('p', [1., 2., 4., 16.])
@pytest.mark.parametrize('relative', [0., .3])
def test_generalized_scaling(p, relative):
    forces = np.random.default_rng(12).normal(size=(4, 7, 3)).astype(np.float32)
    mean = forces.astype(float).mean(0)
    sigma = np.sqrt(np.var(forces.astype(float), axis=0, ddof=1).sum(-1))
    expected = sigma / (.2**p + (relative * np.linalg.norm(mean, axis=-1))**p)**(1/p)
    result = jax.jit(lambda f: _score(f, .2, relative, p, 5.))(forces)
    assert isinstance(result, ModelDeviation)
    assert result._fields == ('atomic', 'smooth_max', 'max')
    assert result.atomic.shape == (7,)
    assert result.max.shape == result.smooth_max.shape == ()
    np.testing.assert_allclose(result.atomic, expected, rtol=2e-6)
    np.testing.assert_allclose(result.max, jnp.max(result.atomic), rtol=1e-6)
    np.testing.assert_allclose(result.smooth_max, smooth_max(result.atomic, beta=5.), rtol=1e-6)
    if relative == 0:
        # Exactly the same absolute denominator for every exponent.
        np.testing.assert_array_equal(_scaled_deviation(jnp.asarray(mean), jnp.asarray(sigma), .2, 0., p), jnp.asarray(sigma) / .2)


@pytest.mark.parametrize('p', [1., 2., 4.])
@pytest.mark.parametrize('relative', [0., .2])
def test_zero_mean_gradient(p, relative):
    forces = jnp.array([[[-1., 0., 0.]], [[1., 0., 0.]]])
    value, gradient = jax.jit(jax.value_and_grad(lambda f: _score(f, .1, relative, p, 3.).smooth_max))(forces)
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 0


@pytest.mark.parametrize('name', ['synthetic_l4', 'synthetic_flexible', 'synthetic_universal',
                                 'synthetic_typewise', 'synthetic_multi', 'synthetic_skew_standard'])
def test_physical_committee(name, tmp_path):
    model, data = load_case(tmp_path, name)
    other = replace(model, output_weights=model.output_weights * 1.07)
    if model.zbl_mode == 'flexible':
        zbl = model.zbl.copy()
        zbl[..., 2:] *= 1.02
        other = replace(other, zbl=zbl)
    provider, score = nep_model_deviation([model, other], data['box'], data['species'],
        delta_abs=.1, delta_rel=.2, beta=5., disable_cell_list=True)
    r = jnp.asarray(data['positions'])
    n = provider.allocate(r)
    assert not n.did_buffer_overflow
    actual = jax.jit(score.forces)(r, neighbor=n)
    explicit = jnp.stack([jax.jit(jax.grad(lambda x: -Energy(m, provider)(x, neighbor=n)))(r)
                          for m in (model, other)])
    assert actual.shape == (2, len(r), 3)
    np.testing.assert_allclose(actual, explicit, rtol=TOLERANCES['forces'][0], atol=TOLERANCES['forces'][1])
    result = jax.jit(score)(r, neighbor=n)
    reference = _score(explicit, .1, .2, 2., 5.)
    np.testing.assert_allclose(result.atomic, reference.atomic, rtol=2e-4, atol=2e-5)
    gradient = jax.jit(jax.grad(lambda x: score(x, neighbor=n).smooth_max))(r)
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 0
    direction = jnp.asarray(np.random.default_rng(8).normal(size=r.shape), r.dtype)
    evaluate = jax.jit(lambda x: score(x, neighbor=n).smooth_max)
    h = .001
    finite = (evaluate(r + h*direction) - evaluate(r - h*direction)) / (2*h)
    np.testing.assert_allclose(jnp.sum(gradient*direction), finite, rtol=1e-2, atol=3e-4)
    foreign, _ = nep_neighbor_list(model, data['box'], data['species'], disable_cell_list=True)
    with pytest.raises(ValueError, match='matching factory'):
        score(r, neighbor=foreign.allocate(r))
    jax.clear_caches()


@pytest.mark.parametrize('change,match', [
    (lambda m: replace(m, species=m.species[::-1]), 'static structure'),
    (lambda m: replace(m, cutoffs=m.cutoffs * 1.01), 'static structure'),
    (lambda m: replace(m, orders=(2, 1)), 'static structure'),
    (lambda m: replace(m, basis_orders=(2, 3)), 'static structure'),
    (lambda m: replace(m, l_max=3), 'static structure'),
    (lambda m: replace(m, invariants=()), 'static structure'),
    (lambda m: replace(m, atomic_numbers=m.atomic_numbers[::-1]), 'static structure'),
    (lambda m: replace(m, zbl_mode='flexible'), 'static structure'),
    (lambda m: replace(m, zbl=np.ones((2, 2, 10), np.float32)), 'static structure'),
    (lambda m: replace(m, input_weights=m.input_weights[..., :-1]), 'parameter shapes'),
    (lambda m: replace(m, scale=m.scale.astype(np.float64)), 'parameter dtypes'),
])
def test_incompatible_committee(tmp_path, change, match):
    model, _ = load_case(tmp_path)
    with pytest.raises(ValueError, match=match):
        stack_models([model, change(model)])


def test_stack_and_fixed_zbl(tmp_path):
    model, _ = load_case(tmp_path, 'synthetic_universal')
    other = replace(model, output_weights=model.output_weights * 1.01)
    stacked = stack_models([model, other])
    assert jax.tree_util.tree_structure(stacked) == jax.tree_util.tree_structure(model)
    for a, b, c in zip(jax.tree_util.tree_leaves(stacked), jax.tree_util.tree_leaves(model), jax.tree_util.tree_leaves(other)):
        np.testing.assert_array_equal(a, np.stack([b, c]))
    with pytest.raises(ValueError, match='static structure'):
        stack_models([model, replace(other, zbl=model.zbl * 1.01)])
    with pytest.raises(ValueError, match='static structure'):
        stack_models([model, replace(other, zbl_mode='typewise')])
    for members in ([], [model]):
        with pytest.raises(ValueError, match='at least two'):
            stack_models(members)
    with pytest.raises(TypeError, match='NEPModel'):
        stack_models([model, object()])


@pytest.mark.parametrize('settings', [dict(delta_abs=0), dict(delta_abs=-1), dict(delta_abs=np.inf),
    dict(delta_abs=[.1]), dict(delta_rel=-1), dict(delta_rel=np.nan), dict(exponent=.5),
    dict(exponent=np.inf), dict(beta=0), dict(beta=np.nan)])
def test_invalid_settings(tmp_path, settings):
    model, data = load_case(tmp_path)
    kwargs = dict(delta_abs=.1, beta=5.) | settings
    with pytest.raises(ValueError):
        nep_model_deviation([model, model], data['box'], data['species'], **kwargs)


def test_float64_stacking(tmp_path):
    model, _ = load_case(tmp_path)
    model = jax.tree.map(lambda x: np.asarray(x, np.float64), model)
    with jax.enable_x64(False), pytest.raises(ValueError, match='x64'):
        stack_models([model, model])
    with jax.enable_x64():
        assert all(x.dtype == np.float64 for x in jax.tree_util.tree_leaves(stack_models([model, model])))


def test_fitted_nibn_reference_and_position_gradient(tmp_path):
    data = np.load(FIXTURES / 'active/nibn.npz')
    models = []
    for i, text in enumerate(data['model_texts']):
        path = tmp_path / f'{i}_nep.txt'
        path.write_text(str(text))
        models.append(load_model(path))
    provider, score = nep_model_deviation(models, data['box'], data['species'],
        delta_abs=.1, beta=10., disable_cell_list=True)
    r = jnp.asarray(data['positions'])
    neighbor = provider.allocate(r)
    assert not neighbor.did_buffer_overflow
    forces = jax.jit(score.forces)(r, neighbor=neighbor)
    np.testing.assert_allclose(forces, data['forces'], rtol=TOLERANCES['forces'][0], atol=TOLERANCES['forces'][1])
    raw = force_deviation(forces)
    np.testing.assert_allclose(raw, data['sigma'], rtol=2e-5, atol=1e-4)
    result = jax.jit(score)(r, neighbor=neighbor)
    np.testing.assert_allclose(result.atomic, data['sigma'] / .1, rtol=2e-5, atol=1e-3)
    # Small configuration, real independently fitted models: second derivatives.
    small_r = r[:6]
    small_provider, small_score = nep_model_deviation(models[:2], data['box'], data['species'][:6],
        delta_abs=.1, delta_rel=.1, beta=10., disable_cell_list=True)
    small_n = small_provider.allocate(small_r)
    gradient = jax.jit(jax.grad(lambda x: small_score(x, neighbor=small_n).smooth_max))(small_r)
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 0
    jax.clear_caches()


@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('split', [False, True])
def test_shared_force_neighbor_domains(tmp_path, sparse, split):
    from jax_md import partition
    from jax_nep.neighbors import NeighborProvider
    from jax_nep.active import _shared_forces

    model, _ = load_case(tmp_path)
    models = [model, replace(model, output_weights=model.output_weights * 1.07)]
    r = jnp.array([[x, y, z] for x in (0., 2.6) for y in (0., 2.6) for z in (0., 2.6)])
    species = np.arange(len(r), dtype=np.int32) % 2
    settings = dict(disable_cell_list=True, representation=partition.Sparse if sparse else partition.Dense)
    provider = NeighborProvider(20., species, model.cutoff, **settings)
    if split:
        provider.angular_provider = NeighborProvider(20., species, 3., **settings)
    n = provider.allocate(r)
    energy = Energy(model, provider)
    actual = jax.jit(lambda x: _shared_forces(energy, stack_models(models), x, n))(r)
    expected = jnp.stack([jax.jit(jax.grad(lambda x: -Energy(m, provider)(x, neighbor=n)))(r) for m in models])
    np.testing.assert_allclose(actual, expected, rtol=TOLERANCES['forces'][0], atol=TOLERANCES['forces'][1])
    jax.clear_caches()
