"""Fitted leaves transform independently of the static physical architecture."""
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_nep import load_model, nep_neighbor_list


@pytest.mark.parametrize('name,leaves', [
    ('synthetic_l4', 7), ('synthetic_flexible', 8),
    ('synthetic_universal', 7), ('synthetic_typewise', 7),
])
def test_parameter_pytree(name, leaves, tmp_path):
    data = np.load(Path(__file__).parent / 'fixtures' / (name + '.npz'))
    path = tmp_path / 'nep.txt'
    path.write_text(str(data['model_text']))
    model = load_model(path)
    flat, tree = jax.tree_util.tree_flatten(model)
    assert len(flat) == leaves
    assert all(np.asarray(x).dtype == np.float32 for x in flat)
    names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(model)[0]]
    assert not any('cutoffs' in n or 'atomic_numbers' in n for n in names)
    assert any('zbl' in n for n in names) == (model.zbl_mode == 'flexible')
    restored = jax.tree_util.tree_unflatten(tree, flat)
    assert restored._structure == model._structure
    for a, b in zip(flat, jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(a, b)
    provider, energy = nep_neighbor_list(model, data['box'], data['species'], disable_cell_list=True)
    r = jnp.asarray(data['positions'])
    neighbor = provider.allocate(r)
    evaluate = jax.jit(lambda m: energy.apply(m, r, neighbor=neighbor))
    expected = jax.jit(energy)(r, neighbor=neighbor)
    np.testing.assert_array_equal(evaluate(restored), evaluate(model))
    # Dynamic arguments and closed constants can fuse reductions differently.
    np.testing.assert_allclose(evaluate(restored) / len(r), expected / len(r),
                               rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(expected / len(r), data['energy'] / len(r), rtol=2e-6, atol=2e-5)
    gradient = jax.jit(jax.grad(lambda m: energy.apply(m, r, neighbor=neighbor)))(model)
    assert jax.tree_util.tree_structure(gradient) == tree
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(gradient))
    np.testing.assert_allclose(gradient.output_bias, -len(r))
    # Check a fitted-parameter directional derivative, beyond a structural smoke test.
    direction = np.ones_like(model.output_weights)
    h = .001
    finite = (evaluate(replace(model, output_weights=model.output_weights + h * direction))
              - evaluate(replace(model, output_weights=model.output_weights - h * direction))) / (2*h)
    np.testing.assert_allclose(finite, jnp.sum(gradient.output_weights), rtol=2e-3, atol=2e-4)
    stacked = jax.tree.map(lambda a: jnp.stack((a, a)), model)
    assert stacked.dimension == model.dimension
    np.testing.assert_allclose(jax.jit(jax.vmap(evaluate))(stacked) / len(r),
                               jnp.repeat(expected, 2) / len(r), rtol=2e-6, atol=2e-5)
    with pytest.raises(ValueError, match='static structure'):
        evaluate(replace(model, cutoffs=model.cutoffs * 1.01))
    jax.clear_caches()
