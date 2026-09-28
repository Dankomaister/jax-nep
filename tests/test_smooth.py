import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_nep.smooth import smooth_max


@pytest.mark.parametrize('axis', [None, 0, 1, (0, 1), -1])
@pytest.mark.parametrize('keepdims', [False, True])
def test_values_axes(axis, keepdims):
    x = np.array([[.1, -.3, .8], [.5, .7, -.2]], np.float32)
    beta = 2.5
    expected = np.log(np.exp(beta * x).sum(axis=axis, keepdims=keepdims)) / beta
    f = jax.jit(lambda x, b: smooth_max(x, beta=b, axis=axis, keepdims=keepdims))
    np.testing.assert_allclose(f(x, beta), expected, rtol=2e-6, atol=1e-7)


def test_stability_gradient_and_limit():
    x = jnp.array([1000., 1001., 999.])
    got = jax.jit(lambda x: smooth_max(x, beta=2.))(x)
    expected = 1001 + np.log(np.exp(np.array([-2., 0., -4.])).sum()) / 2
    np.testing.assert_allclose(got, expected, rtol=1e-7)
    gradient = jax.jit(jax.grad(lambda x: smooth_max(x, beta=2.)))(x)
    weights = np.exp(np.array([-2., 0., -4.]))
    np.testing.assert_allclose(gradient, weights / weights.sum(), rtol=2e-6)
    values = np.array([smooth_max(x, beta=b) for b in (1., 2., 10., 100.)])
    assert np.all(np.diff(values) <= 0)
    assert np.all(values >= 1001.)
    np.testing.assert_allclose(values[-1], 1001., atol=1e-4)
    # Raw log-sum-exp must retain its finite-size offset, including at agreement.
    np.testing.assert_allclose(smooth_max(jnp.zeros(7), beta=3.), np.log(7)/3, rtol=1e-6)
