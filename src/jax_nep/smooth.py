"""Generic differentiable smooth mathematical functions."""

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def smooth_max(x, *, beta, axis=None, keepdims=False):
    """Return log(sum(exp(beta * x))) / beta with stable log-sum-exp.

    ``beta`` must be a finite positive scalar. This is a caller precondition,
    allowing beta to be a traced JAX argument. The reduction is unnormalized:
    for n finite elements, max(x) <= result <= max(x) + log(n) / beta.
    ``axis`` and ``keepdims`` follow JAX reduction conventions.
    """
    return logsumexp(jnp.asarray(x) * beta, axis=axis, keepdims=keepdims) / beta
