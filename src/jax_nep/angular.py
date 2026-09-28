"""NEP real harmonics and one authoritative invariant contraction."""

import jax.numpy as jnp
from .tables import Z_POLYNOMIALS, C3B
from .formulas import FUNCTIONS


def harmonics(direction, lmax):
    x, y, z = (direction[..., i] for i in range(3))
    powers = [jnp.ones_like(z)]
    for k in range(1, lmax + 1):
        powers.append(z if k == 1 else powers[k // 2] * powers[k - k // 2])
    xy = [(jnp.ones_like(x), jnp.zeros_like(x))]
    for k in range(1, lmax + 1):
        if k == 1:
            xy.append((x, y))
        else:
            re, im = xy[k // 2]
            re2, im2 = xy[k - k // 2]
            xy.append((re * re2 - im * im2, re * im2 + im * re2))
    channels = []
    for ell in range(1, lmax + 1):
        for m, coefficients in enumerate(Z_POLYNOMIALS[ell - 1]):
            polynomial = sum(c * powers[k] for k, c in enumerate(coefficients) if c)
            channels.append(polynomial * xy[m][0])
            if m:
                channels.append(polynomial * xy[m][1])
    return jnp.stack(channels, axis=0)


def invariants(moments, lmax, higher_body):
    """Return (..., invariant, radial channel), in the model's descriptor order."""
    values = []
    for ell in range(1, lmax + 1):
        start, end = ell * ell - 1, (ell + 1) ** 2 - 1
        weights = jnp.asarray(
            [C3B[start], *[2 * c for c in C3B[start + 1 : end]]], moments.dtype
        )
        values.append(jnp.sum(moments[..., start:end] ** 2 * weights, axis=-1))
    for name in higher_body:
        values.append(
            FUNCTIONS[name]([moments[..., i] for i in range(moments.shape[-1])])
        )
    return jnp.stack(values, axis=-2)
