"""Fixed row-lattice geometry and finite, exact periodic search bounds."""

import itertools
import numpy as np
import jax
import jax.numpy as jnp


class Lattice:
    def __init__(self, box, search_radius, dtype=np.float32):
        cell = np.asarray(box, dtype=np.float64)
        if cell.ndim == 0:
            cell = np.eye(3) * cell
        if cell.shape == (3,):
            cell = np.diag(cell)
        if (
            cell.shape != (3, 3)
            or not np.isfinite(cell).all()
            or np.linalg.det(cell) <= 0
        ):
            raise ValueError("box must be a finite right-handed row lattice")
        inverse = np.linalg.inv(cell)
        heights = 1 / np.linalg.norm(inverse, axis=0)
        # Any lattice vector shorter than a basis vector has |n_i| <= bound/h_i.
        bound = np.linalg.norm(cell, axis=1).min()
        extent = np.floor(np.nextafter(bound / heights, np.inf)).astype(int)
        shortest = bound
        for n in itertools.product(*(range(-e, e + 1) for e in extent)):
            if any(n):
                shortest = min(shortest, np.linalg.norm(np.array(n) @ cell))
        self.minimum_image_radius = shortest / 2
        self.cell, self.inverse = jnp.asarray(cell, dtype), jnp.asarray(inverse, dtype)
        self.diagonal = np.array_equal(cell, np.diag(np.diag(cell)))
        self.lengths = jnp.asarray(np.diag(cell), dtype)
        self.reciprocal = jnp.asarray(np.diag(inverse), dtype)
        self.search_scale = float(heights.min())
        gram = cell @ cell.T
        orthogonal = np.array_equal(gram, np.diag(np.diag(gram)))
        extent = np.floor(0.5 + search_radius / heights).astype(int)
        shifts = (
            []
            if orthogonal
            else [
                n
                for n in itertools.product(*(range(-e, e + 1) for e in extent))
                if any(n)
            ]
        )
        self.corrections = jnp.asarray(np.array(shifts).reshape(-1, 3) @ cell, dtype)

    def fractional(self, r):
        return (
            r * self.reciprocal
            if self.diagonal
            else jnp.matmul(r, self.inverse, precision="highest")
        )

    def cartesian(self, r):
        return (
            r * self.lengths
            if self.diagonal
            else jnp.matmul(r, self.cell, precision="highest")
        )

    def wrap(self, r):
        return r - self.cartesian(jnp.floor(self.fractional(r)))

    def minimum_image(self, delta):
        base = delta - self.cartesian(jnp.round(self.fractional(delta)))
        if not len(self.corrections):
            return base

        def closer(i, best):
            trial = base + self.corrections[i]
            use = jnp.sum(trial * trial, axis=-1) < jnp.sum(best * best, axis=-1)
            return jnp.where(use[..., None], trial, best)

        return jax.lax.fori_loop(0, len(self.corrections), closer, base)
