"""Setup-time domain work estimate; no benchmark or atom/species thresholds."""

from dataclasses import dataclass
import numpy as np
from jax_md import partition
from .neighbors import NeighborProvider


@dataclass(frozen=True)
class ExecutionPlan:
    separate_angular: bool
    representation: object


class PlannedNeighbors:
    """Choose once per allocation; update never changes the static plan."""

    def __init__(self, model, box, species, **settings):
        self.model = model
        self.arguments = (box, species, model.cutoff)
        self.settings = settings
        self.shared = NeighborProvider(*self.arguments, **settings)
        self.types, self.backend = self.shared.types, self.shared.backend
        angular_cutoff = float(model.cutoffs[:, 1].max())
        self.angular = (
            NeighborProvider(box, species, angular_cutoff, **settings)
            if angular_cutoff < model.cutoff
            else None
        )
        self.providers = [self.shared]
        self.plans = {}

    def allocate(self, positions, extra_capacity=0):
        radial = self.shared.allocate(positions, extra_capacity)
        angular = (
            radial
            if self.angular is None
            else self.angular.allocate(positions, extra_capacity)
        )
        # Work saved by omitting padded angular basis projections and moment
        # contractions, against a second geometry pass and moment workspace.
        ca = self.model.orders[1] + 1
        ka = self.model.basis_orders[1] + 1
        h = self.model.l_max * (self.model.l_max + 2)
        saved = (radial.idx.size - angular.idx.size) * ca * (ka + h)
        # Three displacement and direction components, norm (5 + sqrt),
        # padding masks, and 12 scalar minimum-image operations for a diagonal
        # cell (two general 3x3 products instead for a skew cell).
        lattice = (self.shared if self.angular is None else self.angular).lattice
        geometry_work = 3 + 3 + 5 + 1 + 3 + (12 if lattice.diagonal else 54)
        geometry_work += len(lattice.corrections) * (3 + 5 + 1 + 3)
        extra = angular.idx.size * (geometry_work + ca * h)
        separate = saved > extra
        # Native Sparse stores two indices per edge; Dense stores one per slot.
        # Compare their actual capacity formulas, including caller headroom.
        n = len(self.types)
        indices = np.asarray(radial.idx)
        live = int(np.count_nonzero((indices >= 0) & (indices < n)))
        sparse_capacity = min(
            int(
                live * self.settings.get("capacity_multiplier", 1.25)
                + n * extra_capacity
            ),
            n * (n - 1),
        )
        sparse_saves_indices = 2 * sparse_capacity < radial.idx.size
        representation = (
            partition.Sparse
            if self.backend == "STANDARD" and (separate or sparse_saves_indices)
            else partition.Dense
        )
        plan = ExecutionPlan(separate, representation)
        if not separate and representation == partition.Dense:
            return radial
        if plan not in self.plans:
            provider = NeighborProvider(
                *self.arguments, **self.settings, representation=representation
            )
            if separate:
                provider.angular_provider = NeighborProvider(
                    self.arguments[0],
                    self.arguments[1],
                    float(self.model.cutoffs[:, 1].max()),
                    **self.settings,
                    representation=representation,
                )
            self.plans[plan] = provider
            self.providers.append(provider)
        return self.plans[plan].allocate(positions, extra_capacity)

    def validate(self, neighbor):
        if not hasattr(neighbor, "owner") or neighbor.owner not in self.providers:
            raise ValueError(
                "neighbor must come from the matching factory; raw/foreign list rejected"
            )

    def update(self, positions, neighbor):
        self.validate(neighbor)
        return neighbor.owner.update(positions, neighbor)

    def pair_domains(self, positions, neighbor):
        self.validate(neighbor)
        return neighbor.owner.pair_domains(positions, neighbor)
