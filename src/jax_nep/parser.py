"""Strict text parsing and parameter unpacking for scalar NEP4."""

from pathlib import Path
import numpy as np
from .model import NEPModel
from .tables import ELEMENTS, COVALENT_RADIUS


class _Reader:
    def __init__(self, text):
        self.lines = iter(text.splitlines())

    def header(self, name=None, counts=None, cast=str):
        tokens = next(self.lines, "").split()
        if not tokens or (name is not None and tokens.pop(0) != name):
            raise ValueError(f'missing or invalid {name or "model"} header')
        if counts is not None and len(tokens) not in counts:
            raise ValueError(f"invalid field count for {name}")
        return tuple(map(cast, tokens))


def _zbl_table(spec, values, numbers, angular_cutoffs, dtype):
    nt = len(numbers)
    if spec is None:
        return None
    table = np.empty((nt, nt, 10), dtype=dtype)
    inner, outer, *factor = spec
    if inner == outer == 0:
        for row, (i, j) in zip(values.reshape(-1, 10), zip(*np.triu_indices(nt))):
            table[i, j] = table[j, i] = row
    else:
        table[:] = [
            inner,
            outer,
            0.18175,
            3.1998,
            0.50986,
            0.94229,
            0.28022,
            0.4029,
            0.02817,
            0.20162,
        ]
        if factor:
            radius = np.asarray(COVALENT_RADIUS)[numbers - 1]
            table[:, :, 0] = 0
            table[:, :, 1] = np.minimum(
                outer, factor[0] * (radius[:, None] + radius[None, :])
            )
    if (
        np.any(table[:, :, 0] < 0)
        or np.any(table[:, :, 1] <= table[:, :, 0])
        or np.any(table[:, :, 2:] < 0)
    ):
        raise ValueError("invalid ZBL radii or screening coefficients")
    if np.any(
        table[:, :, 1] > (angular_cutoffs[:, None] + angular_cutoffs[None, :]) * 0.5
    ):
        raise ValueError("ZBL outer cutoff exceeds angular cutoff")
    return table


def load_model(path, *, dtype=np.float32):
    """Load NEP4 or NEP4_ZBL; reject unsupported families and malformed data."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    reader = _Reader(Path(path).read_text())
    header = reader.header()
    if header[0] not in ("nep4", "nep4_zbl"):
        raise ValueError(f"Unsupported model {header[0]!r}")
    try:
        nt = int(header[1])
        species = header[2:]
        if not 1 <= nt <= 94 or len(species) != nt or len(set(species)) != nt:
            raise ValueError("invalid species count or duplicate symbols")
        numbers = np.array([ELEMENTS.index(s) + 1 for s in species], np.int32)
        zbl = (
            reader.header("zbl", (2, 3), float) if header[0].endswith("_zbl") else None
        )
        if zbl is not None:
            if (
                not np.isfinite(zbl).all()
                or min(zbl) < 0
                or (zbl[:2] != (0, 0) and zbl[1] <= zbl[0])
                or (len(zbl) == 3 and (zbl[2] <= 0 or zbl[:2] == (0, 0)))
            ):
                raise ValueError("invalid ZBL header")
        cutoff = reader.header("cutoff", (4, 2 * nt + 2))
        cutoffs = np.broadcast_to(
            np.array(cutoff[:-2], dtype=dtype).reshape(-1, 2), (nt, 2)
        ).copy()
        if (
            not np.isfinite(cutoffs).all()
            or np.any(cutoffs <= 0)
            or np.any(cutoffs[:, 1] > cutoffs[:, 0])
            or min(map(int, cutoff[-2:])) < 0
        ):
            raise ValueError("invalid cutoffs or neighbor hints")
        orders = reader.header("n_max", (2,), int)
        basis_orders = reader.header("basis_size", (2,), int)
        if min(orders + basis_orders) < 0 or max(orders + basis_orders) > 16:
            raise ValueError("basis_size and n_max must be in 0..16")
        angular = reader.header("l_max", range(3, 8), int)
        ell, flags = angular[0], angular[1:]
        if (
            not 1 <= ell <= 8
            or flags[0] not in (0, 1, 2)
            or any(f not in (0, 1) for f in flags[1:])
        ):
            raise ValueError("invalid angular order or flags")
        names = ("222", "1111", "112", "123", "233", "134")
        invariants = tuple(name for name, f in zip(names, flags) if f)
        if any(ell < required for f, required in zip(flags, (2, 1, 2, 3, 3, 4)) if f):
            raise ValueError("harmonic order too small for requested invariant")
        hidden, output = reader.header("ANN", (2,), int)
        if hidden <= 0 or output != 0:
            raise ValueError("ANN requires positive hidden width and output field zero")
        dim = orders[0] + 1 + (orders[1] + 1) * (ell + len(invariants))
        counts = [
            nt * hidden * (dim + 2),
            1,
            *[nt * nt * (n + 1) * (k + 1) for n, k in zip(orders, basis_orders)],
            dim,
        ]
        flexible = zbl is not None and zbl[:2] == (0, 0)
        expected = sum(counts) + (5 * nt * (nt + 1) if flexible else 0)
        values = np.array(
            [float(v) for line in reader.lines for v in line.split()], dtype=dtype
        )
        if values.size != expected:
            raise ValueError(
                f"parameter count: expected {expected}, found {values.size}"
            )
        if not np.isfinite(values).all():
            raise ValueError("nonfinite model parameters")
        network, bias, radial, angular, scale, tail = np.split(
            values, np.cumsum(counts)
        )
        packed = network.reshape(nt, -1)
        w = packed[:, : hidden * dim].reshape(nt, hidden, dim).swapaxes(1, 2)
        b = packed[:, hidden * dim : hidden * (dim + 1)]
        out = packed[:, hidden * (dim + 1) :]
        coefficients = tuple(
            a.reshape(n + 1, k + 1, nt, nt).transpose(2, 3, 1, 0)
            for a, n, k in zip((radial, angular), orders, basis_orders)
        )
        if np.any(scale <= 0):
            raise ValueError("descriptor scalers must be positive")
        table = _zbl_table(zbl, tail, numbers, cutoffs[:, 1], dtype)
        arrays = [cutoffs, w, b, out, bias, scale, numbers, *coefficients]
        if table is not None:
            arrays.append(table)
        for value in arrays:
            value.setflags(write=False)
        return NEPModel(
            species,
            cutoffs,
            orders,
            basis_orders,
            ell,
            invariants,
            w,
            b,
            out,
            bias[0],
            coefficients,
            scale,
            table,
            numbers,
        )
    except (IndexError, OverflowError, TypeError) as error:
        raise ValueError(f"malformed NEP4 file {path}: {error}") from error
