from pathlib import Path
import numpy as np
import pytest
from jax_nep import load_model

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "fixture", sorted(FIXTURES.glob("*.npz")), ids=lambda p: p.stem
)
def test_models(fixture, tmp_path):
    data = np.load(fixture)
    path = tmp_path / "nep.txt"
    path.write_text(str(data["model_text"]))
    m = load_model(path)
    assert m.dimension == data["descriptors"].shape[1]
    assert m.cutoff > 0
    np.testing.assert_array_equal(m.map_species(m.species), np.arange(len(m.species)))
    with pytest.raises(ValueError):
        m.map_species(["not-an-element"])


@pytest.mark.parametrize(
    "kind",
    [
        "nep1",
        "nep2",
        "nep3",
        "nep5",
        "qnep",
        "nep4_charge1",
        "nep4_dipole",
        "nep4_polarizability",
    ],
)
def test_unsupported(kind, tmp_path):
    p = tmp_path / "nep.txt"
    p.write_text(kind + " 1 C\n")
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        load_model(p)


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace("nep4 2 C Si", "nep4 2 C C"),
        lambda s: s.replace("nep4 2 C Si", "nep4 2 C Xx"),
        lambda s: s.replace("cutoff 4.0 3.0", "cutoff -4 3"),
        lambda s: s.replace("cutoff 4.0 3.0", "cutoff 3 4"),
        lambda s: s.replace("n_max 1 1", "n_max 17 1"),
        lambda s: s.replace("basis_size 3 3", "basis_size 3 -1"),
        lambda s: s.replace("l_max 4 1 1 1 1 1 1", "l_max 3 1 1 1 1 1 1"),
        lambda s: s.replace("ANN 5 0", "ANN 0 0"),
        lambda s: s.replace("ANN 5 0", "ANN 5 1"),
        lambda s: s + "0\n",
        lambda s: "\n".join(s.splitlines()[:-1]),
        lambda s: s.rsplit("\n", 2)[0] + "\nnan\n",
        lambda s: "",
    ],
)
def test_invalid(change, tmp_path):
    text = str(np.load(FIXTURES / "synthetic_l4.npz")["model_text"])
    p = tmp_path / "nep.txt"
    p.write_text(change(text))
    with pytest.raises(ValueError):
        load_model(p)


@pytest.mark.parametrize("dtype", [np.int32, np.float16])
def test_dtype(dtype, tmp_path):
    p = tmp_path / "nep.txt"
    p.write_text(str(np.load(FIXTURES / "synthetic_l4.npz")["model_text"]))
    with pytest.raises(ValueError):
        load_model(p, dtype=dtype)
