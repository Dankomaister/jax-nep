"""Regenerate NEP constants and invariant polynomials from pinned NEP_CPU.

Development dependency: sympy. Generated data retains upstream GPL attribution.
This script extracts mathematical data; it does not translate an evaluator.
"""

import argparse
import ast
from pathlib import Path
import re
import pprint

p = argparse.ArgumentParser()
p.add_argument("header", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
text = a.header.read_text()
data = {}
for name in [
    "C3B",
    "C4B",
    "C5B",
    "C4B2",
    "C4B_123",
    "C4B_233",
    "C4B_134",
    *["Z_COEFFICIENT_" + str(i) for i in range(1, 9)],
    "ELEMENTS",
    "COVALENT_RADIUS",
]:
    raw = re.search(r"\b" + name + r"\[[^;]+?=\s*(\{.*?\});", text, re.S).group(1)
    data[name] = ast.literal_eval(raw.replace("{", "[").replace("}", "]"))
output = [
    '"""Mathematical tables from NEP_CPU af615ce0 (nep_utilities.h).',
    "Copyright 2022 Zheyong Fan, Junjie Wang, Eric Lindgren.",
    "GPL-3.0-or-later. Regenerate with tools/extract_tables.py.",
    '"""',
    "",
]
for name, value in [
    ("ELEMENTS", tuple(data["ELEMENTS"])),
    ("COVALENT_RADIUS", data["COVALENT_RADIUS"]),
    ("C3B", data["C3B"]),
    ("Z_POLYNOMIALS", [data["Z_COEFFICIENT_" + str(i)] for i in range(1, 9)]),
]:
    output.append(
        name
        + " = "
        + pprint.pformat(value, width=100, compact=True, sort_dicts=False)
        + "\n"
    )
a.output.write_text("\n".join(output))
