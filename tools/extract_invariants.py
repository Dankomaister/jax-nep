"""Extract the six mathematical invariant equations from pinned NEP_CPU.

Use the same header as extract_tables.py. This translates only the scalar
reference formulas, not any implementation, control flow, or performance path.
"""

import argparse
import ast
from pathlib import Path
import re
import sympy as sp

parser = argparse.ArgumentParser()
parser.add_argument("header", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
text = args.header.read_text()
output = ['"""NEP_CPU scalar invariants; GPL-3.0-or-later. See docs/reference.md."""']
constants = ["C4B", "C5B", "C4B2", "C4B_123", "C4B_233", "C4B_134"]
for name in constants:
    raw = re.search(r"\b" + name + r"\[[^;]+?=\s*(\{.*?\});", text, re.S).group(1)
    output.append(
        name + " = " + repr(ast.literal_eval(raw.replace("{", "[").replace("}", "]")))
    )
body = text[text.rindex("void find_q(") :]
for flag in ("222", "1111", "112", "123", "233", "134"):
    chunk = re.search(r"if \(has_q_" + flag + r"\) \{(.*?)\n  \}", body, re.S).group(1)
    output.append("\ndef q" + flag + "(s):")
    if flag == "112":
        # Expanded q112 avoids an expensive fused lowering in the universal graph.
        # The scalar equation remains identical for every model and backend.
        syms = sp.symbols("s0:24")
        constants_table = {}
        for name in constants:
            raw = re.search(r"\b" + name + r"\[[^;]+?=\s*(\{.*?\});", text, re.S).group(
                1
            )
            constants_table[name] = ast.literal_eval(
                raw.replace("{", "[").replace("}", "]")
            )
        rhs = chunk.split("=", 1)[1].split(";")[0]
        expression = eval(
            " ".join(rhs.split()), {"__builtins__": {}}, {**constants_table, "s": syms}
        )
        terms = sp.Poly(sp.expand(expression), syms).terms()
        output.append(
            "    return "
            + " + ".join(
                " * ".join(
                    [repr(float(c))]
                    + [
                        f"s[{i}]"
                        for i, count in enumerate(powers)
                        for _ in range(count)
                    ]
                )
                for powers, c in terms
            )
        )
        continue
    for statement in chunk.split(";"):
        statement = " ".join(statement.split()).replace("double ", "")
        if not statement:
            continue
        if statement.startswith("q["):
            statement = "return " + statement.split("=", 1)[1].strip()
        output.append("    " + statement)
output += [
    "",
    "FUNCTIONS = {"
    + ", ".join(repr(n) + ":q" + n for n in ("222", "1111", "112", "123", "233", "134"))
    + "}",
]
args.output.write_text("\n".join(output) + "\n")
