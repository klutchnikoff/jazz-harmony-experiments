"""Article data for Section 4.4, "What the reading separates".

Every step of Lemma 4.1 and Proposition 4.2 that the reader is asked to take on
a computation is checked here, in exact rational arithmetic.

  rank nine        Scaling each row by its denominator turns W_0 into the
                   integer matrix of numerators in Table 2.  Its columns
                   1,2,3,4,5,8,9,10,11 have determinant -2, so the rank is nine
                   and the kernel is a plane.  On 70 W_0 the same minor would
                   read -230592040, which is why the article rescales row by row.
  the kernel       W_0 u = W_0 v = 0, the eighteen dot products of the proof,
                   and u, v independent, so they span that plane.
  five values      Every nonzero a u + b v takes at least five distinct nonzero
                   values, tested on the two cases the proof distinguishes.
  injectivity      Phi_1 separates the 2047 nonzero kinds, which is the first
                   part of the proposition over all of K minus zero rather than
                   over the vocabulary.

Exports only what the manuscript prints: the two kernel vectors, the columns of
the minor, its determinant, and the coordinate lists of the two cases.

Run:  LSA_LOCAL=1 .venv/bin/python ART-what-the-reading-separates.py
"""
import itertools
from fractions import Fraction
from math import gcd

import numpy as np

from article_data import export
from chord_scale import SYSTEM

U = [-3, -5, -1, 0, 0, 0, 1, 5, 3, 0, 0]
V = [-32, -40, 8, 3, -15, -2, 8, 24, 0, 15, 24]
MINOR = (1, 2, 3, 4, 5, 8, 9, 10, 11)


def exact_system():
    """W_0 as fractions, and the integer matrix of numerators of Table 2."""
    rows, numerators = [], []
    for row in SYSTEM:
        fractions = [Fraction(x).limit_denominator(10 ** 6) for x in row]
        assert sum(fractions) == 1, "a mode does not sum to one"
        d = 1
        for f in fractions:
            d = d * f.denominator // gcd(d, f.denominator)
        rows.append(fractions)
        numerators.append([int(f * d) for f in fractions])
    return rows, numerators


def determinant(matrix):
    """Exact determinant by fraction-free elimination."""
    a = [[Fraction(x) for x in row] for row in matrix]
    n, det = len(a), Fraction(1)
    for c in range(n):
        pivot = next((r for r in range(c, n) if a[r][c] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != c:
            a[c], a[pivot] = a[pivot], a[c]
            det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            factor = a[r][c] / a[c][c]
            for k in range(c, n):
                a[r][k] -= factor * a[c][k]
    return det


def main():
    W, N = exact_system()

    det = determinant([[N[r][c - 1] for c in MINOR] for r in range(9)])
    assert det != 0, "the stated minor is singular, so the rank claim fails"
    assert np.linalg.matrix_rank(SYSTEM) == 9, "W_0 is not of rank nine"
    print(f"minor on columns {','.join(map(str, MINOR))}: determinant {det}")

    for name, vec in (("u", U), ("v", V)):
        products = [sum(W[j][i] * vec[i] for i in range(11)) for j in range(9)]
        assert all(p == 0 for p in products), f"W_0 {name} is not zero: {products}"
    print(f"the eighteen dot products vanish exactly")

    # u and v are independent, so the plane they span is the whole kernel
    zeros_u = [i for i in range(11) if U[i] == 0]
    assert any(V[i] != 0 for i in zeros_u) and any(U[i] != 0 for i in range(11)
                                                   if V[i] == 0)
    assert [i + 1 for i in zeros_u] == [4, 5, 6, 10, 11]

    case_b = [V[i] for i in zeros_u]
    case_a = [U[i] for i in range(11) if U[i] != 0]
    assert len(set(case_b)) == 5 and 0 not in case_b
    assert len(set(case_a)) == 6 and 0 not in case_a
    print(f"b != 0 gives {case_b} at positions {[i+1 for i in zeros_u]}")
    print(f"b  = 0 gives {case_a} at the other six")

    # Proposition 4.2 at p = 1, over K minus zero and not merely the vocabulary
    seen = {}
    for bits in itertools.product((0, 1), repeat=11):
        if not any(bits):
            continue
        raw = [sum(W[j][i] for i in range(11) if bits[i]) for j in range(9)]
        total = sum(raw)
        assert total > 0, "a nonzero kind reads as the zero vector"
        key = tuple(r / total for r in raw)
        assert key not in seen, f"Phi_1 confuses {bits} with {seen[key]}"
        seen[key] = bits
    print(f"Phi_1 separates all {len(seen)} nonzero kinds")

    export("what-the-reading-separates", {
        "kernel_u": ",".join(map(str, U)),
        "kernel_v": ",".join(map(str, V)),
        "minor_columns": ",".join(map(str, MINOR)),
        "minor_determinant": str(det),
        # the coordinate lists the two cases of the lemma produce
        "case_b": ",".join(f"{c}b" for c in case_b).replace("1b", "b"),
        "case_a": ",".join(f"{c}a" for c in case_a).replace("1a", "a"),
    })


if __name__ == "__main__":
    main()
