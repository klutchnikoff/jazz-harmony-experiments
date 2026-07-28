"""Article data for Section 4.3, "Injectivity".

Every finite computation used in the proofs is checked here in exact rational
arithmetic.

  six rows         Their numerators contain no 4.  Their four occurrences of 3
                   and the selected coefficients of 2 and 1 give exactly the
                   eleven equations used in Proposition 4.1.  Those equations
                   have rank eleven.
  linear order     Clearing the row denominators gives an integer matrix of rank
                   nine.  The stated u and v belong to its kernel and are
                   independent, so they form a basis.
  distinct values  The two cases in the proof give respectively five and six
                   distinct nonzero values.
  exhaustive check Phi_1 separates all 2047 nonzero kinds.

Exports only the two kernel vectors and the coordinate lists printed in the
manuscript.

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


def indicator(*coordinates):
    """Row vector selecting one-based interval coordinates."""
    return [int(i + 1 in coordinates) for i in range(11)]


# The eleven coefficient equations used to prove delta = 0.
ELIMINATION = [
    indicator(3),
    indicator(4),
    indicator(6),
    indicator(7),
    indicator(7, 9),
    indicator(7, 10),
    indicator(1, 3),
    indicator(2, 3, 7, 8),
    indicator(4, 5, 7, 11),
    indicator(2, 9),
    indicator(5, 10),
]


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


def rank(matrix):
    """Exact row rank over the rationals."""
    a = [[Fraction(x) for x in row] for row in matrix]
    n_rows, n_cols = len(a), len(a[0])
    pivot_row = 0
    for column in range(n_cols):
        pivot = next(
            (row for row in range(pivot_row, n_rows)
             if a[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        a[pivot_row], a[pivot] = a[pivot], a[pivot_row]
        scale = a[pivot_row][column]
        a[pivot_row] = [entry / scale for entry in a[pivot_row]]
        for row in range(n_rows):
            if row == pivot_row or a[row][column] == 0:
                continue
            scale = a[row][column]
            a[row] = [
                entry - scale * pivot_entry
                for entry, pivot_entry in zip(a[row], a[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == n_rows:
            break
    return pivot_row


def assert_prime_power_proof(numerators):
    """Check the coefficient equations in the 1, 2^p, 3^p proof."""
    first_six = numerators[:6]
    assert all(4 not in row for row in first_six)

    beta_locations = {
        (row, column)
        for row in range(6)
        for column in range(11)
        if numerators[row][column] == 3
    }
    assert beta_locations == {(0, 5), (2, 3), (3, 2), (5, 6)}
    assert {
        tuple(indicator(column + 1))
        for _, column in beta_locations
    } == {tuple(row) for row in ELIMINATION[:4]}

    alpha_equations = [
        indicator(*(i + 1 for i, entry in enumerate(numerators[row])
                    if entry == 2))
        for row in (3, 2, 5, 4, 1)
    ]
    rational_equations = [
        indicator(*(i + 1 for i, entry in enumerate(numerators[row])
                    if entry == 1))
        for row in (1, 4)
    ]
    assert alpha_equations == ELIMINATION[4:9]
    assert rational_equations == ELIMINATION[9:]
    assert rank(ELIMINATION) == 11
    print("prime-power proof: six row patterns and rank-11 elimination verified")


def main():
    W, N = exact_system()
    assert_prime_power_proof(N)
    assert rank(N) == 9, "the integer matrix does not have rank nine"
    assert np.linalg.matrix_rank(SYSTEM) == 9, "W_0 is not of rank nine"
    print("the row-rescaled integer matrix has rank nine")

    for name, vec in (("u", U), ("v", V)):
        products = [sum(N[j][i] * vec[i] for i in range(11)) for j in range(9)]
        assert all(p == 0 for p in products), f"B_1 {name} is not zero: {products}"
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
        # the coordinate lists produced by the two cases at p = 1
        "case_b": ",".join(f"{c}b" for c in case_b).replace("1b", "b"),
        "case_a": ",".join(f"{c}a" for c in case_a).replace("1a", "a"),
    })


if __name__ == "__main__":
    main()
