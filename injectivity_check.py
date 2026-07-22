"""
Verification of the injectivity proof for the diatonic system W_D (Section 3).

Two kinds k, k' collide under Phi_{W_D} iff W_D (k - k') = 0, and k - k' ranges
over the nonzero vectors of {-1,0,1}^11 (conversely every such vector v is the
difference of the two binary kinds max(v,0) and max(-v,0)).  The article settles
this by an arithmetic argument rather than by enumeration, and this script
checks that argument step by step, in exact rational arithmetic:

  1. W_D has rank 7 with pivots on the first seven coordinates, so an element of
     ker W_D is determined by (delta_8, ..., delta_11);
  2. solving for the first coordinate gives
        delta_1 = (-489 d8 + 530 d9 - 440 d10 + 384 d11) / 285,  285 = 3*5*19;
  3. reduced modulo 3, 5 and 19 the numerator has coefficients too small for the
     sum to reach a nonzero multiple of the modulus, so each congruence forces
     an exact identity;
  4. those identities leave only delta = 0.

The 3^11 - 1 = 177,146 enumeration that the article previously relied on is kept
at the end as an independent control, no longer as the argument.

Note on row order: the article lists the modes by brightness (Table 3) while the
package lists them by degree.  A row permutation changes neither the rank, nor
the kernel, nor d_W, and the script verifies this rather than assuming it.

Run:  python py-code/injectivity_check.py
"""
from fractions import Fraction as F
from itertools import product
from math import gcd

import numpy as np

import article_setup  # noqa: F401  (puts the package on sys.path)
from leadsheetanalyser.constants import W_DIATONIC

# Ten times the weights, so that every entry is an integer.
WD = (np.asarray(W_DIATONIC, dtype=float) * 10).round().astype(int)
assert WD.shape == (7, 11)

# Table 3 of the article, ordered from brightest to darkest.
TABLE3 = np.array([
    [0, 1, 0, 2, 0, 3, 2, 0, 1, 0, 1],  # Lydian
    [0, 1, 0, 2, 2, 0, 2, 0, 1, 0, 2],  # Ionian
    [0, 1, 0, 3, 1, 0, 2, 0, 1, 2, 0],  # Mixolydian
    [0, 1, 3, 0, 1, 0, 2, 0, 2, 1, 0],  # Dorian
    [0, 2, 2, 0, 1, 0, 2, 2, 0, 1, 0],  # Aeolian
    [2, 0, 2, 0, 1, 0, 3, 1, 0, 1, 0],  # Phrygian
    [1, 0, 2, 0, 1, 4, 0, 1, 0, 1, 0],  # Locrian
], dtype=int)

FREE = (8, 9, 10, 11)               # one-based free coordinates
NUMERATOR = {8: -489, 9: 530, 10: -440, 11: 384}
DENOMINATOR = 285
REDUCED = {3: [0, -1, 1, 0], 5: [1, 0, 0, -1], 19: [5, -2, -3, 4]}

failures = []


def check(label, condition):
    print(f"  [{'ok' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)
    return condition


def rational_rref(matrix):
    """Exact reduced row echelon form; returns the rows and the pivot columns."""
    rows = [[F(int(x)) for x in row] for row in matrix]
    n_rows, n_cols = len(rows), len(rows[0])
    pivots, r = [], 0
    for c in range(n_cols):
        p = next((i for i in range(r, n_rows) if rows[i][c] != 0), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        pivot = rows[r][c]
        rows[r] = [x / pivot for x in rows[r]]
        for i in range(n_rows):
            if i != r and rows[i][c] != 0:
                factor = rows[i][c]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[r])]
        pivots.append(c)
        r += 1
    return rows, pivots


def kernel_basis(matrix):
    """Basis of ker(matrix), one exact vector per free coordinate."""
    rows, pivots = rational_rref(matrix)
    free = [c for c in range(matrix.shape[1]) if c not in pivots]
    basis = {}
    for f in free:
        v = [F(0)] * matrix.shape[1]
        v[f] = F(1)
        for i, c in enumerate(pivots):
            v[c] = -rows[i][f]
        basis[f + 1] = v
    return basis, pivots


print("Step 0 -- the package weights are those of Table 3")
check("same rows up to permutation",
      sorted(map(tuple, WD)) == sorted(map(tuple, TABLE3)))
sample = np.random.default_rng(0).integers(0, 2, size=(300, 11))
differences = sample[:, None, :] - sample[None, :, :]
check("the row order does not affect d_W",
      np.allclose(np.linalg.norm(differences @ WD.T, axis=2),
                  np.linalg.norm(differences @ TABLE3.T, axis=2)))

print("\nStep 1 -- rank and pivot structure")
basis, pivots = kernel_basis(WD)
check("rank is 7, with pivots on coordinates 1-7", pivots == list(range(7)))
check("the kernel is four-dimensional", sorted(basis) == list(FREE))

print("\nStep 2 -- the closed form for delta_1")
lcm = 1
for f in FREE:
    d = basis[f][0].denominator
    lcm = lcm * d // gcd(lcm, d)
check(f"common denominator is {DENOMINATOR} = 3*5*19", lcm == DENOMINATOR)
check("numerator coefficients match the article",
      all(int(basis[f][0] * DENOMINATOR) == NUMERATOR[f] for f in FREE))
check("each basis vector lies in ker(W_D)",
      all(np.allclose(WD @ np.array([float(x) for x in basis[f]]), 0)
          for f in FREE))

print("\nStep 3 -- each congruence forces an exact identity")
for p, expected in REDUCED.items():
    reduced = []
    for f in FREE:
        r = NUMERATOR[f] % p
        reduced.append(r - p if r > p // 2 else r)
    bound = sum(abs(r) for r in reduced)
    check(f"modulo {p:2d}: coefficients {reduced}, bound {bound} < {p}",
          reduced == expected and bound < p)

print("\nStep 4 -- the free coordinates vanish")
solutions = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if 9 * a == 5 * b]
check(f"9a = 5b has only the trivial solution in {{-1,0,1}}: {solutions}",
      solutions == [(0, 0)])
tails = [t for t in product((-1, 0, 1), repeat=4) if any(t)]
integral = [t for t in tails
            if sum(NUMERATOR[f] * v for f, v in zip(FREE, t)) % DENOMINATOR == 0]
check(f"none of the {len(tails)} nonzero tails makes delta_1 an integer",
      integral == [])

print(f"\nControl -- independent enumeration of the {3**11 - 1} difference vectors")
kernel_hits = [v for v in product((-1, 0, 1), repeat=11)
               if any(v) and not np.any(WD @ np.asarray(v))]
for v in kernel_hits:
    print(f"  collision: {tuple(np.maximum(v, 0))} vs {tuple(np.maximum(-np.asarray(v), 0))}")
check("no nonzero difference vector lies in ker W_D", kernel_hits == [])

if failures:
    raise SystemExit(f"\n{len(failures)} check(s) FAILED: {failures}")
print("\nAll checks passed: W_D is injective on {0,1}^11, so d_{W_D} is a metric.")
