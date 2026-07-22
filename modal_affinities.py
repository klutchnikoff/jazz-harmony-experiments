"""Heat maps of Phi_W(k) = W k over the twenty diagnostic chord kinds, showing
each kind's affinity to the seven diatonic modes.  Larger values mean greater
affinity.

Kinds are rows, grouped by intervallic family in the order of Table 2; modes are
columns, ordered from brightest to darkest.  Twenty kinds do not fit as columns
at the article's cell size, but they fit as rows.

Two panels on one colour scale.  The left uses the theory-guided weights W_D of
Table 3.  The right keeps the seven diatonic *supports* and replaces the weights
by uniform ones, so that [Phi_W(k)]_j can only count how many degrees of mode j
the chord occupies.  Cell values are deliberately not printed: the argument is
carried by the outlined maxima, and the few numbers the text quotes are printed
below.  Counting ties, 14 of the 20 kinds keep a unique most-affine mode under
W_D against 6 under uniform weights -- the figure quoted in Section 7.2.

The script also prints the agreement with the chord-scale pairing.

Run:  python py-code/modal_affinities.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")

from article_setup import output_directory
from diagnostic_vocabulary import VOCABULARY
from figure_style import (
    HEATMAP_ASPECT,
    HEATMAP_CMAP,
    paired_heatmap_axes,
    save_article_figure,
)
from leadsheetanalyser.constants import W_PYTHAGOREAN, PYTHAGOREAN_MODE_NAMES

OUT = output_directory()

WD = np.asarray(W_PYTHAGOREAN, float)
SUPPORT = (WD > 0).astype(float)
W_UNIFORM = SUPPORT / SUPPORT.sum(axis=1, keepdims=True)

# display order: brightest to darkest (as in the paper's Table 3)
BRIGHT = ["Lydian", "Ionian", "Mixolydian", "Dorian", "Aeolian", "Phrygian", "Locrian"]
order = [PYTHAGOREAN_MODE_NAMES.index(m) for m in BRIGHT]

from diagnostic_vocabulary import FAMILY_ORDER

# rows: the diagnostic vocabulary, grouped by family as in Table 2
BY_FAMILY = sorted(VOCABULARY, key=lambda t: FAMILY_ORDER.index(t[2]))
KIND_LABELS = [name.replace("C", "", 1) or "maj" for name, _, _ in BY_FAMILY]
KIND_VECTORS = [np.asarray(kind, float) for _, kind, _ in BY_FAMILY]
FAMILY_BREAKS = [i for i in range(1, len(BY_FAMILY))
                 if BY_FAMILY[i][2] != BY_FAMILY[i - 1][2]]
SHORT_MODES = ["Lyd", "Ion", "Mix", "Dor", "Aeo", "Phr", "Loc"]

PANELS = [
    (WD, r"theory-guided $W_D$"),
    (W_UNIFORM, "uniform on the same supports"),
]


def profiles(W):
    """Kinds as rows, modes as columns, modes ordered brightest to darkest."""
    return np.asarray([(W @ k)[order] for k in KIND_VECTORS])


def maxima(row, tol=1e-9):
    """Indices attaining the row maximum, i.e. the modes the chord prefers."""
    return np.flatnonzero(row >= row.max() - tol)


nrows, ncols = len(KIND_VECTORS), len(BRIGHT)
fig, axes, cax = paired_heatmap_axes(nrows, ncols, cell=0.22, bottom=0.5, title=0.22)

for ax, (W, title) in zip(axes, PANELS):
    P = profiles(W)
    image = ax.imshow(P, cmap=HEATMAP_CMAP, vmin=0, vmax=1.0, aspect=HEATMAP_ASPECT)
    ax.set_title(title, pad=5, fontsize=9)
    ax.set_xticks(range(ncols), labels=SHORT_MODES, rotation=45, ha="right",
                  fontsize=7)
    if ax is axes[0]:
        ax.set_yticks(range(nrows), labels=KIND_LABELS, fontsize=7)
    else:
        ax.set_yticks([])
    # ring every cell attaining its kind's maximum: one ring means the system
    # names a mode, several mean it cannot choose
    for i in range(nrows):
        for j in maxima(P[i]):
            ax.add_patch(matplotlib.patches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="red", linewidth=1.0, zorder=5))
    for b in FAMILY_BREAKS:
        ax.axhline(b - 0.5, color="0.25", linewidth=0.9, zorder=6)

fig.colorbar(image, cax=cax, label=r"$[\Phi_{W}(k)]_j$")
save_article_figure(fig, OUT, "modal_affinities_kinds")
print("saved modal_affinities_kinds.png and modal_affinities_kinds.pdf")

# The count quoted in Section 7.1, over the full diagnostic vocabulary.
K = np.asarray([kind for _, kind, _ in VOCABULARY], dtype=float)
print(f"\n{'system':32s}{'kinds with a unique modal maximum':>36s}")
for W, title in PANELS:
    rows = (W @ K.T).T
    decided = [name for (name, _, _), row in zip(VOCABULARY, rows)
               if len(maxima(row)) == 1]
    print(f"{title:32s}{f'{len(decided)} of {len(VOCABULARY)}':>36s}")
    ties = [(name, len(maxima(row))) for (name, _, _), row in zip(VOCABULARY, rows)
            if len(maxima(row)) > 1]
    if ties:
        print("    ties: " + ", ".join(f"{n} ({w}-way)" for n, w in ties))

# ---------------------------------------------------------------------------
# The external check of Section 7.1: chord-scale theory pairs each of these five
# chord kinds with a mode, and the weights were fixed without reference to it.
CHORD_SCALE = [
    ("Cmaj7",  "Ionian"),
    ("C7",     "Mixolydian"),
    ("Cmi7",   "Dorian"),
    ("Cmi7b5", "Locrian"),
    ("Cmi6",   "Dorian"),
]
BY_NAME = {name: np.asarray(kind, float) for name, kind, _ in VOCABULARY}

print("\nagreement with the chord-scale pairing")
print(f"  {'kind':10s}{'chord-scale':14s}{'maxima under W_D':30s}{'under uniform weights'}")
agreed = {"W_D": 0, "uniform": 0}
unique = {"W_D": 0, "uniform": 0}
for kind_name, expected in CHORD_SCALE:
    cells = {}
    for label, W in (("W_D", WD), ("uniform", W_UNIFORM)):
        names = [PYTHAGOREAN_MODE_NAMES[i] for i in maxima(W @ BY_NAME[kind_name])]
        cells[label] = names
        if expected in names:
            agreed[label] += 1
            if len(names) == 1:
                unique[label] += 1
    print(f"  {kind_name:10s}{expected:14s}"
          f"{'/'.join(cells['W_D']):30s}{'/'.join(cells['uniform'])}")
for label in ("W_D", "uniform"):
    print(f"  {label:8s}: pairing among the maxima {agreed[label]}/5, "
          f"as the unique maximum {unique[label]}/5")
