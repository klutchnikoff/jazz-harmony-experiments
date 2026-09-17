# Reproducing the article's figures and numerical results

This repository holds the scripts that produce every figure and reported
number in the article, from chord-level diagnostics through the corpus
study. It depends on the sibling `leadsheetanalyser` 0.3 source tree (chord
representation and dissimilarity code). The release tag fixes the exact source
revision used for the article.

## Layout

Clone this repository and `leadsheetanalyser` side by side, under any common
parent directory:

```
<parent>/
    leadsheetanalyser/          # provides data/ and the package
    jazz-harmony-experiments/   # this repository
    TeX/                        # optional sibling manuscript repository
    results/                    # created on first run: figures, tables, cache
```

`results/` is created automatically outside both repositories: the figures
are build products, not sources, so a reader with only these two public
repositories can reproduce them without the (private) manuscript repository.

Reusable experiment-level code lives in `article_analysis/`. The executable
`ART-*.py`, `FIG-*.py`, and `ROBUST-*.py` files retain the article's design
choices and orchestration; chord transformations and modal profiles come from
`leadsheetanalyser`.

## Prerequisites

- Python 3.12
- [`pdm`](https://pdm-project.org)

## Setup

1. Clone both repositories as siblings:

   ```sh
   git clone https://github.com/klutchnikoff/leadsheetanalyser.git
   git clone https://github.com/klutchnikoff/jazz-harmony-experiments.git
   ```

2. Install `leadsheetanalyser`'s own dependencies, then fetch the corpus data
   and build the processed pickle (see `leadsheetanalyser/DATA.md` for
   details on the data-fetching step):

   ```sh
   cd leadsheetanalyser
   pdm install
   pdm run python scripts/download_data.py
   pdm run python scripts/build_corpus.py
   cd ..
   ```

3. Install this repository's dependencies:

   ```sh
   cd jazz-harmony-experiments
   pdm install
   ```

## Running

The dependency installed by PDM is the sibling `leadsheetanalyser` 0.3 source
tree. While editing that package, `LSA_LOCAL=1` makes Python use the live tree
rather than PDM's installed copy. Reproduce everything with:

```sh
LSA_LOCAL=1 pdm run python generate_all.py
```

This runs every script in sequence and writes:

- canonical figures (`.pdf` and `.png`) to `../results/figures/`
- cached intermediates (distance matrices, key audits) to `../results/cache/`
- atomic values to the versioned `article-data/*.json` files
- when `../TeX/main.tex` is present, an aggregate LaTeX snapshot to
  `../TeX/generated/article-values.tex`, followed at the end of the full
  pipeline by an atomic copy of the completed figures into `../TeX/fig/`
- an exact run manifest at `../results/run-manifest.json`, containing repository
  revisions and worktree state, runtime versions, and SHA-256 hashes of source
  data, cached audits, versioned JSON, generated macros, and figures

The LaTeX snapshot is refreshed after each `ART-*.py` export. It is generated
atomically and carries a SHA-256 digest of all `article-data` JSON. A normal
LaTeX build only reads that file: it never starts Python or reruns the analysis.
Consequently, the manuscript reflects the last producer run, not necessarily
the current Python source if the producers have not been rerun. Both the JSON
and generated TeX snapshot are versioned so that this state is reviewable.

A full run takes roughly 15 minutes, dominated by key-estimation audits over
the *Real Book* corpus; a timing summary prints at the end. Add `--force` to
rebuild cached audits instead of reusing them.

Each script can also be run on its own, e.g.
`LSA_LOCAL=1 pdm run python ART-the-order-used-below.py`.
Every run announces on stderr which `leadsheetanalyser` it resolved, and where
it read data from and wrote results to.

The prespecified sensitivity analysis over every hundredth from `p=0.14` to
`p=0.20` writes its complete design and results to
`analysis-data/p-sensitivity.json`; the following `ART-the-p-sensitivity.py`
step exports the summary values used in the manuscript:

```sh
LSA_LOCAL=1 pdm run python ROBUST-p-sensitivity.py
```

It writes the full design and results to `analysis-data/p-sensitivity.json`.

## Regression tests

Fast unit tests cover the shared representation code and byte-for-byte SHA-256
guards cover every versioned scientific JSON consumed by the manuscript:

```sh
LSA_LOCAL=1 pdm run python -m unittest discover -s tests -p 'test_*.py'
```

These tests do not recompute the corpus analyses. Run `generate_all.py` for the
full integration check; its final check verifies that the generated TeX is
byte-for-byte current and that every exported key is referenced by
`TeX/main.tex` when the manuscript repository is present as a sibling.

## Using a live, edited `leadsheetanalyser`

After changes to the sibling package, either rerun `pdm install` or use its live
source tree directly:

```sh
LSA_LOCAL=1 pdm run python generate_all.py
```
