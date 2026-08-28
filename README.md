# Reproducing the article's figures and numerical results

This repository holds the scripts that produce every figure and reported
number in the article, from chord-level diagnostics through the corpus
study. It depends on the `leadsheetanalyser` package (chord representation
and dissimilarity code), pinned to the release used for the article.

## Layout

Clone this repository and `leadsheetanalyser` side by side, under any common
parent directory:

```
<parent>/
    leadsheetanalyser/          # provides data/ and the package
    jazz-harmony-experiments/   # this repository
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

On the v0.3 development branches, use the sibling `leadsheetanalyser` source
tree: it contains `modal_profile`, which is not part of the published 0.2.0
release. Reproduce everything with:

```sh
LSA_LOCAL=1 pdm run python generate_all.py
```

This runs every script in sequence and writes:

- figures (`.pdf` and `.png`) to `../results/`
- cached intermediates (distance matrices, key audits) to `../results/cache/`

A full run takes roughly 15 minutes, dominated by key-estimation audits over
the *Real Book* corpus; a timing summary prints at the end. Add `--force` to
rebuild cached audits instead of reusing them.

Each script can also be run on its own, e.g.
`LSA_LOCAL=1 pdm run python ART-the-order-used-below.py`.
Every run announces on stderr which `leadsheetanalyser` it resolved, and where
it read data from and wrote results to.

The prespecified sensitivity analysis over every hundredth from `p=0.14` to
`p=0.20` is separate from the manuscript-number producers until its results are
explicitly incorporated:

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
full integration check; its final producer also verifies every exported value
against `TeX/main.tex` when the manuscript repository is present as a sibling.

## Using a local, edited `leadsheetanalyser`

By default the pinned PyPI release is used. Until `leadsheetanalyser` 0.3.0 is
released and pinned here, the v0.3 scripts require the sibling source tree via
`LSA_LOCAL=1`:

```sh
LSA_LOCAL=1 pdm run python generate_all.py
```
