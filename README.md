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

3. Install this repository's dependencies. This pulls the pinned
   `leadsheetanalyser` release from PyPI, not the sibling source tree:

   ```sh
   cd jazz-harmony-experiments
   pdm install
   ```

## Running

Reproduce everything with:

```sh
pdm run python generate_all.py
```

This runs every script in sequence and writes:

- figures (`.pdf` and `.png`) to `../results/`
- cached intermediates (distance matrices, key audits) to `../results/cache/`

A full run takes roughly 15 minutes, dominated by key-estimation audits over
the *Real Book* corpus; a timing summary prints at the end. Add `--force` to
rebuild cached audits instead of reusing them.

Each script can also be run on its own, e.g. `pdm run python modal_affinities.py`.
Every run announces on stderr which `leadsheetanalyser` it resolved, and where
it read data from and wrote results to.

## Using a local, edited `leadsheetanalyser`

By default the pinned PyPI release is used, matching what the article cites.
To test against the sibling source tree instead, set `LSA_LOCAL=1`:

```sh
LSA_LOCAL=1 pdm run python generate_all.py
```
