# Reproducing the article computations

The scripts in this directory reproduce the figures and numerical diagnostics used in the article.

They use the independently versioned `leadsheetanalyser` package located at the root of the working tree.

The Real Book family-frequency calculation reads the processed 2,846-song corpus from `leadsheetanalyser/data/music_realbook.pkl`.

Run all computations from the article root with:

```sh
python py-code/generate_all.py
```

The figure-producing scripts write their PDF and PNG outputs to `fig/`.

Each script can also be run independently.

The optional argument `--out DIR` changes the output directory of a figure-producing script.
