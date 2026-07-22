"""Shared visual defaults for the article figures."""

import matplotlib.pyplot as plt


HEATMAP_FIGSIZE = (4.4, 2.9)
HEATMAP_CMAP = "Blues"
HEATMAP_ASPECT = "equal"
HEATMAP_FONT_SIZE = 9
HEATMAP_CELL_SIZE = 0.4125
HEATMAP_BOTTOM = 0.62
HEATMAP_HORIZONTAL_LABEL_BOTTOM = 0.32
HEATMAP_TOP = 0.2175
HEATMAP_COLORBAR_PAD = 0.135
HEATMAP_COLORBAR_WIDTH = 0.09
FIGURE_DPI = 180
FIGURE_PAD = 0.03

plt.rcParams.update(
    {
        "font.size": HEATMAP_FONT_SIZE,
        "font.weight": "normal",
        "axes.labelweight": "normal",
    }
)


def annotated_heatmap_axes(nrows, ncols, bottom=HEATMAP_BOTTOM):
    """Return fixed axes giving every heat-map cell the same physical size."""
    figure_width, _ = HEATMAP_FIGSIZE
    matrix_width = ncols * HEATMAP_CELL_SIZE
    matrix_height = nrows * HEATMAP_CELL_SIZE
    figure_height = bottom + matrix_height + HEATMAP_TOP
    group_width = (
        matrix_width
        + HEATMAP_COLORBAR_PAD
        + HEATMAP_COLORBAR_WIDTH
    )
    left = (figure_width - group_width) / 2

    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = fig.add_axes(
        [
            left / figure_width,
            bottom / figure_height,
            matrix_width / figure_width,
            matrix_height / figure_height,
        ]
    )
    cax = fig.add_axes(
        [
            (left + matrix_width + HEATMAP_COLORBAR_PAD) / figure_width,
            bottom / figure_height,
            HEATMAP_COLORBAR_WIDTH / figure_width,
            matrix_height / figure_height,
        ]
    )
    return fig, ax, cax


def paired_heatmap_axes(nrows, ncols, bottom=HEATMAP_BOTTOM, gap=0.30, title=0.28,
                        cell=HEATMAP_CELL_SIZE):
    """Two same-sized heat maps side by side, sharing one colour bar.

    Row labels are drawn on the left panel only, so the two matrices can be
    compared cell by cell without the eye having to travel past a second set of
    mode names.

    `cell` overrides the article-wide cell size: a matrix with many rows needs
    smaller cells to stay within a page.
    """
    matrix_width = ncols * cell
    matrix_height = nrows * cell
    figure_height = bottom + matrix_height + HEATMAP_TOP + title
    group_width = (
        2 * matrix_width
        + gap
        + HEATMAP_COLORBAR_PAD
        + HEATMAP_COLORBAR_WIDTH
    )
    figure_width = group_width + 1.0  # room for the mode names on the left
    left = figure_width - group_width - 0.05

    fig = plt.figure(figsize=(figure_width, figure_height))

    def box(x):
        return [
            x / figure_width,
            bottom / figure_height,
            matrix_width / figure_width,
            matrix_height / figure_height,
        ]

    axes = [fig.add_axes(box(left)), fig.add_axes(box(left + matrix_width + gap))]
    cax = fig.add_axes(
        [
            (left + 2 * matrix_width + gap + HEATMAP_COLORBAR_PAD) / figure_width,
            bottom / figure_height,
            HEATMAP_COLORBAR_WIDTH / figure_width,
            matrix_height / figure_height,
        ]
    )
    return fig, axes, cax


def save_article_figure(fig, output_directory, stem):
    """Save a tightly cropped figure with the article-wide export settings."""
    options = {
        "bbox_inches": "tight",
        "pad_inches": FIGURE_PAD,
    }
    fig.savefig(
        output_directory / f"{stem}.png",
        dpi=FIGURE_DPI,
        **options,
    )
    fig.savefig(output_directory / f"{stem}.pdf", **options)
