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
                        cell=HEATMAP_CELL_SIZE, extra_col_width=0.0, extra_gap=None):
    """Two same-sized heat maps side by side, sharing one colour bar.

    Row labels are drawn on the left panel only, so the two matrices can be
    compared cell by cell without the eye having to travel past a second set of
    mode names.

    `cell` overrides the article-wide cell size: a matrix with many rows needs
    smaller cells to stay within a page.

    `extra_col_width`, if positive, adds a third, narrow axes of that width
    (inches) and the same height as the two panels, placed between them and
    the colour bar and sharing its scale.  Meant for a single derived column
    that does not deserve a full panel of its own -- a 15-mode system would
    otherwise nearly double the figure's width for one value per row.
    `extra_gap` sets the margin between the second panel and this column
    separately from `gap`, since a panel title wider than its own matrix
    (e.g. "uniform on the same supports") overflows to the right and needs
    more room to clear; defaults to `gap` if unset.
    """
    if extra_gap is None:
        extra_gap = gap
    matrix_width = ncols * cell
    matrix_height = nrows * cell
    figure_height = bottom + matrix_height + HEATMAP_TOP + title
    group_width = (
        2 * matrix_width
        + gap
        + (extra_gap + extra_col_width if extra_col_width else 0)
        + HEATMAP_COLORBAR_PAD
        + HEATMAP_COLORBAR_WIDTH
    )
    figure_width = group_width + 1.0  # room for the mode names on the left
    left = figure_width - group_width - 0.05

    fig = plt.figure(figsize=(figure_width, figure_height))

    def box(x, width=matrix_width):
        return [
            x / figure_width,
            bottom / figure_height,
            width / figure_width,
            matrix_height / figure_height,
        ]

    x = left
    axes = [fig.add_axes(box(x))]
    x += matrix_width + gap
    axes.append(fig.add_axes(box(x)))
    x += matrix_width
    if extra_col_width:
        x += extra_gap
        axes.append(fig.add_axes(box(x, extra_col_width)))
        x += extra_col_width
    cax = fig.add_axes(
        [
            (x + HEATMAP_COLORBAR_PAD) / figure_width,
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
