"""Render a results table as a PNG image using matplotlib."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_table_image(headers, rows, title, path, fontsize=8):
    """Render `rows` (list of lists of str) under `headers` (list of str) to a PNG.

    The figure is auto-sized to the number of rows/columns so nothing is clipped.
    """
    n_rows = len(rows)
    n_cols = len(headers)

    # Column widths are estimated from the longest cell in each column (chars).
    col_chars = []
    for j, h in enumerate(headers):
        longest = len(str(h))
        for r in rows:
            if j < len(r):
                longest = max(longest, len(str(r[j])))
        col_chars.append(max(longest, 6))

    # Empirical per-char widths; numeric columns are narrow, text columns wider.
    col_widths = []
    for j, c in enumerate(col_chars):
        col_widths.append(c * 0.11 + 0.3)

    fig_width = max(sum(col_widths) + 1.0, 6.0)
    fig_height = max((n_rows + 1) * 0.35 + 1.2, 3.0)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.5)

    # Shade the header row.
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#dddddd")
        cell.set_text_props(fontweight="bold")

    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=12)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved table: {path}")
