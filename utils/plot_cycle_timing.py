#!/usr/bin/env python3
"""
Plot one observing cycle (load, OFF, OTF, OFF, OTF, ...) from the output of
compare_data_timing.py: when each integration / OTF dump took data, the gaps
in between, and the expected az/el, optionally with the ACU housekeeping and
its differences from the expected position.

A cycle starts with a load (one or more consecutive CAL rows) and ends before
the next load. Files of several pixels / backends with the same time window
are shown once (they share the expected position: pixel offsets are not
applied by compare_data_timing.py).

Panels (time in seconds since the start of the cycle):
  1. data windows per kind (LOAD / OFF / ON / OTF dumps) with the gaps [s]
  2. azimuth: expected at the window centres, housekeeping (--hk) as a line
  3. elevation: likewise
  4. (--samples) housekeeping - expected, d_az cos(el) and d_el [arcsec]
  5. (--samples) OTF along-track lag [s], + = telescope ahead of the data model

Example
-------
python compare_data_timing.py fits/ --log translator.log --log backend.log \\
    --hk acu_hk.csv --out expected.csv --samples-out samples.csv
python plot_cycle_timing.py expected.csv --list
python plot_cycle_timing.py expected.csv --cycle 2 --samples samples.csv --hk acu_hk.csv
"""
import argparse
import csv
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from compare_data_timing import load_housekeeping

# ---------------------------------------------------------------------------
# Colours (reference data-viz palette, light mode): categorical slots in
# fixed order, one per kind of data; housekeeping and chrome in ink tones
# ---------------------------------------------------------------------------
KIND_COLORS = {
    "OTF": "#2a78d6",     # slot 1 blue
    "OFF": "#eb6834",     # slot 2 orange
    "LOAD": "#1baf7a",    # slot 3 aqua
    "ON": "#eda100",      # slot 4 yellow
}
# order of the rows in the timeline panel, top to bottom
KIND_ORDER = ("LOAD", "OFF", "ON", "OTF")
D_EL_COLOR = "#6250d6"    # slot 7 violet (d_az uses the secondary ink)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
FIG_WIDTH = 12.0          # [inch]
TIMELINE_HEIGHT = 1.8     # [inch]
PANEL_HEIGHT = 2.4        # [inch]
MARKER_SIZE = 4.0         # [pt] expected position at the window centres
DPI = 150
# [sec] gaps between data windows shorter than this are not labelled
MIN_LABELLED_GAP = 0.5
# [sec] housekeeping shown this far before / after the cycle
HK_MARGIN = 5.0
# windows sharing kind, dump and t_begin to this many decimals are one window
DEDUP_DECIMALS = 3

STRING_COLUMNS = {"file", "kind", "sobsmode", "start_source", "refraction"}


@dataclass
class Block:
    """Consecutive windows of one data file: one integration or one OTF line."""
    kind: str
    file: str
    scan: int
    t_begin: float      # [unix sec]
    t_end: float
    n_windows: int


# ===========================================================================
# Input
# ===========================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("windows", help="--out CSV of compare_data_timing.py")
    parser.add_argument("--samples", help="--samples-out CSV of compare_data_timing.py")
    parser.add_argument("--hk", help="housekeeping text file: time [unix sec], az, el [deg]")
    parser.add_argument("--hk-cols", default="0,1,2",
                        help="0-based columns of time, az, el (default 0,1,2)")
    parser.add_argument("--hk-delimiter", default=None,
                        help="column delimiter (default: ',' for *.csv, else whitespace)")
    parser.add_argument("--hk-time-offset", type=float, default=0.0,
                        help="[sec] added to the housekeeping times")
    parser.add_argument("--file-filter", default=None,
                        help="use only rows whose file name contains this, e.g. PX00")
    parser.add_argument("--cycle", type=int, default=0,
                        help="cycle to plot, 0-based; negative counts from the end (default 0)")
    parser.add_argument("--list", action="store_true", help="list the cycles and exit")
    parser.add_argument("--out", default=None, help="PNG file (default cycle_<N>.png)")
    return parser.parse_args()


def _convert(key, value):
    if key in STRING_COLUMNS:
        return value
    if value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return value


def read_rows(path, file_filter=None):
    with open(path, newline="") as handle:
        rows = [{key: _convert(key, value) for key, value in row.items()}
                for row in csv.DictReader(handle)]
    if file_filter:
        rows = [row for row in rows if file_filter in row["file"]]
    return rows


def row_kind(row):
    """LOAD, OTF, or the sub-mode (OFF / ON) of a total power row."""
    if row["kind"] == "TP":
        return row["sobsmode"] if row["sobsmode"] in KIND_COLORS else "ON"
    return row["kind"]


def deduplicate(rows):
    """One row per (kind, dump, t_begin): other pixels / backends dropped."""
    seen, unique = set(), []
    for row in sorted(rows, key=lambda r: (r["t_begin"], r["dump"])):
        key = (row_kind(row), int(row["dump"]), round(row["t_begin"], DEDUP_DECIMALS))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


# ===========================================================================
# Cycles and blocks
# ===========================================================================

def split_cycles(rows):
    """
    Cycles: each starts at a load row that follows a non-load row (or the
    first row). Rows before the first load are dropped; without any load all
    rows form one cycle.
    """
    if not any(row_kind(row) == "LOAD" for row in rows):
        return [rows] if rows else []
    cycles, previous = [], None
    for row in rows:
        is_load = row_kind(row) == "LOAD"
        if is_load and (previous is None or row_kind(previous) != "LOAD"):
            cycles.append([])
        if cycles:
            cycles[-1].append(row)
        previous = row
    return cycles


def make_blocks(rows):
    """Consecutive rows of the same file and kind as one block."""
    blocks = []
    for row in rows:
        kind = row_kind(row)
        last = blocks[-1] if blocks else None
        if last is not None and last.file == row["file"] and last.kind == kind:
            last.t_end = max(last.t_end, row["t_end"])
            last.n_windows += 1
        else:
            blocks.append(Block(kind, row["file"], int(row["scan"]),
                                row["t_begin"], row["t_end"], 1))
    return blocks


def utc_text(t):
    return Time(t, format="unix", scale="utc").isot[:23] + "Z"


def print_cycles(cycles):
    for i, cycle in enumerate(cycles):
        blocks = make_blocks(cycle)
        counts = {kind: sum(b.kind == kind for b in blocks) for kind in KIND_ORDER}
        scans = sorted({b.scan for b in blocks})
        print(f"cycle {i:3d}: {utc_text(cycle[0]['t_begin'])}, "
              f"{cycle[-1]['t_end'] - cycle[0]['t_begin']:7.1f} s, scans {scans[0]}-{scans[-1]}, "
              + ", ".join(f"{counts[k]} {k}" for k in KIND_ORDER if counts[k]))


def print_blocks(blocks, t_ref):
    print(f"{'kind':5s} {'file':32s} {'start':>9s} {'end':>9s} {'length':>8s} {'gap before':>10s}")
    previous_end = None
    for block in blocks:
        gap = "" if previous_end is None else f"{block.t_begin - previous_end:10.3f}"
        print(f"{block.kind:5s} {block.file:32s} {block.t_begin - t_ref:9.3f} "
              f"{block.t_end - t_ref:9.3f} {block.t_end - block.t_begin:8.3f} {gap:>10s}")
        previous_end = block.t_end


# ===========================================================================
# Plot
# ===========================================================================

def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=8)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.xaxis.label.set_color(INK_SECONDARY)


def unwrap_az(az, reference):
    """Azimuth continuous around reference [deg], for plotting across 0/360."""
    return reference + (np.asarray(az) - reference + 180.0) % 360.0 - 180.0


def plot_timeline(ax, rows, blocks, t_ref):
    kinds = [k for k in KIND_ORDER if any(row_kind(r) == k for r in rows)]
    y_of = {kind: len(kinds) - 1 - i for i, kind in enumerate(kinds)}
    for row in rows:
        kind = row_kind(row)
        ax.barh(y_of[kind], row["t_end"] - row["t_begin"], left=row["t_begin"] - t_ref,
                height=0.6, color=KIND_COLORS[kind], edgecolor=SURFACE, linewidth=0.5)
    for previous, block in zip(blocks, blocks[1:]):
        gap = block.t_begin - previous.t_end
        if gap >= MIN_LABELLED_GAP:
            x = (previous.t_end + block.t_begin) / 2 - t_ref
            y = (y_of[previous.kind] + y_of[block.kind]) / 2
            ax.annotate(f"{gap:.1f}", (x, y), ha="center", va="center", fontsize=7,
                        color=INK_SECONDARY,
                        bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none"})
    ax.set_yticks([y_of[k] for k in kinds])
    ax.set_yticklabels(kinds)
    ax.set_ylim(-0.6, len(kinds) - 0.4)
    ax.set_ylabel("data (gaps [s])")
    style_axis(ax)
    ax.grid(False, axis="y")


def plot_position(ax, rows, column, hk_t, hk_values, t_ref, label, az_reference=None):
    for kind in KIND_ORDER:
        selected = [r for r in rows if row_kind(r) == kind and np.isfinite(r[column])]
        if not selected:
            continue
        t = np.array([r["t_mid"] for r in selected]) - t_ref
        values = np.array([r[column] for r in selected])
        if az_reference is not None:
            values = unwrap_az(values, az_reference)
        ax.plot(t, values, "o", markersize=MARKER_SIZE, color=KIND_COLORS[kind],
                markeredgecolor=SURFACE, markeredgewidth=0.5, label=f"expected {kind}",
                zorder=3)
    if hk_t is not None and len(hk_t):
        values = hk_values if az_reference is None else unwrap_az(hk_values, az_reference)
        ax.plot(hk_t - t_ref, values, "-", color=INK_SECONDARY, linewidth=1.0,
                label="housekeeping", zorder=2)
    ax.set_ylabel(label)
    style_axis(ax)
    ax.legend(loc="best", fontsize=7, frameon=False, labelcolor=INK_SECONDARY)


def plot_differences(ax, samples, t_ref):
    t = np.array([s["t"] for s in samples]) - t_ref
    ax.plot(t, [s["d_az"] for s in samples], ".", markersize=3, color=INK_SECONDARY,
            label="d_az cos(el)")
    ax.plot(t, [s["d_el"] for s in samples], ".", markersize=3, color=D_EL_COLOR,
            label="d_el")
    ax.axhline(0.0, color=BASELINE, linewidth=0.8)
    ax.set_ylabel("hk - expected [arcsec]")
    style_axis(ax)
    ax.legend(loc="best", fontsize=7, frameon=False, labelcolor=INK_SECONDARY)


def plot_lag(ax, samples, t_ref):
    otf = [s for s in samples if s["kind"] == "OTF" and np.isfinite(s["lag"])]
    if otf:
        ax.plot(np.array([s["t"] for s in otf]) - t_ref, [s["lag"] for s in otf], ".",
                markersize=3, color=KIND_COLORS["OTF"])
    ax.axhline(0.0, color=BASELINE, linewidth=0.8)
    ax.set_ylabel("OTF lag [s]")
    style_axis(ax)


def plot_cycle(cycle, index, samples, hk, out):
    blocks = make_blocks(cycle)
    t_ref = cycle[0]["t_begin"]
    t_last = cycle[-1]["t_end"]
    hk_t = hk_az = hk_el = None
    if hk is not None:
        mask = (hk[0] >= t_ref - HK_MARGIN) & (hk[0] <= t_last + HK_MARGIN)
        hk_t, hk_az, hk_el = hk[0][mask], hk[1][mask], hk[2][mask]
    cycle_samples = []
    if samples is not None:
        cycle_samples = [s for s in samples if t_ref <= s["t"] <= t_last]

    n_panels = 3 + (2 if cycle_samples else 0)
    heights = [TIMELINE_HEIGHT] + [PANEL_HEIGHT] * (n_panels - 1)
    fig, axes = plt.subplots(n_panels, 1, sharex=True, figsize=(FIG_WIDTH, sum(heights)),
                             gridspec_kw={"height_ratios": heights})
    fig.patch.set_facecolor(SURFACE)

    plot_timeline(axes[0], cycle, blocks, t_ref)
    az_values = [r["exp_az"] for r in cycle if np.isfinite(r["exp_az"])]
    az_reference = float(np.median(az_values)) if az_values else 180.0
    plot_position(axes[1], cycle, "exp_az", hk_t, hk_az, t_ref, "azimuth [deg]", az_reference)
    plot_position(axes[2], cycle, "exp_el", hk_t, hk_el, t_ref, "elevation [deg]")
    if cycle_samples:
        plot_differences(axes[3], cycle_samples, t_ref)
        plot_lag(axes[4], cycle_samples, t_ref)
    axes[-1].set_xlabel(f"time since {utc_text(t_ref)} [s]")

    scans = sorted({b.scan for b in blocks})
    fig.suptitle(f"Cycle {index}: scans {scans[0]}-{scans[-1]}, "
                 f"{t_last - t_ref:.1f} s", color=INK_PRIMARY, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print_blocks(blocks, t_ref)
    print(f"written {out}")


def main():
    args = parse_arguments()
    rows = deduplicate(read_rows(args.windows, args.file_filter))
    cycles = split_cycles(rows)
    if not cycles:
        sys.exit("no rows")
    if args.list:
        print_cycles(cycles)
        return
    try:
        cycle = cycles[args.cycle]
    except IndexError:
        sys.exit(f"cycle {args.cycle} out of range: {len(cycles)} cycles (see --list)")
    index = args.cycle % len(cycles)

    samples = None
    if args.samples:
        samples = read_rows(args.samples, args.file_filter)
        samples = list({(s["kind"], round(s["t"], DEDUP_DECIMALS)): s for s in samples}.values())
        samples.sort(key=lambda s: s["t"])
    hk = None
    if args.hk:
        cols = tuple(int(c) for c in args.hk_cols.split(","))
        hk = load_housekeeping(args.hk, cols, args.hk_delimiter, args.hk_time_offset)

    plot_cycle(cycle, index, samples, hk, args.out or f"cycle_{index}.png")


if __name__ == "__main__":
    main()
