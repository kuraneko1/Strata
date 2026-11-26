#!/usr/bin/env python3
"""Generate a clean 16:9 (1920x1080) AI-lab-style X/Twitter benchmark card from JSON or ladder CSVs.

Usage:
  1) Render from JSON (default docs/media/x_benchmark_data.json -> docs/media/bench-rtx4070-x.png):
       python3 docs/media/make_x_benchmark_card.py
  2) Update directly from new ladder CSVs and save the updated numbers back to JSON:
       python3 docs/media/make_x_benchmark_card.py \
         --on-csv path/to/dspark_ladder.csv \
         --off-csv path/to/nodraft_ladder.csv \
         --save-data
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "x_benchmark_data.json"
DEFAULT_OUT = HERE / "bench-rtx4070-x.png"

C_ON = "#2563EB"
C_OFF = "#94A3B8"
C_OFF_TXT = "#475569"
C_INK = "#0F172A"
C_MUTED = "#64748B"
C_LIGHT = "#94A3B8"
C_GRID = "#EEF2F6"
C_RULE = "#E2E8F0"
C_AXIS = "#CBD5E1"
HALO = [pe.withStroke(linewidth=3.2, foreground="#FFFFFF")]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Noto Sans", "DejaVu Sans"],
    "axes.edgecolor": C_AXIS,
    "axes.linewidth": 1.0,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "text.color": C_INK,
})


def load_csv_ladder(path: Path, profile: str = "262k"):
    depths_k, pp, tg = [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["profile"] != profile:
                continue
            d = int(row["depth"])
            if row["kind"] == "sweep" or (row["kind"] == "fresh" and d == 8192):
                depths_k.append(round(d / 1000.0, 3))
                pp.append(round(float(row["cum_eff_pp_tps"]), 1))
                tg.append(round(float(row["steady_dec_tps"]), 2))
    if not depths_k:
        raise ValueError(f"No matching rows for profile={profile!r} in {path}")
    return depths_k, pp, tg


def style_plot_ax(ax, title: str, unit_note: str, ylim: float, comma_y: bool = False):
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(C_AXIS)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.95, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10.2, length=0, pad=8)
    ax.set_xticks([8.192, 65.536, 131.072, 196.608, 261.5])
    ax.set_xticklabels(["8K", "64K", "128K", "192K", "256K"])
    ax.set_xlim(-10, 276)
    ax.set_ylim(0, ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    if comma_y:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Context Depth", fontsize=10.5, color=C_MUTED, labelpad=10)
    ax.set_title(title, fontsize=14.5, fontweight="bold", loc="left", pad=14, color=C_INK)
    ax.text(
        1.0, 1.032, unit_note,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9.8, color=C_MUTED,
    )


def ann(ax, x, y, txt, color, dx=0, dy=8, ha="center", va="bottom", fs=10.2, weight="bold"):
    t = ax.annotate(
        txt, (x, y),
        textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
        fontsize=fs, fontweight=weight, color=color, zorder=10,
    )
    t.set_path_effects(HALO)


def draw_header_legend_item(bg, x_line_start: float, y: float, label: str, color: str, ls: str, lw: float, ms: float):
    x_line_end = x_line_start + 0.026
    x_mid = 0.5 * (x_line_start + x_line_end)
    bg.plot(
        [x_line_start, x_line_end], [y, y],
        linestyle=ls, linewidth=lw, color=color,
        solid_capstyle="round", zorder=4,
    )
    bg.plot(
        [x_mid], [y],
        marker="o", markersize=ms, color=color,
        markeredgecolor="#FFFFFF", markeredgewidth=1.1, zorder=5,
    )
    bg.text(
        x_line_end + 0.009, y, label,
        fontsize=10.2, fontweight="bold" if ls == "-" else "normal",
        color=C_INK if ls == "-" else C_OFF_TXT,
        ha="left", va="center",
    )


def render(cfg: dict, out_path: Path):
    depths = cfg["depths_k"]
    pp_on = cfg["prefill_on"]
    pp_off = cfg["prefill_off"]
    tg_on = cfg["decode_on"]
    tg_off = cfg["decode_off"]
    labels = cfg.get("labels", {})
    on_label = labels.get("on", "DSpark ON (Draft-3)")
    off_label = labels.get("off", "DSpark OFF")

    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor="#FFFFFF")

    # Full-figure background canvas for header, hairline divider, shared legend & footer
    bg = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    bg.set_xticks([])
    bg.set_yticks([])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)

    # Editorial Header (left: title + compact spec line; right: single shared legend)
    bg.text(0.055, 0.918, cfg["title"], fontsize=20.5, fontweight="bold", color=C_INK, va="center")
    bg.text(0.055, 0.866, cfg["subtitle"], fontsize=10.6, color=C_MUTED, va="center")

    draw_header_legend_item(bg, 0.775, 0.916, on_label, C_ON, "-", 2.6, 6.0)
    draw_header_legend_item(bg, 0.775, 0.870, off_label, C_OFF, "--", 2.0, 4.8)

    # Minimal hairline divider
    bg.plot([0.055, 0.945], [0.820, 0.820], color=C_RULE, linewidth=0.9, zorder=1)

    # Side-by-side spacious plot axes
    ax_pp = fig.add_axes([0.082, 0.145, 0.388, 0.575])
    ax_tg = fig.add_axes([0.558, 0.145, 0.388, 0.575])

    idx_map = {round(v, 3): i for i, v in enumerate(depths)}

    # --- Left: Prefill ---
    style_plot_ax(ax_pp, "Prefill", "tok / s · cumulative", labels.get("prefill_ylim", 9000), comma_y=True)
    ax_pp.plot(
        depths, pp_off, "--o", color=C_OFF, linewidth=2.0, markersize=4.6,
        markeredgecolor="#FFFFFF", markeredgewidth=1.0, zorder=3,
    )
    ax_pp.plot(
        depths, pp_on, "-o", color=C_ON, linewidth=2.6, markersize=6.0,
        markeredgecolor="#FFFFFF", markeredgewidth=1.3, zorder=4,
    )
    ax_pp.fill_between(depths, 0, pp_on, color=C_ON, alpha=0.045, zorder=1)

    if pp_off[0] - pp_on[0] >= 120:
        ann(ax_pp, depths[0], pp_off[0], f"{pp_off[0]:,.0f}", C_OFF_TXT, dx=9, dy=4, ha="left", va="bottom", fs=10.2, weight="normal")
        ann(ax_pp, depths[0], pp_on[0], f"{pp_on[0]:,.0f}", C_ON, dx=9, dy=-12, ha="left", va="top", fs=10.8)
    elif pp_on[0] - pp_off[0] >= 120:
        ann(ax_pp, depths[0], pp_on[0], f"{pp_on[0]:,.0f}", C_ON, dx=9, dy=4, ha="left", va="bottom", fs=10.8)
        ann(ax_pp, depths[0], pp_off[0], f"{pp_off[0]:,.0f}", C_OFF_TXT, dx=9, dy=-12, ha="left", va="top", fs=10.2, weight="normal")
    else:
        ann(ax_pp, depths[0], pp_off[0], f"{pp_off[0]:,.0f}", C_OFF_TXT, dx=9, dy=5, ha="left", va="bottom", fs=10.0, weight="normal")
        ann(ax_pp, depths[0], pp_on[0], f"{pp_on[0]:,.0f}", C_ON, dx=10, dy=-8, ha="left", va="top", fs=10.6)
    if 65.536 in idx_map:
        i64 = idx_map[65.536]
        ann(ax_pp, depths[i64], pp_on[i64], f"{pp_on[i64]:,.0f}", C_ON, dx=-6, dy=-14, ha="right", va="top", fs=10.2)
    if 131.072 in idx_map:
        i128 = idx_map[131.072]
        if pp_on[i128] >= pp_off[i128]:
            ann(ax_pp, depths[i128], pp_on[i128], f"{pp_on[i128]:,.0f}", C_ON, dx=4, dy=11, ha="left", va="bottom", fs=10.2)
        else:
            ann(ax_pp, depths[i128], pp_on[i128], f"{pp_on[i128]:,.0f}", C_ON, dx=-4, dy=-13, ha="right", va="top", fs=10.2)

    if pp_off[-1] >= pp_on[-1]:
        ann(ax_pp, depths[-1], pp_off[-1], f"{pp_off[-1]:,.0f}", C_OFF_TXT, dx=-2, dy=11, ha="right", va="bottom", fs=9.8, weight="normal")
        ann(ax_pp, depths[-1], pp_on[-1], f"{pp_on[-1]:,.0f}", C_ON, dx=-2, dy=-14, ha="right", va="top", fs=10.8)
    else:
        ann(ax_pp, depths[-1], pp_on[-1], f"{pp_on[-1]:,.0f}", C_ON, dx=-2, dy=11, ha="right", va="bottom", fs=10.8)
        ann(ax_pp, depths[-1], pp_off[-1], f"{pp_off[-1]:,.0f}", C_OFF_TXT, dx=-2, dy=-14, ha="right", va="top", fs=9.8, weight="normal")

    # --- Right: Decode ---
    style_plot_ax(ax_tg, "Decode", "tok / s · steady-state", labels.get("decode_ylim", 80), comma_y=False)
    ax_tg.plot(
        depths, tg_off, "--o", color=C_OFF, linewidth=2.0, markersize=4.6,
        markeredgecolor="#FFFFFF", markeredgewidth=1.0, zorder=3,
    )
    ax_tg.plot(
        depths, tg_on, "-o", color=C_ON, linewidth=2.6, markersize=6.0,
        markeredgecolor="#FFFFFF", markeredgewidth=1.3, zorder=4,
    )
    ax_tg.fill_between(depths, tg_off, tg_on, color=C_ON, alpha=0.075, zorder=1)

    peak_i = max(range(min(4, len(tg_on))), key=lambda i: tg_on[i])
    ann(ax_tg, depths[peak_i], tg_on[peak_i], f"{tg_on[peak_i]:.1f}", C_ON, dx=2, dy=10, ha="center", va="bottom", fs=10.8)
    ann(ax_tg, depths[0], tg_off[0], f"{tg_off[0]:.1f}", C_OFF_TXT, dx=-1, dy=-14, ha="center", va="top", fs=9.8, weight="normal")
    if 65.536 in idx_map:
        i64 = idx_map[65.536]
        ann(ax_tg, depths[i64], tg_on[i64], f"{tg_on[i64]:.1f}", C_ON, dx=0, dy=11, ha="center", va="bottom", fs=10.2)
    if 131.072 in idx_map:
        i128 = idx_map[131.072]
        ann(ax_tg, depths[i128], tg_on[i128], f"{tg_on[i128]:.1f}", C_ON, dx=-2, dy=-15, ha="center", va="top", fs=10.0)
    if 163.84 in idx_map:
        i160 = idx_map[163.84]
        ann(ax_tg, depths[i160], tg_on[i160], f"{tg_on[i160]:.1f}", C_ON, dx=2, dy=11, ha="left", va="bottom", fs=10.2)

    if tg_on[-1] >= tg_off[-1]:
        ann(ax_tg, depths[-1], tg_on[-1], f"{tg_on[-1]:.1f}", C_ON, dx=-2, dy=11, ha="right", va="bottom", fs=10.8)
        ann(ax_tg, depths[-1], tg_off[-1], f"{tg_off[-1]:.1f}", C_OFF_TXT, dx=-2, dy=-14, ha="right", va="top", fs=9.8, weight="normal")
    else:
        ann(ax_tg, depths[-1], tg_off[-1], f"{tg_off[-1]:.1f}", C_OFF_TXT, dx=-2, dy=11, ha="right", va="bottom", fs=9.8, weight="normal")
        ann(ax_tg, depths[-1], tg_on[-1], f"{tg_on[-1]:.1f}", C_ON, dx=-2, dy=-14, ha="right", va="top", fs=10.8)

    # Quiet Footer
    footer_left = cfg.get("footer_left", cfg.get("footer", ""))
    footer_right = cfg.get("footer_right", "")
    if footer_left:
        bg.text(0.055, 0.042, footer_left, fontsize=8.8, color=C_LIGHT, ha="left", va="center")
    if footer_right:
        bg.text(0.945, 0.042, footer_right, fontsize=8.8, color=C_LIGHT, ha="right", va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Saved: {out_path} ({int(12.8 * 150)}x{int(7.2 * 150)})")


def main():
    ap = argparse.ArgumentParser(description="Generate X-ready Strata benchmark card PNG.")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to benchmark JSON config")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PNG path")
    ap.add_argument("--on-csv", type=Path, help="Optional DSpark ON ladder CSV to overwrite ON curves")
    ap.add_argument("--off-csv", type=Path, help="Optional DSpark OFF ladder CSV to overwrite OFF curves")
    ap.add_argument("--profile", default="262k", help="Profile column to extract when --on-csv/--off-csv is set")
    ap.add_argument("--save-data", action="store_true", help="Write CSV updates back into --data JSON")
    args = ap.parse_args()

    with args.data.open(encoding="utf-8") as f:
        cfg = json.load(f)

    if args.on_csv:
        dk, pp, tg = load_csv_ladder(args.on_csv, args.profile)
        cfg["depths_k"] = dk
        cfg["prefill_on"] = pp
        cfg["decode_on"] = tg
    if args.off_csv:
        dk, pp, tg = load_csv_ladder(args.off_csv, args.profile)
        cfg["depths_k"] = dk
        cfg["prefill_off"] = pp
        cfg["decode_off"] = tg

    if args.save_data and (args.on_csv or args.off_csv):
        with args.data.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Updated JSON: {args.data}")

    render(cfg, args.out)


if __name__ == "__main__":
    main()
