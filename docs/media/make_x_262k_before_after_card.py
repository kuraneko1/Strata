#!/usr/bin/env python3
"""Generate a clean 16:9 (1920x1080) AI-lab-style X/Twitter card for 262K Before vs After."""
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES_DIR = ROOT / "docs/bench-results/2026-11-26-rtx4070-262k-roofline-dspark"
CSV_4WL_BEFORE = RES_DIR / "rtx4070-262k-4wl-before.csv"
CSV_4WL_AFTER = RES_DIR / "rtx4070-262k-4wl-after.csv"
CSV_PRIOR3 = RES_DIR / "rtx4070-262k-dspark-on.csv"
CSV_AFTER6 = RES_DIR / "rtx4070-262k-ladder-dm6-after.csv"

OUT_PNG = Path(__file__).resolve().parent / "bench-rtx4070-262k-before-after-x.png"
OUT_ART = Path("/home/ryu/.gemini/antigravity-cli/brain/fe0c96cd-1f9a-4557-8da6-aa4a410e5b21/bench-rtx4070-262k-before-after-x.png")

XB_6B = [8.192, 16.384, 32.768, 49.152, 65.536, 98.304, 131.072, 163.840, 196.608, 229.376, 261.500]
DB_6B_ON = [49.08, 55.68, 46.78, 46.61, 45.51, 50.05, 37.64, 35.66, 31.45, 24.97, 25.09]

WORKLOADS = ["edit_rag", "code", "prose", "japanese"]
LABELS = ["Edit / RAG", "Code Gen", "English Prose", "Japanese"]

C_ON = "#2563EB"
C_ON_TXT = "#1D4ED8"
C_EM = "#10B981"
C_EM_TXT = "#059669"
C_OFF = "#94A3B8"
C_OFF_TXT = "#64748B"
C_INK = "#0F172A"
C_MUTED = "#64748B"
C_LIGHT = "#94A3B8"
C_GRID = "#EEF2F6"
C_RULE = "#E2E8F0"
C_AXIS = "#CBD5E1"
HALO = [pe.withStroke(linewidth=3.0, foreground="#FFFFFF")]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Noto Sans", "DejaVu Sans"],
    "axes.edgecolor": C_AXIS,
    "axes.linewidth": 1.0,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "text.color": C_INK,
})


def load_4wl(path: Path):
    res = {}
    rss = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["mode"] != "sampling":
                continue
            dm = int(r["draft_max"])
            res[(dm, r["workload"])] = float(r["tg_tps"])
            if "peak_rss_mib" in r and r["peak_rss_mib"]:
                rss[dm] = max(rss.get(dm, 0.0), float(r["peak_rss_mib"]))
    return res, rss


def load_ladder(path: Path):
    x, tg = [], []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["profile"] != "262k":
                continue
            d = int(r["depth"])
            if r["kind"] == "sweep" or (r["kind"] == "fresh" and d == 8192):
                x.append(round(d / 1000.0, 3))
                tg.append(round(float(r["steady_dec_tps"]), 2))
    return x, tg


def ann(ax, x, y, txt, color, dx=0, dy=8, ha="center", va="bottom", fs=10.0, weight="bold"):
    t = ax.annotate(
        txt, (x, y),
        textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
        fontsize=fs, fontweight=weight, color=color, zorder=10,
    )
    t.set_path_effects(HALO)


def draw_header_legend_item(bg, x_start: float, y: float, label: str, color: str, ls: str, lw: float, ms: float):
    x_end = x_start + 0.024
    x_mid = 0.5 * (x_start + x_end)
    bg.plot([x_start, x_end], [y, y], linestyle=ls, linewidth=lw, color=color, solid_capstyle="round", zorder=4)
    bg.plot([x_mid], [y], marker="o", markersize=ms, color=color, markeredgecolor="#FFFFFF", markeredgewidth=1.0, zorder=5)
    bg.text(
        x_end + 0.008, y, label,
        fontsize=9.8, fontweight="bold" if ls == "-" else "normal",
        color=C_INK if ls == "-" else C_OFF_TXT, ha="left", va="center",
    )


def style_clean_ax(ax, title: str, unit_note: str, ylim: float, xlabel: str):
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(C_AXIS)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.95, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10.0, length=0, pad=8)
    ax.set_ylim(0, ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.set_xlabel(xlabel, fontsize=10.2, color=C_MUTED, labelpad=10)
    ax.set_title(title, fontsize=14.0, fontweight="bold", loc="left", pad=14, color=C_INK)
    ax.text(1.0, 1.032, unit_note, transform=ax.transAxes, ha="right", va="bottom", fontsize=9.6, color=C_MUTED)


def main():
    b4, b_rss = load_4wl(CSV_4WL_BEFORE)
    a4, a_rss = load_4wl(CSV_4WL_AFTER)
    xp3, dp3 = load_ladder(CSV_PRIOR3)
    xa6, da6 = load_ladder(CSV_AFTER6)

    b_dm6 = [b4[(6, w)] for w in WORKLOADS]
    a_dm3 = [a4[(3, w)] for w in WORKLOADS]
    a_dm6 = [a4[(6, w)] for w in WORKLOADS]

    avg_b6 = sum(b_dm6) / 4.0
    avg_a3 = sum(a_dm3) / 4.0
    ram_b = b_rss.get(6, 26995.0) / 1024.0
    ram_a = a_rss.get(6, 12514.0) / 1024.0

    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor="#FFFFFF")
    bg = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    bg.set_xticks([])
    bg.set_yticks([])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)

    bg.text(0.055, 0.918, "Inkling-2-Mini 262K on RTX 4070 12 GB", fontsize=20.2, fontweight="bold", color=C_INK, va="center")
    bg.text(
        0.055, 0.866,
        f"Before vs After  ·  Draft-3 Avg {avg_b6:.1f} → {avg_a3:.1f} t/s (+{(avg_a3/avg_b6-1)*100:.1f}% vs Before Draft-6)  ·  Peak RAM {ram_b:.1f} → {ram_a:.1f} GiB ({(ram_a/ram_b-1)*100:.0f}%)",
        fontsize=10.0, color=C_MUTED, va="center",
    )

    draw_header_legend_item(bg, 0.745, 0.928, "After · Draft-6", C_ON, "-", 2.6, 6.0)
    draw_header_legend_item(bg, 0.745, 0.892, "After / Freq-Prior · Draft-3", C_EM, "-", 2.2, 5.2)
    draw_header_legend_item(bg, 0.745, 0.856, "Before", C_OFF, "--", 2.0, 4.8)

    bg.plot([0.055, 0.945], [0.816, 0.816], color=C_RULE, linewidth=0.9, zorder=1)

    ax1 = fig.add_axes([0.075, 0.145, 0.400, 0.570])
    ax2 = fig.add_axes([0.555, 0.145, 0.390, 0.570])

    # --- Left: 262K Practical Sampling ---
    style_clean_ax(ax1, "262K Practical Sampling", "tok / s · temp=1.0, 256 tok", 102, "Workload (2.8K–6.1K prompts)")
    x = np.arange(len(WORKLOADS))
    w = 0.255
    g0 = ax1.bar(x - w, b_dm6, width=w, color=C_OFF, alpha=0.88, zorder=3)
    g1 = ax1.bar(x,     a_dm3, width=w, color=C_EM,  alpha=0.90, zorder=3)
    g2 = ax1.bar(x + w, a_dm6, width=w, color=C_ON,  alpha=0.94, zorder=3)

    for grp, vals, col, wt in [(g0, b_dm6, C_OFF_TXT, "normal"), (g1, a_dm3, C_EM_TXT, "bold"), (g2, a_dm6, C_ON_TXT, "bold")]:
        for rect, val in zip(grp, vals):
            t = ax1.text(
                rect.get_x() + rect.get_width() / 2, val + 1.5,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=9.0, fontweight=wt, color=col, zorder=10,
            )
            t.set_path_effects(HALO)

    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS, fontsize=10.0)
    ax1.add_patch(Rectangle((0, 0), 0, 0))  # keep clean ticks

    # --- Right: 262K Context Depth Ladder ---
    style_clean_ax(ax2, "262K Context Depth", "tok / s · real code 8K–256K", 80, "Context Depth")
    ax2.set_xticks([8.192, 65.536, 131.072, 196.608, 261.5])
    ax2.set_xticklabels(["8K", "64K", "128K", "192K", "256K"])
    ax2.set_xlim(-10, 276)

    ax2.plot(XB_6B, DB_6B_ON, "--o", color=C_OFF, linewidth=2.0, markersize=4.5, markeredgecolor="#FFFFFF", markeredgewidth=1.0, zorder=3)
    ax2.plot(xp3, dp3, "-o", color=C_EM, linewidth=2.1, markersize=4.8, markeredgecolor="#FFFFFF", markeredgewidth=1.1, zorder=3)
    ax2.plot(xa6, da6, "-o", color=C_ON, linewidth=2.7, markersize=6.0, markeredgecolor="#FFFFFF", markeredgewidth=1.3, zorder=4)
    ax2.fill_between(xa6, DB_6B_ON, da6, color=C_ON, alpha=0.07, zorder=1)

    idx3 = {round(v, 3): i for i, v in enumerate(xp3)}
    idx6 = {round(v, 3): i for i, v in enumerate(xa6)}
    ann(ax2, xa6[0], da6[0], f"{da6[0]:.1f}", C_ON_TXT, dx=-8, dy=9, ha="right", va="bottom", fs=10.0)
    ann(ax2, XB_6B[0], DB_6B_ON[0], f"{DB_6B_ON[0]:.1f}", C_OFF_TXT, dx=1, dy=-13, ha="center", va="top", fs=9.5, weight="normal")
    ann(ax2, xa6[idx6[32.768]], da6[idx6[32.768]], f"{da6[idx6[32.768]]:.1f}", C_ON_TXT, dx=4, dy=10, ha="left", va="bottom", fs=10.2)
    ann(ax2, xp3[idx3[65.536]], dp3[idx3[65.536]], f"{dp3[idx3[65.536]]:.1f}", C_EM_TXT, dx=0, dy=10, ha="center", va="bottom", fs=9.8)
    ann(ax2, xa6[idx6[163.840]], da6[idx6[163.840]], f"{da6[idx6[163.840]]:.1f}", C_ON_TXT, dx=0, dy=-14, ha="center", va="top", fs=9.8)
    ann(ax2, xp3[-1], dp3[-1], f"{dp3[-1]:.1f}", C_EM_TXT, dx=-2, dy=11, ha="right", va="bottom", fs=10.2)
    ann(ax2, xa6[-1], da6[-1], f"{da6[-1]:.1f}", C_ON_TXT, dx=-18, dy=-12, ha="right", va="top", fs=9.8)
    ann(ax2, XB_6B[-1], DB_6B_ON[-1], f"{DB_6B_ON[-1]:.1f}", C_OFF_TXT, dx=-2, dy=-12, ha="right", va="top", fs=9.6, weight="normal")

    bg.text(0.055, 0.042, "Strata Engine  ·  UD-Q4_K_M + Q8_0 KV  ·  Zero Precision Loss", fontsize=8.8, color=C_LIGHT, ha="left", va="center")
    bg.text(0.945, 0.042, "RTX 4070 12 GB + Ryzen 7 5700X", fontsize=8.8, color=C_LIGHT, ha="right", va="center")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, facecolor="#FFFFFF")
    plt.close(fig)
    shutil.copyfile(OUT_PNG, OUT_ART)
    print(f"Saved: {OUT_PNG} ({int(12.8 * 150)}x{int(7.2 * 150)})")


if __name__ == "__main__":
    main()
