#!/usr/bin/env python3
"""Generate a clean 16:9 (1920x1080) X/Twitter benchmark card from JSON or ladder CSVs.

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
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "x_benchmark_data.json"
DEFAULT_OUT = HERE / "bench-rtx4070-x.png"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.0,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "text.color": "#0F172A",
})

C_ON = "#2563EB"
C_OFF = "#10B981"
C_GRID = "#E2E8F0"
HALO = [pe.withStroke(linewidth=3.0, foreground="#FFFFFF")]


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


def draw_rounded_box(ax, x, y, w, h, fc, ec, lw=1.0, rad=0.012, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=ax.transAxes, zorder=zorder,
    )
    ax.add_patch(patch)


def style_plot_ax(ax, ylabel: str, ylim: float):
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.85, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10.0, length=0, pad=6)
    ax.set_xticks([8.192, 65.536, 131.072, 196.608, 261.5])
    ax.set_xticklabels(["8K", "64K", "128K", "192K", "256K"])
    ax.set_xlim(-8, 276)
    ax.set_ylim(0, ylim)
    ax.set_xlabel("Context Depth", fontsize=10.2, fontweight="bold", color="#475569", labelpad=6)
    ax.set_ylabel(ylabel, fontsize=10.2, fontweight="bold", color="#475569")


def ann(ax, x, y, txt, color, dx=0, dy=8, ha="center", va="bottom", fs=9.8):
    t = ax.annotate(
        txt, (x, y),
        textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
        fontsize=fs, fontweight="bold", color=color, zorder=10,
    )
    t.set_path_effects(HALO)


def render(cfg: dict, out_path: Path):
    depths = cfg["depths_k"]
    pp_on = cfg["prefill_on"]
    pp_off = cfg["prefill_off"]
    tg_on = cfg["decode_on"]
    tg_off = cfg["decode_off"]
    labels = cfg.get("labels", {})
    on_label = labels.get("on", "DSpark ON (Draft-3)")
    off_label = labels.get("off", "DSpark OFF (T=1)")

    fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor="#FFFFFF")

    # Full-figure background canvas for header badges & KPI cards
    bg = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
    bg.set_xticks([])
    bg.set_yticks([])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)

    # Subtle top accent bar
    draw_rounded_box(bg, 0.0, 0.990, 1.0, 0.010, fc=C_ON, ec="none", lw=0, rad=0.0, zorder=1)

    # Title & subtitle
    bg.text(0.04, 0.940, cfg["title"], fontsize=17.0, fontweight="bold", color="#0F172A", va="center")
    bg.text(0.04, 0.896, cfg["subtitle"], fontsize=10.2, color="#475569", va="center")

    # Spec badges row (4 pills across x=0.04..0.96)
    badges = cfg.get("badges", [])
    if badges:
        n_b = len(badges)
        gap = 0.014
        tot_w = 0.92
        bw = (tot_w - gap * (n_b - 1)) / n_b
        y_b, h_b = 0.816, 0.052
        for i, b in enumerate(badges):
            bx = 0.04 + i * (bw + gap)
            draw_rounded_box(bg, bx, y_b, bw, h_b, fc=b["bg"], ec=b["border"], lw=1.1, rad=0.010)
            cx = bx + bw * 0.5
            bg.text(cx, y_b + h_b * 0.67, b["label"].upper(), fontsize=7.6, fontweight="bold", color=b["color"], ha="center", va="center", alpha=0.85)
            bg.text(cx, y_b + h_b * 0.30, b["value"], fontsize=9.1, fontweight="bold", color="#0F172A", ha="center", va="center")

    # 3 Hero KPI cards (auto-computed from series arrays!)
    kpi_notes = labels.get("kpi_notes", {})
    kpis = [
        (
            "8K FRESH PREFILL",
            f"{pp_on[0]:,.0f} ON  ·  {pp_off[0]:,.0f} OFF  t/s",
            kpi_notes.get("pp_8k", "Smooth 8K–256K curve"),
            C_ON,
        ),
        (
            "256K CUMULATIVE PREFILL",
            f"{pp_on[-1]:,.0f} ON  ·  {pp_off[-1]:,.0f} OFF  t/s",
            kpi_notes.get("pp_256k", "No host-pinning cliffs"),
            C_OFF,
        ),
        (
            "REAL-CODE DECODE (DSPARK ON)",
            f"{tg_on[0]:.1f}  →  {tg_on[-1]:.1f}  t/s",
            kpi_notes.get("decode", "Beats DSpark OFF across 8K–256K"),
            "#1D4ED8",
        ),
    ]
    k_gap = 0.016
    kw = (0.92 - 2 * k_gap) / 3.0
    ky, kh = 0.696, 0.098
    for i, (k_title, k_val, k_sub, k_col) in enumerate(kpis):
        kx = 0.04 + i * (kw + k_gap)
        draw_rounded_box(bg, kx, ky, kw, kh, fc="#F8FAFC", ec="#E2E8F0", lw=1.0, rad=0.010)
        bg.text(kx + 0.014, ky + kh * 0.77, k_title, fontsize=8.0, fontweight="bold", color="#64748B", va="center")
        bg.text(kx + 0.014, ky + kh * 0.46, k_val, fontsize=13.0, fontweight="bold", color=k_col, va="center")
        bg.text(kx + 0.014, ky + kh * 0.17, k_sub, fontsize=8.2, color="#475569", va="center")

    # Side-by-side plot axes
    ax_pp = fig.add_axes([0.068, 0.132, 0.412, 0.485])
    ax_tg = fig.add_axes([0.553, 0.132, 0.412, 0.485])

    idx_map = {round(v, 3): i for i, v in enumerate(depths)}

    # --- Left: Prefill ---
    style_plot_ax(ax_pp, "tok / s", labels.get("prefill_ylim", 9000))
    ax_pp.plot(depths, pp_off, "--o", color=C_OFF, linewidth=2.2, markersize=4.8, alpha=0.90, zorder=3, label=off_label)
    ax_pp.plot(depths, pp_on, "-o", color=C_ON, linewidth=2.8, markersize=5.8, zorder=4, label=on_label)
    ax_pp.fill_between(depths, 0, pp_on, color=C_ON, alpha=0.07, zorder=1)

    ann(ax_pp, depths[0], max(pp_on[0], pp_off[0]), f"{pp_on[0]:,.0f}", C_ON, dx=8, dy=4, ha="left")
    if 65.536 in idx_map:
        i64 = idx_map[65.536]
        ann(ax_pp, depths[i64], pp_on[i64], f"{pp_on[i64]:,.0f}", C_ON, dx=-4, dy=-14, ha="right", va="top")
    if 131.072 in idx_map:
        i128 = idx_map[131.072]
        ann(ax_pp, depths[i128], max(pp_on[i128], pp_off[i128]), f"{pp_on[i128]:,.0f}", C_ON, dx=2, dy=11, ha="left", va="bottom")

    # Place higher 256K Prefill above and lower below so they never collide
    if pp_off[-1] >= pp_on[-1]:
        ann(ax_pp, depths[-1], pp_off[-1], f"{pp_off[-1]:,.0f} OFF", C_OFF, dx=-2, dy=11, ha="right", va="bottom")
        ann(ax_pp, depths[-1], pp_on[-1], f"{pp_on[-1]:,.0f} ON", C_ON, dx=-2, dy=-13, ha="right", va="top")
    else:
        ann(ax_pp, depths[-1], pp_on[-1], f"{pp_on[-1]:,.0f} ON", C_ON, dx=-2, dy=11, ha="right", va="bottom")
        ann(ax_pp, depths[-1], pp_off[-1], f"{pp_off[-1]:,.0f} OFF", C_OFF, dx=-2, dy=-13, ha="right", va="top")

    ax_pp.set_title("Prefill Throughput (Cumulative Effective)", fontsize=11.5, fontweight="bold", loc="left", pad=10, color="#1E293B")
    ax_pp.legend(frameon=False, fontsize=9.4, loc="upper right")

    # --- Right: Decode ---
    style_plot_ax(ax_tg, "tok / s", labels.get("decode_ylim", 72))
    ax_tg.plot(depths, tg_off, "--o", color=C_OFF, linewidth=2.2, markersize=4.8, alpha=0.90, zorder=3, label=off_label)
    ax_tg.plot(depths, tg_on, "-o", color=C_ON, linewidth=2.8, markersize=5.8, zorder=4, label=on_label)
    ax_tg.fill_between(depths, tg_off, tg_on, color=C_ON, alpha=0.09, zorder=1)

    ann(ax_tg, depths[0], tg_on[0], f"{tg_on[0]:.1f}", C_ON, dx=-2, dy=9, ha="right", va="bottom")
    ann(ax_tg, depths[0], tg_off[0], f"{tg_off[0]:.1f}", C_OFF, dx=6, dy=-13, ha="left", va="top")
    if 65.536 in idx_map:
        i64 = idx_map[65.536]
        ann(ax_tg, depths[i64], tg_on[i64], f"{tg_on[i64]:.1f}", C_ON, dx=4, dy=9, ha="left", va="bottom")
    if 131.072 in idx_map:
        i128 = idx_map[131.072]
        ann(ax_tg, depths[i128], tg_on[i128], f"{tg_on[i128]:.1f}", C_ON, dx=0, dy=10, ha="center", va="bottom")
        ann(ax_tg, depths[i128], tg_off[i128], f"{tg_off[i128]:.1f}", C_OFF, dx=0, dy=-13, ha="center", va="top")
    if 196.608 in idx_map:
        i192 = idx_map[196.608]
        ann(ax_tg, depths[i192], tg_on[i192], f"{tg_on[i192]:.1f}", C_ON, dx=0, dy=10, ha="center", va="bottom")

    if tg_on[-1] >= tg_off[-1]:
        ann(ax_tg, depths[-1], tg_on[-1], f"{tg_on[-1]:.1f}", C_ON, dx=-4, dy=10, ha="right", va="bottom")
        ann(ax_tg, depths[-1], tg_off[-1], f"{tg_off[-1]:.1f}", C_OFF, dx=-4, dy=-12, ha="right", va="top")
    else:
        ann(ax_tg, depths[-1], tg_off[-1], f"{tg_off[-1]:.1f}", C_OFF, dx=-4, dy=10, ha="right", va="bottom")
        ann(ax_tg, depths[-1], tg_on[-1], f"{tg_on[-1]:.1f}", C_ON, dx=-4, dy=-12, ha="right", va="top")

    ax_tg.set_title("Decode Throughput (Steady-State)", fontsize=11.5, fontweight="bold", loc="left", pad=10, color="#1E293B")
    ax_tg.legend(frameon=False, fontsize=9.4, loc="upper right")

    # Footer
    bg.text(0.04, 0.032, cfg.get("footer", ""), fontsize=8.4, color="#64748B", va="center")

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
