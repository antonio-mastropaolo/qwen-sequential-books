#!/usr/bin/env python3
"""Build the README figures from results/results.csv.

usage: python assets/make_figures.py [--csv results/results.csv] [--out assets] [--fonts DIR]

Writes hero, method, accuracy_bars, forgetting, tradeoff and per_seed as PNG and SVG.
Needs pandas and matplotlib. Text is set in Inter when static Inter TTFs are found in
--fonts (or $FIGURE_FONTS); otherwise matplotlib falls back to DejaVu Sans.
"""
import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=os.path.join(HERE, "..", "results", "results.csv"))
ap.add_argument("--out", default=HERE)
ap.add_argument("--fonts", default=os.environ.get("FIGURE_FONTS", ""))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

fonts = sorted(glob.glob(os.path.join(args.fonts, "*.ttf"))) if args.fonts else []
if fonts:
    # static instances beat the variable font, which matplotlib can only render at one weight
    font_manager.fontManager.ttflist = [f for f in font_manager.fontManager.ttflist if f.name != "Inter"]
    for f in fonts:
        font_manager.fontManager.addfont(f)

INK, BODY, MUTED, FAINT = "#0f172a", "#334155", "#64748b", "#94a3b8"
GRID, RULE, PANEL = "#e8ecf2", "#cbd5e1", "#f6f8fb"
HERO = "#6d28d9"
METHODS = ["naive", "kl_global", "replay_kl"]
COLOR = {"naive": "#f08a5d", "kl_global": "#3b8fd0", "replay_kl": HERO}
NAME = {"naive": "Naive fine-tuning", "kl_global": "KL on new-book tokens",
        "replay_kl": "Rehearsal + KL on rehearsal prompts"}
SHORT = {"naive": "Naive", "kl_global": "KL on new-book tokens", "replay_kl": "Rehearsal + KL"}

plt.rcParams.update({
    "font.family": ["Inter", "IBM Plex Sans", "DejaVu Sans"],
    "font.size": 11, "text.color": INK, "axes.labelcolor": BODY, "axes.edgecolor": RULE,
    "axes.linewidth": 1.0, "axes.labelsize": 11, "axes.labelpad": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 200,
    "svg.fonttype": "path", "axes.unicode_minus": True,
})


def tint(color, amount):
    """Blend a color towards white; amount=0 is the color, 1 is white."""
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


def num(v, sign=False):
    s = f"{v:+.3f}" if sign else f"{v:.3f}"
    return s.replace("-", "\u2212")


df = pd.read_csv(args.csv)
df = df[~df.smoke_test.astype(str).str.lower().eq("true")]
order = [m for m in METHODS if m in set(df.method)]
seeds = sorted(df.seed.unique())
cols = ["after_A_A", "after_B_A", "after_B_B", "forgetting_A"]
g = df.groupby("method")[cols].agg(["mean", "std"]).reindex(order)  # sample sd, as in summary.md


def mu(m, c):
    return g.loc[m, (c, "mean")]


def sd(m, c):
    return g.loc[m, (c, "std")]


stage1 = df.drop_duplicates("seed").set_index("seed").sort_index()
start_A, start_B = stage1.after_A_A.mean(), stage1.after_A_B.mean()
model = df.model.iloc[0]
foot = f"{model}  \u00b7  LoRA r=64  \u00b7  {len(seeds)} seeds  \u00b7  1,500 held-out questions per book"


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(args.out, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("wrote", os.path.join(args.out, name + ".png"))


def panel_title(ax, tag, text, sub=None, y=1.10):
    ax.text(0, y, tag, transform=ax.transAxes, fontsize=11, fontweight="bold", color="white", va="center",
            ha="center", bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.5", fc=INK, ec="none"))
    ax.text(0.045, y, text, transform=ax.transAxes, fontsize=13.5, fontweight="bold", va="center")
    if sub:
        ax.text(0.045, y - 0.06, sub, transform=ax.transAxes, fontsize=10, color=MUTED, va="center")


def frontier(ax, show_seeds=True, labels=True, label_size=11):
    """Book A retained against book B learned after stage 2, with mean, +-1 sd ellipse and seeds."""
    ax.set_xlim(-0.02, 1.06)
    ax.set_ylim(-0.02, 1.06)
    ticks = [0, 0.25, 0.5, 0.75, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
    ax.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.add_patch(Rectangle((0.85, 0.85), 0.21, 0.21, fc=tint(HERO, 0.93), ec="none", zorder=0))
    ax.plot(1, 1, marker="*", ms=15, color=INK, mec="white", mew=1, zorder=5)
    ax.text(1.0, 1.035, "goal", ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="semibold")

    p = np.array([mu("naive", "after_B_B"), mu("naive", "after_B_A")])
    q = np.array([mu("kl_global", "after_B_B"), mu("kl_global", "after_B_A")])
    d = (q - p) / np.linalg.norm(q - p)
    a, b = p - 0.10 * d, q + 0.22 * d
    ax.plot([a[0], b[0]], [a[1], b[1]], color=FAINT, lw=1.4, ls=(0, (5, 4)), zorder=1)

    for m in order:
        x, y = mu(m, "after_B_B"), mu(m, "after_B_A")
        ax.add_patch(Ellipse((x, y), 2 * sd(m, "after_B_B"), 2 * sd(m, "after_B_A"), fc=tint(COLOR[m], 0.75),
                             ec=COLOR[m], lw=1, alpha=0.8, zorder=2))
        if show_seeds:
            d_m = df[df.method == m]
            ax.scatter(d_m.after_B_B, d_m.after_B_A, s=26, color=COLOR[m], ec="white", lw=0.8, zorder=3)
        ax.scatter([x], [y], s=150 if m == "replay_kl" else 110, color=COLOR[m], ec="white", lw=2, zorder=4)

    ax.scatter([start_B], [start_A], s=70, marker="D", color="white", ec=INK, lw=1.6, zorder=4)

    h = np.array([mu("replay_kl", "after_B_B"), mu("replay_kl", "after_B_A")])
    t = np.dot(h - p, q - p) / np.dot(q - p, q - p)
    foot_pt = p + t * (q - p)
    ax.add_patch(FancyArrowPatch(foot_pt, h, arrowstyle="-|>,head_width=4,head_length=7", color=HERO, lw=2,
                                 shrinkA=2, shrinkB=11, zorder=3))
    if not labels:
        return
    ang = np.degrees(np.arctan2(*(q - p)[::-1])) - 180
    normal = np.array([-d[1], d[0]])
    ax.text(*(p + 0.38 * (q - p) + 0.035 * normal), "naive \u2194 KL trade-off", rotation=ang,
            rotation_mode="anchor", transform_rotates_text=True, ha="center", va="center", fontsize=9.5,
            color=MUTED)
    mid = foot_pt + 0.5 * (h - foot_pt)
    ax.text(mid[0] + 0.03, mid[1] - 0.035, "breaks the\ntrade-off", ha="left", va="top", fontsize=10.5,
            color=HERO, fontweight="bold", linespacing=1.15)
    ax.annotate("after book A\n(start of stage 2)", (start_B, start_A), xytext=(10, -14),
                textcoords="offset points", fontsize=9.5, color=BODY, va="top", linespacing=1.15)
    place = {"naive": ((-12, -6), "right", "top"), "kl_global": ((-14, -8), "right", "top"),
             "replay_kl": ((-44, 10), "right", "center")}
    for m in order:
        off, ha, va = place[m]
        x, y = mu(m, "after_B_B"), mu(m, "after_B_A")
        ax.annotate(f"{SHORT[m]}", (x, y), xytext=off, textcoords="offset points", ha=ha, va=va,
                    fontsize=label_size, fontweight="bold", color=COLOR[m])
        ax.annotate(f"A {num(y)}  \u00b7  B {num(x)}", (x, y), xytext=(off[0], off[1] - label_size * 1.35),
                    textcoords="offset points", ha=ha, va=va, fontsize=label_size - 1.5, color=BODY)


# ---------------------------------------------------------------- hero
fig = plt.figure(figsize=(14, 7.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.2, left=0.05, right=0.98, top=0.74, bottom=0.12)
ax = fig.add_subplot(gs[0])
ax.axvspan(1.5, 3.4, color=PANEL, zorder=0, lw=0)
ax.text(2.0, 1.04, "STAGE 2  \u00b7  train on book B", ha="center", va="bottom", fontsize=9, color=MUTED,
        fontweight="semibold")
ax.text(0.5, 1.04, "STAGE 1  \u00b7  book A", ha="center", va="bottom", fontsize=9, color=MUTED,
        fontweight="semibold")
for m in order:
    d_m = df[df.method == m]
    for _, r in d_m.iterrows():
        ax.plot([1, 2], [r.after_A_A, r.after_B_A], color=COLOR[m], lw=0.9, alpha=0.35, zorder=2)
for _, r in stage1.iterrows():
    ax.plot([0, 1], [r.base_A, r.after_A_A], color=INK, lw=0.9, alpha=0.2, zorder=2)
ax.plot([0, 1], [df.base_A.mean(), start_A], color=INK, lw=3, solid_capstyle="round", zorder=3)
ax.scatter([0, 1], [df.base_A.mean(), start_A], s=60, color=INK, ec="white", lw=1.5, zorder=5)
for m in order:
    y2 = mu(m, "after_B_A")
    ax.plot([1, 2], [start_A, y2], color=COLOR[m], lw=4 if m == "replay_kl" else 3, solid_capstyle="round",
            zorder=4 if m == "replay_kl" else 3)
    ax.errorbar(2, y2, yerr=sd(m, "after_B_A"), color=COLOR[m], lw=1.6, capsize=0, zorder=4)
    ax.scatter([2], [y2], s=90, color=COLOR[m], ec="white", lw=1.8, zorder=5)
    dy = {"naive": 0, "kl_global": -0.02, "replay_kl": 0.025}[m]
    ax.text(2.16, y2 + dy + 0.024, SHORT[m], fontsize=11.5, fontweight="bold", color=COLOR[m], va="center")
    ax.text(2.16, y2 + dy - 0.028, f"{num(y2)}  (forgets {num(mu(m, 'forgetting_A'))})", fontsize=9.5,
            color=BODY, va="center")
ax.annotate(f"base {num(df.base_A.mean())}", (0, df.base_A.mean()), xytext=(10, 4), textcoords="offset points",
            fontsize=9.5, color=BODY)
ax.annotate(num(start_A), (1, start_A), xytext=(-12, 8), textcoords="offset points", ha="right",
            fontsize=9.5, color=BODY)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Base model", "After book A", "After book B"], fontsize=10.5, color=BODY)
ax.set_xlim(-0.15, 3.4)
ax.set_ylim(-0.04, 1.1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
ax.set_ylabel("Book-A accuracy")
ax.yaxis.grid(True, color=GRID, lw=0.9)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
panel_title(ax, "a", "Does the model still know book A?", "mean of 3 seeds (thick), each seed (thin), \u00b11 sd")

ax2 = fig.add_subplot(gs[1])
frontier(ax2)
ax2.set_aspect("equal", adjustable="box", anchor="W")
ax2.set_xlabel("Book B learned  \u2192")
ax2.set_ylabel("Book A retained  \u2192")
panel_title(ax2, "b", "Retention vs. learning after stage 2", "mean (large), seeds (small), \u00b11 sd ellipse")

fig.text(0.05, 0.955, "Learn a second book without forgetting the first", fontsize=25, fontweight="heavy",
         color=INK, va="top")
fig.text(0.05, 0.885, f"{model.split('/')[-1]} learns book A, then book B on a fresh LoRA. ", fontsize=13,
         color=BODY, va="top")
fig.text(0.05, 0.85, f"Rehearsal + KL on rehearsal prompts keeps book A at {num(mu('replay_kl', 'after_B_A'))} "
         f"(forgetting {num(mu('replay_kl', 'forgetting_A'))}) while learning book B to "
         f"{num(mu('replay_kl', 'after_B_B'))}.", fontsize=13, color=HERO, fontweight="semibold", va="top")
fig.text(0.05, 0.035, foot + "  \u00b7  accuracy on held-out questions", fontsize=9.5, color=MUTED)
save(fig, "hero")

# ---------------------------------------------------------------- accuracy bars
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True, gridspec_kw=dict(wspace=0.08))
ys = np.arange(len(order))[::-1]
for ax, c, title in zip(axs, ["after_B_A", "after_B_B"], ["Book A retained", "Book B learned"]):
    for y, m in zip(ys, order):
        v, e = mu(m, c), sd(m, c)
        ax.barh(y, 1.0, 0.62, color=PANEL, zorder=0)
        ax.barh(y, v, 0.62, color=COLOR[m], zorder=2)
        ax.plot([v - e, v + e], [y, y], color="white", lw=1.6, zorder=3, solid_capstyle="butt")
        ax.plot([v - e, v + e], [y, y], color=INK, lw=0.9, zorder=3, alpha=0.6)
        d_m = df[df.method == m]
        ax.scatter(d_m[c], [y + 0.0] * len(d_m), s=18, color="white", ec=INK, lw=0.8, zorder=4)
        ax.text(1.03, y, num(v), va="center", fontsize=12, fontweight="bold",
                color=COLOR[m] if m == "replay_kl" else INK)
        ax.text(1.03, y - 0.27, f"\u00b1 {e:.3f}", va="center", fontsize=9, color=MUTED)
    if c == "after_B_A":
        ax.axvline(start_A, color=INK, lw=1, ls=(0, (3, 3)), zorder=5, ymax=0.97)
        ax.text(start_A, len(order) - 0.45, f"before stage 2: {num(start_A)}", ha="right", va="bottom",
                fontsize=9, color=BODY)
    ax.set_xlim(0, 1.17)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_ylim(-0.55, len(order) - 0.25)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", pad=14)
axs[0].set_yticks(ys)
axs[0].set_yticklabels([NAME[m] for m in order], fontsize=11, color=INK)
for t, m in zip(axs[0].get_yticklabels(), order):
    t.set_fontweight("bold" if m == "replay_kl" else "normal")
    t.set_color(HERO if m == "replay_kl" else INK)
fig.suptitle("Accuracy on each book after training on book B", x=0.125, ha="left", y=1.06, fontsize=17,
             fontweight="heavy")
fig.text(0.125, -0.03, foot + "  \u00b7  bars: mean, white dots: seeds, line: \u00b11 sd", fontsize=9,
         color=MUTED)
save(fig, "accuracy_bars")

# ---------------------------------------------------------------- forgetting (dumbbell)
fig, ax = plt.subplots(figsize=(7.6, 6.2))
ys = np.arange(len(order))[::-1] * 1.0
for y, m in zip(ys, order):
    a, b = mu(m, "after_A_A"), mu(m, "after_B_A")
    ax.plot([0, 1.0], [y, y], color=GRID, lw=1, zorder=0)
    if a - b > 0.05:
        ax.add_patch(FancyArrowPatch((a, y), (b, y), arrowstyle="-|>,head_width=5,head_length=8",
                                     color=COLOR[m], lw=3.2, shrinkA=8, shrinkB=9, zorder=2))
    ax.scatter([a], [y], s=150, color="white", ec=INK, lw=1.8, zorder=3)
    ax.scatter([b], [y], s=90 if abs(a - b) < 0.05 else 140, color=COLOR[m], ec="white", lw=1.5, zorder=4)
    d_m = df[df.method == m]
    ax.scatter(d_m.after_A_A, [y - 0.17] * len(d_m), s=16, color=INK, lw=0, alpha=0.35, zorder=3)
    ax.scatter(d_m.after_B_A, [y - 0.26] * len(d_m), s=16, color=COLOR[m], lw=0, alpha=0.8, zorder=3)
    if abs(a - b) < 0.05:
        ax.text(min(a, b) - 0.03, y, f"{num(a)} \u2192 {num(b)}", ha="right", va="center", fontsize=10.5,
                color=COLOR[m], fontweight="bold")
    else:
        ax.text(a, y + 0.13, num(a), ha="center", va="bottom", fontsize=10, color=BODY)
        ax.text(b, y + 0.13, num(b), ha="center", va="bottom", fontsize=10.5, color=COLOR[m], fontweight="bold")
    hero = m == "replay_kl"
    ax.text(0, y + 0.42, NAME[m], fontsize=12, fontweight="bold", color=HERO if hero else INK, va="bottom")
    f, fs = mu(m, "forgetting_A"), sd(m, "forgetting_A")
    ax.text(0, y + 0.38, f"forgetting {num(f)} \u00b1 {fs:.3f}", fontsize=10.5, va="top",
            color=HERO if hero else BODY, fontweight="semibold" if hero else "normal")
ax.set_xlim(-0.02, 1.04)
ax.set_ylim(-0.5, ys[0] + 0.85)
ax.set_yticks([])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
ax.set_xlabel("Book-A accuracy")
ax.spines["left"].set_visible(False)
ax.xaxis.grid(True, color=GRID, lw=0.9)
ax.set_axisbelow(True)
ax.set_title("Book A, before and after learning book B", loc="left", fontsize=15, fontweight="heavy", pad=34)
ax.scatter([0.008], [1.055], s=60, color="white", ec=INK, lw=1.5, transform=ax.transAxes, clip_on=False)
ax.text(0.03, 1.055, "before stage 2", transform=ax.transAxes, fontsize=10, color=BODY, va="center")
ax.scatter([0.26], [1.055], s=60, color=FAINT, ec="white", lw=1, transform=ax.transAxes, clip_on=False)
ax.text(0.282, 1.055, f"after stage 2   \u00b7   mean of {len(seeds)} seeds, small dots: each seed",
        transform=ax.transAxes, fontsize=10, color=BODY, va="center")
save(fig, "forgetting")

# ---------------------------------------------------------------- trade-off
fig, ax = plt.subplots(figsize=(7.6, 6.2))
frontier(ax, label_size=11)
ax.set_xlabel("Book B learned  \u2192")
ax.set_ylabel("Book A retained  \u2192")
ax.set_title("Stability vs. plasticity", loc="left", fontsize=15, fontweight="heavy", pad=30)
ax.text(0, 1.035, "after stage 2  \u00b7  mean (large), each seed (small), \u00b11 sd ellipse",
        transform=ax.transAxes, fontsize=9.5, color=MUTED)
save(fig, "tradeoff")

# ---------------------------------------------------------------- per seed (small multiples)
fig, axs = plt.subplots(1, len(seeds), figsize=(14, 5.4), sharex=True, sharey=True,
                        gridspec_kw=dict(wspace=0.08))
axs = np.atleast_1d(axs)
for ax, s in zip(axs, seeds):
    ds = df[df.seed == s].set_index("method")
    st = stage1.loc[s]
    ax.set_xlim(-0.02, 1.06)
    ax.set_ylim(-0.02, 1.16)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(["0", ".50", "1"])
    ax.set_yticklabels(["0", ".50", "1"])
    ax.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for k in ("left", "bottom"):
        ax.spines[k].set_visible(False)
    ax.add_patch(Rectangle((0.85, 0.85), 0.21, 0.31, fc=tint(HERO, 0.93), ec="none", zorder=0))
    ax.plot(1, 1, marker="*", ms=12, color=INK, mec="white", mew=1, zorder=5)
    ax.scatter([st.after_A_B], [st.after_A_A], s=55, marker="D", color="white", ec=INK, lw=1.5, zorder=6)
    for m in order:
        r = ds.loc[m]
        ax.add_patch(FancyArrowPatch((st.after_A_B, st.after_A_A), (r.after_B_B, r.after_B_A),
                                     connectionstyle="arc3,rad=-0.15",
                                     arrowstyle="-|>,head_width=3.5,head_length=6", color=COLOR[m],
                                     lw=2.4 if m == "replay_kl" else 1.8, shrinkA=6, shrinkB=7, zorder=3))
        ax.scatter([r.after_B_B], [r.after_B_A], s=90, color=COLOR[m], ec="white", lw=1.8, zorder=5)
    ax.text(0.03, 0.99, f"seed {s}", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    x0, y0, row = 0.035, 0.32, 0.07
    colx = {"A": 0.5, "B": 0.635, "forgets": 0.79}
    for k, x in colx.items():
        ax.text(x, y0, k, transform=ax.transAxes, fontsize=8.5, color=MUTED, ha="right", va="center", zorder=7)
    for i, m in enumerate(order):
        r = ds.loc[m]
        yy = y0 - (i + 1) * row
        hero = m == "replay_kl"
        w = "bold" if hero else "normal"
        ax.scatter([x0 + 0.012], [yy], s=34, color=COLOR[m], transform=ax.transAxes, zorder=7, ec="none")
        ax.text(x0 + 0.04, yy, "KL" if m == "kl_global" else SHORT[m], transform=ax.transAxes, fontsize=9,
                va="center", color=HERO if hero else INK, fontweight=w, zorder=7)
        for k, v in zip(colx, (r.after_B_A, r.after_B_B, r.forgetting_A)):
            ax.text(colx[k], yy, num(v), transform=ax.transAxes, fontsize=9, ha="right", va="center",
                    color=HERO if hero else BODY, fontweight=w, zorder=7)
    ax.add_patch(FancyBboxPatch((0.015, 0.03), 0.8, 0.345, boxstyle="round,pad=0,rounding_size=0.02",
                                transform=ax.transAxes, fc="white", ec=GRID, lw=1, zorder=6.5))
    ax.set_xlabel("Book B learned  \u2192")
axs[0].set_ylabel("Book A retained  \u2192")
fig.suptitle("The same picture in every seed", x=0.125, ha="left", y=1.04, fontsize=17, fontweight="heavy")
fig.text(0.125, 0.96, "arrows run from the book-A model (diamond) to each method after stage 2; values are "
         "accuracy after stage 2 and forgetting of book A", fontsize=10.5, color=MUTED)
save(fig, "per_seed")

# ---------------------------------------------------------------- method diagram
W, H = 150, 58
fig = plt.figure(figsize=(W / 10, H / 10))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")
STAGE1 = "#0f766e"


def card(x, y, w, h, fc="white", ec=RULE, lw=1.2, r=1.6, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z))


def label(x, y, title, sub=None, color=INK, size=12, subsize=9.5):
    ax.text(x, y, title, ha="center", va="center", fontsize=size, fontweight="bold", color=color, zorder=6)
    if sub:
        ax.text(x, y - 2.0, sub, ha="center", va="top", fontsize=subsize, color=MUTED, linespacing=1.35,
                zorder=6)


def arrow(p, q, color=INK, ls="-", lw=1.6, rad=0.0, style="angle3"):
    cs = f"arc3,rad={rad}" if style == "arc" else style
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_width=3.5,head_length=6", color=color, lw=lw,
                                 linestyle=ls, connectionstyle=cs, shrinkA=0, shrinkB=0, zorder=4))


def network_icon(cx, cy, color):
    layers = [(-4, [-2.2, 0, 2.2]), (0, [-3.3, -1.1, 1.1, 3.3]), (4, [-2.2, 0, 2.2])]
    for (x1, ys1), (x2, ys2) in zip(layers, layers[1:]):
        for a in ys1:
            for b in ys2:
                ax.plot([cx + x1, cx + x2], [cy + a, cy + b], color=tint(color, 0.55), lw=0.7, zorder=5)
    for x, ys_ in layers:
        for y in ys_:
            ax.add_patch(Circle((cx + x, cy + y), 0.75, fc=color, ec="white", lw=0.8, zorder=6))


def book_icon(cx, cy, color):
    ax.add_patch(Polygon([(cx, cy - 2.6), (cx - 4.4, cy - 1.8), (cx - 4.4, cy + 3.2), (cx, cy + 2.4)],
                         closed=True, fc=tint(color, 0.8), ec=color, lw=1.2, zorder=5))
    ax.add_patch(Polygon([(cx, cy - 2.6), (cx + 4.4, cy - 1.8), (cx + 4.4, cy + 3.2), (cx, cy + 2.4)],
                         closed=True, fc=tint(color, 0.8), ec=color, lw=1.2, zorder=5))
    for k in range(3):
        o = 0.9 + k * 1.1
        ax.plot([cx - 3.6, cx - 0.8], [cy + o - 1.2 + 0.4, cy + o - 1.2 - 0.1], color=color, lw=0.9, zorder=6)
        ax.plot([cx + 0.8, cx + 3.6], [cy + o - 1.2 - 0.1, cy + o - 1.2 + 0.4], color=color, lw=0.9, zorder=6)


def lora_icon(cx, cy, color):
    ax.add_patch(Rectangle((cx - 4.2, cy - 3), 1.6, 6, fc=color, ec="none", zorder=6))
    ax.text(cx - 1.4, cy, "\u00d7", ha="center", va="center", fontsize=11, color=color, zorder=6)
    ax.add_patch(Rectangle((cx - 0.2, cy + 1.2), 6, 1.6, fc=tint(color, 0.35), ec="none", zorder=6))
    ax.text(cx + 2.8, cy - 1.3, "low rank", ha="center", va="center", fontsize=7.5, color=color, zorder=6)


def snowflake_icon(cx, cy, color, r=2.6):
    for k in range(3):
        t = np.pi / 3 * k + np.pi / 2
        ax.plot([cx - r * np.cos(t), cx + r * np.cos(t)], [cy - r * np.sin(t), cy + r * np.sin(t)], color=color,
                lw=1.6, solid_capstyle="round", zorder=7)
        for sgn in (1, -1):
            tx, ty = cx + sgn * 0.62 * r * np.cos(t), cy + sgn * 0.62 * r * np.sin(t)
            for dt in (0.6, -0.6):
                ax.plot([tx, tx + sgn * 0.8 * np.cos(t + dt)], [ty, ty + sgn * 0.8 * np.sin(t + dt)], color=color,
                        lw=1.2, solid_capstyle="round", zorder=7)


def check_icon(cx, cy, color):
    ax.add_patch(Circle((cx, cy), 3.2, fc=tint(color, 0.85), ec=color, lw=1.4, zorder=5))
    ax.plot([cx - 1.5, cx - 0.4, cx + 1.7], [cy + 0.1, cy - 1.1, cy + 1.3], color=color, lw=2.2,
            solid_capstyle="round", solid_joinstyle="round", zorder=6)


def chip(x, y, text, color, struck=False):
    w = 0.95 * len(text) + 2.8
    ax.add_patch(FancyBboxPatch((x, y - 1.3), w, 2.6, boxstyle="round,pad=0,rounding_size=1.3",
                                fc=tint(color, 0.85) if not struck else "white", ec=color, lw=1, zorder=6))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=9, fontweight="bold", color=color, zorder=7)
    if struck:
        ax.plot([x + 0.9, x + w - 0.9], [y, y], color=color, lw=1.2, zorder=8)
    return x + w + 1.2


def stage_header(x, y, n, title, sub, color):
    ax.add_patch(Circle((x + 1.8, y), 1.8, fc=color, ec="none", zorder=6))
    ax.text(x + 1.8, y - 0.05, str(n), ha="center", va="center", fontsize=11, fontweight="bold", color="white",
            zorder=7)
    ax.text(x + 5, y + 0.1, title, va="center", fontsize=13, fontweight="bold", color=color, zorder=7)
    ax.text(x + 5, y - 3.0, sub, va="center", fontsize=9.5, color=MUTED, zorder=7)


card(1, 1, 61, 56, fc=tint(STAGE1, 0.95), ec="none", r=2.5, z=0)
card(65, 1, 84, 56, fc=tint(HERO, 0.95), ec="none", r=2.5, z=0)
stage_header(4, 52, 1, "Learn book A", "LoRA on the base model, then merged", STAGE1)
stage_header(68, 52, 2, "Learn book B, protect book A", "fresh LoRA on the frozen book-A model", HERO)

card(4, 26, 17, 18)
network_icon(12.5, 37.5, INK)
label(12.5, 30.5, "Qwen2.5-7B", "base model")
card(4, 4, 17, 18)
book_icon(12.5, 16, STAGE1)
label(12.5, 10.5, "Book A", "300 towns \u00d7 5 facts\n\u00d7 4 paraphrases", subsize=8.5)
card(25, 15, 15, 18, ec=STAGE1, lw=1.6)
lora_icon(31.5, 26.5, STAGE1)
label(32.5, 19.8, "LoRA A", "r=64, CE loss", subsize=9)
card(44, 15, 15, 18, fc=INK, ec=INK)
snowflake_icon(51.5, 26.5, "white")
ax.text(51.5, 20.0, "Book-A model", ha="center", va="center", fontsize=11, fontweight="bold", color="white",
        zorder=6)
ax.text(51.5, 17.4, "frozen", ha="center", va="center", fontsize=9, color=tint(INK, 0.55), zorder=6)
arrow((21, 35), (25, 28), style="arc", rad=0.0)
arrow((21, 13), (25, 20), style="arc", rad=0.0)
arrow((40, 24), (44, 24))

card(68, 6, 40, 38, fc="white", ec=tint(HERO, 0.6), lw=1.2, r=2)
ax.text(70, 41.2, "EVERY TRAINING BATCH", fontsize=8.5, fontweight="bold", color=MUTED, va="center")
card(70.5, 26, 35, 12, fc=PANEL, ec=RULE)
book_icon(76.5, 32, MUTED)
ax.text(82.5, 34.6, "32 book-B statements", fontsize=10.5, fontweight="bold", va="center", color=INK, zorder=6)
nx = chip(82.5, 30.3, "CE", BODY)
chip(nx, 30.3, "no KL", FAINT, struck=True)
card(70.5, 9, 35, 15, fc=tint(HERO, 0.9), ec=HERO, lw=1.4)
book_icon(76.5, 17.5, HERO)
ax.text(82.5, 21.0, "13 book-A rehearsal", fontsize=10.5, fontweight="bold", va="center", color=INK, zorder=6)
ax.text(82.5, 18.6, "statements", fontsize=10.5, fontweight="bold", va="center", color=INK, zorder=6)
nx = chip(82.5, 14.0, "CE", HERO)
chip(nx, 14.0, "KL to book-A model", HERO)
ax.text(70, 3.4, "KL on the rehearsal tokens only; book-B tokens get none", fontsize=9, color=HERO, va="center",
        fontweight="semibold")

arrow((59, 24), (68, 24), lw=1.8)
ax.add_patch(FancyArrowPatch((51.5, 15), (70.5, 12.0), arrowstyle="-|>,head_width=3.5,head_length=6",
                             color=HERO, lw=1.6, linestyle=(0, (4, 3)), connectionstyle="angle,angleA=-90,angleB=180,rad=3",
                             shrinkA=0, shrinkB=0, zorder=4))
ax.text(60, 9.6, "KL reference", fontsize=9, color=HERO, ha="center", va="center", fontweight="semibold")

card(112, 15, 15, 18, ec=HERO, lw=2)
lora_icon(118.5, 26.5, HERO)
label(119.5, 19.8, "LoRA B", "fresh, r=64", color=HERO, subsize=9)
arrow((108, 24), (112, 24), lw=1.8)
card(131, 11, 16, 26, ec=RULE)
check_icon(139, 30.5, INK)
label(139, 24.2, "Evaluate", "1,500 held-out\nquestions per book;\ngold answer within\n12 greedy tokens",
      subsize=8.3)
arrow((127, 24), (131, 24), lw=1.8)
save(fig, "method")
