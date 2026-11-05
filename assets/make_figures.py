#!/usr/bin/env python3
"""Build the README figures from results/results.csv.

usage: python assets/make_figures.py [--csv results/results.csv] [--out assets] [--only hero,method,...]

Writes hero, method, accuracy_bars, forgetting, tradeoff and per_seed as PNG (200 dpi) and SVG.
Needs numpy, pandas and matplotlib. Type is set in Inter, Instrument Serif and IBM Plex Mono,
which ship as static TTFs in assets/fonts (SIL Open Font License), so the output does not depend
on what is installed on the machine. Every number drawn comes from the CSV.
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
from matplotlib import font_manager, patheffects
from matplotlib.colors import to_rgb
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon,
                                Rectangle)
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = ["hero", "method", "accuracy_bars", "forgetting", "tradeoff", "per_seed"]
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=os.path.join(HERE, "..", "results", "results.csv"))
ap.add_argument("--out", default=HERE)
ap.add_argument("--fonts", default=os.path.join(HERE, "fonts"))
ap.add_argument("--only", default=",".join(FIGURES))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

for f in sorted(glob.glob(os.path.join(args.fonts, "*.ttf"))):
    font_manager.fontManager.addfont(f)
# a variable Inter, if installed, renders at a single weight; keep only the static files
font_manager.fontManager.ttflist = [
    f for f in font_manager.fontManager.ttflist
    if f.name not in ("Inter", "Instrument Serif", "IBM Plex Mono") or os.path.dirname(f.fname) == args.fonts]

SANS, SERIF, MONO = "Inter", "Instrument Serif", "IBM Plex Mono"
HALO = [patheffects.withStroke(linewidth=3.5, foreground="white")]

# warm neutrals, two quiet baselines and one loud colour for the method that works
INK = "#14161a"
BODY = "#3e434b"
MUTED = "#737882"
FAINT = "#a9adb4"
HAIR = "#e4e1da"
PAPER = "#f6f4ef"
ACCENT = "#2337e6"
COLOR = {"naive": "#d9835a", "kl_global": "#8f9c92", "replay_kl": ACCENT}
DEEP = {"naive": "#b45f37", "kl_global": "#5f6d63", "replay_kl": "#1a2ab8"}

METHODS = ["naive", "kl_global", "replay_kl"]
SHORT = {"naive": "Naive fine-tuning", "kl_global": "KL on new-book tokens", "replay_kl": "Rehearsal + KL"}
TINY = {"naive": "Naive", "kl_global": "KL on B tokens", "replay_kl": "Rehearsal + KL"}

plt.rcParams.update({
    "font.family": SANS, "font.size": 11, "text.color": INK,
    "axes.labelcolor": BODY, "axes.edgecolor": HAIR, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.major.pad": 7, "ytick.major.pad": 7,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "none",
    "svg.fonttype": "path", "svg.hashsalt": "sequential-books", "axes.unicode_minus": True,
    "path.simplify": False,
})

df = pd.read_csv(args.csv)
df = df[~df.smoke_test.astype(str).str.lower().eq("true")]
order = [m for m in METHODS if m in set(df.method)]
seeds = sorted(df.seed.unique())
cols = ["after_A_A", "after_A_B", "after_B_A", "after_B_B", "forgetting_A", "stage2_sec"]
g = df.groupby("method")[cols].agg(["mean", "std"]).reindex(order)  # sample sd, as in summary.md
stage1 = df.drop_duplicates("seed").set_index("seed").sort_index()
model = df.model.iloc[0]
model_short = model.split("/")[-1]
BASE_A = df.base_A.mean()
START_A, START_B = stage1.after_A_A.mean(), stage1.after_A_B.mean()


def mu(m, c):
    return g.loc[m, (c, "mean")]


def sd(m, c):
    return g.loc[m, (c, "std")]


def num(v, sign=False, nd=3):
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "\u2212")


def tint(color, amount):
    """Blend towards white: 0 keeps the colour, 1 is white."""
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


class Sheet:
    """A figure laid out in inches from the top-left corner, like a page."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.fig.patch.set_facecolor("white")
        self.page = self.fig.add_axes([0, 0, 1, 1])
        self.page.set_xlim(0, w)
        self.page.set_ylim(h, 0)
        self.page.axis("off")
        self.page.set_zorder(-1)

    def axes(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h], **kw)

    def text(self, x, y, s, size=11, color=INK, family=SANS, weight="regular", style="normal", ha="left",
             va="baseline", **kw):
        return self.page.text(x, y, s, fontsize=size, color=color, family=family, fontweight=weight,
                              fontstyle=style, ha=ha, va=va, **kw)

    def width(self, t):
        r = self.fig.canvas.get_renderer()
        return t.get_window_extent(r).width / self.fig.dpi

    def runs(self, x, y, parts, **common):
        """Set consecutive text runs with different styles on one baseline; returns the end x."""
        for s, kw in parts:
            t = self.text(x, y, s, **{**common, **kw})
            x += self.width(t)
        return x

    def rule(self, x0, x1, y, color=HAIR, lw=1.0):
        self.page.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt")

    def vrule(self, x, y0, y1, color=HAIR, lw=1.0):
        self.page.plot([x, x], [y0, y1], color=color, lw=lw, solid_capstyle="butt")

    def eyebrow(self, x, y, s, color=MUTED, size=8.6, square=None):
        if square:
            self.page.add_patch(Rectangle((x, y - 0.085), 0.085, 0.085, fc=square, ec="none"))
            x += 0.17
        return self.text(x, y, s.upper(), size=size, color=color, family=MONO, weight="medium")

    def save(self, name):
        for ext in ("png", "svg"):
            meta = {"Software": None} if ext == "png" else {"Date": None, "Creator": None}
            self.fig.savefig(os.path.join(args.out, f"{name}.{ext}"), dpi=200, metadata=meta)
        plt.close(self.fig)
        print("wrote", os.path.join(args.out, name + ".png"))


def frame(ax, xlim, ylim, xticks, yticks, xfmt=None, yfmt=None, grid="both", size=9):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xfmt or [f"{t:g}" for t in xticks], family=MONO, fontsize=size)
    ax.set_yticklabels(yfmt or [f"{t:g}" for t in yticks], family=MONO, fontsize=size)
    for s in ax.spines.values():
        s.set_visible(False)
    if grid in ("both", "x"):
        for t in xticks:
            ax.axvline(t, color=HAIR, lw=0.9, zorder=0)
    if grid in ("both", "y"):
        for t in yticks:
            ax.axhline(t, color=HAIR, lw=0.9, zorder=0)


FRAC = lambda ts: ["0" if t == 0 else "1" if t == 1 else f"{t:.2f}"[1:] for t in ts]  # noqa: E731


def glow(ax, xlim, ylim, center, color, strength=0.16, radius=0.75, clip=None, zorder=0):
    """A soft radial wash of colour, optionally clipped to a patch."""
    xs = np.linspace(*xlim, 400)
    ys = np.linspace(*ylim, 400)
    X, Y = np.meshgrid(xs, ys)
    d = np.hypot(X - center[0], Y - center[1]) / radius
    a = strength * np.clip(1 - d, 0, 1) ** 1.6
    img = np.zeros(X.shape + (4,))
    img[..., :3] = to_rgb(color)
    img[..., 3] = a
    im = ax.imshow(img, extent=(*xlim, *ylim), origin="lower", aspect="auto", zorder=zorder,
                   interpolation="bilinear")
    if clip is not None:
        im.set_clip_path(clip)
    return im


def frontier(ax, data=None, start=None, big=True, seed_dots=True, labels=True, scale=1.0):
    """Book A retained (y) against book B learned (x) after stage 2."""
    xlim, ylim = (-0.03, 1.08), (-0.03, 1.08)
    ts = [0, 0.25, 0.5, 0.75, 1]
    frame(ax, xlim, ylim, ts, ts, FRAC(ts), FRAC(ts), size=9 * scale)

    pts = {}
    for m in order:
        d_m = df[df.method == m] if data is None else data[data.method == m]
        pts[m] = (d_m.after_B_B.mean(), d_m.after_B_A.mean(), d_m)
    if start is None:
        start = (START_B, START_A)

    # shade the side of the line through the two baselines that faces the goal
    p = np.array(pts["naive"][:2])
    q = np.array(pts["kl_global"][:2])
    slope = (q[1] - p[1]) / (q[0] - p[0])
    y_at = lambda x: p[1] + slope * (x - p[0])  # noqa: E731
    x_top = p[0] + (ylim[1] - p[1]) / slope
    beyond = Polygon([(x_top, ylim[1]), (xlim[1], ylim[1]), (xlim[1], y_at(xlim[1]))], closed=True,
                     fc="none", ec="none", transform=ax.transData)
    ax.add_patch(beyond)
    glow(ax, xlim, ylim, (1.0, 1.0), ACCENT, strength=0.13, radius=1.05, clip=beyond)
    ax.plot([x_top, xlim[1]], [ylim[1], y_at(xlim[1])], color=FAINT, lw=1.1 * scale, ls=(0, (1.2, 3.2)),
            dash_capstyle="round", zorder=1)

    # goal corner
    ax.scatter([1], [1], s=260 * scale ** 2, marker="+", color=INK, lw=1.0 * scale, zorder=6)
    ax.scatter([1], [1], s=34 * scale ** 2, fc="white", ec=INK, lw=1.0 * scale, zorder=7)

    # trajectories from the shared stage-1 model
    bend = {"naive": 0.14, "kl_global": -0.22, "replay_kl": -0.06}
    for m in order:
        x, y, _ = pts[m]
        hero = m == "replay_kl"
        ax.add_patch(FancyArrowPatch(
            start, (x, y), connectionstyle=f"arc3,rad={bend[m]}",
            arrowstyle=f"-|>,head_width={(3.6 if hero else 2.6) * scale},head_length={(7 if hero else 5.5) * scale}",
            color=COLOR[m], lw=(3.0 if hero else 1.5) * scale, shrinkA=9 * scale, shrinkB=(15 if hero else 11) * scale,
            zorder=4 if hero else 3, alpha=1 if hero else 0.9, capstyle="round"))

    for m in order:
        x, y, d_m = pts[m]
        hero = m == "replay_kl"
        if len(d_m) > 1:
            ax.add_patch(Ellipse((x, y), 2 * d_m.after_B_B.std(), 2 * d_m.after_B_A.std(),
                                 fc=tint(COLOR[m], 0.82), ec="none", zorder=2))
        if seed_dots and len(d_m) > 1:
            ax.scatter(d_m.after_B_B, d_m.after_B_A, s=22 * scale ** 2, fc="white", ec=COLOR[m],
                       lw=1.2 * scale, zorder=5)
        if hero:
            ax.scatter([x], [y], s=520 * scale ** 2, fc="none", ec=tint(ACCENT, 0.6), lw=1.0 * scale, zorder=5)
        ax.scatter([x], [y], s=(150 if hero else 95) * scale ** 2, fc=COLOR[m], ec="white", lw=2 * scale,
                   zorder=6)

    ax.scatter([start[0]], [start[1]], s=95 * scale ** 2, fc="white", ec=INK, lw=1.8 * scale, zorder=6)
    ax.scatter([start[0]], [start[1]], s=12 * scale ** 2, fc=INK, ec="none", zorder=7)
    return pts, (p, q, y_at)


def annotate_frontier(ax, pts, line, start=None, s=1.0, lab=None, zone=(0.765, 0.705), start_note=True,
                      line_note=True):
    p, q, y_at = line
    sx, sy = start or (START_B, START_A)
    note = "after book A  \u00b7  every method starts here" if start_note else "after book A"
    ax.annotate("", (sx, sy), xytext=(sx, sy + 0.075), arrowprops=dict(arrowstyle="-", color=INK, lw=0.8 * s,
                                                                      shrinkA=0, shrinkB=6 * s))
    ax.text(sx - 0.035, sy + 0.085, note, fontsize=9.6 * s, color=INK, va="bottom", ha="left",
            fontweight="medium", path_effects=HALO)
    lab = lab or {"naive": (6, -14, "right"), "kl_global": (0, -14, "center"), "replay_kl": (-22, -18, "right")}
    for m in order:
        x, y, _ = pts[m]
        dx, dy, ha = lab[m]
        hero_m = m == "replay_kl"
        ax.annotate(SHORT[m], (x, y), xytext=(dx * s, dy * s), textcoords="offset points", ha=ha, va="top",
                    fontsize=(14 if hero_m else 11.5) * s, fontweight="bold" if hero_m else "semibold",
                    color=DEEP[m], path_effects=HALO)
        ax.annotate(f"A {num(y)}   B {num(x)}", (x, y), xytext=(dx * s, (dy - (19 if hero_m else 16)) * s),
                    textcoords="offset points", ha=ha, va="top", fontsize=9.8 * s, family=MONO, color=BODY, path_effects=HALO)
    ax.text(1.0, 1.06, "goal", ha="center", va="bottom", fontsize=8.6 * s, family=MONO, color=INK,
            fontweight="medium")
    if line_note:
        xm = 0.5 * (p[0] + q[0]) - 0.02
        ang = np.degrees(np.arctan2(*(ax.transData.transform(q) - ax.transData.transform(p))[::-1])) - 180
        ax.text(xm, y_at(xm) - 0.05, "straight line through the two baselines", rotation=ang,
                rotation_mode="anchor", ha="center", va="center", fontsize=9.2 * s, color=MUTED, style="italic", path_effects=HALO)
    if zone:
        ax.text(zone[0], zone[1], "beyond the", ha="left", va="baseline", fontsize=18 * s, family=SERIF,
                style="italic", color=ACCENT)
        ax.text(zone[0], zone[1] - 0.065, "trade-off line", ha="left", va="baseline", fontsize=18 * s,
                family=SERIF, style="italic", color=ACCENT)


def header(S, eyebrow, parts, deck, size=34, L=0.75, y0=0.62):
    S.eyebrow(L, y0, eyebrow, square=ACCENT)
    S.runs(L - 0.02, y0 + size / 72 * 0.98, parts, size=size, family=SERIF)
    y = y0 + size / 72 * 0.98 + 0.44
    for line in deck:
        S.text(L, y, line, size=12.2, color=BODY)
        y += 0.25
    return y - 0.25


def footer(S, y, L=0.75, R=None):
    R = R or S.w - 0.75
    S.rule(L, R, y - 0.26)
    S.text(R, y, "github.com/antonio-mastropaolo/qwen-sequential-books", size=7.8, family=MONO, color=FAINT,
           ha="right")
    parts = ["DATA  results/results.csv", model, f"seeds {', '.join(map(str, seeds))}",
             "1,500 held-out questions per book", "mean \u00b1 1 sd"]
    room = R - L - S.width(S.page.texts[-1]) - 0.4
    while True:
        t = S.text(L, y, "  \u00b7  ".join(parts), size=7.8, family=MONO, color=FAINT)
        if S.width(t) <= room or len(parts) == 1:
            break
        t.remove()
        parts.pop()


ITALIC = dict(style="italic", color=ACCENT)


# ------------------------------------------------------------------ hero
def hero():
    S = Sheet(16, 9)
    L, R = 0.75, 15.25

    S.eyebrow(L, 0.70, "Sequential Books  \u00b7  catastrophic forgetting in sequential LoRA fine-tuning",
              square=ACCENT)
    S.eyebrow(R, 0.70, f"{model_short}  \u00b7  {len(seeds)} seeds", color=FAINT).set_ha("right")
    S.runs(L - 0.02, 1.45, [("Learn the second book. ", {}), ("Keep the first.", dict(style="italic",
                                                                                         color=ACCENT))],
           size=50, family=SERIF)
    deck = [
        f"{model_short} learns book A, then a closely related book B on a fresh LoRA. Naive fine-tuning learns B and",
        "erases A. A KL penalty on B\u2019s tokens protects A but stalls B. Rehearsing A, with KL on those rehearsal tokens only, keeps both.",
    ]
    for i, line in enumerate(deck):
        S.text(L, 1.93 + i * 0.27, line, size=13.2, color=BODY)
    S.rule(L, R, 2.5)

    S.text(L, 2.88, "Where every run lands after learning book B", size=13, weight="semibold")
    S.text(L, 3.12, "mean (solid), each seed (ring), \u00b11 sd (halo)  \u00b7  arrows run from the book-A model",
           size=10.2, color=MUTED)
    ax = S.axes(L + 0.5, 3.62, 8.2, 4.2)
    pts, (p, q, y_at) = frontier(ax)
    ax.text(1.08, -0.16, "Book B learned  \u2192", transform=ax.transData, ha="right", va="top", fontsize=10.5,
            color=BODY, fontweight="medium")
    ax.text(-0.03, 1.115, "\u2191  Book A retained", transform=ax.transData, ha="left", va="bottom",
            fontsize=10.5, color=BODY, fontweight="medium")

    annotate_frontier(ax, pts, (p, q, y_at))

    # right column
    cx = 10.55
    S.vrule(cx - 0.45, 2.78, 8.2)
    S.eyebrow(cx, 2.88, "Forgetting of book A  \u00b7  rehearsal + KL", color=MUTED)
    f = mu("replay_kl", "forgetting_A")
    S.text(cx - 0.05, 4.35, num(f), size=104, family=SERIF, color=ACCENT)
    worst = df[df.method == "replay_kl"].forgetting_A.max()
    S.text(cx, 4.76, f"\u00b1 {num(sd('replay_kl', 'forgetting_A'))} over {len(seeds)} seeds; no seed forgets more "
           f"than {num(worst)}.", size=10.8, color=BODY)
    S.text(cx, 5.0, "Below zero means book A got slightly better.", size=10.8, color=MUTED)

    top = max(mu(m, "forgetting_A") for m in order)
    bx, bw = cx + 1.55, 2.55
    for i, m in enumerate(["naive", "kl_global", "replay_kl"]):
        y = 5.42 + i * 0.29
        v = mu(m, "forgetting_A")
        S.text(cx, y, TINY[m], size=10, color=DEEP[m] if m == "replay_kl" else BODY,
               weight="semibold" if m == "replay_kl" else "regular", va="center")
        S.page.add_patch(Rectangle((bx, y - 0.075), bw, 0.15, fc=PAPER, ec="none"))
        if v > 0:
            S.page.add_patch(Rectangle((bx, y - 0.075), bw * v / top, 0.15, fc=COLOR[m], ec="none"))
        else:
            S.page.add_patch(Rectangle((bx - 0.02, y - 0.11), 0.04, 0.22, fc=COLOR[m], ec="none"))
        S.text(bx + bw + 0.12, y, num(v), size=9.6, family=MONO, color=BODY, va="center")

    S.text(cx, 6.5, "Book-A accuracy, stage by stage", size=11.2, weight="semibold")
    ia = S.axes(cx + 0.38, 6.72, 2.55, 1.32)
    frame(ia, (-0.12, 2.12), (-0.04, 1.08), [0, 1, 2], [0, 0.5, 1], ["base", "after A", "after B"],
          ["0", ".5", "1"], grid="y", size=8.4)
    ia.axvspan(1, 2.12, color=PAPER, zorder=0, lw=0)
    for _, r in stage1.iterrows():
        ia.plot([0, 1], [r.base_A, r.after_A_A], color=INK, lw=0.7, alpha=0.18, zorder=2)
    ia.plot([0, 1], [BASE_A, START_A], color=INK, lw=2.0, zorder=3, solid_capstyle="round")
    for m in order:
        y2 = mu(m, "after_B_A")
        ia.plot([1, 2], [START_A, y2], color=COLOR[m], lw=2.8 if m == "replay_kl" else 1.8, zorder=4,
                solid_capstyle="round")
        ia.scatter([2], [y2], s=26, fc=COLOR[m], ec="white", lw=1, zorder=5)
        dy = {"naive": 0, "kl_global": -0.045, "replay_kl": 0.045}[m]
        ia.text(2.2, y2 + dy, f"{num(y2)}", fontsize=8.8, family=MONO, color=BODY, va="center")
        ia.text(2.68, y2 + dy, TINY[m], fontsize=8.8, color=DEEP[m], va="center",
                fontweight="semibold" if m == "replay_kl" else "regular")
    ia.scatter([0, 1], [BASE_A, START_A], s=22, fc=INK, ec="white", lw=1, zorder=5)
    for t in ia.get_xticklabels():
        t.set_fontfamily(SANS)

    S.rule(L, R, 8.5)
    S.text(L, 8.76, f"DATA  results/results.csv  \u00b7  {model}  \u00b7  LoRA r=64  \u00b7  seeds "
           f"{', '.join(map(str, seeds))}  \u00b7  1,500 held-out questions per book  \u00b7  mean \u00b1 1 sd",
           size=8.2, family=MONO, color=FAINT)
    S.text(R, 8.76, "github.com/antonio-mastropaolo/qwen-sequential-books", size=8.2, family=MONO, color=FAINT,
           ha="right")
    S.save("hero")


# ------------------------------------------------------------------ method
BOOK_A, BOOK_B = ACCENT, "#c9bfae"
ICE = "#eef1fb"


def card_stack(ax, x, y, w, h, n=5, fc="white", ec=INK, lw=1.1, step=0.075, zorder=3):
    """A stack of layers seen slightly from above; (x, y) is the top-left of the front card."""
    for i in reversed(range(n)):
        dx, dy = i * step, -i * step
        ax.add_patch(FancyBboxPatch((x + dx, y + dy), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=fc, ec=ec, lw=lw, zorder=zorder + (n - i) * 0.01))


def open_book(ax, cx, cy, w, color, lines=4, zorder=4):
    """An open book drawn from two curved pages."""
    h = w * 0.62
    for side in (-1, 1):
        x0, x1 = cx, cx + side * w / 2
        verts = [(x0, cy - h / 2 + 0.03), (x0 + side * w * 0.18, cy - h / 2 - 0.06), (x1, cy - h / 2 + 0.01),
                 (x1, cy + h / 2), (x0 + side * w * 0.2, cy + h / 2 - 0.05), (x0, cy + h / 2 + 0.03),
                 (x0, cy - h / 2 + 0.03)]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), fc=tint(color, 0.86), ec=color, lw=1.3, zorder=zorder,
                               joinstyle="round"))
        for k in range(lines):
            t = (k + 1) / (lines + 1)
            yy = cy - h / 2 + 0.05 + t * (h - 0.12)
            xa, xb = x0 + side * w * 0.08, x0 + side * w * (0.40 if k == lines - 1 else 0.42)
            ax.plot([xa, xb], [yy - 0.02 * (1 - t), yy - 0.02 * (1 - t)], color=color, lw=1.0, alpha=0.75,
                    solid_capstyle="round", zorder=zorder + 0.1)
    ax.plot([cx, cx], [cy - h / 2 + 0.03, cy + h / 2 + 0.03], color=color, lw=1.3, zorder=zorder + 0.2)


def low_rank(ax, x, y, color, h=0.78, thin=0.12, w=0.78, label=None, zorder=4):
    """LoRA as the product of a tall-thin and a short-wide matrix; (x, y) is the top-left."""
    ax.add_patch(FancyBboxPatch((x, y), thin, h, boxstyle="round,pad=0,rounding_size=0.025", fc=color,
                                ec="none", zorder=zorder))
    ax.text(x + thin + 0.11, y + 0.13, "\u00d7", fontsize=12, color=MUTED, ha="center", va="center",
            zorder=zorder)
    ax.add_patch(FancyBboxPatch((x + thin + 0.22, y + 0.06), w, thin, boxstyle="round,pad=0,rounding_size=0.025",
                                fc=color, ec="none", zorder=zorder))


def snowflake(ax, cx, cy, r, color, lw=1.4, zorder=6):
    for k in range(6):
        a = np.pi / 2 + k * np.pi / 3
        ux, uy = np.cos(a), np.sin(a)
        ax.plot([cx, cx + r * ux], [cy, cy + r * uy], color=color, lw=lw, solid_capstyle="round", zorder=zorder)
        for sgn in (-1, 1):
            b = a + sgn * np.pi / 4
            px, py = cx + 0.6 * r * ux, cy + 0.6 * r * uy
            ax.plot([px, px + 0.32 * r * np.cos(b)], [py, py + 0.32 * r * np.sin(b)], color=color, lw=lw,
                    solid_capstyle="round", zorder=zorder)


def chip(S, x, y, s, fc, color, ec="none", size=10, weight="semibold", family=SANS, strike=False, pad=0.09):
    t = S.text(x, y, s, size=size, color=color, weight=weight, family=family, va="center", zorder=8)
    w = S.width(t)
    S.page.add_patch(FancyBboxPatch((x - pad, y - 0.13), w + 2 * pad, 0.26, boxstyle="round,pad=0,rounding_size=0.13",
                                    fc=fc, ec=ec, lw=1.0, zorder=7))
    if strike:
        S.page.plot([x - 0.02, x + w + 0.02], [y + 0.01, y + 0.01], color=color, lw=1.0, zorder=9)
    return x + w + pad


def arrow(ax, a, b, color=INK, lw=1.3, rad=0.0, style="-|>", ls="-", head=(3.2, 6), shrink=(2, 2), zorder=4):
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}", linestyle=ls,
                                 arrowstyle=f"{style},head_width={head[0]},head_length={head[1]}" if style != "-" else "-",
                                 color=color, lw=lw, shrinkA=shrink[0], shrinkB=shrink[1], zorder=zorder,
                                 capstyle="round", joinstyle="round"))


def method():
    S = Sheet(16, 9.0)
    ax = S.page
    L, R = 0.75, 15.25
    n_b, n_a = 32, int(round(32 * 0.4))

    S.eyebrow(L, 0.70, "Method  \u00b7  two LoRA stages, three stage-2 losses compared", square=ACCENT)
    S.runs(L - 0.02, 1.42, [("Constrain the model ", {}), ("only where book A lives.", dict(style="italic",
                                                                                            color=ACCENT))],
           size=44, family=SERIF)
    S.text(L, 1.9, "Stage 1 teaches book A and freezes the result. Stage 2 trains a fresh adapter on book B, rehearses "
           "book A, and applies KL to the", size=13.2, color=BODY)
    S.text(L, 2.17, "book-A model on the rehearsal tokens alone. Book-B tokens are left free to change.", size=13.2,
           color=BODY)
    S.rule(L, R, 2.55)
    top, bot = 2.95, 7.4

    # stage I
    w1 = 4.2
    ax.add_patch(FancyBboxPatch((L, top), w1, bot - top, boxstyle="round,pad=0,rounding_size=0.18", fc=PAPER,
                                ec="none", zorder=0))
    S.text(L + 0.3, top + 0.62, "I", size=34, family=SERIF, color=INK)
    S.text(L + 0.72, top + 0.42, "Learn book A", size=13.5, weight="semibold")
    S.text(L + 0.72, top + 0.64, "LoRA on the base model, then merged", size=10, color=MUTED)

    card_stack(ax, L + 0.4, 4.2, 1.25, 0.74, n=4, fc="white", ec=INK)
    S.text(L + 1.02, 4.6, model_short, size=10.5, weight="semibold", ha="center", va="center", zorder=9)
    S.text(L + 1.02, 5.18, "base model", size=9.5, color=MUTED, ha="center")
    open_book(ax, L + 1.05, 6.15, 1.15, BOOK_A)
    S.text(L + 1.05, 6.74, "Book A", size=10.5, weight="semibold", ha="center", color=DEEP["replay_kl"])
    S.text(L + 1.05, 6.97, "300 towns \u00d7 5 facts \u00d7 4", size=8.4, color=MUTED, ha="center", family=MONO)

    lx, ly = L + 2.45, 4.95
    low_rank(ax, lx, ly, tint(BOOK_A, 0.2), h=0.8, w=0.66)
    S.text(lx, ly + 1.12, "LoRA A", size=10.5, weight="semibold")
    S.text(lx, ly + 1.33, "r=64  \u00b7  CE", size=8.6, color=MUTED, family=MONO)
    arrow(ax, (L + 1.75, 4.75), (lx - 0.1, ly + 0.25), color=INK, rad=-0.2)
    arrow(ax, (L + 1.72, 6.1), (lx - 0.1, ly + 0.6), color=INK, rad=0.2)

    # the frozen book-A model sits between the stages: stage 1 makes it, stage 2 builds on it
    bx, by = 5.4, 4.62
    card_stack(ax, bx, by, 1.5, 0.92, n=5, fc=ICE, ec=INK, lw=1.2)
    snowflake(ax, bx + 0.32, by + 0.46, 0.17, INK, lw=1.2, zorder=9)
    S.text(bx + 0.6, by + 0.36, "Book-A", size=11, weight="bold", va="center", zorder=9)
    S.text(bx + 0.6, by + 0.59, "model", size=11, weight="bold", va="center", zorder=9)
    S.text(bx + 0.02, by + 1.3, "merged and frozen", size=9.4, color=BODY)
    S.text(bx + 0.02, by + 1.5, "the KL reference", size=9.4, color=MUTED)
    arrow(ax, (lx + 1.0, ly + 0.2), (bx - 0.08, by + 0.4), color=INK, shrink=(4, 2))
    S.text(lx + 1.08, ly - 0.06, "merge", size=8.4, color=MUTED, family=MONO)

    # stage II
    x2 = 7.3
    ax.add_patch(FancyBboxPatch((x2, top), R - x2, bot - top, boxstyle="round,pad=0,rounding_size=0.18",
                                fc=tint(ACCENT, 0.955), ec="none", zorder=0))
    S.text(x2 + 0.3, top + 0.62, "II", size=34, family=SERIF, color=ACCENT)
    S.text(x2 + 0.86, top + 0.42, "Learn book B, protect book A", size=13.5, weight="semibold",
           color=DEEP["replay_kl"])
    S.text(x2 + 0.86, top + 0.64, "a fresh LoRA B on top of the frozen book-A model", size=10, color=MUTED)

    tx, ty, cell, gap, split = x2 + 0.45, 4.12, 0.078, 0.026, 0.13
    S.text(tx, ty - 0.18, "EVERY TRAINING BATCH, DRAWN TO SCALE", size=8.2, family=MONO, color=MUTED,
           weight="medium")
    for i in range(n_b + n_a):
        is_a = i >= n_b
        ax.add_patch(FancyBboxPatch((tx + i * (cell + gap) + (split if is_a else 0), ty), cell, 0.34,
                                    boxstyle="round,pad=0,rounding_size=0.018",
                                    fc=BOOK_A if is_a else BOOK_B, ec="none", zorder=4))
    xb1 = tx + n_b * (cell + gap) - gap
    xa0 = tx + n_b * (cell + gap) + split
    xa1 = xa0 + n_a * (cell + gap) - gap
    for x0, x1 in ((tx, xb1), (xa0, xa1)):
        ax.plot([x0, x0, x1, x1], [ty + 0.44, ty + 0.5, ty + 0.5, ty + 0.44], color=FAINT, lw=1.0, zorder=4)
    S.text(tx, ty + 0.74, f"{n_b} book-B statements", size=10.5, weight="semibold", color=INK)
    S.text(xa0, ty + 0.74, f"{n_a} book-A rehearsal statements", size=10.5, weight="semibold",
           color=DEEP["replay_kl"])
    cy = ty + 1.1
    e = chip(S, tx + 0.09, cy, "CE", "white", INK, ec=HAIR)
    chip(S, e + 0.15, cy, "no KL", "none", FAINT, ec=HAIR, weight="medium", strike=True)
    e = chip(S, xa0 + 0.09, cy, "CE", "white", INK, ec=HAIR)
    kl_x = e + 0.15
    chip(S, kl_x, cy, "+ KL to book-A model", ACCENT, "white")

    # the model being trained: frozen book-A weights plus the fresh adapter
    mx, my = tx + 0.02, 6.25
    card_stack(ax, mx, my, 1.55, 0.62, n=4, fc=ICE, ec=INK, lw=1.1, step=0.06)
    snowflake(ax, mx + 0.25, my + 0.31, 0.13, INK, lw=1.1, zorder=9)
    S.text(mx + 0.47, my + 0.36, "book-A model", size=9.4, weight="semibold", va="center", zorder=9)
    S.text(mx + 1.86, my + 0.33, "+", size=16, color=MUTED, ha="center", va="center")
    lbx = mx + 2.12
    low_rank(ax, lbx, my - 0.1, INK, h=0.74, thin=0.11, w=0.6)
    S.text(lbx + 1.12, my + 0.2, "LoRA B", size=10.5, weight="semibold", va="center")
    S.text(lbx + 1.12, my + 0.42, "fresh  \u00b7  r=64", size=8.6, color=MUTED, family=MONO, va="center")
    S.text(lbx + 1.12, my + 0.62, "the only weights that train", size=8.6, color=MUTED, va="center")
    arrow(ax, (bx + 1.85, by + 0.62), (mx - 0.12, my + 0.3), color=INK, rad=0.12, shrink=(4, 3))
    arrow(ax, (e - 0.3, cy + 0.17), (lbx + 0.3, my - 0.18), color=INK, rad=0.0, shrink=(2, 4))
    arrow(ax, (tx + 0.35, cy + 0.17), (lbx + 0.06, my - 0.16), color=INK, rad=0.0, shrink=(2, 4))

    # the KL reference: the same weights with the adapter switched off
    rx = kl_x + 0.3
    arrow(ax, (mx + 1.6, my - 0.16), (rx, cy + 0.17), color="white", lw=6, rad=-0.25, style="-", shrink=(3, 3),
          zorder=5)
    arrow(ax, (mx + 1.6, my - 0.16), (rx, cy + 0.17), color=ACCENT, lw=1.5, rad=-0.25, ls=(0, (3, 2.5)),
          shrink=(3, 3), zorder=6)
    S.text(rx + 0.3, 5.56, "reference = the same model", size=9.2, color=ACCENT, style="italic")
    S.text(rx + 0.3, 5.74, "with LoRA B switched off", size=9.2, color=ACCENT, style="italic")
    S.text(rx + 0.3, 5.92, "no second copy of weights", size=9.2, color=MUTED)

    ex, ew, ey = 13.95, 1.1, 6.1
    ax.add_patch(FancyBboxPatch((ex, ey), ew, 1.14, boxstyle="round,pad=0,rounding_size=0.12", fc="white",
                                ec=HAIR, lw=1.0, zorder=3))
    S.text(ex + ew / 2, ey + 0.27, "Evaluate", size=10.5, weight="semibold", ha="center")
    for k, (lbl, c) in enumerate((("A", BOOK_A), ("B", BOOK_B))):
        yy = ey + 0.52 + k * 0.36
        ax.add_patch(Circle((ex + 0.27, yy), 0.12, fc=c, ec="none", zorder=4))
        S.text(ex + 0.27, yy + 0.005, lbl, size=8.4, weight="bold", color="white" if k == 0 else INK, ha="center",
               va="center", zorder=5)
        S.text(ex + 0.5, yy - 0.05, "1,500", size=8.6, family=MONO, color=INK, va="center")
        S.text(ex + 0.5, yy + 0.12, "questions", size=7.4, color=MUTED, va="center")
    arrow(ax, (lbx + 2.8, my + 0.5), (ex - 0.06, my + 0.5), color=INK, shrink=(2, 2))

    S.rule(L, R, 7.8)
    y = 8.3
    x = S.runs(L, y, [("loss  =  ", dict(color=MUTED)), ("CE", dict(weight="semibold")),
                      (" (book B)", dict(color=BODY)), ("    +    ", dict(color=MUTED)),
                      ("CE", dict(weight="semibold", color=DEEP["replay_kl"])),
                      (" (book-A rehearsal)", dict(color=BODY)), ("    +    ", dict(color=MUTED)),
                      ("kl_w \u00b7 KL", dict(weight="semibold", color=DEEP["replay_kl"])),
                      (" (book-A model  \u2016  model)", dict(color=BODY))], size=14, family=SANS)
    S.text(x + 0.3, y, "on the rehearsal tokens only,  kl_w = 1.0", size=12.5, color=ACCENT, style="italic",
           family=SANS)
    S.text(L, 8.66, "Baselines share stage 1, the fresh LoRA, learning rate and steps:    naive = CE (book B)    "
           "\u00b7    KL on new-book tokens = CE (book B) + kl_w \u00b7 KL on those same book-B tokens",
           size=9.6, color=MUTED)
    S.save("method")


# ------------------------------------------------------------------ results
def accuracy_bars():
    S = Sheet(12, 6.3)
    L, R = 0.75, 11.25
    y = header(S, "Results  \u00b7  accuracy after stage 2",
               [("Only one method ", {}), ("keeps both books.", ITALIC)],
               ["Accuracy on 1,500 held-out questions per book after learning book B. Bar: mean of 3 seeds;",
                "ring: each seed; whisker: \u00b11 sd. The tick on the left panel marks book A before stage 2."])
    names_w = 2.1
    pw, gapw = (R - L - names_w - 0.5) / 2, 0.5
    rows = list(reversed(order))
    top = y + 0.75
    row_h, bar_h = 0.86, 0.34
    S.page.add_patch(FancyBboxPatch((L - 0.2, top + 0.06), R - L + 0.4, row_h - 0.12,
                                    boxstyle="round,pad=0,rounding_size=0.1", fc=tint(ACCENT, 0.95), ec="none",
                                    zorder=0))
    for k, (col, title) in enumerate((("after_B_A", "Book A retained"), ("after_B_B", "Book B learned"))):
        x0 = L + names_w + k * (pw + gapw)
        S.text(x0, top - 0.22, title, size=12, weight="semibold")
        S.text(x0 + pw, top - 0.22, "higher is better", size=8.6, color=FAINT, ha="right", family=MONO)
        for t in (0, 0.25, 0.5, 0.75, 1):
            S.vrule(x0 + t * pw, top, top + row_h * 3, color=HAIR, lw=0.8)
            S.text(x0 + t * pw, top + row_h * 3 + 0.22, FRAC([t])[0], size=8.4, family=MONO, color=MUTED,
                   ha="center")
        for i, m in enumerate(rows):
            yc = top + row_h * (i + 0.5)
            v, e = mu(m, col), sd(m, col)
            hero_m = m == "replay_kl"
            S.page.add_patch(Rectangle((x0, yc - bar_h / 2), pw, bar_h, fc=PAPER, ec="none", zorder=1))
            S.page.add_patch(Rectangle((x0, yc - bar_h / 2), v * pw, bar_h, fc=COLOR[m], ec="none", zorder=2))
            S.page.plot([x0 + (v - e) * pw, x0 + (v + e) * pw], [yc, yc], color=INK, lw=1.1, zorder=4,
                        solid_capstyle="butt")
            for sv in df[df.method == m][col]:
                S.page.scatter([x0 + sv * pw], [yc], s=26, fc="white", ec=INK, lw=0.9, zorder=5)
            S.text(x0 + 0.1, yc, num(v), size=11 if hero_m else 10, family=MONO,
                   weight="semibold" if hero_m else "medium", color="white", va="center_baseline", zorder=6)
            if k == 0:
                S.text(L, yc - 0.02, SHORT[m], size=11.5 if hero_m else 10.8, va="center_baseline",
                       weight="bold" if hero_m else "medium", color=DEEP[m])
        if col == "after_B_A":
            xs = x0 + START_A * pw
            S.page.plot([xs, xs], [top - 0.05, top + row_h * 3 + 0.05], color=INK, lw=1.2, ls=(0, (2, 2)), zorder=7)
            S.text(xs, top + row_h * 3 + 0.48, f"before stage 2: {num(START_A)}", size=8.6, color=INK, ha="center",
                   family=MONO)
    footer(S, 6.02)
    S.save("accuracy_bars")


def forgetting():
    S = Sheet(12, 6.2)
    L, R = 0.75, 11.25
    y = header(S, "Results  \u00b7  forgetting of book A",
               [("Naive fine-tuning drops book A ", {}), (f"from {num(START_A)} to {num(mu('naive', 'after_B_A'))}.",
                                                          ITALIC)],
               ["Book-A accuracy after stage 2 (dot: mean of 3 seeds; rings: each seed) against the book-A model",
                "before it (dashed). Forgetting = before \u2212 after; below zero means book A got slightly better."])
    names_w, fw = 2.2, 1.7
    x0, x1 = L + names_w, R - fw - 0.35
    pw = x1 - x0
    top, row_h = y + 0.85, 0.9
    rows = list(reversed(order))
    S.text(x0, top - 0.3, "Book-A accuracy", size=12, weight="semibold")
    S.text(R, top - 0.3, "forgetting", size=12, weight="semibold", ha="right")
    for t in (0, 0.25, 0.5, 0.75, 1):
        S.vrule(x0 + t * pw, top, top + row_h * 3, color=HAIR, lw=0.8)
        S.text(x0 + t * pw, top + row_h * 3 + 0.22, FRAC([t])[0], size=8.4, family=MONO, color=MUTED, ha="center")
    X = lambda v: x0 + v * pw  # noqa: E731
    xs = X(START_A)
    S.page.plot([xs, xs], [top - 0.02, top + row_h * 3 + 0.02], color=INK, lw=1.2, ls=(0, (2, 2)), zorder=4)
    S.text(xs, top - 0.08, f"before stage 2: {num(START_A)}", size=8.4, family=MONO, color=INK, ha="right",
           va="bottom")
    for i, m in enumerate(rows):
        yc = top + row_h * (i + 0.5)
        hero_m = m == "replay_kl"
        a, b = START_A, mu(m, "after_B_A")
        if b < a - 0.03:
            S.page.add_patch(FancyArrowPatch((X(a), yc), (X(b), yc), arrowstyle="-|>,head_width=4,head_length=7",
                                             color=COLOR[m], lw=2.4, shrinkA=2, shrinkB=8, zorder=3))
        for v in df[df.method == m].after_B_A:
            S.page.scatter([X(v)], [yc], s=24, fc="white", ec=COLOR[m], lw=1.1, zorder=5)
        S.page.scatter([X(b)], [yc], s=150 if hero_m else 100, fc=COLOR[m], ec="white", lw=2, zorder=6)
        right = b > a - 0.03
        S.text(X(b) + (0.2 if right else -0.2), yc, num(b), size=10 if hero_m else 9.4, family=MONO,
               color=DEEP[m] if hero_m else BODY, weight="semibold" if hero_m else "regular",
               ha="left" if right else "right", va="center_baseline")
        S.text(L, yc, SHORT[m], size=11.5 if hero_m else 10.8, va="center_baseline",
               weight="bold" if hero_m else "medium", color=DEEP[m])
        f = mu(m, "forgetting_A")
        S.text(R, yc + 0.12, num(f), size=34 if hero_m else 28, family=SERIF, color=ACCENT if hero_m else DEEP[m],
               ha="right")
        S.text(R, yc + 0.34, f"\u00b1 {num(sd(m, 'forgetting_A'))}", size=8.4, family=MONO, color=MUTED, ha="right")
    S.vrule(R - fw + 0.15, top, top + row_h * 3, color=HAIR)
    footer(S, 5.92)
    S.save("forgetting")


def tradeoff():
    S = Sheet(11, 10.15)
    L = 0.75
    y = header(S, "Results  \u00b7  the stability\u2013plasticity plane",
               [("Rehearsal + KL lands ", {}), ("beyond the trade-off.", ITALIC)],
               ["Book A retained against book B learned after stage 2. Mean (solid), seeds (rings), \u00b11 sd",
                "(halo). Arrows start at the book-A model; the dashed line joins the two baselines."], size=32)
    ax = S.axes(L + 0.5, y + 0.95, 9.5, 6.2)
    pts, line = frontier(ax)
    annotate_frontier(ax, pts, line, zone=(0.73, 0.70))
    ax.text(1.08, -0.1, "Book B learned  \u2192", ha="right", va="top", fontsize=10.5, color=BODY,
            fontweight="medium")
    ax.text(-0.03, 1.1, "\u2191  Book A retained", ha="left", va="bottom", fontsize=10.5, color=BODY,
            fontweight="medium")
    footer(S, 9.88, R=10.25)
    S.save("tradeoff")


def per_seed():
    S = Sheet(16, 7.95)
    L, R = 0.75, 15.25
    y = header(S, "Results  \u00b7  seed by seed",
               [("The same picture ", {}), ("in every seed.", ITALIC)],
               ["One panel per seed: each method\u2019s book-A and book-B accuracy after stage 2, with arrows from "
                "that seed\u2019s book-A model. Vertical: book A retained; horizontal: book B learned."], size=32)
    pw, gap = (R - L - 2 * 0.55 - 0.4) / 3, 0.55
    for k, sd_ in enumerate(seeds):
        d = df[df.seed == sd_]
        st = (d.after_A_B.iloc[0], d.after_A_A.iloc[0])
        x0 = L + 0.4 + k * (pw + gap)
        S.text(x0 - 0.4, y + 0.62, f"Seed {sd_}", size=13, weight="semibold")
        f = d.set_index("method").forgetting_A
        S.text(x0 - 0.4 + pw, y + 0.62, f"rehearsal + KL forgets {num(f['replay_kl'])}", size=8.6, family=MONO,
               color=ACCENT, ha="right")
        ax = S.axes(x0, y + 1.15, pw - 0.4, 4.25)
        pts, line = frontier(ax, data=d, start=st, scale=0.9)
        annotate_frontier(ax, pts, line, start=st, s=0.9, start_note=False, line_note=False, zone=None,
                          lab={"naive": (6, -12, "right"), "kl_global": (0, -14, "center"),
                               "replay_kl": (-12, -16, "center")})
        ax.text(1.08, -0.12, "Book B learned  \u2192", ha="right", va="top", fontsize=9, color=BODY,
                fontweight="medium")
    footer(S, 7.72)
    S.save("per_seed")


if __name__ == "__main__":
    wanted = args.only.split(",")
    for name in FIGURES:
        if name in wanted and name in globals():
            globals()[name]()
