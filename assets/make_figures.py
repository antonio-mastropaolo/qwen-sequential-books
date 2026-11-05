#!/usr/bin/env python3
"""Build every README figure from results/results.csv.

    python assets/make_figures.py                  # all figures, PNG + SVG, into assets/
    python assets/make_figures.py --only hero      # just one

Figures: hero, method, accuracy_bars, forgetting, tradeoff, per_seed.

Design system (from the redesign brief): opaque white canvas, #E5E7EB grid behind the data, near-black
#111827 headings and #4B5563 secondary text; one colour per method (naive #D97757 terracotta, KL on
new-book tokens #5E8B7E sage, rehearsal + KL #4F46E5 indigo) and neutral grey for the book-A model.
Headlines in Instrument Serif, prose and axes in Inter, every number in IBM Plex Mono. Points are
labelled directly (no legends); ±1 sd is a translucent halo on the 2D planes and a whisker on 1D plots.

Needs numpy, pandas, matplotlib and Pillow. Fonts load from assets/fonts (SIL OFL), so output does not
depend on the machine. Rebuilding with the same CSV and library versions gives byte-identical files:
PNGs carry no timestamp or software tag, and SVG ids are hashed with a fixed salt.

Every number drawn on a results figure is read from the CSV. The method figure only shows the
training configuration (batch sizes, LoRA rank, question counts) described in the README.
"""
import argparse
import glob
import io
import logging
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = ["hero", "method", "accuracy_bars", "forgetting", "tradeoff", "per_seed"]

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--csv", default=os.path.join(HERE, "..", "results", "results.csv"))
ap.add_argument("--out", default=HERE)
ap.add_argument("--only", default=",".join(FIGURES), help="comma list of figure names")
ap.add_argument("--dpi", type=int, default=200)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# ---------------------------------------------------------------- tokens

FONT_DIR = os.path.join(HERE, "fonts")
for f in sorted(glob.glob(os.path.join(FONT_DIR, "*.ttf"))):
    font_manager.fontManager.addfont(f)
font_manager.fontManager.ttflist = [
    f for f in font_manager.fontManager.ttflist
    if f.name not in ("Inter", "Instrument Serif", "IBM Plex Mono") or os.path.dirname(f.fname) == FONT_DIR]
SANS, SERIF, MONO = "Inter", "Instrument Serif", "IBM Plex Mono"

CANVAS = "#FFFFFF"     # brief says #FAFAFA; pure opaque white keeps GitHub light and dark mode clean
GRID = "#E5E7EB"
INK = "#111827"
BODY = "#4B5563"
AXIS = "#6B7280"
ORIGIN = "#9CA3AF"
ORIGIN_EDGE = "#4B5563"
NAIVE, KLG, REH = "#D97757", "#5E8B7E", "#4F46E5"

COLOR = {"naive": NAIVE, "kl_global": KLG, "replay_kl": REH}
METHODS = ["naive", "kl_global", "replay_kl"]
NAME = {"naive": "Naive fine-tuning", "kl_global": "KL on new-book tokens", "replay_kl": "Rehearsal + KL"}

# type scale, in points at 200 dpi (the brief's px sizes on a 3200 px canvas)
PX = 72 / 200
T_HERO, T_PROSE, T_AXIS, T_MONO = 110 * PX, 44 * PX, 32 * PX, 28 * PX
T_TITLE = 30            # headline on the secondary figures
T_LABEL = 14            # method names on the planes

plt.rcParams.update({
    "font.family": SANS, "font.size": T_AXIS, "text.color": INK,
    "axes.labelcolor": AXIS, "axes.edgecolor": GRID, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.major.pad": 10, "ytick.major.pad": 10,
    "figure.facecolor": CANVAS, "savefig.facecolor": CANVAS, "axes.facecolor": "none",
    "svg.fonttype": "path", "svg.hashsalt": "sequential-books", "axes.unicode_minus": True,
    "path.simplify": False,
})

# ---------------------------------------------------------------- data

df = pd.read_csv(args.csv)
df = df[~df.smoke_test.astype(str).str.lower().eq("true")]
ORDER = [m for m in METHODS if m in set(df.method)]
SEEDS = sorted(df.seed.unique())
COLS = ["base_A", "base_B", "after_A_A", "after_A_B", "after_B_A", "after_B_B", "forgetting_A"]
stats = df.groupby("method")[COLS].agg(["mean", "std"]).reindex(ORDER)  # sample sd, as in summary.md
stage1 = df.drop_duplicates("seed").set_index("seed").sort_index()
MODEL = df.model.iloc[0].split("/")[-1]
BASE_A, BASE_B = df.base_A.mean(), df.base_B.mean()
START_A, START_B = stage1.after_A_A.mean(), stage1.after_A_B.mean()
START_A_SD, START_B_SD = stage1.after_A_A.std(), stage1.after_A_B.std()


def mu(m, c):
    return stats.loc[m, (c, "mean")]


def sd(m, c):
    return stats.loc[m, (c, "std")]


def run(m, seed):
    return df[(df.method == m) & (df.seed == seed)].iloc[0]


def num(v, nd=3, sign=False):
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "\u2212")


def pm(m, c):
    return f"{num(mu(m, c))}\u00b1{num(sd(m, c))}"


def tint(color, amount):
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


# ---------------------------------------------------------------- page helpers

class Sheet:
    """A figure laid out in inches from the top-left corner."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.page = self.fig.add_axes([0, 0, 1, 1])
        self.page.set_xlim(0, w)
        self.page.set_ylim(h, 0)
        self.page.axis("off")
        self.page.set_zorder(-1)

    def axes(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h], **kw)

    def text(self, x, y, s, size=T_PROSE, color=INK, family=SANS, weight="regular", style="normal",
             ha="left", va="baseline", **kw):
        return self.page.text(x, y, s, fontsize=size, color=color, family=family, fontweight=weight,
                              fontstyle=style, ha=ha, va=va, **kw)

    def width(self, t):
        return t.get_window_extent(self.fig.canvas.get_renderer()).width / self.fig.dpi

    def measure(self, s, **kw):
        t = self.text(0, 0, s, **kw)
        w = self.width(t)
        t.remove()
        return w

    def runs(self, x, y, parts, **common):
        for s, kw in parts:
            x += self.width(self.text(x, y, s, **{**common, **kw}))
        return x

    def para(self, x, y, s, width, size=T_PROSE, lh=1.4, **kw):
        """Wrap prose to a measured width. Returns the baseline y after the last line."""
        words, line, lines = s.split(), "", []
        for w in words:
            trial = (line + " " + w).strip()
            if line and self.measure(trial, size=size, **kw) > width:
                lines.append(line)
                line = w
            else:
                line = trial
        lines.append(line)
        step = size * lh / 72
        for i, ln in enumerate(lines):
            self.text(x, y + i * step, ln, size=size, **kw)
        return y + (len(lines) - 1) * step

    def rule(self, x0, x1, y, color=GRID, lw=1.0, **kw):
        self.page.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", **kw)

    def vrule(self, x, y0, y1, color=GRID, lw=1.0, **kw):
        self.page.plot([x, x], [y0, y1], color=color, lw=lw, solid_capstyle="butt", **kw)

    def box(self, x, y, w, h, fc, ec="none", lw=1.0, r=0.08, z=0):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                           lw=lw, zorder=z)
        self.page.add_patch(p)
        return p

    def save(self, name):
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=args.dpi, metadata={"Software": None})
        Image.open(buf).convert("RGB").save(os.path.join(args.out, name + ".png"), optimize=True)
        self.fig.savefig(os.path.join(args.out, name + ".svg"), format="svg",
                         metadata={"Date": None, "Creator": None})
        plt.close(self.fig)
        print("wrote", os.path.join(args.out, name + ".png"))


def header(S, title, deck, L, width, y=1.0):
    S.text(L, y, title, size=T_TITLE, family=SERIF)
    return S.para(L, y + 0.5, deck, width, size=T_PROSE * 0.8, color=BODY)


def footer(S, L, R, y, note="mean \u00b1 1 sd over seeds"):
    S.rule(L, R, y - 0.3)
    S.runs(L, y, [("DATA  ", {"color": ORIGIN}), ("results/results.csv", {"color": BODY}),
                  (f"   \u00b7   {MODEL}   \u00b7   seeds {', '.join(str(s) for s in SEEDS)}   \u00b7   {note}", {})],
           size=T_MONO * 0.85, family=MONO, color=AXIS)
    S.text(R, y, "github.com/antonio-mastropaolo/qwen-sequential-books", size=T_MONO * 0.85, family=MONO,
           color=AXIS, ha="right")


def bezier(ax, a, b, color, rad, lw=6 * PX, alpha=0.6, shrink=(9, 13), head=1.0, z=4, ls="-"):
    """Smooth quadratic Bezier arrow (arc3) at 60 % opacity, as the brief asks."""
    p = FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                        arrowstyle=f"-|>,head_length={0.75 * head},head_width={0.42 * head}",
                        color=color, lw=lw, alpha=alpha, shrinkA=shrink[0], shrinkB=shrink[1], zorder=z,
                        capstyle="round", joinstyle="round", mutation_scale=12, ls=ls)
    ax.add_patch(p)
    return p


def plane_axes(ax, size=T_AXIS, step=0.1, label_every=0.2, xt="BOOK B ACCURACY  \u2192  LEARNED",
               yt="BOOK A ACCURACY  \u2192  RETAINED", lo=-0.03, hi=1.03, titles=True):
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    grid = np.round(np.arange(0, 1 + 1e-9, step), 3)
    for t in grid:
        ax.axvline(t, color=GRID, lw=1.0, zorder=0)
        ax.axhline(t, color=GRID, lw=1.0, zorder=0)
    ticks = np.round(np.arange(0, 1 + 1e-9, label_every), 3)
    lab = [f"{t:.1f}" for t in ticks]
    ax.set_xticks(ticks, lab, family=SANS, fontweight="medium", fontsize=size * 0.85, color=AXIS)
    ax.set_yticks(ticks, lab, family=SANS, fontweight="medium", fontsize=size * 0.85, color=AXIS)
    for s in ax.spines.values():
        s.set_visible(False)
    if titles:
        ax.set_xlabel(xt, family=SANS, fontweight="medium", fontsize=size, color=AXIS, labelpad=12)
        ax.set_ylabel(yt, family=SANS, fontweight="medium", fontsize=size, color=AXIS, labelpad=12)


def point(ax, b, a, color, s=150, edge="white", z=8, lw=2.0):
    ax.scatter([b], [a], s=s, color=color, edgecolor=edge, linewidth=lw, zorder=z)


def halo(ax, b, a, sb, sa, color, alpha=0.22, z=6):
    ax.add_patch(Ellipse((b, a), 2 * sb, 2 * sa, fc=color, ec="none", alpha=alpha, zorder=z))
    ax.add_patch(Ellipse((b, a), 2 * sb, 2 * sa, fc="none", ec=color, lw=1.0, alpha=0.45, zorder=z))


def tag(ax, b, a, name, coords, color, ha="left", va="top", size=T_LABEL, mono=T_MONO, dx=0, dy=0):
    """Direct label: method name (Inter SemiBold) over 'A: 0.xxx  B: 0.xxx' (Plex Mono)."""
    off = 1.55 * mono
    y0, y1 = (0, -off) if va == "top" else (off, 0)
    kw = dict(xycoords="data", textcoords="offset points", ha=ha, va="baseline", zorder=12, color=color)
    ax.annotate(name, (b, a), xytext=(dx, dy + y0 - (size if va == "top" else 0)), family=SANS,
                fontweight="semibold", fontsize=size, **kw)
    ax.annotate(coords, (b, a), xytext=(dx, dy + y1 - (size if va == "top" else 0)), family=MONO,
                fontsize=mono, **kw)


def coords(a, b):
    return f"A: {num(a)}  B: {num(b)}"


# where each terminal label sits relative to its point, and how each path bends
LABEL = {"replay_kl": dict(ha="right", va="top", dx=-4, dy=-16),
         "kl_global": dict(ha="center", va="top", dx=0, dy=-18),
         "naive": dict(ha="right", va="top", dx=6, dy=-16)}
BEND = {"replay_kl": -0.05, "kl_global": 0.10, "naive": 0.22}

# ---------------------------------------------------------------- hero

def hero():
    W, H = 16, 8
    S = Sheet(W, H)
    L, CW = 0.8, 5.25            # narrative column (grid columns 1-4)
    PL, PR, PT, PB = 7.35, 15.3, 0.6, 6.75   # data column (grid columns 5-12)

    S.text(L, 1.3, "Learn the new book,", size=T_HERO, family=SERIF)
    S.text(L, 1.97, "keep the old one.", size=T_HERO, family=SERIF)
    y = S.para(L, 2.75,
               f"{MODEL} learns book A, then a closely related book B on a fresh LoRA adapter. "
               "Naive fine-tuning erases A, and a KL penalty on B\u2019s tokens saves A but stalls B. "
               "Rehearsing A, with KL on the rehearsal tokens only, keeps A and learns most of B.",
               CW, size=T_PROSE, color=BODY)

    # Distill-style table: exact mean ± sd for the four states
    ty = y + 0.62
    cols = [L + 3.0, L + 4.15, L + CW]
    S.text(L, ty, "STATE", size=T_MONO * 0.9, family=MONO, color=AXIS)
    for x, h in zip(cols, ["BOOK A", "BOOK B", "FORGET A"]):
        S.text(x, ty, h, size=T_MONO * 0.9, family=MONO, color=AXIS, ha="right")
    S.rule(L, L + CW, ty + 0.14, color=INK, lw=1.0)
    rows = [("Book-A model", ORIGIN_EDGE, f"{num(START_A)}\u00b1{num(START_A_SD)}",
             f"{num(START_B)}\u00b1{num(START_B_SD)}", "\u2014")]
    rows += [(NAME[m] if m != "kl_global" else "KL on B tokens", COLOR[m], pm(m, "after_B_A"),
              pm(m, "after_B_B"), pm(m, "forgetting_A")) for m in ORDER]
    for i, (nm, c, a, b, f) in enumerate(rows):
        yy = ty + 0.48 + i * 0.4
        S.page.scatter([L + 0.07], [yy - 0.065], s=46, color=ORIGIN if i == 0 else c,
                       edgecolor=ORIGIN_EDGE if i == 0 else "none", lw=1.2)
        S.text(L + 0.24, yy, nm, size=T_MONO * 1.05, weight="semibold", color=c)
        for x, v in zip(cols, [a, b, f]):
            S.text(x, yy, v, size=T_MONO, family=MONO, color=c, ha="right")
        if i < len(rows) - 1:
            S.rule(L, L + CW, yy + 0.15, color=GRID)
    S.rule(L, L + CW, ty + 0.48 + (len(rows) - 1) * 0.4 + 0.15, color=INK, lw=1.0)
    S.text(L, ty + 0.48 + len(rows) * 0.4 + 0.05,
           f"Accuracy on 1,500 held-out questions per book. Mean \u00b1 sd, {len(SEEDS)} seeds.",
           size=T_MONO * 0.85, family=MONO, color=AXIS)

    # the plane
    ax = S.axes(PL, PT, PR - PL, PB - PT)
    plane_axes(ax, hi=1.09)
    # stage 1: base model -> book-A model (quiet, dashed)
    ax.plot([BASE_B, START_B], [BASE_A, START_A], color=ORIGIN, lw=1.4, ls=(0, (3, 3)), zorder=3)
    point(ax, BASE_B, BASE_A, "white", s=70, edge=ORIGIN_EDGE, lw=1.4)
    tag(ax, BASE_B, BASE_A, "Base model", coords(BASE_A, BASE_B), BODY, ha="left", va="bottom",
        dx=12, dy=4, size=T_LABEL * 0.85)
    ax.annotate("stage 1 \u00b7 learn book A", ((BASE_B + START_B) / 2, (BASE_A + START_A) / 2),
                xytext=(7, 0), textcoords="offset points", rotation=90, ha="center", va="top",
                fontsize=T_MONO * 0.85, family=MONO, color=AXIS,
                rotation_mode="anchor")
    for m in ORDER:
        b, a = mu(m, "after_B_B"), mu(m, "after_B_A")
        bezier(ax, (START_B, START_A), (b, a), COLOR[m], BEND[m])
    for m in ORDER:
        b, a = mu(m, "after_B_B"), mu(m, "after_B_A")
        halo(ax, b, a, sd(m, "after_B_B"), sd(m, "after_B_A"), COLOR[m])
        point(ax, b, a, COLOR[m])
        tag(ax, b, a, NAME[m], coords(a, b), COLOR[m], **LABEL[m])
    halo(ax, START_B, START_A, START_B_SD, START_A_SD, ORIGIN, alpha=0.25)
    point(ax, START_B, START_A, ORIGIN, s=190, edge=ORIGIN_EDGE, lw=2.0, z=9)
    tag(ax, START_B, START_A, "Book-A model \u00b7 every run starts here", coords(START_A, START_B),
        ORIGIN_EDGE, ha="left", va="bottom", dx=4, dy=22)

    footer(S, L, PR, 7.62)
    S.save("hero")


# ---------------------------------------------------------------- method

def blocks(S, x, y, n, fc, ec, w=0.125, gap=0.035, h=0.36):
    for i in range(n):
        S.page.add_patch(Rectangle((x + i * (w + gap), y), w, h, fc=fc, ec=ec, lw=0.9, zorder=3))
    return x + n * (w + gap) - gap


def bracket(S, x0, x1, y, color, text, size=T_MONO, depth=0.09, lw=1.6):
    S.page.plot([x0, x0, x1, x1], [y, y + depth, y + depth, y], color=color, lw=lw, zorder=3,
                solid_joinstyle="miter")
    S.text((x0 + x1) / 2, y + depth + 0.24, text, size=size, family=MONO, color=color, ha="center")


def card(S, x, y, w, h, title, sub, fc="white", ec=INK, tcolor=INK, lw=1.3):
    S.box(x, y, w, h, fc=fc, ec=ec, lw=lw, z=2)
    S.text(x + 0.2, y + 0.4, title, size=T_LABEL, weight="semibold", color=tcolor)
    S.text(x + 0.2, y + 0.7, sub, size=T_MONO * 0.9, family=MONO, color=BODY)


def page_arrow(S, a, b, color=INK, lw=1.6, rad=0.0, alpha=1.0):
    S.page.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-|>,head_length=0.7,head_width=0.38", color=color,
                                     lw=lw, alpha=alpha, mutation_scale=12, zorder=4, shrinkA=0, shrinkB=0))


def method():
    W, H = 16, 8.9
    S = Sheet(W, H)
    L, R = 0.8, 15.2
    header(S, "Constrain the model only where book A lives.",
           "Stage 1 teaches book A and freezes it. Stage 2 trains a fresh LoRA on book B under one of three "
           "losses. The three losses differ only in which tokens carry a KL penalty to the frozen book-A model.",
           L, 11.5)

    BA_FC, BA_EC = "#374151", "#374151"     # book-A rehearsal statements
    BB_FC, BB_EC = "#F3F4F6", "#9CA3AF"     # book-B statements
    top = 2.35
    for x, t in [(L, "STAGE 1 \u00b7 LEARN BOOK A"), (4.95, "STAGE 2 \u00b7 LEARN BOOK B, ONE BATCH DRAWN TO SCALE"),
                 (12.95, "STAGE 3 \u00b7 EVALUATE")]:
        S.text(x, top, t, size=T_MONO * 0.95, family=MONO, weight="medium", color=AXIS)
    S.rule(L, R, top + 0.17, color=INK)

    # stage 1
    card(S, L, 2.95, 3.45, 0.95, f"{MODEL}", "base model")
    page_arrow(S, (L + 1.0, 3.9), (L + 1.0, 5.3))
    S.text(L + 1.2, 4.42, "LoRA A \u00b7 r=64", size=T_MONO, family=MONO, color=INK)
    S.text(L + 1.2, 4.72, "CE on 6,000 book-A", size=T_MONO * 0.9, family=MONO, color=BODY)
    S.text(L + 1.2, 4.98, "statements, 4 epochs", size=T_MONO * 0.9, family=MONO, color=BODY)
    card(S, L, 5.3, 3.45, 0.95, "Book-A model", "LoRA A merged \u00b7 frozen", fc="#F3F4F6", ec=ORIGIN_EDGE)
    S.para(L, 6.75, "Every stage-2 run starts from this model and uses it as the KL reference "
           "(the same network with LoRA B switched off).", 3.45, size=T_MONO * 0.95, color=BODY, lh=1.45)
    S.page.plot([L + 3.45, 4.5], [5.78, 5.78], color=ORIGIN_EDGE, lw=1.6)
    S.page.plot([4.5, 4.5], [3.45 + 0.63, 6.55 + 0.63], color=ORIGIN_EDGE, lw=1.6, solid_capstyle="butt")
    for yb in (3.45, 5.0, 6.55):
        page_arrow(S, (4.5, yb + 0.63), (4.72, yb + 0.63), color=ORIGIN_EDGE)

    # stage 2 legend
    ly = 2.92
    S.page.add_patch(Rectangle((4.95, ly - 0.15), 0.15, 0.17, fc=BB_FC, ec=BB_EC, lw=0.9))
    S.text(5.2, ly, "book-B statement \u00d7 32", size=T_MONO, family=MONO, color=BODY)
    S.page.add_patch(Rectangle((8.05, ly - 0.15), 0.15, 0.17, fc=BA_FC, ec=BA_EC, lw=0.9))
    S.text(8.3, ly, "book-A rehearsal statement \u00d7 13", size=T_MONO, family=MONO, color=BODY)
    S.text(12.55, top, "fresh LoRA B (r=64) is all that trains", size=T_MONO * 0.9, family=MONO,
           color=AXIS, ha="right")

    X0 = 4.95
    rows = {"naive": 3.45, "kl_global": 5.0, "replay_kl": 6.55}
    loss = {"naive": "CE(B)", "kl_global": "CE(B) + KL(B)", "replay_kl": "CE(B) + CE(A) + KL(A)"}
    for m, y in rows.items():
        c = COLOR[m]
        S.page.add_patch(Rectangle((X0 - 0.17, y), 0.06, 1.25, fc=c, ec="none"))
        S.text(X0, y + 0.24, NAME[m], size=T_LABEL, weight="semibold", color=c)
        S.text(X0 + S.measure(NAME[m], size=T_LABEL, weight="semibold") + 0.18, y + 0.24, m,
               size=T_MONO * 0.9, family=MONO, color=AXIS)
        S.text(12.55, y + 0.24, loss[m], size=T_MONO, family=MONO, color=c, ha="right")
        by = y + 0.45
        xb = blocks(S, X0, by, 32, BB_FC, BB_EC)
        if m == "kl_global":
            bracket(S, X0, xb, by + 0.47, c, "CE + KL to book-A model")
            S.page.add_patch(Rectangle((X0 - 0.02, by - 0.03), xb - X0 + 0.04, 0.42, fc="none", ec=c, lw=1.8,
                                       zorder=4))
        if m == "replay_kl":
            xa0 = xb + 0.2
            xa = blocks(S, xa0, by, 13, BA_FC, BA_EC)
            bracket(S, xa0, xa, by + 0.47, c, "CE + KL to book-A model")
            S.page.add_patch(Rectangle((xa0 - 0.02, by - 0.03), xa - xa0 + 0.04, 0.42, fc="none", ec=c,
                                       lw=1.8, zorder=4))
        if m != "kl_global":
            bracket(S, X0, xb, by + 0.47, BODY, "CE only, no constraint")
    # naive/kl rows: show where rehearsal would sit, empty
    for m in ("naive", "kl_global"):
        by = rows[m] + 0.45
        xa0 = X0 + 32 * 0.16 - 0.035 + 0.2
        S.text(xa0, by + 0.25, "no rehearsal", size=T_MONO * 0.9, family=MONO, color=ORIGIN)

    # stage 3
    for i, (bk, fc, ec) in enumerate([("A", BA_FC, BA_EC), ("B", BB_FC, BB_EC)]):
        y = 3.0 + i * 1.3
        S.box(12.95, y, 2.25, 1.05, fc="white", ec=GRID, lw=1.3, z=2)
        S.page.add_patch(Rectangle((13.12, y + 0.2), 0.15, 0.17, fc=fc, ec=ec, lw=0.9, zorder=3))
        S.text(13.38, y + 0.36, f"Book {bk}", size=T_LABEL * 0.9, weight="semibold")
        S.text(13.12, y + 0.72, "1,500 held-out", size=T_MONO * 0.9, family=MONO, color=BODY)
        S.text(13.12, y + 0.95, "questions", size=T_MONO * 0.9, family=MONO, color=BODY)
    S.text(12.95, 5.95, "Unseen template:", size=T_MONO * 0.9, color=BODY)
    S.text(12.95, 6.25, "Question: Who founded", size=T_MONO * 0.85, family=MONO, color=INK)
    S.text(12.95, 6.48, "the town of X?", size=T_MONO * 0.85, family=MONO, color=INK)
    S.text(12.95, 6.71, "Answer:", size=T_MONO * 0.85, family=MONO, color=INK)
    S.para(12.95, 7.15, "Greedy, 12 new tokens; correct if the gold string appears.", 2.25,
           size=T_MONO * 0.9, color=BODY, lh=1.45)
    S.vrule(12.75, 2.75, 7.85, color=GRID)

    S.rule(L, R, 8.2, color=GRID)
    S.text(L, 8.55, "loss  =  CE(book B)  +  CE(book-A rehearsal)  +  kl_w \u00b7 KL(book-A model || model)"
           "   on rehearsal tokens only,  kl_w = 1.0", size=T_MONO, family=MONO, color=REH)
    S.text(R, 8.55, "stage-2 loss of rehearsal + KL", size=T_MONO * 0.85, family=MONO, color=AXIS, ha="right")
    S.save("method")


# ---------------------------------------------------------------- accuracy bars

def accuracy_bars():
    W, H = 13, 7.6
    S = Sheet(W, H)
    L, R = 0.8, 12.2
    header(S, "Rehearsal + KL keeps book A and learns most of book B.",
           "Accuracy after stage 2. Solid bars are book A (retained), light bars book B (learned). "
           "Whiskers are \u00b11 sd; short dashes mark the book-A model before stage 2.", L, 11.2)
    ax = S.axes(1.55, 2.25, 10.6, 4.1)
    ax.set_xlim(-0.55, len(ORDER) - 0.45)
    ax.set_ylim(0, 1.16)
    yt = np.round(np.arange(0, 1.01, 0.2), 1)
    for t in np.round(np.arange(0, 1.01, 0.1), 1):
        ax.axhline(t, color=GRID, lw=1.0, zorder=0)
    ax.set_yticks(yt, [f"{t:.1f}" for t in yt], family=SANS, fontweight="medium", fontsize=T_AXIS * 0.85)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("ACCURACY AFTER STAGE 2", family=SANS, fontweight="medium", fontsize=T_AXIS, labelpad=12)
    bw, off = 0.3, 0.17
    for i, m in enumerate(ORDER):
        c = COLOR[m]
        for j, (col, ref, book) in enumerate([("after_B_A", START_A, "A"), ("after_B_B", START_B, "B")]):
            x = i + (-off if j == 0 else off)
            v, e = mu(m, col), sd(m, col)
            ax.bar(x, v, bw, color=c if j == 0 else tint(c, 0.72), ec=c, lw=0 if j == 0 else 1.6, zorder=2)
            ax.errorbar(x, v, yerr=e, color=INK, lw=1.4, capsize=6, capthick=1.4, zorder=4)
            ax.plot([x - bw / 2 - 0.03, x + bw / 2 + 0.03], [ref, ref], color=ORIGIN_EDGE, lw=1.4,
                    ls=(0, (2.5, 2)), zorder=5)
            top = v + e
            if top < ref < top + 0.13:
                top = ref
            ax.annotate(f"{num(v)}", (x, top), xytext=(0, 9), textcoords="offset points", ha="center",
                        family=MONO, fontsize=T_MONO, color=c if j == 0 else INK, zorder=6,
                        fontweight="medium")
            ax.annotate(f"\u00b1{num(e)}", (x, top), xytext=(0, 9 + T_MONO * 1.25), textcoords="offset points",
                        ha="center", family=MONO, fontsize=T_MONO * 0.85, color=AXIS, zorder=6)
            ax.annotate(f"Book {book}", (x, 0), xytext=(0, -20), textcoords="offset points", ha="center",
                        family=SANS, fontweight="medium", fontsize=T_AXIS * 0.85, color=AXIS,
                        annotation_clip=False)
        ax.annotate(NAME[m], (i, 0), xytext=(0, -46), textcoords="offset points", ha="center", family=SANS,
                    fontweight="semibold", fontsize=T_LABEL, color=c, annotation_clip=False)
    S.text(R, 2.05, f"- -  book-A model:  A {num(START_A)}   B {num(START_B)}", size=T_MONO * 0.9,
           family=MONO, color=ORIGIN_EDGE, ha="right")
    footer(S, L, R, 7.3)
    S.save("accuracy_bars")


# ---------------------------------------------------------------- forgetting (forest plot)

def forgetting():
    W, H = 13, 6.2
    S = Sheet(W, H)
    L, R = 0.8, 12.2
    header(S, "Rehearsal + KL sits at zero: no measurable forgetting of book A.",
           "Forgetting of book A = accuracy after stage 1 \u2212 accuracy after stage 2. "
           "Dot: mean; bar: \u00b11 sd; open circles below: the individual seeds.", L, 11.2)
    AX0, AX1 = 3.7, 9.9
    top, rowh = 2.45, 0.78
    ax = S.axes(AX0, top - 0.35, AX1 - AX0, rowh * len(ORDER) + 0.35)
    lo, hi = -0.1, 0.75
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(ORDER) - 0.5 + 0.45, -0.5)
    xt = np.round(np.arange(-0.1, 0.71, 0.1), 1)
    for t in xt:
        ax.axvline(t, color=GRID, lw=1.0, zorder=0)
    ax.axvline(0, color=ORIGIN_EDGE, lw=1.6, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(xt, [num(t, 1) if t else "0" for t in xt], family=SANS, fontweight="medium",
                  fontsize=T_AXIS * 0.85)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("FORGETTING OF BOOK A", family=SANS, fontweight="medium", fontsize=T_AXIS, labelpad=12)
    rows = sorted(ORDER, key=lambda m: -mu(m, "forgetting_A"))
    S.text(L, top - 0.2, "METHOD", size=T_MONO * 0.9, family=MONO, color=AXIS)
    S.text(R, top - 0.2, "MEAN \u00b1 SD", size=T_MONO * 0.9, family=MONO, color=AXIS, ha="right")
    for i, m in enumerate(rows):
        c, v, e = COLOR[m], mu(m, "forgetting_A"), sd(m, "forgetting_A")
        ax.plot([v - e, v + e], [i, i], color=c, lw=3.2, solid_capstyle="round", zorder=3)
        seeds = df[df.method == m].forgetting_A.values
        ax.scatter(seeds, [i + 0.27] * len(seeds), s=40, fc="white", ec=c, lw=1.4, zorder=4)
        ax.scatter([v], [i], s=190, color=c, ec="white", lw=2, zorder=5)
        yy = top + i * rowh + 0.1
        S.text(L, yy, NAME[m], size=T_LABEL, weight="semibold", color=c)
        S.text(R, yy, f"{num(v)} \u00b1 {num(e)}", size=T_MONO * 1.1, family=MONO, color=c, ha="right")
    ax.annotate("\u2190 book A improved", (0, len(ORDER) - 0.5 + 0.3), xytext=(-8, 0),
                textcoords="offset points", ha="right", va="center", family=SANS, fontsize=T_MONO, color=BODY)
    ax.annotate("book A forgotten \u2192", (0, len(ORDER) - 0.5 + 0.3), xytext=(8, 0),
                textcoords="offset points", ha="left", va="center", family=SANS, fontsize=T_MONO, color=BODY)
    footer(S, L, R, 5.9)
    S.save("forgetting")


# ---------------------------------------------------------------- tradeoff deep dive

def tradeoff():
    W, H = 16, 8.6
    S = Sheet(W, H)
    L, R = 0.8, 15.2
    header(S, "Where every run goes, checkpoint by checkpoint.",
           "Each run is measured at three checkpoints: the base model, after book A, after book B. Thin paths "
           "are the nine individual runs; bold dots are method means with a \u00b11 sd halo.", L, 14.2)
    ax = S.axes(1.65, 2.3, 6.4, 5.15)
    plane_axes(ax, hi=1.1)
    for seed in SEEDS:
        s1 = stage1.loc[seed]
        ax.plot([s1.base_B, s1.after_A_B], [s1.base_A, s1.after_A_A], color=ORIGIN, lw=1.0,
                ls=(0, (3, 3)), zorder=2)
        for m in ORDER:
            r = run(m, seed)
            bezier(ax, (r.after_A_B, r.after_A_A), (r.after_B_B, r.after_B_A), COLOR[m], BEND[m], lw=1.3,
                   alpha=0.45, shrink=(3, 4), head=0.7, z=3)
            ax.scatter([r.after_B_B], [r.after_B_A], s=30, fc="white", ec=COLOR[m], lw=1.2, zorder=7)
        ax.scatter([s1.after_A_B], [s1.after_A_A], s=30, fc="white", ec=ORIGIN_EDGE, lw=1.2, zorder=7)
    for m in ORDER:
        b, a = mu(m, "after_B_B"), mu(m, "after_B_A")
        halo(ax, b, a, sd(m, "after_B_B"), sd(m, "after_B_A"), COLOR[m])
        point(ax, b, a, COLOR[m], s=120)
        tag(ax, b, a, NAME[m], coords(a, b), COLOR[m], size=T_LABEL * 0.85, mono=T_MONO * 0.9, **LABEL[m])
    point(ax, START_B, START_A, ORIGIN, s=140, edge=ORIGIN_EDGE, lw=1.8, z=9)
    tag(ax, START_B, START_A, "Book-A model (mean of 3 seeds)", coords(START_A, START_B), ORIGIN_EDGE,
        ha="left", va="bottom", dx=2, dy=24, size=T_LABEL * 0.85, mono=T_MONO * 0.9)
    point(ax, BASE_B, BASE_A, "white", s=50, edge=ORIGIN_EDGE, lw=1.2)
    tag(ax, BASE_B, BASE_A, "Base model", coords(BASE_A, BASE_B), BODY, ha="left", va="bottom", dx=10,
        dy=4, size=T_LABEL * 0.8, mono=T_MONO * 0.85)

    # right: the same runs as slopes across the three checkpoints
    stages = ["base", "after A", "after B"]
    for k, (book, cols) in enumerate([("BOOK A ACCURACY", ["base_A", "after_A_A", "after_B_A"]),
                                     ("BOOK B ACCURACY", ["base_B", "after_A_B", "after_B_B"])]):
        x0 = 9.35 + k * 3.05
        a2 = S.axes(x0, 2.3, 2.45, 5.15)
        a2.set_xlim(-0.25, 2.25 + 0.95)
        a2.set_ylim(-0.03, 1.03)
        for t in np.round(np.arange(0, 1.01, 0.1), 1):
            a2.axhline(t, color=GRID, lw=1.0, zorder=0)
        for t in range(3):
            a2.axvline(t, color=GRID, lw=1.0, zorder=0)
        yt = np.round(np.arange(0, 1.01, 0.2), 1)
        a2.set_yticks(yt, [f"{t:.1f}" for t in yt] if k == 0 else [], family=SANS, fontweight="medium",
                      fontsize=T_AXIS * 0.85)
        a2.set_xticks(range(3), stages, family=SANS, fontweight="medium", fontsize=T_AXIS * 0.8)
        for s in a2.spines.values():
            s.set_visible(False)
        a2.set_title(book, family=SANS, fontweight="medium", fontsize=T_AXIS, color=AXIS, loc="left", pad=14)
        for m in ORDER:
            sub = df[df.method == m]
            for _, r in sub.iterrows():
                a2.plot([0, 1], [r[c] for c in cols[:2]], color=ORIGIN, lw=0.8, alpha=0.5, zorder=2)
                a2.plot([1, 2], [r[c] for c in cols[1:]], color=COLOR[m], lw=0.9, alpha=0.4, zorder=2)
            means = [sub[c].mean() for c in cols]
            a2.plot([0, 1], means[:2], color=ORIGIN, lw=2.2, zorder=3)
            a2.plot([1, 2], means[1:], color=COLOR[m], lw=2.6, zorder=4, solid_capstyle="round")
            a2.scatter([2], [means[2]], s=60, color=COLOR[m], ec="white", lw=1.5, zorder=5)
        a2.scatter([0, 1], [df[cols[0]].mean(), stage1[cols[1]].mean()], s=60, color=ORIGIN, ec=ORIGIN_EDGE,
                   lw=1.2, zorder=5)
        # end labels, nudged apart so they never collide
        ends = sorted([(mu(m, cols[2]), m) for m in ORDER])
        ys, last = [], -1
        for v, m in ends:
            y = max(v, last + 0.075)
            ys.append((y, v, m))
            last = y
        for y, v, m in ys:
            a2.annotate(num(v), (2, v), xytext=(2.18, y), textcoords="data", va="center", ha="left",
                        family=MONO, fontsize=T_MONO, color=COLOR[m], annotation_clip=False)
    footer(S, L, R, 8.3)
    S.save("tradeoff")


# ---------------------------------------------------------------- per seed

def per_seed():
    W, H = 16, 8.3
    S = Sheet(W, H)
    L, R = 0.8, 15.2
    header(S, "The same picture in every seed.",
           "One plane per seed: book A retained against book B learned, each path starting from that seed\u2019s "
           "book-A model. The rehearsal + KL result is not one lucky initialization.", L, 14.2)
    pw, gap = 4.2, 0.8
    for k, seed in enumerate(SEEDS):
        x0 = 1.55 + k * (pw + gap)
        ax = S.axes(x0, 2.55, pw, 3.6)
        plane_axes(ax, size=T_AXIS * 0.8, label_every=0.5, titles=(k == 0),
                   xt="BOOK B  \u2192", yt="BOOK A  \u2192")
        if k > 0:
            ax.set_xlabel("BOOK B  \u2192", family=SANS, fontweight="medium", fontsize=T_AXIS * 0.8, color=AXIS,
                          labelpad=12)
        s1 = stage1.loc[seed]
        for m in ORDER:
            r = run(m, seed)
            bezier(ax, (s1.after_A_B, s1.after_A_A), (r.after_B_B, r.after_B_A), COLOR[m], BEND[m], lw=3.5 * PX,
                   shrink=(6, 8), head=0.8)
            point(ax, r.after_B_B, r.after_B_A, COLOR[m], s=80, lw=1.5)
        point(ax, s1.after_A_B, s1.after_A_A, ORIGIN, s=90, edge=ORIGIN_EDGE, lw=1.5, z=9)
        rk = run("replay_kl", seed)
        S.text(x0, 2.3, f"SEED {seed}", size=T_MONO, family=MONO, weight="semibold", color=INK)
        # per-seed table under each plane
        ty = 7.0
        S.text(x0 + pw, 2.3, f"book-A model  A {num(s1.after_A_A)}", size=T_MONO * 0.85, family=MONO,
               color=ORIGIN_EDGE, ha="right")
        for i, m in enumerate(ORDER):
            r = run(m, seed)
            yy = ty + i * 0.27
            S.text(x0, yy, NAME[m], size=T_MONO * 0.95, weight="semibold", color=COLOR[m])
            S.text(x0 + pw, yy, f"A {num(r.after_B_A)}  B {num(r.after_B_B)}  forget {num(r.forgetting_A, sign=True)}",
                   size=T_MONO * 0.85, family=MONO, color=COLOR[m], ha="right")
        del rk
    footer(S, L, R, 8.05, note="one run per method per seed")
    S.save("per_seed")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    for name in args.only.split(","):
        globals()[name.strip()]()
