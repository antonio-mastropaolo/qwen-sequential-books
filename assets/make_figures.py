#!/usr/bin/env python3
"""Build every README figure from results/results.csv.

    python assets/make_figures.py                  # all figures, PNG + SVG, into assets/
    python assets/make_figures.py --only hero      # just one

Figures: hero, method, accuracy_bars, forgetting, tradeoff, per_seed.

Needs numpy, pandas, matplotlib and Pillow. Text is set in Inter, Instrument Serif and IBM Plex
Mono, loaded from assets/fonts (SIL OFL), so the output does not depend on the fonts installed on
the machine. Rebuilding with the same CSV and library versions gives byte-identical files: PNGs
carry no timestamp or software tag, and SVG ids are hashed with a fixed salt.

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
from matplotlib import font_manager, patheffects  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
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

# ---------------------------------------------------------------- type and colour

FONT_DIR = os.path.join(HERE, "fonts")
for f in sorted(glob.glob(os.path.join(FONT_DIR, "*.ttf"))):
    font_manager.fontManager.addfont(f)
# if a variable build of one of these families is installed system-wide, ignore it
font_manager.fontManager.ttflist = [
    f for f in font_manager.fontManager.ttflist
    if f.name not in ("Inter", "Instrument Serif", "IBM Plex Mono") or os.path.dirname(f.fname) == FONT_DIR]
SANS, SERIF, MONO = "Inter", "Instrument Serif", "IBM Plex Mono"

INK = "#101217"
BODY = "#3b4049"
MUTED = "#6c717a"
FAINT = "#a4a8af"
HAIR = "#e7e4dd"
PAPER = "#f6f3ec"      # warm off-white for quiet panels
ACCENT = "#2236e8"     # the one loud colour: the method that works
ACCENT_DEEP = "#1424b0"
NAIVE = "#c9734d"
KLG = "#7f8a84"

COLOR = {"naive": NAIVE, "kl_global": KLG, "replay_kl": ACCENT}
DEEP = {"naive": "#a2532f", "kl_global": "#56605a", "replay_kl": ACCENT_DEEP}
METHODS = ["naive", "kl_global", "replay_kl"]
NAME = {"naive": "Naive fine-tuning", "kl_global": "KL on new-book tokens", "replay_kl": "Rehearsal + KL"}
SHORT = {"naive": "Naive", "kl_global": "KL on B tokens", "replay_kl": "Rehearsal + KL"}

plt.rcParams.update({
    "font.family": SANS, "font.size": 11, "text.color": INK,
    "axes.labelcolor": BODY, "axes.edgecolor": HAIR, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.major.pad": 8, "ytick.major.pad": 8,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "none",
    "svg.fonttype": "path", "svg.hashsalt": "sequential-books", "axes.unicode_minus": True,
    "path.simplify": False, "image.composite_image": True,
})

# ---------------------------------------------------------------- data

df = pd.read_csv(args.csv)
df = df[~df.smoke_test.astype(str).str.lower().eq("true")]
ORDER = [m for m in METHODS if m in set(df.method)]
SEEDS = sorted(df.seed.unique())
COLS = ["after_A_A", "after_A_B", "after_B_A", "after_B_B", "forgetting_A", "stage2_sec"]
stats = df.groupby("method")[COLS].agg(["mean", "std"]).reindex(ORDER)  # sample sd, as in summary.md
stage1 = df.drop_duplicates("seed").set_index("seed").sort_index()
MODEL = df.model.iloc[0].split("/")[-1]
BASE_A, BASE_B = df.base_A.mean(), df.base_B.mean()
START_A, START_B = stage1.after_A_A.mean(), stage1.after_A_B.mean()
START_A_SD = stage1.after_A_A.std()


def mu(m, c):
    return stats.loc[m, (c, "mean")]


def sd(m, c):
    return stats.loc[m, (c, "std")]


def runs_of(m):
    return df[df.method == m].set_index("seed").sort_index()


def num(v, nd=3, sign=False):
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "\u2212")


def tint(color, amount):
    """Blend a colour towards white: 0 keeps it, 1 is white."""
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


def frac_ticks(ts):
    return ["0" if t == 0 else "1" if t == 1 else f"{t:.2f}"[1:] for t in ts]


# ---------------------------------------------------------------- page helpers

class Sheet:
    """A figure laid out in inches from the top-left corner, like a printed page."""

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

    def text(self, x, y, s, size=11, color=INK, family=SANS, weight="regular", style="normal",
             ha="left", va="baseline", **kw):
        return self.page.text(x, y, s, fontsize=size, color=color, family=family, fontweight=weight,
                              fontstyle=style, ha=ha, va=va, **kw)

    def width(self, t):
        return t.get_window_extent(self.fig.canvas.get_renderer()).width / self.fig.dpi

    def runs(self, x, y, parts, **common):
        """Consecutive text runs in different styles on one baseline. Returns the end x."""
        for s, kw in parts:
            x += self.width(self.text(x, y, s, **{**common, **kw}))
        return x

    def rule(self, x0, x1, y, color=HAIR, lw=1.0, **kw):
        self.page.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", **kw)

    def vrule(self, x, y0, y1, color=HAIR, lw=1.0, **kw):
        self.page.plot([x, x], [y0, y1], color=color, lw=lw, solid_capstyle="butt", **kw)

    def box(self, x, y, w, h, fc, ec="none", lw=1.0, r=0.12, z=0, **kw):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                           lw=lw, zorder=z, **kw)
        self.page.add_patch(p)
        return p

    def eyebrow(self, x, y, s, color=MUTED, size=9, square=None, ha="left"):
        if square:
            self.page.add_patch(Rectangle((x, y - 0.1), 0.1, 0.1, fc=square, ec="none"))
            x += 0.2
        return self.text(x, y, s.upper(), size=size, color=color, family=MONO, weight="medium", ha=ha)

    def save(self, name):
        """PNG (opaque RGB, no metadata) and SVG (no date), both reproducible byte for byte."""
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=args.dpi, metadata={"Software": None})
        Image.open(buf).convert("RGB").save(os.path.join(args.out, name + ".png"), optimize=True)
        self.fig.savefig(os.path.join(args.out, name + ".svg"), format="svg",
                         metadata={"Date": None, "Creator": None})
        plt.close(self.fig)
        print("wrote", os.path.join(args.out, name + ".png"))


def header(S, eyebrow, title, deck, L, y=0.82, size=40, width=None):
    """Mono eyebrow, a serif headline in two voices, one line or two of plain deck text."""
    S.eyebrow(L, y, eyebrow, square=ACCENT)
    S.runs(L, y + 0.86, [(title[0], {}), (title[1], {"color": ACCENT, "style": "italic"})],
           size=size, family=SERIF)
    for i, line in enumerate(deck):
        S.text(L, y + 1.36 + i * 0.27, line, size=12.5, color=BODY)


def footer(S, y, L, R, extra="", stacked=False):
    S.rule(L, R, y - 0.28 - (0.24 if stacked else 0))
    if stacked:
        y -= 0.24
    S.runs(L, y, [("DATA  ", {"color": FAINT}), ("results/results.csv", {"color": BODY}),
                  (f"   ·   {MODEL}   ·   seeds {', '.join(str(s) for s in SEEDS)}{extra}", {})],
           size=8.6, family=MONO, color=MUTED)
    S.text(L if stacked else R, y + (0.24 if stacked else 0), "github.com/antonio-mastropaolo/qwen-sequential-books",
           size=8.6, family=MONO, color=MUTED, ha="left" if stacked else "right")


def arrow(ax, a, b, color=INK, lw=1.3, rad=0.0, head=(4, 7), shrink=(0, 0), ls="-", z=5, alpha=1):
    p = FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                        arrowstyle=f"-|>,head_length={head[0] / 10},head_width={head[1] / 20}",
                        color=color, lw=lw, ls=ls, shrinkA=shrink[0], shrinkB=shrink[1], zorder=z,
                        alpha=alpha, capstyle="round", joinstyle="round", mutation_scale=10)
    ax.add_patch(p)
    return p


def quiet_axes(ax, xlim, ylim, xticks, yticks, size=9.5, grid=True, xfmt=None, yfmt=None):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xfmt or frac_ticks(xticks), family=MONO, fontsize=size)
    ax.set_yticklabels(yfmt or frac_ticks(yticks), family=MONO, fontsize=size)
    for s in ax.spines.values():
        s.set_visible(False)
    if grid:
        for t in xticks:
            ax.axvline(t, color=HAIR, lw=0.9, zorder=0)
        for t in yticks:
            ax.axhline(t, color=HAIR, lw=0.9, zorder=0)


def radial_wash(ax, center, radius, color, strength, xlim, ylim, z=0, clip=None):
    xs, ys = np.linspace(*xlim, 300), np.linspace(*ylim, 300)
    X, Y = np.meshgrid(xs, ys)
    a = strength * np.clip(1 - np.hypot(X - center[0], Y - center[1]) / radius, 0, 1) ** 2
    img = np.zeros(X.shape + (4,))
    img[..., :3] = to_rgb(color)
    img[..., 3] = a
    im = ax.imshow(img, extent=(*xlim, *ylim), origin="lower", aspect="auto", zorder=z,
                   interpolation="bilinear")
    if clip is not None:
        im.set_clip_path(clip)
    return im


HALO = [patheffects.withStroke(linewidth=4, foreground="white")]


# ---------------------------------------------------------------- the trade-off plane

def plane_points(seed=None):
    """(B learned, A retained) per method after stage 2: mean, sd and the seeds; or one seed."""
    pts = {}
    for m in ORDER:
        r = runs_of(m)
        if seed is None:
            pts[m] = dict(x=mu(m, "after_B_B"), y=mu(m, "after_B_A"), sx=sd(m, "after_B_B"),
                          sy=sd(m, "after_B_A"), xs=r.after_B_B.values, ys=r.after_B_A.values)
        else:
            pts[m] = dict(x=r.after_B_B[seed], y=r.after_B_A[seed], sx=0, sy=0, xs=[], ys=[])
    if seed is None:
        start = (START_B, START_A)
    else:
        start = (stage1.after_A_B[seed], stage1.after_A_A[seed])
    return pts, start


def tradeoff_line(pts):
    (x0, y0), (x1, y1) = (pts["naive"]["x"], pts["naive"]["y"]), (pts["kl_global"]["x"], pts["kl_global"]["y"])
    k = (y1 - y0) / (x1 - x0)
    return lambda x: y0 + k * (x - x0)


def plane(ax, pts, start, base=None, s=1.0, tick=9.5, detail=True):
    """Book A retained (y) against book B learned (x). The region the two baselines trade along is
    shaded; the goal is the top-right corner."""
    xlim, ylim = (-0.035, 1.075), (-0.035, 1.075)
    ts = [0, 0.25, 0.5, 0.75, 1]
    quiet_axes(ax, xlim, ylim, ts, ts, size=tick)
    line = tradeoff_line(pts)

    # below the straight line through the two baselines: what trading one book for the other buys
    xs = np.array([xlim[0], xlim[1]])
    under = Polygon([(xlim[0], ylim[0]), (xlim[0], line(xlim[0])), (xlim[1], line(xlim[1])),
                     (xlim[1], ylim[0])], closed=True, fc=PAPER, ec="none", zorder=0.2)
    ax.add_patch(under)
    for t in ts:  # grid on top of the paper tint, so it reads as one surface
        ax.plot([t, t], ylim, color=HAIR, lw=0.9, zorder=0.3)
        ax.plot(xlim, [t, t], color=HAIR, lw=0.9, zorder=0.3)
    ax.plot(xs, line(xs), color=FAINT, lw=1.1 * s, ls=(0, (2.5, 3)), zorder=1, solid_capstyle="butt")

    # a soft wash around the goal corner, clipped to the region beyond the line
    beyond = Polygon([(xlim[0], ylim[1]), (xlim[0], line(xlim[0])), (xlim[1], line(xlim[1])),
                      (xlim[1], ylim[1])], closed=True, fc="none", ec="none", transform=ax.transData)
    ax.add_patch(beyond)
    radial_wash(ax, (1.0, 1.0), 0.95, ACCENT, 0.13, xlim, ylim, z=0.4, clip=beyond)

    # goal
    gx, gy = 1.0, 1.0
    ax.plot([gx - 0.035, gx + 0.035], [gy, gy], color=INK, lw=0.9 * s, zorder=6)
    ax.plot([gx, gx], [gy - 0.035, gy + 0.035], color=INK, lw=0.9 * s, zorder=6)
    ax.scatter([gx], [gy], s=90 * s ** 2, fc="white", ec=INK, lw=0.9 * s, zorder=7)

    # stage 1: from the base model, which knows neither book, to the book-A model
    sx, sy = start
    if base is not None:
        bx, by = base
        arrow(ax, (bx, by), (sx, sy), color=INK, lw=1.0 * s, ls=(0, (1.5, 2.5)), head=(3.5 * s, 6 * s),
              shrink=(5 * s, 11 * s), z=2, alpha=0.7)
        ax.scatter([bx], [by], s=45 * s ** 2, fc="white", ec=INK, lw=1.1 * s, zorder=8)

    # stage 2: every method starts from the book-A model
    for m, rad in (("naive", 0.2), ("kl_global", 0.14), ("replay_kl", 0.04)):
        p = pts[m]
        b = m == "replay_kl"
        arrow(ax, (sx, sy), (p["x"], p["y"]), color=COLOR[m], lw=(2.8 if b else 1.4) * s, rad=rad,
              head=(8 * s, 13 * s) if b else (4.5 * s, 7 * s), shrink=(7 * s, 9 * s), z=4.8,
              alpha=1 if b else 0.85)
    ax.scatter([sx], [sy], s=150 * s ** 2, fc="white", ec=INK, lw=1.4 * s, zorder=8)
    ax.scatter([sx], [sy], s=22 * s ** 2, fc=INK, ec="none", zorder=9)

    for m in ORDER:
        p, c = pts[m], COLOR[m]
        big = m == "replay_kl"
        if detail and len(p["xs"]):
            # the ±1 sd box as a soft ellipse, the seeds as rings
            ax.add_patch(Ellipse((p["x"], p["y"]), 2 * max(p["sx"], 0.012), 2 * max(p["sy"], 0.012),
                                 fc=tint(c, 0.8 if big else 0.85), ec="none", zorder=4.5))
            ax.scatter(p["xs"], p["ys"], s=(30 if big else 22) * s ** 2, fc="white", ec=c, lw=1.2 * s,
                       zorder=5)
        ax.scatter([p["x"]], [p["y"]], s=(150 if big else 70) * s ** 2, fc=c, ec="white",
                   lw=1.8 * s, zorder=6)
    return line


def plane_labels(ax, pts, start, size=12.5, start_note="book-A model", notes=True, compact=False):
    """Direct labels below and left of each mean: name, its two numbers, and a one-line reading."""
    note = {"naive": "learns B, loses A", "kl_global": "keeps A, stalls B",
            "replay_kl": "beyond the trade-off line"}
    gap = size * 1.25
    for m in ORDER:
        p = pts[m]
        big = m == "replay_kl"
        x0 = -{"naive": 0.9, "kl_global": 1.5, "replay_kl": 0.9}[m] * size
        y0 = -1.9 * size - (2 if big else 0)
        kw = dict(xy=(p["x"], p["y"]), textcoords="offset points", ha="right", va="baseline",
                  path_effects=HALO, zorder=10, annotation_clip=False)
        ax.annotate(SHORT[m] if compact else NAME[m], xytext=(x0, y0), fontsize=size * (1.14 if big else 1),
                    color=DEEP[m], fontweight="semibold" if big else "medium", **kw)
        ax.annotate(f"A {num(p['y'])}   B {num(p['x'])}", xytext=(x0, y0 - gap), fontsize=size * 0.8,
                    family=MONO, color=BODY if big else MUTED, **kw)
        if notes:
            ax.annotate(note[m], xytext=(x0, y0 - 2.15 * gap), fontsize=size * (1.18 if big else 1.05),
                        family=SERIF, style="italic", color=ACCENT if big else MUTED, **kw)
    ax.annotate(start_note, xy=start, xytext=(-0.5 * size, 1.25 * size), textcoords="offset points",
                ha="left", va="baseline", fontsize=size * 0.82, color=INK, fontweight="medium",
                path_effects=HALO, zorder=10)
    ax.annotate("goal", xy=(1, 1), xytext=(0, 1.25 * size), textcoords="offset points", ha="center",
                va="baseline", fontsize=size * 0.74, family=MONO, color=INK, path_effects=HALO, zorder=10)


def region(ax, x, y, text, color, size=10):
    """A letter-spaced caption in small caps naming a region of the plane."""
    spaced = "\n".join("\u2009".join(line.upper()) for line in text.split("\n"))
    ax.text(x, y, spaced, ha="center", va="center", fontsize=size, family=SANS, color=color,
            fontweight="semibold", linespacing=1.5, zorder=2)


def line_label(ax, line, x, text, size=10.5, dy=0.018):
    """Write text along the trade-off line, rotated to match it on screen."""
    ax.figure.canvas.draw_idle()
    p0 = ax.transData.transform((x - 0.05, line(x - 0.05)))
    p1 = ax.transData.transform((x + 0.05, line(x + 0.05)))
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(x, line(x) + dy, text, rotation=ang, rotation_mode="anchor", ha="center", va="bottom",
            fontsize=size, family=SERIF, style="italic", color=MUTED, zorder=2)


# ---------------------------------------------------------------- hero

def hero():
    W, H, L, R = 18, 10, 0.95, 17.05
    S = Sheet(W, H)
    pts, start = plane_points()
    ours = "replay_kl"

    S.eyebrow(L, 0.8, "Sequential Books  ·  catastrophic forgetting in sequential LoRA fine-tuning",
              square=ACCENT)
    S.eyebrow(R, 0.8, f"{MODEL}  ·  LoRA r=64  ·  {len(SEEDS)} seeds", ha="right")

    # ---- left column: the claim and the number
    S.text(L, 1.98, "Learn the new book.", size=66, family=SERIF)
    S.text(L, 2.95, "Keep the old one.", size=66, family=SERIF, style="italic", color=ACCENT)
    deck = ["Qwen2.5-7B learns book A, then a closely related",
            "book B. Naive fine-tuning erases A. A KL penalty on",
            "B's tokens saves A but stalls B. Rehearsing A, with KL",
            "on the rehearsal tokens only, keeps both."]
    for i, line in enumerate(deck):
        S.text(L, 3.62 + i * 0.31, line, size=13.5, color=BODY)

    y0 = 5.2
    S.eyebrow(L, y0, "Forgetting of book A  ·  rehearsal + KL", color=ACCENT)
    big = S.text(L - 0.08, y0 + 2.0, num(mu(ours, "forgetting_A")), size=150, family=SERIF, color=ACCENT)
    bx = L - 0.08 + S.width(big) + 0.22
    S.text(bx, y0 + 1.42, f"\u00b1 {num(sd(ours, 'forgetting_A'))}", size=17, family=MONO, color=ACCENT)
    S.text(bx, y0 + 1.74, f"sd over {len(SEEDS)} seeds", size=11.5, color=MUTED)
    S.text(bx, y0 + 1.98, "below zero: A improved", size=11.5, color=MUTED)

    # forgetting on one scale: three methods, zero marked
    ax = S.axes(L, y0 + 2.5, 6.2, 1.2)
    ax.set_xlim(-0.15, 0.74)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([-0.08, 0.74], [0.62, 0.62], color=HAIR, lw=1.2, solid_capstyle="butt")
    ax.plot([0, 0], [0.46, 0.78], color=INK, lw=1.0)
    ax.text(0, 0.86, "0", ha="center", va="bottom", family=MONO, fontsize=9.5, color=INK)
    for t in (0.2, 0.4):
        ax.plot([t, t], [0.58, 0.66], color=FAINT, lw=0.9)
        ax.text(t, 0.86, f"{t:.1f}", ha="center", va="bottom", family=MONO, fontsize=9.5, color=FAINT)
    ax.text(0.74, 0.86, "more forgetting \u2192", ha="right", va="bottom", family=MONO, fontsize=9.5,
            color=MUTED)
    place = {"naive": ("center", 0), "kl_global": ("left", -0.012), "replay_kl": ("right", 0.012)}
    for m in ORDER:
        f = mu(m, "forgetting_A")
        b = m == ours
        ax.plot([min(0, f), max(0, f)], [0.62, 0.62], color=COLOR[m], lw=5 if b else 3,
                solid_capstyle="butt", zorder=3)
        ax.scatter([f], [0.62], s=110 if b else 55, fc=COLOR[m], ec="white", lw=1.6, zorder=4)
        ha, dx = place[m]
        ax.text(f + dx, 0.32, num(f), ha=ha, va="top", family=MONO, fontsize=11,
                color=DEEP[m], fontweight="semibold" if b else "regular")
        ax.text(f + dx, 0.04, SHORT[m], ha=ha, va="top", fontsize=10, color=DEEP[m],
                fontweight="semibold" if b else "regular")

    # ---- right: where every run lands
    px, py, pw, ph = 8.6, 1.5, 8.45, 7.55
    S.eyebrow(px, 1.22, "Where every run lands after learning book B")
    S.text(R, 1.22, "dot: mean  \u00b7  rings: seeds  \u00b7  shading: \u00b11 sd", size=9, family=MONO,
           color=MUTED, ha="right")
    ax = S.axes(px, py, pw, ph)
    line = plane(ax, pts, start, base=(BASE_B, BASE_A), s=1.25, tick=10)
    plane_labels(ax, pts, start, size=14, start_note="book-A model  \u00b7  every run starts here")
    ax.text(-0.02, 1.06, "\u2191 Book A retained", ha="left", va="top", fontsize=12, color=INK,
            fontweight="medium", zorder=10)
    ax.text(1.065, 0.0, "Book B learned \u2192", ha="right", va="bottom", fontsize=12, color=INK,
            fontweight="medium", zorder=10)
    line_label(ax, line, 0.66, "the straight line through the two baselines", size=12)
    region(ax, 0.47, 0.17, "trade one book\nfor the other", MUTED, size=10.5)
    region(ax, 0.8, 0.7, "keep both", ACCENT, size=10.5)
    ax.annotate("base model", xy=(BASE_B, BASE_A), xytext=(12, 4), textcoords="offset points",
                fontsize=11.5, color=INK, fontweight="medium", path_effects=HALO, zorder=10)
    ax.annotate("knows neither book", xy=(BASE_B, BASE_A), xytext=(12, -12), textcoords="offset points",
                fontsize=12.5, family=SERIF, style="italic", color=MUTED, zorder=10)
    mid = (0.5 * (BASE_B + START_B), 0.5 * (BASE_A + START_A))
    ax.annotate("stage 1: learn book A", xy=mid, xytext=(-8, 0), textcoords="offset points", rotation=90,
                rotation_mode="anchor", ha="center", va="bottom", fontsize=10, family=MONO, color=MUTED,
                path_effects=HALO, zorder=10)

    footer(S, 9.66, L, R, extra="   ·   1,500 held-out questions per book   ·   mean \u00b1 1 sd")
    S.save("hero")


# ---------------------------------------------------------------- method

SAND = "#d3b98c"        # book B statements
SAND_DEEP = "#8a6a36"


def open_book(S, cx, top, w, h, color, lines=4, lw=1.3):
    """An open book seen from above, drawn in page inches (y grows downward)."""
    sag = 0.07
    for side in (-1, 1):
        edge = cx + side * w / 2
        S.page.add_patch(Polygon([(cx, top + sag), (edge, top), (edge, top + h), (cx, top + h + sag)],
                                 closed=True, fc=tint(color, 0.9), ec=color, lw=lw, joinstyle="round", zorder=4))
        for k in range(lines):
            f = (k + 1) / (lines + 1)
            y_in, y_out = top + sag + f * h * 0.92, top + f * h * 0.92
            a, b = cx + side * 0.09, edge - side * 0.09
            S.page.plot([a, b], [y_in - 0.012 * (1 if side else 0), y_out], color=color, lw=0.9, zorder=5,
                        solid_capstyle="round")


def model_card(S, x, y, w, h, title, sub=None, fc="white", ec=INK, layers=3, frozen=False, tcolor=INK):
    """A model as a short stack of rounded cards."""
    for k in reversed(range(layers)):
        d = 0.07 * k
        S.box(x + d, y - d, w, h, fc=fc if k == 0 else tint(ec, 0.92), ec=ec, lw=1.1, r=0.08, z=3 + (layers - k) * 0.1)
    S.text(x + (0.48 if frozen else 0.2), y + h / 2 - (0.08 if sub else 0) + 0.06, title, size=12.5,
           weight="semibold", color=tcolor, va="baseline", zorder=6)
    if sub:
        S.text(x + (0.48 if frozen else 0.2), y + h / 2 + 0.22, sub, size=9, family=MONO, color=MUTED, zorder=6)
    if frozen:
        snowflake(S, x + 0.26, y + h / 2, 0.11, ec)


def snowflake(S, cx, cy, r, color, lw=1.2):
    for k in range(3):
        a = np.pi / 2 + k * np.pi / 3
        dx, dy = r * np.cos(a), r * np.sin(a)
        S.page.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, lw=lw, zorder=7, solid_capstyle="round")
        for sgn in (-1, 1):
            tx, ty = cx + sgn * dx * 0.62, cy + sgn * dy * 0.62
            for t in (a + np.pi / 4 * 3, a - np.pi / 4 * 3):
                ex, ey = sgn * 0.38 * r * np.cos(t), sgn * 0.38 * r * np.sin(t)
                S.page.plot([tx, tx + ex], [ty, ty + ey], color=color, lw=lw * 0.8, zorder=7,
                            solid_capstyle="round")


def low_rank(S, x, y, color, size=0.62, thin=0.1, label=None, sub=None):
    """LoRA as the product of a tall thin and a wide thin matrix."""
    S.box(x, y, thin, size, fc=color, r=0.02, z=5)
    S.text(x + thin + 0.07, y + 0.1, "\u00d7", size=10, color=MUTED, va="center")
    S.box(x + thin + 0.2, y, size, thin, fc=color, r=0.02, z=5)
    if label:
        S.text(x, y + size + 0.32, label, size=12, weight="semibold", color=color)
    if sub:
        S.text(x, y + size + 0.56, sub, size=9, family=MONO, color=MUTED)


def page_arrow(S, a, b, color=INK, lw=1.2, rad=0.0, ls="-", head=(7, 11)):
    return arrow(S.page, a, b, color=color, lw=lw, rad=-rad, head=head, ls=ls, z=6)


def chip(S, x, y, s, fc, color, size=10, family=MONO, weight="medium", ec="none"):
    t = S.text(x, y, s, size=size, family=family, color=color, weight=weight, va="center", zorder=8)
    w = S.width(t) * 1.04
    S.box(x - 0.13, y - 0.155, w + 0.26, 0.31, fc=fc, ec=ec, r=0.155, z=7)
    return x + w + 0.13


def method():
    W, H, L, R = 18, 10.2, 0.95, 17.05
    S = Sheet(W, H)
    header(S, "Method  \u00b7  two LoRA stages, three stage-2 losses compared",
           ("Constrain the model ", "only where book A lives."),
           ["Stage I teaches book A and freezes the result. Stage II trains a fresh adapter on book B, "
            "rehearses book A, and applies KL to the",
            "book-A model on the rehearsal tokens alone. Book-B tokens are left free to change."], L, size=40)
    top, bot = 2.85, 7.6
    x1, x2 = 5.45, 13.95          # column dividers
    S.box(x1 + 0.2, top - 0.12, x2 - x1 - 0.4, bot - top + 0.12, fc=tint(ACCENT, 0.965), r=0.14)
    row = 6.35                    # the model row, shared by stages I and II

    def stage(x, numeral, title, sub, color=INK):
        S.text(x, top + 0.42, numeral, size=34, family=SERIF, color=color)
        S.text(x + 0.12 + 0.24 * len(numeral), top + 0.2, title, size=13.5, weight="semibold", color=color)
        S.text(x + 0.12 + 0.24 * len(numeral), top + 0.45, sub, size=10, color=MUTED)

    # ---- stage I: base model + book A -> LoRA A -> merged, frozen book-A model
    stage(L, "I", "Learn book A", "LoRA on the base model, then merged")
    model_card(S, L + 0.05, 3.95, 2.05, 0.74, MODEL, sub="base model")
    open_book(S, L + 1.05, 5.05, 1.75, 0.66, ACCENT)
    S.text(L + 1.05, 6.08, "Book A", size=12, weight="semibold", color=ACCENT, ha="center")
    S.text(L + 1.05, 6.32, "300 towns \u00d7 5 facts \u00d7 4", size=9, family=MONO, color=MUTED, ha="center")
    low_rank(S, L + 2.95, 4.35, ACCENT, size=0.55, thin=0.09)
    S.text(L + 2.95, 5.22, "LoRA A", size=12, weight="semibold", color=ACCENT)
    S.text(L + 2.95, 5.45, "r=64 \u00b7 CE", size=9, family=MONO, color=MUTED)
    page_arrow(S, (L + 2.25, 4.3), (L + 2.85, 4.55), rad=0.0)
    page_arrow(S, (L + 2.0, 5.4), (L + 2.85, 4.85), rad=0.15)
    S.box(L + 2.15, row, 2.1, 0.78, fc=PAPER, ec=INK, lw=1.1, r=0.08, z=3)
    snowflake(S, L + 2.42, row + 0.39, 0.11, INK)
    S.text(L + 2.65, row + 0.36, "Book-A model", size=12, weight="semibold")
    S.text(L + 2.65, row + 0.58, "merged \u00b7 frozen", size=9, family=MONO, color=MUTED)
    page_arrow(S, (L + 3.25, 5.6), (L + 3.25, row - 0.08))
    S.text(L + 3.38, 5.98, "merge", size=9, family=MONO, color=MUTED)
    page_arrow(S, (L + 4.3, row + 0.39), (x1 + 0.4, row + 0.39))

    # ---- stage II
    gx = x1 + 0.55
    stage(gx, "II", "Learn book B, protect book A", "a fresh LoRA B on top of the frozen book-A model", ACCENT)
    S.eyebrow(gx, 3.95, "Every training batch, drawn to scale", size=8.5)
    nb, na = 32, 13   # bs and round(replay_ratio * bs), as run
    cell, pad, split = 0.128, 0.032, 0.14
    for k in range(nb + na):
        x = gx + k * (cell + pad) + (split if k >= nb else 0)
        S.box(x, 4.12, cell, 0.42, fc=SAND if k < nb else ACCENT, r=0.025, z=4)
    xb_end = gx + nb * (cell + pad) - pad
    xa = xb_end + pad + split
    xa_end = xa + na * (cell + pad) - pad
    for a_, b_, c in ((gx, xb_end, SAND_DEEP), (xa, xa_end, ACCENT)):
        S.page.plot([a_, a_, b_, b_], [4.62, 4.68, 4.68, 4.62], color=c, lw=1.0, zorder=4)
    S.text(gx, 4.95, "32 book-B statements", size=11.5, weight="semibold", color=SAND_DEEP)
    S.text(xa, 4.95, "13 book-A rehearsal", size=11.5, weight="semibold", color=ACCENT)
    e = chip(S, gx + 0.1, 5.3, "CE", "white", INK, ec=HAIR)
    S.text(e + 0.2, 5.3, "and nothing else", size=12, family=SERIF, style="italic", color=MUTED, va="center")
    e = chip(S, xa + 0.1, 5.3, "CE", "white", INK, ec=HAIR)
    chip(S, e + 0.24, 5.3, "+ KL to book-A model", ACCENT, "white", weight="semibold")

    # the network that trains: the frozen book-A model plus LoRA B
    S.box(gx, row, 2.45, 0.78, fc=PAPER, ec=INK, lw=1.1, r=0.08, z=3)
    snowflake(S, gx + 0.27, row + 0.39, 0.11, INK)
    S.text(gx + 0.5, row + 0.36, "Book-A model", size=12, weight="semibold")
    S.text(gx + 0.5, row + 0.58, "frozen", size=9, family=MONO, color=MUTED)
    S.text(gx + 2.72, row + 0.45, "+", size=18, color=MUTED, va="center")
    low_rank(S, gx + 3.12, row + 0.1, ACCENT, size=0.55, thin=0.09)
    S.text(gx + 4.05, row + 0.26, "LoRA B", size=12.5, weight="semibold", color=ACCENT)
    S.text(gx + 4.05, row + 0.5, "fresh \u00b7 r=64", size=9, family=MONO, color=MUTED)
    S.text(gx + 4.05, row + 0.74, "the only weights that train", size=10, color=MUTED)
    page_arrow(S, (gx + 1.25, 5.55), (gx + 1.25, row - 0.08), color=SAND_DEEP)
    page_arrow(S, (xa + 0.3, 5.55), (gx + 3.55, row - 0.06), color=ACCENT, rad=-0.12)
    # the KL reference costs no second copy of the weights
    S.text(xa + 0.62, 5.72, "reference: the same network", size=9.5, color=ACCENT)
    S.text(xa + 0.62, 5.94, "with LoRA B switched off", size=9.5, color=ACCENT)

    # ---- stage III
    ex = x2 + 0.5
    stage(ex, "III", "Evaluate both books", "an unseen question template")
    for k, (book, color) in enumerate((("A", ACCENT), ("B", SAND_DEEP))):
        y = 3.95 + k * 1.3
        S.box(ex, y, R - ex, 1.1, fc="white", ec=HAIR, lw=1.0, r=0.1, z=2)
        S.page.add_patch(Circle((ex + 0.32, y + 0.33), 0.14, fc=color, ec="none", zorder=4))
        S.text(ex + 0.32, y + 0.34, book, size=10, weight="bold", color="white", ha="center", va="center",
               zorder=5)
        S.text(ex + 0.58, y + 0.39, f"Book {book}: 1,500 questions", size=11, weight="semibold")
        S.text(ex + 0.22, y + 0.72, "Question: Who founded the", size=8.6, family=MONO, color=BODY)
        S.text(ex + 0.22, y + 0.92, "town of X?  Answer:", size=8.6, family=MONO, color=BODY)
    S.text(ex, 6.98, "greedy, 12 new tokens; correct if", size=10, color=MUTED)
    S.text(ex, 7.2, "the answer string appears", size=10, color=MUTED)
    # a bus from the trained model to both question sets
    bus = ex - 0.25
    S.page.plot([gx + 6.35, bus, bus], [row + 0.39, row + 0.39, 4.5], color=INK, lw=1.2, zorder=5,
                solid_joinstyle="round")
    for yy in (4.5, 5.8):
        page_arrow(S, (bus, yy), (ex - 0.04, yy))

    # ---- the loss, set as an equation with a reading under each term
    S.rule(L, R, 8.0)
    y = 8.62
    x = S.runs(L, y, [("loss", {"color": MUTED, "style": "italic"}), ("  =  ", {"color": MUTED})], size=28,
               family=SERIF)
    terms = [
        ([("CE(book B)", {})], SAND_DEEP, "the new book, unconstrained"),
        ([("CE(book-A rehearsal)", {})], ACCENT, "a positive target for the old book"),
        ([("kl_w \u00b7 KL(book-A model ", {}), ("\u2016", {"family": SANS, "size": 22}), (" model)", {})],
         ACCENT, "on the rehearsal tokens only, kl_w = 1.0"),
    ]
    for i, (parts, c, reading) in enumerate(terms):
        if i:
            x = S.runs(x, y, [("   +   ", {"color": MUTED})], size=28, family=SERIF)
        t0 = x
        x = S.runs(x, y, parts, size=28, family=SERIF, color=c)
        S.page.plot([t0, t0, x, x], [y + 0.17, y + 0.23, y + 0.23, y + 0.17], color=tint(c, 0.35), lw=1.0)
        S.text(t0, y + 0.5, reading, size=10.5, color=c)
    S.runs(L, H - 0.42, [("BASELINES   ", {"color": FAINT}), ("naive = CE(book B)", {"color": BODY}),
                        ("      KL on new-book tokens = CE(book B) + kl_w \u00b7 KL on those same book-B tokens",
                         {"color": BODY})], size=9, family=MONO)
    S.text(R, H - 0.42, "github.com/antonio-mastropaolo/qwen-sequential-books", size=8.6, family=MONO,
           color=MUTED, ha="right")
    S.save("method")


# ---------------------------------------------------------------- results

def tradeoff():
    W, H, L, R = 11, 11.35, 0.8, 10.2
    S = Sheet(W, H)
    header(S, "Results  \u00b7  the stability-plasticity trade-off", ("Beyond ", "the trade-off line."),
           ["Book A retained against book B learned after stage 2. Arrows run from the book-A",
            "model to each method's mean; the dashed line joins the two baselines."], L, size=38)
    pts, start = plane_points()
    ax = S.axes(L + 0.45, 2.85, R - L - 0.45, 7.0)
    line = plane(ax, pts, start, base=(BASE_B, BASE_A), s=1.0, tick=9.5)
    plane_labels(ax, pts, start, size=12, start_note="book-A model")
    ax.text(-0.02, 1.06, "\u2191 Book A retained", ha="left", va="top", fontsize=11, color=INK,
            fontweight="medium", zorder=10)
    ax.text(1.065, 0.0, "Book B learned \u2192", ha="right", va="bottom", fontsize=11, color=INK,
            fontweight="medium", zorder=10)
    ax.annotate("base model", xy=(BASE_B, BASE_A), xytext=(10, 4), textcoords="offset points",
                fontsize=10.5, color=INK, fontweight="medium", path_effects=HALO, zorder=10)
    line_label(ax, line, 0.66, "the straight line through the two baselines", size=11)
    region(ax, 0.47, 0.17, "trade one book\nfor the other", MUTED, size=9.5)
    region(ax, 0.8, 0.7, "keep both", ACCENT, size=9.5)
    footer(S, 10.97, L, R, extra="   \u00b7   mean \u00b1 1 sd", stacked=True)
    S.save("tradeoff")


def per_seed():
    W, H, L, R = 18, 9.45, 0.95, 17.05
    S = Sheet(W, H)
    header(S, "Results  \u00b7  seed by seed", ("The same picture ", "in every seed."),
           ["One panel per seed: book A retained against book B learned after stage 2, with arrows from "
            "that seed's book-A model."], L, size=38)
    n, gap = len(SEEDS), 0.6
    pw = (R - L - gap * (n - 1)) / n
    for i, seed in enumerate(SEEDS):
        pts, start = plane_points(seed)
        x = L + i * (pw + gap)
        f = runs_of("replay_kl").forgetting_A[seed]
        S.text(x, 2.55, f"Seed {seed}", size=19, family=SERIF)
        S.text(x + pw, 2.55, f"rehearsal + KL forgets {num(f)}", size=9.5, family=MONO, color=ACCENT,
               ha="right")
        S.rule(x, x + pw, 2.7, color=INK, lw=0.8)
        ax = S.axes(x + 0.35, 3.15, pw - 0.35, 4.95)
        plane(ax, pts, start, s=0.8, tick=8.5, detail=False)
        plane_labels(ax, pts, start, size=11.5, start_note="book-A model", notes=False, compact=True)
        ax.text(1.065, 0.0, "B learned \u2192", ha="right", va="bottom", fontsize=9.5, color=INK, zorder=10)
        ax.text(-0.035, 1.1, "\u2191 A retained", ha="left", va="bottom", fontsize=9.5, color=INK, zorder=10)
    footer(S, H - 0.38, L, R, extra="   \u00b7   1,500 held-out questions per book")
    S.save("per_seed")


def accuracy_bars():
    W, H, L, R = 16, 8.0, 0.95, 15.05
    S = Sheet(W, H)
    header(S, "Results  \u00b7  accuracy after stage 2", ("Only one method ", "keeps both books."),
           ["Accuracy on 1,500 held-out questions per book after learning book B. Bar: mean of "
            f"{len(SEEDS)} seeds. Under each bar, a tick per seed and a line for \u00b11 sd.",
            "The dashed mark is the same model before stage 2: the book-A model."], L, size=40)
    rows = list(reversed(ORDER))           # rehearsal + KL on top
    names_w, gutter, val_w = 2.75, 0.45, 0.95
    col_w = (R - L - names_w - 2 * val_w - gutter) / 2
    top, row_h = 3.45, 1.0
    S.box(L - 0.25, top + 0.02, R - L + 0.5, row_h - 0.04, fc=tint(ACCENT, 0.94), r=0.1)
    cols = [("Book A retained", "after_B_A", START_A), ("Book B learned", "after_B_B", START_B)]
    for j, (title, col, before) in enumerate(cols):
        x0 = L + names_w + j * (col_w + val_w + gutter)
        S.text(x0, top - 0.25, title, size=13, weight="semibold")
        S.text(x0 + col_w, top - 0.25, "higher is better \u2192", size=9, family=MONO, color=FAINT, ha="right")
        ax = S.axes(x0, top, col_w, row_h * len(rows))
        ax.set_xlim(0, 1)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.axis("off")
        for t in (0.25, 0.5, 0.75):
            ax.plot([t, t], [-0.5, len(rows) - 0.5], color=HAIR, lw=0.9, zorder=0)
        for i, m in enumerate(rows):
            b = m == "replay_kl"
            r = runs_of(m)
            v, e = mu(m, col), sd(m, col)
            ax.add_patch(Rectangle((0, i - 0.17), 1, 0.3, fc="white" if b else PAPER, ec="none", zorder=1))
            ax.add_patch(Rectangle((0, i - 0.17), v, 0.3, fc=COLOR[m], ec="none", zorder=2))
            ax.plot([v - e, v + e], [i + 0.25, i + 0.25], color=DEEP[m], lw=1.1, zorder=3, solid_capstyle="butt")
            for sv in r[col].values:
                ax.plot([sv, sv], [i + 0.19, i + 0.31], color=DEEP[m], lw=1.1, zorder=3, solid_capstyle="butt")
            S.text(x0 + col_w + 0.14, top + (i + 0.5) * row_h - 0.03, num(v), size=15 if b else 13.5,
                   family=MONO, color=DEEP[m], weight="semibold" if b else "regular", va="center")
        ax.plot([before, before], [-0.42, len(rows) - 0.5], color=INK, lw=1.0, ls=(0, (2, 2.2)), zorder=6)
        S.text(x0 + before * col_w + (-0.08 if j == 0 else 0.08), top + row_h * len(rows) + 0.3,
               f"before stage 2: {num(before)}", size=9.5, family=MONO, color=INK, ha="right" if j == 0 else "left")
        for t in (0, 0.25, 0.5, 0.75, 1.0):
                    S.text(x0 + t * col_w, top + row_h * len(rows) + 0.62, frac_ticks([t])[0], size=9,
                   family=MONO, color=FAINT, ha="center")
    for i, m in enumerate(rows):
        b = m == "replay_kl"
        S.text(L, top + (i + 0.5) * row_h - 0.03, NAME[m], size=13.5 if b else 12.5, color=DEEP[m],
               weight="semibold" if b else "medium", va="center")
    footer(S, H - 0.38, L, R, extra="   \u00b7   mean \u00b1 1 sd")
    S.save("accuracy_bars")


def forgetting():
    W, H, L, R = 16, 8.15, 0.95, 15.05
    S = Sheet(W, H)
    header(S, "Results  \u00b7  forgetting of book A", ("Book A, stage by stage. ", "One method holds it."),
           ["Book-A accuracy of the base model, after stage 1 (book A) and after stage 2 (book B), mean of "
            f"{len(SEEDS)} seeds, thin lines",
            "per seed. Right: forgetting of A, the stage-1 score minus the stage-2 score. Below zero means book A improved."],
           L, size=40)
    # left: slope chart
    ax = S.axes(L + 0.45, 3.35, 5.5, 3.6)
    quiet_axes(ax, (-0.12, 2.12), (-0.04, 1.08), [0, 1, 2], [0, 0.25, 0.5, 0.75, 1], size=9.5)
    ax.set_xticklabels(["base model", "after book A", "after book B"], family=SANS, fontsize=11)
    ax.add_patch(Rectangle((1, -0.04), 1.12, 1.12, fc=PAPER, ec="none", zorder=0.1))
    ax.text(1.06, 1.05, "STAGE 2", family=MONO, fontsize=9, color=MUTED, va="top", fontweight="medium")
    ax.plot([0, 1], [BASE_A, START_A], color=INK, lw=2.0, zorder=3, solid_capstyle="round")
    for m in ORDER:  # each seed, faintly, under its mean
        r = runs_of(m)
        for seed in SEEDS:
            ax.plot([1, 2], [r.after_A_A[seed], r.after_B_A[seed]], color=tint(COLOR[m], 0.6), lw=0.9,
                    zorder=2, solid_capstyle="round")
    ax.errorbar([1], [START_A], yerr=[START_A_SD], fmt="none", ecolor=FAINT, elinewidth=1.2, zorder=2)
    for m in ORDER:
        b = m == "replay_kl"
        v, e = mu(m, "after_B_A"), sd(m, "after_B_A")
        ax.plot([1, 2], [START_A, v], color=COLOR[m], lw=3.2 if b else 1.8, zorder=4 if b else 3,
                solid_capstyle="round")
        ax.scatter([2], [v], s=80 if b else 45, fc=COLOR[m], ec="white", lw=1.5, zorder=5)
    ax.scatter([0, 1], [BASE_A, START_A], s=45, fc="white", ec=INK, lw=1.4, zorder=6)
    ax.annotate(num(BASE_A), xy=(0, BASE_A), xytext=(10, -3), textcoords="offset points", ha="left",
                va="center", family=MONO, fontsize=10, color=INK)
    ax.annotate(num(START_A), xy=(1, START_A), xytext=(-12, 6), textcoords="offset points", ha="right",
                family=MONO, fontsize=10, color=INK, path_effects=HALO)
    for m in ORDER:
        b = m == "replay_kl"
        v = mu(m, "after_B_A")
        ax.annotate(f"{num(v)}  {SHORT[m]}", xy=(2, v), xytext=(12, -4 if m == "kl_global" else (4 if b else 0)),
                    textcoords="offset points", ha="left", va="center", family=MONO, fontsize=10.5,
                    color=DEEP[m], fontweight="semibold" if b else "regular", annotation_clip=False)
    S.text(L, 3.1, "\u2191 Book-A accuracy", size=11, weight="medium")

    # right: forgetting per seed, one row per method
    x0, w, y0, h = 10.55, R - 10.55 - 1.05, 3.35, 3.6
    S.text(L + 8.15, 3.1, "Forgetting of book A, each seed", size=11, weight="medium")
    ax = S.axes(x0, y0, w, h)
    ax.set_xlim(-0.08, 0.72)
    rows = list(reversed(ORDER))
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.axis("off")
    for t in (0.2, 0.4, 0.6):
        ax.plot([t, t], [-0.5, len(rows) - 0.5], color=HAIR, lw=0.9, zorder=0)
        ax.text(t, len(rows) - 0.4, f"{t:.1f}", ha="center", va="top", family=MONO, fontsize=9, color=FAINT)
    ax.plot([0, 0], [-0.5, len(rows) - 0.5], color=INK, lw=1.0, zorder=1)
    ax.text(0, len(rows) - 0.4, "0", ha="center", va="top", family=MONO, fontsize=9, color=INK)
    S.text(x0 + w, 3.1, "more forgetting \u2192", size=9, family=MONO, color=MUTED, ha="right")
    for i, m in enumerate(rows):
        b = m == "replay_kl"
        r = runs_of(m).forgetting_A
        v, e = mu(m, "forgetting_A"), sd(m, "forgetting_A")
        ax.plot([v - e, v + e], [i, i], color=tint(COLOR[m], 0.55), lw=9, solid_capstyle="round", zorder=2)
        ax.scatter(r.values, [i] * len(r), s=44 if b else 34, fc="white", ec=COLOR[m], lw=1.4, zorder=3)
        ax.plot([v, v], [i - 0.17, i + 0.17], color=DEEP[m], lw=2, zorder=4, solid_capstyle="butt")
        cy = y0 + h * (i + 0.5) / len(rows)
        S.text(L + 8.15, cy, SHORT[m], size=11.5, color=DEEP[m], weight="semibold" if b else "regular",
               va="center")
        S.text(R, cy, num(v), size=13 if b else 12, family=MONO, color=DEEP[m],
               weight="semibold" if b else "regular", va="center", ha="right")
    S.text(R, 3.1, "mean", size=9, family=MONO, color=MUTED, ha="right")
    footer(S, H - 0.35, L, R, extra="   \u00b7   mean \u00b1 1 sd")
    S.save("forgetting")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    for name in args.only.split(","):
        globals()[name.strip()]()
