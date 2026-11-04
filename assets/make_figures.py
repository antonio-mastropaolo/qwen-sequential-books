#!/usr/bin/env python3
"""Build the README figures from results/results.csv.

usage: python assets/make_figures.py [--csv results/results.csv] [--out assets]
Writes hero.png, accuracy_bars.png, forgetting.png, tradeoff.png, per_seed.png,
method.png and method.svg. Needs pandas and matplotlib only.
"""
import argparse, os
import pandas as pd
import logging
import matplotlib; matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=os.path.join(HERE, "..", "results", "results.csv"))
ap.add_argument("--out", default=HERE)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# palette (matches the results one-pager)
NAVY, STEEL, ORANGE = "#2f4b6e", "#9db4cc", "#b5651d"
INK, MUTED, GRID, PALE = "#1f2933", "#5f6b7a", "#e3e7ec", "#fcf3ea"
GRAY = "#9aa5b1"
METHODS = ["naive", "kl_global", "replay_kl"]
LABEL = {"naive": "Naive fine-tuning",
         "kl_global": "KL on new-book tokens",
         "replay_kl": "Rehearsal + KL on\nrehearsal prompts"}
LABEL1 = {k: v.replace("\n", " ") for k, v in LABEL.items()}
MCOL = {"naive": GRAY, "kl_global": "#5b7fa6", "replay_kl": ORANGE}
SEED_MARK = {0: "o", 1: "s", 2: "^"}

plt.rcParams.update({
    "font.family": ["Source Sans Pro", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "xtick.color": INK, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 200, "savefig.facecolor": "white",
    "svg.fonttype": "path",
})

df = pd.read_csv(args.csv)
df = df[~df.smoke_test.astype(str).str.lower().eq("true")]
order = [m for m in METHODS if m in set(df.method)]
cols = ["after_A_A", "after_B_A", "after_B_B", "forgetting_A"]
g = df.groupby("method")[cols].agg(["mean", "std"]).reindex(order)  # sd: ddof=1, as summary.md
mu = lambda m, c: g.loc[m, (c, "mean")]
sd = lambda m, c: g.loc[m, (c, "std")]
n_seeds = df.seed.nunique()
model = df.model.iloc[0]
base_A, base_B = df.base_A.mean(), df.base_B.mean()
afterA_A, afterA_B = df.drop_duplicates("seed").after_A_A, df.drop_duplicates("seed").after_A_B
foot = f"{model} · LoRA r=64 · {n_seeds} seeds, mean ± sd · 1,500 held-out questions per book"


def save(fig, name, svg=False):
    fig.savefig(os.path.join(args.out, name + ".png"), bbox_inches="tight", pad_inches=0.25)
    if svg:
        fig.savefig(os.path.join(args.out, name + ".svg"), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", os.path.join(args.out, name + ".png"))


def grid_y(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.9); ax.set_axisbelow(True)


def bars(ax, show_vals=True, fs=10.5):
    """Grouped bars: book A retained and book B learned after stage 2."""
    w, gap = 0.36, 0.04
    for i, m in enumerate(order):
        if m == "replay_kl":
            ax.axvspan(i - 0.48, i + 0.48, color=PALE, zorder=0)
        for j, (c, col) in enumerate([("after_B_A", NAVY), ("after_B_B", STEEL)]):
            x = i + (-(w + gap) / 2 if j == 0 else (w + gap) / 2)
            ax.bar(x, mu(m, c), w, color=col, zorder=2)
            ax.errorbar(x, mu(m, c), yerr=sd(m, c), color=INK, capsize=3.5, lw=1.1, zorder=3)
            if show_vals:
                ax.text(x, mu(m, c) + sd(m, c) + 0.02, f"{mu(m, c):.3f}", ha="center", va="bottom",
                        fontsize=fs, color=INK, fontweight="semibold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL[m] for m in order])
    for t, m in zip(ax.get_xticklabels(), order):
        if m == "replay_kl":
            t.set_fontweight("bold")
    ax.set_ylim(0, 1.12); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlim(-0.55, len(order) - 0.45)
    grid_y(ax)


# ---------------------------------------------------------------- hero
fig = plt.figure(figsize=(13, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.22)
ax = fig.add_subplot(gs[0])
stages = ["Base model", "After book A", "After book B"]
for m in order:
    d = df[df.method == m]
    y = [d.base_A.mean(), d.after_A_A.mean(), d.after_B_A.mean()]
    e = [d.base_A.std(), d.after_A_A.std(), d.after_B_A.std()]
    ax.errorbar(range(3), y, yerr=e, color=MCOL[m], marker="o", ms=7, lw=2.6 if m == "replay_kl" else 2,
                capsize=3, zorder=3 if m == "replay_kl" else 2)
    ax.annotate(f"{y[2]:.3f}", (2, y[2]), xytext=(10, 0), textcoords="offset points", va="center",
                color=MCOL[m] if m != "naive" else MUTED, fontweight="bold", fontsize=11.5)
ax.set_xticks(range(3)); ax.set_xticklabels(stages)
ax.set_xlim(-0.2, 2.45); ax.set_ylim(-0.03, 1.05)
ax.set_ylabel("Book-A accuracy (held-out questions)")
ax.set_title("Does the model still know book A?", loc="left")
ax.axvspan(1.5, 2.45, color="#f5f7fa", zorder=0)
ax.text(1.98, 0.06, "stage 2:\ntrain on book B", ha="center", fontsize=10, color=MUTED)
grid_y(ax)
ax.legend(handles=[Line2D([], [], color=MCOL[m], marker="o", lw=2, label=LABEL1[m]) for m in order],
          loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=10.5, ncol=2)
ax2 = fig.add_subplot(gs[1])
bars(ax2)
ax2.set_ylabel("Accuracy after book B")
ax2.set_title("After book B: book A kept vs book B learned", loc="left")
ax2.legend(handles=[Patch(color=NAVY, label="Book A (retained)"), Patch(color=STEEL, label="Book B (newly learned)")],
           loc="upper center", ncol=2, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.17))
fig.suptitle("Learn a second book without forgetting the first", x=0.065, ha="left", y=1.04,
             fontsize=19, fontweight="bold", color="#16222f")
fmt = lambda v: f"{v:.3f}".replace("-", "−")
fig.text(0.065, 0.955, f"Forgetting of book A:  naive {fmt(mu('naive','forgetting_A'))}  ·  KL on new-book tokens "
         f"{fmt(mu('kl_global','forgetting_A'))}  ·  rehearsal + KL on rehearsal prompts "
         f"{fmt(mu('replay_kl','forgetting_A'))}", fontsize=12, color=ORANGE, fontweight="semibold")
fig.text(0.065, -0.16, foot, fontsize=9.5, color=MUTED)
save(fig, "hero")

# ---------------------------------------------------------------- accuracy bars
fig, ax = plt.subplots(figsize=(8.6, 5))
bars(ax)
ax.axhline(mu("naive", "after_A_A"), color=NAVY, ls=(0, (4, 3)), lw=1, zorder=1)

ax.set_ylabel("Accuracy on 1,500 held-out questions")
ax.set_title("Accuracy on each book after training on book B", loc="left")
ax.legend(handles=[Patch(color=NAVY, label="Book A (retained)"), Patch(color=STEEL, label="Book B (newly learned)"),
                   Line2D([], [], color=NAVY, ls=(0, (4, 3)), lw=1,
                          label=f"Book A before stage 2 ({mu('naive','after_A_A'):.3f})")],
          loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.16))
fig.text(0.01, -0.14, foot + " · error bars ± sd", fontsize=9, color=MUTED, transform=fig.transFigure)
save(fig, "accuracy_bars")

# ---------------------------------------------------------------- forgetting
fig, ax = plt.subplots(figsize=(8.6, 3.9))
ys = list(range(len(order)))[::-1]
for y, m in zip(ys, order):
    v = mu(m, "forgetting_A")
    ax.barh(y, v, 0.56, color=MCOL[m], zorder=2)
    ax.errorbar(v, y, xerr=sd(m, "forgetting_A"), color=INK, capsize=3.5, lw=1.1, zorder=3)
    for _, r in df[df.method == m].iterrows():
        ax.scatter(r.forgetting_A, y + 0.36, marker=SEED_MARK.get(r.seed, "o"), s=22, color=INK,
                   facecolor="white", lw=1, zorder=4)
    txt = f"{v:+.3f} ± {sd(m,'forgetting_A'):.3f}".replace("-", "−")
    xt = max(v + sd(m, "forgetting_A"), 0) + 0.02
    ax.text(xt, y, txt, va="center", fontsize=11, color=INK, fontweight="semibold")
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(ys); ax.set_yticklabels([LABEL1[m] for m in order])
ax.set_xlim(-0.1, 0.92)
ax.set_xlabel("Forgetting of book A = acc(A after stage 1) − acc(A after stage 2)   (lower is better)")
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.set_title("How much of book A is lost while learning book B", loc="left")
ax.legend(handles=[Line2D([], [], marker=SEED_MARK[s], ls="", color=INK, mfc="white", label=f"seed {s}")
                   for s in sorted(df.seed.unique())], loc="lower right", frameon=False, fontsize=9.5, ncol=3)
save(fig, "forgetting")

# ---------------------------------------------------------------- trade-off scatter
fig, ax = plt.subplots(figsize=(7.6, 6.4))
ax.add_patch(plt.Rectangle((0.85, 0.85), 0.2, 0.2, color="#eef6ee", zorder=0))
ax.text(1.04, 0.835, "ideal corner:\nkeep A, learn B", ha="right", va="top", fontsize=10.5, color="#3c6e47",
        fontweight="semibold")
for m in order:
    d = df[df.method == m]
    for _, r in d.iterrows():
        ax.scatter(r.after_B_B, r.after_B_A, marker=SEED_MARK.get(r.seed, "o"), s=40, color=MCOL[m],
                   alpha=0.55, lw=0, zorder=2)
    ax.errorbar(mu(m, "after_B_B"), mu(m, "after_B_A"), xerr=sd(m, "after_B_B"), yerr=sd(m, "after_B_A"),
                fmt="o", ms=13, color=MCOL[m], mec="white", mew=1.8, capsize=0, elinewidth=1.4, zorder=3)
ann = {"naive": ((0.975, 0.30), (-16, 22), "right"),
       "kl_global": ((0.44, 0.82), (0, -38), "center"),
       "replay_kl": ((0.886, 0.957), (-30, -34), "right")}
for m in order:
    (x, y), off, ha = ann[m]
    ax.annotate(f"{LABEL1[m]}\nA {mu(m,'after_B_A'):.3f} · B {mu(m,'after_B_B'):.3f}",
                (mu(m, "after_B_B"), mu(m, "after_B_A")), xytext=off, textcoords="offset points", ha=ha,
                va="center", fontsize=10.5, color=INK if m != "replay_kl" else ORANGE, fontweight="semibold")
ax.scatter([afterA_B.mean()], [afterA_A.mean()], marker="D", s=60, color=NAVY, zorder=3)
ax.annotate(f"after book A only\n(starting point of stage 2)", (afterA_B.mean(), afterA_A.mean()),
            xytext=(12, -6), textcoords="offset points", fontsize=10, color=NAVY, va="top")
ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
ax.set_xlabel("Book B learned (accuracy after stage 2)")
ax.set_ylabel("Book A retained (accuracy after stage 2)")
ax.set_title("Stability vs plasticity: retained A vs learned B", loc="left")
ax.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], marker="o", ms=11, ls="", color=MUTED, mec="white", label="mean ± sd"),
                   *[Line2D([], [], marker=SEED_MARK[s], ls="", color=MUTED, alpha=0.6, label=f"seed {s}")
                     for s in sorted(df.seed.unique())]],
          loc="lower left", frameon=False, fontsize=9.5, bbox_to_anchor=(0.0, 0.08))
save(fig, "tradeoff")

# ---------------------------------------------------------------- per-seed
panels = [("after_B_A", "Book A after stage 2", (0, 1.05)), ("after_B_B", "Book B after stage 2", (0, 1.05)),
          ("forgetting_A", "Forgetting of book A", (-0.1, 0.75))]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, (c, title, yl) in zip(axs, panels):
    for i, m in enumerate(order):
        d = df[df.method == m].sort_values("seed")
        for k, (_, r) in enumerate(d.iterrows()):
            ax.scatter(i + (k - 1) * 0.12, r[c], marker=SEED_MARK.get(r.seed, "o"), s=46, color=MCOL[m],
                       zorder=3, edgecolor="white", lw=0.6)
        ax.hlines(mu(m, c), i - 0.28, i + 0.28, color=INK, lw=1.6, zorder=2)
        ax.text(i + 0.31, mu(m, c), f"{mu(m,c):.3f}".replace("-", "−"), va="center", fontsize=9.5, color=INK)
    if c == "forgetting_A":
        ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([LABEL[m] for m in order], fontsize=9.5)
    ax.set_xlim(-0.5, len(order) - 0.35); ax.set_ylim(*yl)
    ax.set_title(title, loc="left", fontsize=12); grid_y(ax)
axs[0].legend(handles=[Line2D([], [], marker=SEED_MARK[s], ls="", color=MUTED, label=f"seed {s}")
                       for s in sorted(df.seed.unique())] +
              [Line2D([], [], color=INK, lw=1.6, label="mean")], loc="lower left", frameon=False,
              fontsize=9.5, ncol=2)
fig.suptitle("Every seed, every method", x=0.06, ha="left", fontsize=15, fontweight="bold", color="#16222f")
fig.tight_layout()
save(fig, "per_seed")

# ---------------------------------------------------------------- method diagram
fig, ax = plt.subplots(figsize=(13, 5.4))
ax.set_xlim(0, 130); ax.set_ylim(0, 54); ax.axis("off")


def box(x, y, w, h, title, sub="", fc="white", ec=NAVY, tc=INK, lw=1.6, tsize=12.5, ssize=10.2, ty=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.6", fc=fc, ec=ec, lw=lw))
    if not sub:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=tc)
        return
    ty = y + h - 3.2 if ty is None else ty
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=tc)
    ax.text(x + w / 2, (y + ty - 2.2) / 2, sub, ha="center", va="center", fontsize=ssize, color=MUTED,
            linespacing=1.4)


def arrow(x1, y1, x2, y2, col=INK, ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=col, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0))

ax.text(1, 52, "STAGE 1 · learn book A", fontsize=11, color=NAVY, fontweight="bold")
ax.text(44, 52, "STAGE 2 · learn book B on a fresh LoRA while protecting book A", fontsize=11, color=ORANGE,
        fontweight="bold")
ax.plot([40.8, 40.8], [1, 53], color=GRID, lw=1.2)
# stage 1
box(1, 38, 17, 10, "Qwen2.5-7B", "base model")
box(1, 20, 17, 12, "Book A text", "300 towns × 5 facts\n× 4 paraphrases")
box(22, 29, 15, 13, "LoRA A", "r=64, CE loss,\nthen merged")
box(22, 3, 15, 15, "Book-A model", "frozen; KL\nreference in\nstage 2", fc="#eef2f7")
arrow(18.6, 41, 21.6, 38); arrow(18.6, 27, 21.6, 32); arrow(29.5, 28.4, 29.5, 18.6)
# stage 2 batch
ax.add_patch(FancyBboxPatch((45, 2), 40, 45, boxstyle="round,pad=0.3,rounding_size=2", fc="#fafbfc",
                            ec="#d3d9e0", lw=1.2))
ax.text(65, 44.3, "every training batch", ha="center", fontsize=10.5, color=MUTED, style="italic")
box(47.5, 28, 35, 13, "32 book-B statements", "cross-entropy only\n(no KL on new-book tokens)", fc="#eef3f8",
    ec=STEEL)
box(47.5, 4.5, 35, 18, "13 book-A rehearsal statements", "taken from the original book-A text\ncross-entropy + KL(book-A model ‖ model)\n"
    "on these rehearsal examples only", fc=PALE, ec=ORANGE, tsize=12)
arrow(37.6, 10.5, 47, 12, col=ORANGE, ls="--")
ax.text(42.2, 13.4, "KL target", fontsize=9.5, color=ORANGE, ha="center")
# fresh LoRA + eval
box(90, 21, 17, 14, "Fresh LoRA B", "r=64, on top of the\nfrozen book-A model", ec=ORANGE, lw=2)
arrow(85.6, 28, 89.6, 28)
box(111.5, 15, 17.5, 26, "Evaluate", "1,500 held-out\nquestions per book\n\ncorrect if the gold\nanswer appears in\n"
    "12 greedy tokens", ec=MUTED)
arrow(107.6, 28, 111.1, 28)
save(fig, "method", svg=True)
