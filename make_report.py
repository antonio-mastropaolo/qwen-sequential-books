#!/usr/bin/env python3
import argparse, glob, json, os
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results"); a = ap.parse_args()
rows = []
for p in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
    r = json.load(open(p))
    row = {"model": r["model"], "method": r["method"], "seed": r["seed"], "smoke_test": r["smoke_test"]}
    for s in ("base", "after_A", "after_B"):
        for k in ("A", "B"):
            row[f"{s}_{k}"] = r[s][k]
    row["forgetting_A"] = r["after_A"]["A"] - r["after_B"]["A"]
    row["stage2_sec"] = r["stage2_sec"]
    rows.append(row)
if not rows:
    raise SystemExit(f"no result JSONs in {a.dir}")
df = pd.DataFrame(rows).sort_values(["model", "method", "seed"])
df.to_csv(os.path.join(a.dir, "results.csv"), index=False)
smoke = bool(df.smoke_test.any())
cols = ["after_A_A", "after_B_A", "after_B_B", "forgetting_A"]
order = [m for m in ["naive", "kl_global", "replay_kl"] if m in set(df.method)]
g = df.groupby("method")[cols].agg(["mean", "std"]).reindex(order)
nseed = df.groupby("method").seed.nunique().reindex(order)

font_dir = os.environ.get("FIGURE_FONTS")  # optional dir of static Inter TTFs, as for assets/make_figures.py
fonts = glob.glob(os.path.join(font_dir, "*.ttf")) if font_dir else []
if fonts:
    font_manager.fontManager.ttflist = [f for f in font_manager.fontManager.ttflist if f.name != "Inter"]
    for f in fonts:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": "#cbd5e1",
                     "xtick.color": "#64748b", "ytick.color": "#64748b", "xtick.major.size": 0,
                     "ytick.major.size": 0, "text.color": "#0f172a", "axes.labelcolor": "#334155"})
color = {"naive": "#f08a5d", "kl_global": "#3b8fd0", "replay_kl": "#6d28d9"}
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
x = ["base", "after A", "after B"]
for m in order:
    d = df[df.method == m]
    for i, k in enumerate("AB"):
        mu = [d[f"{s}_{k}"].mean() for s in ("base", "after_A", "after_B")]
        sd = [d[f"{s}_{k}"].std(ddof=0) for s in ("base", "after_A", "after_B")]
        ax[i].errorbar(x, mu, yerr=sd, marker="o", ms=6, lw=2.4, capsize=0, label=m, color=color.get(m),
                       mec="white", mew=1.2)
for i, k in enumerate("AB"):
    ax[i].set_title(f"Book-{k} accuracy", loc="left", fontweight="bold")
    ax[i].set_ylim(-0.02, 1.02); ax[i].yaxis.grid(True, color="#e8ecf2"); ax[i].set_axisbelow(True)
ax[0].set_ylabel("held-out QA accuracy"); ax[1].legend(frameon=False)
title = f"{df.model.iloc[0]} (mean ± sd over seeds)"
if smoke:
    title = "SMOKE TEST ONLY, NOT RESULTS: " + title
fig.suptitle(title, color="red" if smoke else "#0f172a", x=0.06, ha="left", fontweight="bold"); fig.tight_layout()
fig.savefig(os.path.join(a.dir, "forgetting.png"), dpi=200, facecolor="white")

f = lambda c, m: f"{g.loc[m, (c, 'mean')]:.3f} ± {0 if pd.isna(g.loc[m, (c, 'std')]) else g.loc[m, (c, 'std')]:.3f}"
lines = []
if smoke:
    lines += ["> **SMOKE TEST ONLY.** Tiny model, tiny data, a few steps on CPU. These numbers only show the",
              "> pipeline runs end to end. They are NOT experimental results and must not be cited.", ""]
lines += [f"# Summary: {df.model.iloc[0]}", "",
          f"Base: A={df.base_A.mean():.3f}, B={df.base_B.mean():.3f}. After stage 1 (book A): "
          f"A={df.after_A_A.mean():.3f}, B={df.after_A_B.mean():.3f}.", "",
          "| method | seeds | A after A | A after B | B after B | forgetting of A |", "|---|---|---|---|---|---|"]
for m in order:
    lines.append(f"| {m} | {nseed[m]} | " + " | ".join(f(c, m) for c in cols) + " |")
lines += ["", "forgetting of A = acc_A(after A) - acc_A(after B). Values are mean ± sd over seeds."]
open(os.path.join(a.dir, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
