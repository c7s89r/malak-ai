# -*- coding: utf-8 -*-
# makes the train.png graph from the loss numbers train.py wrote down.
# left side = the loss going down. right side = a little cheat-sheet telling
# you what every line/color means so you're not just staring at squiggles.
# run: python plot_train.py --out_dir=out-egyptian
import os
import sys
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# grab --out_dir if you passed one
out_dir = "out-egyptian"
for a in sys.argv[1:]:
    if a.startswith("--out_dir="):
        out_dir = a.split("=", 1)[1]

csv_path = os.path.join(out_dir, "loss_log.csv")
if not os.path.exists(csv_path):
    raise SystemExit(f"no loss log at {csv_path}. Run training first.")

iters, train_loss, val_loss = [], [], []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        iters.append(int(row["iter"]))
        train_loss.append(float(row["train_loss"]))
        val_loss.append(float(row["val_loss"]))

best_val = min(val_loss)
best_i = iters[val_loss.index(best_val)]
final_gap = val_loss[-1] - train_loss[-1]

fig, (ax, ax_txt) = plt.subplots(
    1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [3, 2]}
)

# left side: the actual graph
ax.plot(iters, train_loss, color="#1f77b4", lw=2, marker="o", ms=3, label="train loss")
ax.plot(iters, val_loss, color="#d62728", lw=2, marker="s", ms=3, label="val loss")

# star on the best spot (lowest val = best brain)
ax.scatter([best_i], [best_val], color="green", zorder=5, s=90, marker="*")
ax.annotate(
    f"best val = {best_val:.3f}\n(checkpoint saved here)",
    xy=(best_i, best_val),
    xytext=(best_i + (max(iters) * 0.08), best_val + 0.4),
    arrowprops=dict(arrowstyle="->", color="green"),
    color="green", fontsize=9,
)

# point at where it started (knows nothing yet)
ax.annotate(
    "start: random weights\n(loss ≈ ln(vocab_size))",
    xy=(iters[0], train_loss[0]),
    xytext=(iters[0] + max(iters) * 0.1, train_loss[0] - 0.2),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=9, color="gray",
)

# color in the warmup bit at the start
warmup = 100
ax.axvspan(0, warmup, color="orange", alpha=0.12)
ax.text(warmup / 2, max(train_loss) * 0.97, "lr\nwarmup", ha="center",
        va="top", fontsize=8, color="darkorange")

ax.set_xlabel("training iteration", fontsize=11)
ax.set_ylabel("cross-entropy loss (lower = better)", fontsize=11)
ax.set_title("Egyptian-Arabic baby GPT — training curve", fontsize=13, weight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)

# right side: the cheat-sheet text
ax_txt.axis("off")
explanation = (
    "HOW TO READ THIS PLOT (line by line)\n"
    "------------------------------------------------\n\n"
    "X axis  : training iteration (one optimizer step\n"
    "          over a batch of 64 sequences).\n\n"
    "Y axis  : cross-entropy loss. It's how 'surprised'\n"
    "          the model is by the next character.\n"
    "          Lower = it predicts Egyptian text better.\n\n"
    "BLUE line  (train loss):\n"
    "   error on data the model is learning from.\n"
    "   Goes down as it memorizes patterns.\n\n"
    "RED line  (val loss):\n"
    "   error on held-out data it never trains on.\n"
    "   This is the honest measure of 'speaking well'.\n\n"
    "GREEN star:\n"
    "   lowest val loss = best model. train.py saves\n"
    "   the checkpoint exactly at this point.\n\n"
    "ORANGE band (warmup):\n"
    "   learning rate ramps up slowly here so training\n"
    "   starts stable, then cosine-decays after.\n\n"
    "GAP between red and blue:\n"
    "   small gap = healthy. If red rises while blue\n"
    "   keeps falling -> overfitting (dataset too small,\n"
    "   so generate more: N_CONVOS=60000).\n\n"
    "------------------------------------------------\n"
    f"final train loss : {train_loss[-1]:.3f}\n"
    f"final val   loss : {val_loss[-1]:.3f}\n"
    f"best  val   loss : {best_val:.3f}  @ iter {best_i}\n"
    f"train/val gap    : {final_gap:+.3f}\n"
    "------------------------------------------------\n"
    "DATASET format the model learned:\n"
    "   <user> <question in Egyptian>\n"
    "   <assistant> <reply in Egyptian>\n"
)
ax_txt.text(
    0.0, 1.0, explanation, va="top", ha="left", fontsize=9.5,
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="#f7f7f7", edgecolor="#cccccc"),
)

plt.tight_layout()
out_png = os.path.join(out_dir, "train.png")
fig.savefig(out_png, dpi=130, bbox_inches="tight")
fig.savefig("train.png", dpi=130, bbox_inches="tight")
print(f"wrote {out_png} and ./train.png")
