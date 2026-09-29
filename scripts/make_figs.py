import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import os
os.makedirs("figs", exist_ok=True)
import json, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
r2 = json.load(open("results2.json"))
# per-fold "all labels" N differs slightly (~10-11k); pool those into one point at their mean N
small = sorted({b["N"] for b in r2["budget"] if b["N"] <= 2048})
allb = [b for b in r2["budget"] if b["N"] > 2048]
Ns = small + [int(np.mean([b["N"] for b in allb]))]
ap = [np.mean([b["auprc"] for b in r2["budget"] if b["N"] == N]) for N in small] + [np.mean([b["auprc"] for b in allb])]
inc = np.mean([b["inc_auprc"] for b in r2["budget"]])
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.semilogx(Ns, ap, "o-", label="linear probe on frozen DINO tokens (N labeled tokens)")
ax.axhline(inc, color="r", ls="--", label="incumbent: 1 prototype (256 clicks) + per-chunk min-max")
ax.set_xlabel("labeled tokens used to fit the probe (held-out segments, 4 folds)"); ax.set_ylabel("AUPRC (ink vs non-ink surface tokens)")
ax.set_title("Label-budget curve — PHerc. Paris 4, 8 segments, held-out by segment"); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figs/label_budget_curve.png", dpi=110)
fl = json.load(open("fulllabel_results.json"))
fig, ax = plt.subplots(figsize=(7, 4.2)); names = ["UNet x incumbent S", "UNet x probe", "UNet alone"]
for i, (k, c) in enumerate(zip(["inc", "probe", "unet"], ["#c44", "#27a", "#888"])):
    ax.bar(np.arange(2) + i * 0.27, [fl[k]["auprc"], fl[k]["auroc"]], 0.27, label=names[i], color=c)
ax.set_xticks(np.arange(2) + .27, ["AUPRC", "AUROC"]); ax.set_ylim(0, 1); ax.legend(fontsize=8)
ax.set_title("Full DINO-guided label (UNet x S) at held-out label points"); plt.tight_layout(); plt.savefig("figs/full_label.png", dpi=110)
print("figs ok")
