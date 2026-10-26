"""Results consolidation (Akshat Yuvan).

Builds every table and summary figure used in the final report from the
results files the Kaggle kernels wrote, and from nothing else (locked
decision 8: no number enters the report without a saved results file).

ENV-A (local Mac). Run from the repo root:
    python results/consolidate.py

Inputs  : kaggle/*/output/*.json, the kernel outputs as committed.
Outputs : results/tables/*.csv and *.md, results/figures/*.png.
A missing input is reported and its row left out; it is never filled in.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "kaggle"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MODELS = ["M1", "M2", "M3", "M4"]
NAMES = {"M1": "M1 (MLP)", "M2": "M2 (CNN)", "M3": "M3 (CRNN)", "M4": "M4 (AST)"}
FOLDS = ["MusicGen_medium", "stable_audio_open", "mustango"]
missing = []


def load(path):
    """Return the parsed JSON, or None (and record it) if the file is absent."""
    p = K / path
    if not p.exists():
        missing.append(str(p.relative_to(ROOT)))
        return None
    return json.loads(p.read_text())


def write(df, name, floatfmt=".4f"):
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=floatfmt) + "\n")
    print(f"\n== {name}\n{df.to_string(index=False)}")


# ---------------------------------------------------------------- Table I
# Closed-set comparison of the four models on the common protocol. M2's
# results file predates the per-generator fields, so those come only from
# the models that recorded them.
main = {
    "M1": load("03b-m1-mlp-rerun/output/m1_results.json"),
    "M2": None,  # m2_results.json lives in the fmc-m2-mels dataset, mirrored below
    "M3": load("05-m3-crnn/output/m3_results.json"),
    "M4": load("09-m4-ast/output/m4_results.json"),
}
m2_local = ROOT / "results" / "inputs" / "m2_results.json"
if m2_local.exists():
    main["M2"] = json.loads(m2_local.read_text())
else:
    missing.append(str(m2_local.relative_to(ROOT)))

rows = []
for m in MODELS:
    r = main[m]
    if r is None:
        continue
    rows.append({"Model": NAMES[m], "Bal. acc": r["balanced_accuracy"], "Precision": r["precision"],
                 "Recall": r["recall"], "F1": r["f1"], "AUC": r["auc"], "EER": r["eer"],
                 "Params": r.get("params"), "Train time (s)": r.get("train_time_s"),
                 "Device": "CPU" if r.get("device") == "cpu" else "T4"})
write(pd.DataFrame(rows), "table1_closed_set")

# ---------------------------------------------------------------- Table II
# E1 leave-one-generator-out, M1-M3 from 06-e1-logo, M4 from 06b-e1-m4.
rows = []
for fold in FOLDS:
    for m in MODELS:
        src = "06b-e1-m4" if m == "M4" else "06-e1-logo"
        r = load(f"{src}/output/e1_{fold}_{m}.json")
        if r is None:
            continue
        rows.append({"Withheld": fold, "Model": m,
                     "Closed bal. acc": r["closed_set"]["balanced_accuracy"],
                     "Open bal. acc": r["open_set"]["balanced_accuracy"],
                     "Delta": r["delta_bal_acc_open_minus_closed"],
                     "Withheld recall": r["open_set"]["recall"],
                     "Open AUC": r["open_set"]["auc"], "Open EER": r["open_set"]["eer"]})
e1 = pd.DataFrame(rows)
write(e1, "table2_e1_logo")
if len(e1):
    mean = (e1.groupby("Model")[["Closed bal. acc", "Open bal. acc", "Delta", "Withheld recall", "Open AUC"]]
              .mean().reindex([m for m in MODELS if m in set(e1.Model)]).reset_index())
    write(mean, "table2b_e1_mean_over_folds")

    fig, axes = plt.subplots(1, len(FOLDS), figsize=(12, 3.6), sharey=True)
    for ax, fold in zip(axes, FOLDS):
        d = e1[e1.Withheld == fold]
        x = range(len(d))
        ax.bar([i - 0.2 for i in x], d["Closed bal. acc"], 0.4, label="closed-set", color="#4C72B0")
        ax.bar([i + 0.2 for i in x], d["Open bal. acc"], 0.4, label="open-set (withheld)", color="#DD8452")
        ax.set_xticks(list(x)); ax.set_xticklabels(d.Model)
        ax.set_title(f"withheld: {fold}", fontsize=10)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--"); ax.set_ylim(0.3, 1.02); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("balanced accuracy"); axes[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "fig_e1_closed_vs_open.png", dpi=200); plt.close()

# ---------------------------------------------------------------- Table III
# E2 test-time ablation, M1-M3 from 07-e2-shortcut-audit, M4 from 07b-e2-m4.
e2 = load("07-e2-shortcut-audit/output/e2_results.json")
e2m4 = load("07b-e2-m4/output/e2_m4_results.json")
rows = []
order = []
if e2:
    order = list(e2["conditions"])
    for c in order:
        for m in ["M1", "M2", "M3"]:
            e = e2["conditions"][c][m]
            rows.append({"Condition": c, "Model": m, "Bal. acc": e["balanced_accuracy"],
                         "Spec. (real)": e["specificity_real"], "Recall (synth)": e["recall"],
                         "AUC": e["auc"]})
if e2m4:
    for c, v in e2m4["conditions"].items():
        e = v["M4"]
        rows.append({"Condition": c, "Model": "M4", "Bal. acc": e["balanced_accuracy"],
                     "Spec. (real)": e["specificity_real"], "Recall (synth)": e["recall"], "AUC": e["auc"]})
e2df = pd.DataFrame(rows)
if len(e2df):
    write(e2df, "table3_e2_long")
    wide = e2df.pivot(index="Condition", columns="Model", values="Bal. acc").reindex(order).reset_index()
    write(wide, "table3b_e2_bal_acc_wide")

    lp = ["intact"] + [c for c in order if c.startswith("lowpass")]
    mp = ["intact"] + [c for c in order if c.startswith("mp3")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, gridspec_kw={"width_ratios": [7, 4]})
    for ax, conds, title in [(axes[0], lp, "low-pass cutoff"), (axes[1], mp, "MP3 bitrate")]:
        for m in MODELS:
            d = e2df[e2df.Model == m].set_index("Condition")
            if not len(d):
                continue
            ax.plot(range(len(conds)), d.loc[conds, "Bal. acc"], "o-", label=NAMES[m])
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c.replace("lowpass_", "").replace("mp3_", "") for c in conds])
        ax.set_title(title, fontsize=10); ax.axhline(0.5, color="grey", lw=0.8, ls="--"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("balanced accuracy (test)"); axes[0].legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(FIG / "fig_e2_degradation.png", dpi=200); plt.close()

# ---------------------------------------------------------------- Interpretability
gc = load("08-m2-gradcam/output/gradcam_m2_results.json")
if gc:
    rows = [{"Group": g, "n": v["n"], "Share of CAM in top-2 rows (5.3-8 kHz)": v["share_top2_freq_rows"],
             "Modal peak row (Hz)": f'{v["peak_row_mode_hz"][0]}-{v["peak_row_mode_hz"][1]}',
             "Clips peaking there": v["frac_clips_peak_at_mode"]} for g, v in gc["groups"].items()]
    write(pd.DataFrame(rows), "table4_gradcam_m2")
att = load("09-m4-ast/output/m4_attention_rollout.json")
if att:
    rows = [{"Group": g, "Share top row": v["share_top_row"], "Share top two rows": v["share_top_two_rows"]}
            for g, v in att["groups"].items()]
    write(pd.DataFrame(rows), "table5_m4_attention_rollout")

# M3 time-order ablation
ab = load("05-m3-crnn/output/m3_time_ablation.json")
if ab:
    rows = [{"Condition": c, **{k: ab[c][k] for k in ["balanced_accuracy", "precision", "recall", "auc"]}}
            for c in ["intact", "block_shuffle_50", "frame_shuffle"]]
    write(pd.DataFrame(rows), "table6_m3_time_ablation")

print("\nmissing inputs:", missing if missing else "none")
