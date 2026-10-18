"""
ORION3 - IEEE-CIS benchmark in ONE script (replaces the scattered notebook cells).

What it does
  * Chronological split by TransactionDT: 70% train / 15% validation / 15% FINAL HOLDOUT.
  * Same 58 features for every model: dummy, logistic regression, decision tree, LightGBM.
  * Validation set: model comparison, threshold choice, feature-group ablation (3 seeds).
  * Final holdout: scored ONCE at the end with the threshold chosen on validation.
  * Saves CSVs and a ready-to-paste README snippet. Nothing is typed in by hand.

Run on Kaggle (after adding the IEEE-CIS competition data to the notebook):
    !git clone https://github.com/SANJANAA-V/ORION3.git
    %cd ORION3
    !python ieee_cis_benchmark.py --data /kaggle/input/competitions/ieee-fraud-detection/train_transaction.csv --out results
Quick test (about 10x faster):  add  --seeds 1
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

FEATURES = (["TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1"]
            + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]
            + [f"M{i}" for i in range(1, 10)] + [f"V{i}" for i in range(1, 11)])
GROUPS = {
    "V features": lambda f: [c for c in f if c.startswith("V")],
    "D features": lambda f: [c for c in f if c.startswith("D")],
    "M features": lambda f: [c for c in f if c.startswith("M")],
    "C features": lambda f: [c for c in f if c.startswith("C")],
    "card attributes": lambda f: [c for c in f if c in ("card1", "card2", "card3", "card5")],
    "address and distance": lambda f: [c for c in f if c in ("addr1", "addr2", "dist1")],
    "time and amount": lambda f: [c for c in f if c in ("TransactionDT", "TransactionAmt")],
}
LGBM_PARAMS = dict(n_estimators=350, learning_rate=0.05, num_leaves=31, subsample=0.9, subsample_freq=1,
                   colsample_bytree=0.9, n_jobs=-1, verbosity=-1)


def load(path):
    header = pd.read_csv(path, nrows=0).columns
    feats = [c for c in FEATURES if c in header]
    if "isFraud" not in header or "TransactionDT" not in header or len(feats) < 20:
        raise SystemExit("This does not look like IEEE-CIS train_transaction.csv")
    df = pd.read_csv(path, usecols=["isFraud"] + feats)
    for c in feats:
        df[c] = df[c].astype("float32") if pd.api.types.is_numeric_dtype(df[c]) else df[c].astype("category")
    return df, feats


def metrics(y, p):
    return {"PR_AUC": float(average_precision_score(y, p)), "ROC_AUC": float(roc_auc_score(y, p))}


def thr_at_fpr(y, p, fpr=0.01):
    return float(np.quantile(p[y == 0], 1 - fpr))             # from VALIDATION negatives only


def at_threshold(y, p, t):
    pred = p >= t; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    return {"threshold": t, "precision": tp / max(1, tp + fp), "recall": tp / max(1, int((y == 1).sum())),
            "FPR": fp / max(1, int((y == 0).sum())), "TP": tp, "FP": fp}


def prep_linear(Xtr, Xs):
    """Median-impute, standardise (clipped) numeric columns, one-hot the categorical ones. Fitted on TRAIN only."""
    num = [c for c in Xtr.columns if str(Xtr[c].dtype) != "category"]
    cat = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
    med = Xtr[num].median(); filled = Xtr[num].fillna(med); mu, sd = filled.mean(), filled.std().replace(0, 1)
    dummies_cols = pd.get_dummies(Xtr[cat].astype(object).fillna("NA"), dtype="float32").columns if cat else []

    def tf(X):
        A = ((X[num].fillna(med) - mu) / sd).clip(-10, 10).astype("float32").values
        if not cat: return A
        D = pd.get_dummies(X[cat].astype(object).fillna("NA"), dtype="float32").reindex(columns=dummies_cols, fill_value=0).values
        return np.hstack([A, D])
    return [tf(X) for X in Xs]


def lgbm(seed=42): return LGBMClassifier(random_state=seed, **LGBM_PARAMS)


def md_table(df, floatfmt="{:.4f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=3, help="seeds for the ablation (use 1 for a quick run)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t_start = time.time()

    df, feats = load(a.data)
    order = np.argsort(df["TransactionDT"].to_numpy(), kind="stable"); n = len(order)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    idx = {"train": order[:i1], "val": order[i1:i2], "holdout": order[i2:]}
    y = {k: df["isFraud"].to_numpy()[v] for k, v in idx.items()}
    X = {k: df.iloc[v][feats].reset_index(drop=True) for k, v in idx.items()}
    del df
    print(f"rows: train={len(y['train']):,} val={len(y['val']):,} holdout={len(y['holdout']):,} | features={len(feats)} | "
          f"fraud rate: " + ", ".join(f"{k}={v.mean():.2%}" for k, v in y.items()))

    # ---------------- models (fit on TRAIN only) ----------------
    preds = {}
    dummy = DummyClassifier(strategy="prior").fit(X["train"].iloc[:, :1], y["train"])
    preds["Dummy (base rate)"] = {k: dummy.predict_proba(X[k].iloc[:, :1])[:, 1] for k in ("val", "holdout")}
    Ltr, Lva, Lho = prep_linear(X["train"], [X["train"], X["val"], X["holdout"]])
    lr = LogisticRegression(max_iter=300, C=1.0).fit(Ltr, y["train"])
    preds["Logistic regression"] = {"val": lr.predict_proba(Lva)[:, 1], "holdout": lr.predict_proba(Lho)[:, 1]}
    dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=100, random_state=42).fit(Ltr, y["train"])
    preds["Decision tree"] = {"val": dt.predict_proba(Lva)[:, 1], "holdout": dt.predict_proba(Lho)[:, 1]}
    t0 = time.time(); gb = lgbm().fit(X["train"], y["train"]); print(f"LightGBM fit: {time.time() - t0:.0f}s")
    preds["LightGBM"] = {k: gb.predict_proba(X[k])[:, 1] for k in ("val", "holdout")}

    rows = []
    for name, p in preds.items():
        v, h = metrics(y["val"], p["val"]), metrics(y["holdout"], p["holdout"])
        rows.append({"model": name, "val_PR_AUC": v["PR_AUC"], "val_ROC_AUC": v["ROC_AUC"],
                     "holdout_PR_AUC": h["PR_AUC"], "holdout_ROC_AUC": h["ROC_AUC"]})
    models = pd.DataFrame(rows); models.to_csv(f"{a.out}/ieee_models.csv", index=False)
    print("\n=== MODEL COMPARISON (same 58 features, chronological) ==="); print(models.round(4).to_string(index=False))

    # ---------------- threshold: chosen on VALIDATION, applied once to HOLDOUT ----------------
    pv, ph = preds["LightGBM"]["val"], preds["LightGBM"]["holdout"]
    thr = thr_at_fpr(y["val"], pv, 0.01)
    op = pd.DataFrame([{"set": "validation", **at_threshold(y["val"], pv, thr)}, {"set": "final holdout", **at_threshold(y["holdout"], ph, thr)}])
    op.to_csv(f"{a.out}/ieee_operating_point.csv", index=False)
    table = pd.DataFrame([{"threshold": t, **{k: v for k, v in at_threshold(y["val"], pv, t).items() if k != "threshold"}}
                          for t in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
    table.to_csv(f"{a.out}/ieee_thresholds_validation.csv", index=False)
    print("\n=== OPERATING POINT (threshold at 1% FPR on validation) ==="); print(op.round(4).to_string(index=False))

    rng = np.random.default_rng(0); boots = []
    for _ in range(200):
        i = rng.integers(0, len(ph), len(ph)); boots.append(average_precision_score(y["holdout"][i], ph[i]))
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f"LightGBM holdout PR-AUC 95% CI (bootstrap): [{ci[0]:.3f}, {ci[1]:.3f}]")

    # ---------------- feature-group ablation on VALIDATION, several seeds ----------------
    seeds = list(range(a.seeds)); abl = []

    def val_pr(cols, seed): return average_precision_score(y["val"], lgbm(seed).fit(X["train"][cols], y["train"]).predict_proba(X["val"][cols])[:, 1])
    full = [val_pr(feats, s) for s in seeds]
    abl.append({"removed": "none (full)", "features": len(feats), "PR_AUC_mean": np.mean(full), "PR_AUC_std": np.std(full), "delta_vs_full": 0.0, "clear_effect": ""})
    for g, pick in GROUPS.items():
        drop = pick(feats); cols = [c for c in feats if c not in drop]
        if not drop or len(cols) < 2: continue
        sc = [val_pr(cols, s) for s in seeds]; d = float(np.mean(sc) - np.mean(full))
        noise = 2 * max(np.std(full), np.std(sc), 0.002)
        abl.append({"removed": g, "features": len(cols), "PR_AUC_mean": np.mean(sc), "PR_AUC_std": np.std(sc), "delta_vs_full": d,
                    "clear_effect": ("use --seeds 3" if len(seeds) < 3 else "yes" if abs(d) > noise else "within noise")})
        print(f"ablation: without {g:22s} PR-AUC {np.mean(sc):.4f}  delta {d:+.4f}")
    abl = pd.DataFrame(abl); abl.to_csv(f"{a.out}/ieee_ablation.csv", index=False)

    gain = pd.DataFrame({"feature": feats, "gain": gb.booster_.feature_importance(importance_type="gain")}).sort_values("gain", ascending=False)
    gain["share_of_total_gain"] = gain.gain / gain.gain.sum(); gain.head(15).to_csv(f"{a.out}/ieee_top_features.csv", index=False)

    # ---------------- ready-to-paste snippet ----------------
    h = op[op.set == "final holdout"].iloc[0]; lg = models[models.model == "LightGBM"].iloc[0]; dm = models[models.model == "Dummy (base rate)"].iloc[0]
    snippet = f"""### IEEE-CIS benchmark (real data, generated by ieee_cis_benchmark.py)

Chronological split by TransactionDT: {len(y['train']):,} train / {len(y['val']):,} validation / {len(y['holdout']):,} final holdout. Same {len(feats)} features for every model.

{md_table(models[['model', 'val_PR_AUC', 'val_ROC_AUC', 'holdout_PR_AUC', 'holdout_ROC_AUC']], '{:.3f}')}

Final holdout, LightGBM: PR-AUC {lg.holdout_PR_AUC:.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f}), ROC-AUC {lg.holdout_ROC_AUC:.3f}; dummy baseline PR-AUC {dm.holdout_PR_AUC:.3f}.
Threshold chosen on validation for 1% false-positive rate ({thr:.3f}): on the holdout, recall {h.recall:.3f}, precision {h.precision:.3f}, false-positive rate {h.FPR:.4f}.

Feature-group ablation (validation, mean of {a.seeds} seeds):

{md_table(abl[['removed', 'features', 'PR_AUC_mean', 'PR_AUC_std', 'delta_vs_full', 'clear_effect']], '{:.4f}')}
"""
    open(f"{a.out}/ieee_readme_snippet.md", "w").write(snippet)
    json.dump({"holdout_PR_AUC_CI95": ci, "val_threshold_1pct_FPR": thr, "rows": {k: int(len(v)) for k, v in y.items()}, "seeds": a.seeds},
              open(f"{a.out}/ieee_summary.json", "w"), indent=1)
    print("\n" + snippet); print(f"done in {(time.time() - t_start) / 60:.1f} min; files in {a.out}/")


if __name__ == "__main__":
    main()
