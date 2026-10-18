"""
ORION3 - evaluate the MESSAGE model on text it never saw, written by other people.

Inputs (all optional, but give at least --classmates):
  --classmates  CSV with columns  message,label   (label = scam or legit)  written by classmates/friends
  --smishing    a public SMS-phishing CSV (see --smishing-text-col / --smishing-label-col / --smishing-pos)
  --uci-path    UCI SMSSpamCollection file (tab separated). If omitted the script downloads it.

It reports, for each test set:  rows, scam share, PR-AUC, ROC-AUC, precision / recall / false-positive rate at 0.5,
for TWO text models:
  1. "Demo model"  : the message model inside the demo (trained on SIMULATED template messages)
  2. "UCI model"   : TF-IDF + logistic regression trained on the real UCI SMS Spam Collection
and saves every wrong prediction so you can read the mistakes.

Run on Kaggle / Colab after cloning the repo:
    !python text_external_eval.py --classmates classmates.csv
"""
import argparse
import io
import os
import urllib.request
import zipfile
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def load_uci(path=None):
    if path:
        raw = pd.read_csv(path, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    else:
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(UCI_URL, timeout=30).read()))
        name = next(n for n in z.namelist() if n.endswith("SMSSpamCollection"))
        raw = pd.read_csv(z.open(name), sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    raw = raw.dropna().drop_duplicates("message")
    return raw["message"].astype(str), (raw["label"].str.lower() == "spam").astype(int)


def to_binary(series, positives):
    pos = {p.strip().lower() for p in positives}
    return series.astype(str).str.strip().str.lower().isin(pos).astype(int)


def evaluate(name, test_name, texts, y, p, out_dir):
    pred = (p >= 0.5).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    row = {"test_set": test_name, "model": name, "rows": len(y), "scam_share": float(np.mean(y)),
           "PR_AUC": float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
           "ROC_AUC": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
           "precision@0.5": tp / max(1, tp + fp), "recall@0.5": tp / max(1, tp + fn), "FPR@0.5": fp / max(1, fp + tn)}
    wrong = pd.DataFrame({"message": list(texts), "true_label": y, "model_score": p})[pred != y]
    wrong.sort_values("model_score").to_csv(os.path.join(out_dir, f"errors_{test_name}_{name.split()[0].lower()}.csv"), index=False)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classmates"); ap.add_argument("--smishing"); ap.add_argument("--uci-path")
    ap.add_argument("--smishing-text-col", default="TEXT"); ap.add_argument("--smishing-label-col", default="LABEL")
    ap.add_argument("--smishing-pos", default="smishing", help="comma separated labels that count as scam")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if not (a.classmates or a.smishing): raise SystemExit("Give --classmates and/or --smishing")

    tests = {}
    if a.classmates:
        d = pd.read_csv(a.classmates).dropna(subset=["message", "label"]).drop_duplicates("message")
        tests["classmates"] = (d["message"].astype(str), to_binary(d["label"], ["scam", "1", "spam", "fraud"]))
    if a.smishing:
        d = pd.read_csv(a.smishing, encoding="utf-8", encoding_errors="ignore").dropna(subset=[a.smishing_text_col, a.smishing_label_col])
        d = d.drop_duplicates(a.smishing_text_col)
        tests["smishing_public"] = (d[a.smishing_text_col].astype(str), to_binary(d[a.smishing_label_col], a.smishing_pos.split(",")))

    # model 2: trained on real UCI spam
    uci_x, uci_y = load_uci(a.uci_path)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    uci_model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(vec.fit_transform(uci_x), uci_y)
    # model 1: the text model inside the demo (trained on simulated templates)
    import orion3_core as core
    demo = core.load_bundle()["models"]

    rows = []
    for tname, (texts, y) in tests.items():
        y = y.to_numpy()
        rows.append(evaluate("Demo model (simulated training)", tname, texts, y, demo["txt"].predict_proba(demo["vec"].transform(texts))[:, 1], a.out))
        rows.append(evaluate("UCI model (real spam training)", tname, texts, y, uci_model.predict_proba(vec.transform(texts))[:, 1], a.out))
    res = pd.DataFrame(rows); res.to_csv(os.path.join(a.out, "text_external_results.csv"), index=False)
    print(res.round(3).to_string(index=False))
    cols = list(res.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in res.iterrows(): md.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    open(os.path.join(a.out, "text_external_snippet.md"), "w").write("### Message model on text written by other people\n\n" + "\n".join(md) + "\n")
    print(f"\nSaved results and the wrong predictions (errors_*.csv) in {a.out}/. READ the errors: they show what the model misses.")


if __name__ == "__main__":
    main()
