"""
ORION3 - test the MESSAGE models on REAL messages written by other people (no classmates needed).

Data (stored in data/text_train.csv.gz, see data/ATTRIBUTION.md):
  * real smishing: "Fishing for Smishing" dataset, IMC 2025 (Agarwal, Papasavva, Suarez-Tangil, Vasek), CC BY 4.0.
    The scam test set is split BY TEMPLATE GROUP: test messages come from templates the model never saw in training.
  * real legitimate SMS: UCI SMS Spam Collection "ham" (casual messages, older, UK/Singapore).
  * AUTHOR-WRITTEN legitimate bank / UPI / bill notifications (NOT real messages): set B is held out for testing.

Three message models, all with the SAME threshold rule (the score giving 1% false alarms on real validation ham):
  1. "Real-data model"          : the message model inside the demo (trained on the real corpus, train split)
  2. "Simulated-template model" : trained on the simulator's old template messages (the earlier demo model)
  3. "UCI spam model"           : TF-IDF + logistic regression trained on UCI spam vs ham only

Reports recall on real unseen-template smishing (overall, Indian networks, by scam type) and false alarms on
(a) real casual legitimate SMS and (b) author-written formal legitimate notifications.
PR-AUC is not reported on purpose: scam and legitimate messages come from different sources, so a ranking score would
partly measure the source.

Run:  python text_real_eval.py          (about 1 minute; downloads UCI spam once)
"""
import os
import urllib.request
import warnings

import numpy as np
import pandas as pd

import orion3_text as OT

warnings.filterwarnings("ignore")
OUT = "results"


def uci_spam_model(corpus):
    path = "/tmp/orion3_uci.tsv"
    if not os.path.exists(path): urllib.request.urlretrieve(OT.UCI_URL, path)
    uci = pd.read_csv(path, sep="\t", header=None, names=["label", "message"]).dropna().drop_duplicates("message")
    spam = uci[uci.label == "spam"].message.map(OT.normalize_text)
    ham_train = corpus[(corpus.source == "uci_ham") & (corpus.split == "train")].text_norm       # never sees val/test ham
    texts = list(spam) + list(ham_train)
    return OT.build_text_model(texts, [1] * len(spam) + [0] * len(ham_train))


def simulated_template_model():
    """The earlier demo model: trained on hand-written template scam messages like the old simulator used (comparison only)."""
    scam = ["{b} KYC update pending. Verify now or your account will be blocked {l}", "Dear customer your {b} account will be suspended today, complete KYC {l}",
            "KYC expired. Update PAN and Aadhaar details immediately {l}", "{b}: account on hold, verify your details at {l} within 24 hours",
            "Congratulations! You won Rs {a} in the lucky draw. Claim your reward {l}", "You are selected for a cashback reward of Rs {a}. Pay processing fee to claim",
            "Prize notification: claim your gift voucher before it expires {l}", "{b} customer care: your refund of Rs {a} is pending, share details to receive it",
            "Support team here. We detected an issue with your account, call {l} to fix", "Your complaint is being processed, please confirm card details with our executive",
            "To resolve your issue install AnyDesk and share the code with our agent", "Our technician needs remote access to your phone to reverse the payment, install the app {l}",
            "Download the support app and give us the 9 digit code to refund your money", "Pay Rs {a} immediately to avoid a penalty and legal action",
            "Your electricity will be cut tonight, transfer Rs {a} now to this number", "Last notice: send the pending amount today or your account will be reported"]
    legit = ["rent for october", "thanks for dinner", "chai", "groceries", "split the bill", "electricity bill", "school fees", "mobile recharge",
             "movie tickets", "birthday gift", "auto fare", "milk", "lunch", "returning the money you lent", "tuition fee", "gym fee",
             "Your {b} statement is ready, verify account details at the branch", "Reward points credited to your account",
             "Urgent: electricity bill due tomorrow, pay at the office counter", "Your refund of Rs {a} has been processed to your account",
             "Dear customer, complete your KYC at the nearest {b} branch with documents", "Payment reminder: insurance premium is due, pay via the official app"]
    rng = np.random.default_rng(0); banks = ["SBI", "HDFC", "ICICI", "Axis", "Canara"]
    fill = lambda t: t.format(b=rng.choice(banks), a=int(rng.integers(500, 50000)), l="bit.ly/" + "".join(rng.choice(list("abcdefghjkmnp23456789"), 6)))
    texts = [fill(t) for t in scam for _ in range(60)] + [fill(t) for t in legit for _ in range(60)]
    return OT.build_text_model(texts, [1] * (len(scam) * 60) + [0] * (len(legit) * 60))


def main():
    os.makedirs(OUT, exist_ok=True)
    c = OT.load_corpus()
    if c is None: raise SystemExit("data/text_train.csv.gz missing. Run: python orion3_text.py --rebuild")
    val_ham = c[(c.split == "val") & (c.source == "uci_ham")].text_norm
    test_pos = c[(c.split == "test") & (c.label == 1)]
    test_ham = c[(c.split == "test") & (c.source == "uci_ham")].text_norm
    test_formal = c[(c.split == "test") & (c.source == "author_legit_B")].text_norm
    print(f"test: {len(test_pos):,} real smishing from unseen templates ({int(test_pos.india.sum())} Indian-network), "
          f"{len(test_ham):,} real casual legit SMS, {len(test_formal)} author-written formal legit messages")

    models = {"Real-data model (used in demo)": OT.default_text_model(),
              "Simulated-template model (old demo)": simulated_template_model(),
              "UCI spam model": uci_spam_model(c)}

    rows = []
    for name, m in models.items():
        thr = float(np.quantile(m.predict_proba(list(val_ham))[:, 1], 0.99))        # 1% false alarms on real validation ham
        sc = lambda t: m.predict_proba(list(t))[:, 1]
        p_pos = sc(test_pos.text_norm); flag = p_pos >= thr
        fa_ham, fa_formal = float((sc(test_ham) >= thr).mean()), float((sc(test_formal) >= thr).mean())
        base = {"model": name, "false alarms, real casual SMS": fa_ham, "false alarms, formal legit (author-written)": fa_formal, "threshold": thr}
        rows.append({**base, "slice": "all unseen-template smishing", "n": len(test_pos), "recall": float(flag.mean())})
        rows.append({**base, "slice": "Indian-network smishing", "n": int(test_pos.india.sum()), "recall": float(flag[test_pos.india.values].mean())})
        if name.startswith("Real-data"):
            for t, g in test_pos.assign(flag=flag).groupby("scam_type"):
                if len(g) >= 20: rows.append({**base, "slice": f"type: {t}", "n": len(g), "recall": float(g.flag.mean())})
            test_pos.assign(score=p_pos)[~flag].sample(min(60, int((~flag).sum())), random_state=0)[["scam_type", "text_norm", "score"]] \
                .to_csv(f"{OUT}/missed_smishing_realdata.csv", index=False)
            fm = pd.DataFrame({"message": list(test_formal), "score": sc(test_formal)})
            fm[fm.score >= thr].to_csv(f"{OUT}/false_alarms_formal_legit.csv", index=False)

    cols = ["model", "slice", "n", "recall", "false alarms, real casual SMS", "false alarms, formal legit (author-written)"]
    res = pd.DataFrame(rows)[cols + ["threshold"]]; res.to_csv(f"{OUT}/text_real_results.csv", index=False)
    print("\n" + res[cols].round(3).to_string(index=False))
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in res.iterrows(): md.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    open(f"{OUT}/text_real_snippet.md", "w").write("### Message models on real unseen-template smishing\n\n" + "\n".join(md) + "\n")
    print(f"\nSaved to {OUT}/. Read missed_smishing_realdata.csv and false_alarms_formal_legit.csv: they show what the model gets wrong.")


if __name__ == "__main__":
    main()
