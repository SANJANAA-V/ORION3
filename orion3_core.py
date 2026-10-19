"""
ORION3 core: train the models once, save them, and score ONE payment.
The demo app uses exactly these saved models (the same ones evaluated in orion3_sim_v3.py).

    python orion3_core.py          # trains, saves orion3_bundle.joblib, runs the smoke tests
"""
import os
import time
import numpy as np
import pandas as pd
import joblib

import orion3_sim_v3 as sim

BUNDLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orion3_bundle.joblib")


def train_bundle(hard=True, path=BUNDLE_PATH):
    """Fit all models on simulated data and save them with thresholds picked on VALIDATION."""
    df = sim.simulate(hard=hard)
    R = sim.fit_all(df)
    va, Pva = R["va"], R["Pva"]
    y_va = va.is_scam.values
    warn_thr = sim.thr_at_fpr(y_va, Pva.p_fusion.values, 0.01)      # ~1% of legit payments warned
    block_thr = max(sim.thr_at_fpr(y_va, Pva.p_fusion.values, 0.001), warn_thr)  # ~0.1% blocked
    neg_idx = va.index[y_va == 0]
    neutral_feats = {c: float(va.loc[neg_idx, c].median()) for c in sim.BEH + sim.NET}   # a "typical legitimate payment"
    neutral_txt = float(Pva.loc[neg_idx, "p_txt"].dropna().median())
    bundle = {
        "models": R["models"],
        "warn_thr": warn_thr, "block_thr": block_thr,
        "neutral": {"feats": neutral_feats, "p_txt": neutral_txt},
        "meta": {"simulator_hard_mode": hard, "message_model_trained_on": R["models"]["text_source"], "train_rows": len(R["tr"]), "val_scams": int(y_va.sum()),
                 "note": "Payment behaviour is SIMULATED; the message model uses real public smishing data. Not for real payment decisions."},
    }
    try:
        joblib.dump(bundle, path)
    except OSError:      # read-only folder (e.g. some hosting setups): keep the models in memory only
        pass
    return bundle


def load_bundle(path=BUNDLE_PATH):
    return joblib.load(path) if os.path.exists(path) else train_bundle(path=path)


def _fusion_proba(bundle, p_bn, p_txt, has_msg):
    row = pd.DataFrame([{"p_beh_net": p_bn, "p_txt": p_txt, "has_message": has_msg}])[sim.FUS]
    return float(bundle["models"]["fusion"].predict_proba(row)[0, 1])


def score_payment(bundle, f):
    """f keys: amount, typical_amount, n_prior_payments, new_receiver, rcv_unique_senders_60m,
    rcv_txns_60m, rcv_out_60m, rcv_age_min (-1 = never seen), hour, receiver_is_merchant, message."""
    t0 = time.perf_counter()
    m = bundle["models"]
    msg = (f.get("message") or "").strip()
    n_prior = int(f["n_prior_payments"])
    ratio = f["amount"] / f["typical_amount"] if (n_prior >= 3 and f["typical_amount"] > 0) else np.nan
    row = {"log_amount": np.log1p(f["amount"]), "amount_ratio": ratio, "sender_n_prior": n_prior,
           "is_new_receiver": int(f["new_receiver"]), "hour": int(f["hour"]), "is_merchant": int(f["receiver_is_merchant"]),
           "rcv_unique_senders_60m": f["rcv_unique_senders_60m"], "rcv_txns_60m": f["rcv_txns_60m"],
           "rcv_out_60m": f["rcv_out_60m"], "rcv_age_min": f["rcv_age_min"]}
    bn = pd.DataFrame([row])[sim.BEH + sim.NET]
    p_bn = float(m["beh_net"].predict_proba(bn)[0, 1])
    p_beh = float(m["beh"].predict_proba(bn[sim.BEH])[0, 1])          # shown for information
    p_net = float(m["net"].predict_proba(bn[sim.NET])[0, 1])
    has = int(bool(msg))
    p_txt = float(m["txt"].predict_proba([msg])[0, 1]) if has else np.nan
    p = _fusion_proba(bundle, p_bn, p_txt, has)
    action = "BLOCK" if p >= bundle["block_thr"] else "WARN" if p >= bundle["warn_thr"] else "ALLOW"

    # Reasons come from the models themselves: replace one group of inputs with a typical legitimate payment and see how far
    # the log-odds of "scam" fall (weight of the fusion model x change in the behaviour+network score). Only strong pushes are explained.
    lr = bundle["models"]["fusion"][-1]
    w = dict(zip(sim.FUSION_NAMES, lr.coef_[0]))
    lg, nf = sim._logit, bundle["neutral"]["feats"]

    def p_bn_without(cols):
        b2 = bn.copy()
        for c in cols: b2[c] = nf[c]
        return float(m["beh_net"].predict_proba(b2)[0, 1])
    contrib = {
        "network": float(w["behaviour_network"] * (lg(p_bn) - lg(p_bn_without(sim.NET)))),
        "behaviour": float(w["behaviour_network"] * (lg(p_bn) - lg(p_bn_without(sim.BEH)))),
        "message": float(w["message"] * (lg(p_txt) - lg(bundle["neutral"]["p_txt"]))) if has else 0.0,
    }
    reasons = []
    for sig, c in sorted(contrib.items(), key=lambda kv: -kv[1]):
        if c < 1.0: continue
        if sig == "network":
            n_s, n_out, age = int(f["rcv_unique_senders_60m"]), int(f["rcv_out_60m"]), f["rcv_age_min"]
            if n_s >= 3 or n_out >= 1:
                s_ = f"The receiving account got money from {n_s} different people in the last hour"
                if n_out >= 1: s_ += f" and has already sent money onward {n_out} time(s)"
                reasons.append(s_ + ". This pattern is typical of accounts that collect and pass on scam money.")
            elif 0 <= age < 120:
                reasons.append("The receiving account is very new.")
        elif sig == "behaviour":
            if f["new_receiver"] and not np.isnan(ratio) and ratio >= 3:
                reasons.append(f"First payment to this receiver, and the amount is {ratio:.1f}x your usual payment.")
            elif f["new_receiver"]:
                reasons.append("First payment to this receiver.")
            elif not np.isnan(ratio) and ratio >= 3:
                reasons.append(f"The amount is {ratio:.1f}x your usual payment.")
            else:
                reasons.append("The payment pattern is unusual for this sender.")
        else:
            reasons.append("The message wording resembles known scam messages.")
    if action == "ALLOW":
        reasons = ["No strong risk signal found."]
    elif not reasons:
        reasons = ["Combined risk is elevated, but no single signal stands out."]
    return {"action": action, "risk_score": p, "reasons": reasons,
            "signals": {"behaviour": p_beh, "receiver_network": p_net, "behaviour_and_network": p_bn, "message": None if not has else p_txt},
            "contributions": contrib,
            "thresholds": {"warn": bundle["warn_thr"], "block": bundle["block_thr"]},
            "latency_ms": (time.perf_counter() - t0) * 1000}


PRESETS = {
    "Ordinary payment to a regular shop": dict(
        amount=450.0, typical_amount=400.0, n_prior_payments=25, new_receiver=False, rcv_unique_senders_60m=0,
        rcv_txns_60m=1, rcv_out_60m=0, rcv_age_min=20000, hour=19, receiver_is_merchant=True, message=""),
    "Fake KYC message + new receiver + many payers": dict(
        amount=4800.0, typical_amount=400.0, n_prior_payments=25, new_receiver=True, rcv_unique_senders_60m=7,
        rcv_txns_60m=8, rcv_out_60m=2, rcv_age_min=90, hour=15, receiver_is_merchant=False,
        message="HDFC KYC update pending. Verify now or your account will be blocked bit.ly/x2k9a"),
    "Legitimate fundraiser (many people paying one account)": dict(
        amount=300.0, typical_amount=400.0, n_prior_payments=25, new_receiver=True, rcv_unique_senders_60m=12,
        rcv_txns_60m=12, rcv_out_60m=0, rcv_age_min=400, hour=11, receiver_is_merchant=False,
        message="class gift contribution"),
    "No message, large payment to a new person": dict(
        amount=5200.0, typical_amount=400.0, n_prior_payments=25, new_receiver=True, rcv_unique_senders_60m=0,
        rcv_txns_60m=0, rcv_out_60m=0, rcv_age_min=-1, hour=23, receiver_is_merchant=False, message=""),
    "Scam type never seen in training (relative in hospital)": dict(
        amount=6000.0, typical_amount=400.0, n_prior_payments=25, new_receiver=True, rcv_unique_senders_60m=4,
        rcv_txns_60m=4, rcv_out_60m=1, rcv_age_min=200, hour=21, receiver_is_merchant=False,
        message="Hi it's Ravi, new number. I am in hospital, please send Rs 6000 urgently, will return tonight"),
    "Real-looking bank notice, regular receiver": dict(
        amount=500.0, typical_amount=400.0, n_prior_payments=25, new_receiver=False, rcv_unique_senders_60m=0,
        rcv_txns_60m=1, rcv_out_60m=0, rcv_age_min=15000, hour=14, receiver_is_merchant=True,
        message="Payment reminder: insurance premium is due, pay via the official app"),
}


def smoke_test(bundle):
    """Real checks (not just 'returns a number'). Prints a table and raises if an expectation fails."""
    rows, ok = [], True
    expect = {"Ordinary payment to a regular shop": ("ALLOW",),
              "Fake KYC message + new receiver + many payers": ("WARN", "BLOCK"),
              "Real-looking bank notice, regular receiver": ("ALLOW",)}
    for name, f in PRESETS.items():
        r = score_payment(bundle, f)
        good = name not in expect or r["action"] in expect[name]
        ok &= good
        rows.append({"scenario": name, "action": r["action"], "risk": round(r["risk_score"], 4),
                     "expected": "/".join(expect.get(name, ("(informational)",))), "pass": good})
    print(pd.DataFrame(rows).to_string(index=False))
    if not ok: raise AssertionError("A smoke-test expectation failed - do not publish the demo yet.")


if __name__ == "__main__":
    b = train_bundle()
    print("saved:", BUNDLE_PATH, "| warn>=%.4f block>=%.4f" % (b["warn_thr"], b["block_thr"]))
    smoke_test(load_bundle())
    print("OK")
