"""
ORION3 simulator v3 + honest evaluation.

Why this exists: the master-build simulator injected features directly from the
label (e.g. receiver_txns_10m was overwritten for scam rows), so models only
recovered the generator. Here features are COMPUTED from a generated event
stream, in time order, using only past events. Scams are campaigns (one mule
account, many victims, forwarding of money), and hard negatives (fundraiser
bursts, legit alarming bank messages, legit high-value payments) are present.

Run:  python orion3_sim_v3.py
Everything below is SYNTHETIC. State that in the README.
"""
import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
DAYS, DAY_MIN = 30, 1440
HELDOUT = "impersonation"          # only appears in the last 6 days (test period)
TRAIN_END, VAL_END = 18, 24        # day index cut-offs: train <18, val 18-23, test >=24
BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Canara"]

NORMAL_NOTES = ["rent for october", "thanks for dinner", "chai", "groceries", "split the bill",
                "electricity bill", "school fees", "mobile recharge", "movie tickets", "birthday gift",
                "auto fare", "milk", "lunch", "returning the money you lent", "tuition fee", "gym fee"]
CONTRIB_NOTES = ["contribution for temple fund", "class gift contribution", "flood relief donation",
                 "wedding gift pooling", "farewell party contribution"]
LEGIT_ALARMING = [
    "Your {bank} statement is ready, verify account details at the branch",
    "Reward points credited to your account",
    "Urgent: electricity bill due tomorrow, pay at the office counter",
    "Your refund of Rs {amt} has been processed to your account",
    "Dear customer, complete your KYC at the nearest {bank} branch with documents",
    "Payment reminder: insurance premium is due, pay via the official app"]
SCAM_TEMPLATES = {
    "fake_kyc": ["{bank} KYC update pending. Verify now or your account will be blocked {link}",
                 "Dear customer your {bank} account will be suspended today, complete KYC {link}",
                 "KYC expired. Update PAN and Aadhaar details immediately {link}",
                 "{bank}: account on hold, verify your details at {link} within 24 hours"],
    "prize": ["Congratulations! You won Rs {amt} in the lucky draw. Claim your reward {link}",
              "You are selected for a cashback reward of Rs {amt}. Pay processing fee to claim",
              "Prize notification: claim your gift voucher before it expires {link}"],
    "fake_support": ["{bank} customer care: your refund of Rs {amt} is pending, share details to receive it",
                     "Support team here. We detected an issue with your account, call {phone} to fix",
                     "Your complaint is being processed, please confirm card details with our executive"],
    "remote_access": ["To resolve your issue install AnyDesk and share the code with our agent",
                      "Our technician needs remote access to your phone to reverse the payment, install the app {link}",
                      "Download the support app and give us the 9 digit code to refund your money"],
    "urgent_transfer": ["Pay Rs {amt} immediately to avoid a penalty and legal action",
                        "Your electricity will be cut tonight, transfer Rs {amt} now to this number",
                        "Last notice: send the pending amount today or your account will be reported"],
    "impersonation": ["Hi, this is {name}, I changed my number. I am in hospital, please send Rs {amt} urgently, will return tonight",
                      "Beta it's me, phone broke, using a friend's phone. Need Rs {amt} for a small emergency, will repay tomorrow",
                      "Hello sir, I am calling for {name} ji, he had an accident, please transfer Rs {amt} to this account"],
}
HOUR_P = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 10, 9, 7, 4, 2], float)
HOUR_P /= HOUR_P.sum()


def fill(rng, tpl):
    s = tpl.format(bank=rng.choice(BANKS), amt=int(rng.integers(500, 50000)),
                   link="bit.ly/" + "".join(rng.choice(list("abcdefghjkmnp23456789"), 6)),
                   phone="1800" + str(int(rng.integers(100000, 999999))),
                   name=rng.choice(["Ravi", "Anita", "Suresh", "Priya", "Karthik"]))
    if rng.random() < 0.3: s = s.lower()
    if rng.random() < 0.2: s = s.replace(",", "").replace(".", "")
    return s


def simulate(seed=SEED, hard=False, n_users=3000, n_merchants=300, n_collectors=40, n_mules=40, n_campaigns=None):
    """hard=True: low-and-slow campaigns, small scam amounts, fewer messages, more legit bursts and alarming legit messages."""
    n_campaigns = n_campaigns or (80 if hard else 45)
    rng = np.random.default_rng(seed)
    P = n_users; M0 = P; C0 = M0 + n_merchants; U0 = C0 + n_collectors; K0 = U0 + n_mules
    mules = list(range(U0, U0 + n_mules)); cashouts = list(range(K0, K0 + 30))
    scale = np.exp(rng.normal(np.log(400), 0.7, P))
    rate = np.exp(rng.normal(np.log(0.6), 0.5, P))
    pop = 1 / np.arange(1, n_merchants + 1) ** 0.8; pop /= pop.sum()
    regulars = []
    for i in range(P):
        k = int(rng.integers(4, 10)); r = []
        for _ in range(k):
            r.append(int(M0 + rng.choice(n_merchants, p=pop)) if rng.random() < 0.6 else int(rng.integers(0, P)))
        regulars.append(r)
    ev = []  # (minute, sender, receiver, amount, message, family, kind)

    def add(t, s, r, a, m, fam, kind): ev.append((int(t), int(s), int(r), float(a), m, fam, kind))

    # normal payments
    n_norm = rng.poisson(rate * DAYS)
    for s in range(P):
        for _ in range(n_norm[s]):
            t = int(rng.integers(0, DAYS)) * DAY_MIN + int(rng.choice(24, p=HOUR_P)) * 60 + int(rng.integers(0, 60))
            if rng.random() < 0.78: r = regulars[s][int(rng.integers(0, len(regulars[s])))]
            else: r = int(M0 + rng.choice(n_merchants, p=pop)) if rng.random() < 0.4 else int(rng.integers(0, P))
            if r == s: r = (s + 1) % P
            a = scale[s] * np.exp(rng.normal(0, 0.5)) * (1.5 if r >= M0 else 1)
            if rng.random() < 0.025: a = scale[s] * rng.uniform(8, 30)       # legit high-value
            msg = ""
            if rng.random() < 0.35:
                msg = rng.choice(NORMAL_NOTES) if rng.random() > (0.25 if hard else 0.08) else fill(rng, rng.choice(LEGIT_ALARMING))
            add(t, s, r, a, msg, "normal", "payment")
    # legit fundraiser bursts (hard negatives: many new senders -> one person)
    for c in range(C0, U0):
        for _ in range(int(rng.integers(2, 5)) if hard else int(rng.integers(1, 3))):
            t0 = int(rng.integers(0, DAYS)) * DAY_MIN + int(rng.integers(8 * 60, 18 * 60))
            for s in rng.choice(P, int(rng.integers(20, 80)) if hard else int(rng.integers(15, 60)), replace=False):
                add(t0 + int(rng.integers(0, 360)), s, c, scale[s] * rng.uniform(0.3, 1.0),
                    rng.choice(CONTRIB_NOTES) if rng.random() < 0.6 else "", "normal", "payment")
    # scam campaigns
    fams = [f for f in SCAM_TEMPLATES]
    for _ in range(n_campaigns):
        fam = fams[int(rng.integers(0, len(fams)))]
        day = int(rng.integers(24, DAYS)) if fam == HELDOUT else int(rng.integers(0, DAYS))
        t0 = day * DAY_MIN + int(rng.integers(8 * 60, 20 * 60)); dur = int(rng.integers(1440, 4320)) if (hard and rng.random() < 0.5) else int(rng.integers(30, 300))
        mule = mules[int(rng.integers(0, len(mules)))]
        slow = rng.random() < (0.5 if hard else 0.3)
        for v in rng.choice(P, int(rng.integers(4, 14)) if hard else int(rng.integers(8, 26)), replace=False):
            t = t0 + int(rng.integers(0, dur))
            a = scale[v] * (rng.uniform(3, 12) if rng.random() < (0.4 if hard else 0.7) else rng.uniform(0.5, 2.5))
            msg = fill(rng, rng.choice(SCAM_TEMPLATES[fam])) if rng.random() < (0.45 if hard else 0.6) else ""
            add(t, v, mule, min(a, 50000), msg, fam, "payment")
            if rng.random() < (0.6 if hard else 0.8):   # mule forwards the money
                delay = int(rng.integers(1, 40)) if not slow else int(rng.integers(60, 360))
                add(t + delay, mule, cashouts[int(rng.integers(0, len(cashouts)))], a * rng.uniform(0.7, 0.95), "", "normal", "forward")
    df = pd.DataFrame(ev, columns=["minute", "sender", "receiver", "amount", "message", "family", "kind"])
    df = df.sort_values(["minute", "sender"], kind="stable").reset_index(drop=True)
    df["is_scam"] = ((df.family != "normal") & (df.kind == "payment")).astype(int)
    df["is_merchant"] = ((df.receiver >= M0) & (df.receiver < C0)).astype(int)
    return add_features(df)


def add_features(df):
    """Chronological features: every value uses ONLY events strictly before the current row."""
    s_n = defaultdict(int); s_sum = defaultdict(float); s_rcv = defaultdict(set)
    r_in = defaultdict(deque); r_out = defaultdict(deque); first_seen = {}
    cols = {k: [] for k in ["sender_n_prior", "amount_ratio", "is_new_receiver", "rcv_unique_senders_60m",
                            "rcv_txns_60m", "rcv_out_60m", "rcv_age_min"]}
    for t, s, r, a in zip(df.minute.values, df.sender.values, df.receiver.values, df.amount.values):
        n = s_n[s]
        cols["sender_n_prior"].append(n)
        cols["amount_ratio"].append(a / (s_sum[s] / n) if n >= 3 else np.nan)
        cols["is_new_receiver"].append(int(r not in s_rcv[s]))
        din, dout = r_in[r], r_out[r]
        while din and din[0][0] < t - 60: din.popleft()
        while dout and dout[0] < t - 60: dout.popleft()
        cols["rcv_unique_senders_60m"].append(len({x[1] for x in din}))
        cols["rcv_txns_60m"].append(len(din)); cols["rcv_out_60m"].append(len(dout))
        cols["rcv_age_min"].append(t - first_seen[r] if r in first_seen else -1)
        # update state AFTER computing features
        s_n[s] += 1; s_sum[s] += a; s_rcv[s].add(r); din.append((t, s)); r_out[s].append(t)
        first_seen.setdefault(r, t); first_seen.setdefault(s, t)
    for k, v in cols.items(): df[k] = v
    df["hour"] = (df.minute // 60) % 24
    df["day"] = df.minute // DAY_MIN
    df["log_amount"] = np.log1p(df.amount)
    df["has_message"] = (df.message != "").astype(int)
    return df


BEH = ["log_amount", "amount_ratio", "sender_n_prior", "is_new_receiver", "hour", "is_merchant"]
NET = ["rcv_unique_senders_60m", "rcv_txns_60m", "rcv_out_60m", "rcv_age_min"]


def hgb(): return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=SEED)


def recall_at_threshold(y, p, thr): return float(((p >= thr) & (y == 1)).sum() / max(1, (y == 1).sum()))
def fpr_at_threshold(y, p, thr): return float(((p >= thr) & (y == 0)).sum() / max(1, (y == 0).sum()))
def thr_at_fpr(y, p, fpr=0.01): return float(np.quantile(p[y == 0], 1 - fpr))   # chosen on VALIDATION negatives only


def main(hard=False):
    df = simulate(hard=hard)
    sc = df[df.kind == "payment"].copy()     # forwards feed the graph features but are not scored
    tr, va, te = sc[sc.day < TRAIN_END], sc[(sc.day >= TRAIN_END) & (sc.day < VAL_END)], sc[sc.day >= VAL_END]
    tr = tr[tr.family != HELDOUT]
    print(f"events={len(df):,} scored={len(sc):,} scam_rate={sc.is_scam.mean():.3%}")
    for n, d in [("train", tr), ("val", va), ("test", te)]:
        print(f"{n}: rows={len(d):,} scams={int(d.is_scam.sum())} families={sorted(d[d.is_scam == 1].family.unique())}")

    # --- base models (fit on train only) ---
    m_beh = hgb().fit(tr[BEH], tr.is_scam)
    m_net = hgb().fit(tr[NET], tr.is_scam)
    m_beh_net = hgb().fit(tr[BEH + NET], tr.is_scam)
    msg_tr = tr[tr.has_message == 1]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    m_txt = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced").fit(vec.fit_transform(msg_tr.message), msg_tr.is_scam)

    def base_preds(d):
        out = pd.DataFrame(index=d.index)
        out["p_beh"] = m_beh.predict_proba(d[BEH])[:, 1]
        out["p_net"] = m_net.predict_proba(d[NET])[:, 1]
        out["p_beh_net"] = m_beh_net.predict_proba(d[BEH + NET])[:, 1]
        p = np.full(len(d), np.nan); has = d.has_message.values == 1
        if has.any(): p[has] = m_txt.predict_proba(vec.transform(d.message[has]))[:, 1]
        out["p_txt"] = p                         # NaN when there is no message (no fake 0.75 score)
        out["has_message"] = d.has_message.values
        return out

    # --- fusion trained on VALIDATION predictions (stacking), evaluated on TEST ---
    Pva, Pte = base_preds(va), base_preds(te)
    FUS = ["p_beh", "p_net", "p_txt", "has_message"]
    fus = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.08, max_leaf_nodes=6, random_state=SEED).fit(Pva[FUS], va.is_scam)
    Pva["p_fusion"], Pte["p_fusion"] = fus.predict_proba(Pva[FUS])[:, 1], fus.predict_proba(Pte[FUS])[:, 1]

    # rules baseline (hand weights)
    kw = ["verify", "blocked", "reward", "prize", "remote", "urgent", "send", "suspended", "claim", "refund"]
    def rules(d):
        k = d.message.str.lower().apply(lambda m: int(any(w in m for w in kw)))
        return (0.4 * (d.rcv_unique_senders_60m >= 3) + 0.3 * ((d.is_new_receiver == 1) & (d.amount_ratio.fillna(0) >= 3)) + 0.3 * k).astype(float).values
    Pva["p_rules"], Pte["p_rules"] = rules(va), rules(te)

    y_va, y_te = va.is_scam.values, te.is_scam.values
    rows = []
    for name, col in [("Rules (hand weights)", "p_rules"), ("Text only", "p_txt"), ("Sender behaviour only", "p_beh"),
                      ("Receiver/network only", "p_net"), ("Behaviour + network", "p_beh_net"), ("Learned fusion", "p_fusion")]:
        pv, pt = Pva[col].fillna(-1).values, Pte[col].fillna(-1).values   # missing text -> lowest score
        thr = thr_at_fpr(y_va, pv, 0.01)                                  # threshold from VALIDATION
        rows.append({"model": name, "PR_AUC": average_precision_score(y_te, pt), "ROC_AUC": roc_auc_score(y_te, pt),
                     "val_thr@1%FPR": round(thr, 4), "test_recall": recall_at_threshold(y_te, pt, thr),
                     "test_FPR": fpr_at_threshold(y_te, pt, thr)})
    res = pd.DataFrame(rows).round(4)
    print("\n=== TEST RESULTS (threshold chosen on validation) ==="); print(res.to_string(index=False))

    # --- per-family (the bug fix: family scams vs ALL test negatives, so both classes exist) ---
    thr = thr_at_fpr(y_va, Pva.p_fusion.values, 0.01); neg = te.is_scam == 0
    fam_rows = []
    for f in sorted(te[te.is_scam == 1].family.unique()):
        m = (te.family == f) | neg; y = (te.family[m] == f).astype(int).values; p = Pte.p_fusion[m].values
        fam_rows.append({"family": f + (" (HELD OUT)" if f == HELDOUT else ""), "scams": int(y.sum()),
                         "PR_AUC": average_precision_score(y, p), "recall@val_thr": recall_at_threshold(y, p, thr)})
    print("\n=== FUSION BY SCAM FAMILY ==="); print(pd.DataFrame(fam_rows).round(4).to_string(index=False))

    # --- subgroups ---
    for name, mask in [("no-message", te.has_message == 0), ("with-message", te.has_message == 1)]:
        y, p = y_te[mask.values], Pte.p_fusion[mask].values
        print(f"\n{name}: n={int(mask.sum())} scam_rate={y.mean():.3%} PR_AUC={average_precision_score(y, p):.4f} "
              f"recall={recall_at_threshold(y, p, thr):.3f} FPR={fpr_at_threshold(y, p, thr):.4f}")
    hn = te[(te.family == "normal") & (te.rcv_unique_senders_60m >= 3)]
    print(f"legit receiver bursts (>=3 senders/60m): n={len(hn)}  flagged by fusion={float((Pte.p_fusion[hn.index] >= thr).mean()):.3f}")

    # --- bootstrap CI for fusion PR-AUC ---
    rng = np.random.default_rng(0); b = []
    pf = Pte.p_fusion.values
    for _ in range(200):
        i = rng.integers(0, len(y_te), len(y_te)); b.append(average_precision_score(y_te[i], pf[i]))
    print(f"\nFusion PR-AUC 95% CI (bootstrap): [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]")

    # --- cost curve (illustrative units: missed scam = 1, false alarm = 0.1) ---
    print("\nthreshold  recall   FPR    cost")
    for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        fn = ((pf < t) & (y_te == 1)).sum(); fp = ((pf >= t) & (y_te == 0)).sum()
        print(f"{t:>8}  {recall_at_threshold(y_te, pf, t):.3f}  {fpr_at_threshold(y_te, pf, t):.4f}  {fn + 0.1 * fp:.1f}")

    # --- single-event latency (feature lookup excluded: assume precomputed) ---
    row = te.iloc[[0]]; lat = []
    for _ in range(300):
        t0 = time.perf_counter(); b_ = base_preds(row); fus.predict_proba(b_[FUS]); lat.append((time.perf_counter() - t0) * 1000)
    print(f"\nscoring latency ms: p50={np.percentile(lat, 50):.1f}  p95={np.percentile(lat, 95):.1f}  (models only)")


if __name__ == "__main__":
    for h in (False, True):
        print("\n" + "#" * 70 + f"\n# SIMULATOR DIFFICULTY: {'HARD' if h else 'EASY'}\n" + "#" * 70)
        main(h)
