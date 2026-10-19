"""
ORION3 simulator v3 + honest evaluation.

Why this exists: the master-build simulator injected features directly from the
label (e.g. receiver_txns_10m was overwritten for scam rows), so models only
recovered the generator. Here features are COMPUTED from a generated event
stream, in time order, using only past events. Scams are campaigns (one mule
account, many victims, forwarding of money), and hard negatives (fundraiser
bursts, formal legit bank messages, legit high-value payments) are present.
Message text comes from real public data (orion3_text.py); payments, accounts and campaigns are simulated.

Run:  python orion3_sim_v3.py
Payments, accounts and campaigns are SYNTHETIC. State that in the README.
"""
import time
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import orion3_text as OT

SEED = 42
DAYS, DAY_MIN = 30, 1440
HELDOUT = "impersonation"          # only appears in the last 6 days (test period)
TRAIN_END, VAL_END = 18, 24        # day index cut-offs: train <18, val 18-23, test >=24
# Scam families = real smishing types from the public IMC 2025 dataset. "impersonation" (hey mum/dad + wrong number)
# is held out: it never appears in the training or validation days and is excluded from message-model training.
FAMILY_TYPES = {"banking": ["banking"], "delivery": ["delivery"], "government": ["government"], "telecom": ["telecom"],
                "impersonation": ["hey mum/dad", "wrong number"]}
FORMAL_SOURCES = ("author_legit_A", "author_legit_B", "author_legit_C")


def build_pools():
    """Message pools per period, taken from the real-data corpus (orion3_text.py). Scam messages in a period come from
    that period's template-group split, so the message model never trained on the templates it is tested on."""
    c = OT.load_corpus()
    if c is None:
        raise FileNotFoundError("data/text_train.csv.gz is missing. Upload it to the repo (folder data/), or run: python orion3_text.py --rebuild")
    pools = {"scam": {}, "casual": {}, "formal": {}}
    for split in ("train", "val", "test"):
        d = c[c.split == split]; pos = d[d.label == 1]; neg = d[d.label == 0]
        pools["scam"][split] = {f: pos[pos.scam_type.isin(types)].text_norm.tolist() for f, types in FAMILY_TYPES.items()}
        pools["casual"][split] = neg[neg.source == "uci_ham"].text_norm.tolist()
        pools["formal"][split] = neg[neg.source.isin(FORMAL_SOURCES)].text_norm.tolist()
    pools["scam"]["train"]["impersonation"] = pools["scam"]["val"]["impersonation"] = []
    return pools


def period(minute):
    day = minute // DAY_MIN
    return "train" if day < TRAIN_END else "val" if day < VAL_END else "test"


HOUR_P = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 10, 10, 9, 7, 4, 2], float)
HOUR_P /= HOUR_P.sum()


def simulate(seed=SEED, hard=False, n_users=3000, n_merchants=300, n_collectors=40, n_mules=40, n_campaigns=None):
    """hard=True: low-and-slow campaigns, small scam amounts, fewer messages, more legit bursts and more formal legit messages.
    Messages are real text from public data (see build_pools): scam messages are real smishing, legitimate ones are real
    casual SMS plus author-written bank/UPI notifications."""
    n_campaigns = n_campaigns or (80 if hard else 45)
    rng = np.random.default_rng(seed)
    pools = build_pools(); q_formal = 0.5 if hard else 0.25

    def legit_msg(t):
        per = period(t); kind = "formal" if (rng.random() < q_formal and pools["formal"][per]) else "casual"
        pool = pools[kind][per]
        return pool[int(rng.integers(0, len(pool)))]

    def scam_msg(fam, t):
        pool = pools["scam"][period(t)][fam]
        return pool[int(rng.integers(0, len(pool)))] if pool else ""
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
            large_new = rng.random() < 0.015
            if large_new:                                                    # legit large payment to a NEW person
                r = int(rng.integers(0, P)); a = scale[s] * rng.uniform(5, 15)  # (second-hand purchase, plumber, ...)
                if r == s: r = (s + 1) % P
            msg = ""
            if rng.random() < 0.35:
                msg = legit_msg(t)
            add(t, s, r, a, msg, "normal", "payment")
    # legit fundraiser bursts (hard negatives: many new senders -> one person)
    for c in range(C0, U0):
        for _ in range(int(rng.integers(2, 5)) if hard else int(rng.integers(1, 3))):
            t0 = int(rng.integers(0, DAYS)) * DAY_MIN + int(rng.integers(8 * 60, 18 * 60))
            for s in rng.choice(P, int(rng.integers(20, 80)) if hard else int(rng.integers(15, 60)), replace=False):
                tt = t0 + int(rng.integers(0, 360))
                add(tt, s, c, scale[s] * rng.uniform(0.3, 1.0), legit_msg(tt) if rng.random() < 0.6 else "", "normal", "payment")
    # scam campaigns
    fams = list(FAMILY_TYPES)
    for _ in range(n_campaigns):
        fam = fams[int(rng.integers(0, len(fams)))]
        day = int(rng.integers(24, DAYS)) if fam == HELDOUT else int(rng.integers(0, DAYS))
        t0 = day * DAY_MIN + int(rng.integers(8 * 60, 20 * 60)); dur = int(rng.integers(1440, 4320)) if (hard and rng.random() < 0.5) else int(rng.integers(30, 300))
        mule = mules[int(rng.integers(0, len(mules)))]
        slow = rng.random() < (0.5 if hard else 0.3)
        for v in rng.choice(P, int(rng.integers(4, 14)) if hard else int(rng.integers(8, 26)), replace=False):
            t = t0 + int(rng.integers(0, dur))
            a = scale[v] * (rng.uniform(3, 12) if rng.random() < (0.4 if hard else 0.7) else rng.uniform(0.5, 2.5))
            msg = scam_msg(fam, t) if rng.random() < (0.45 if hard else 0.6) else ""
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
FUS = ["p_beh_net", "p_txt", "has_message"]
KEYWORDS = ["verify", "blocked", "reward", "prize", "remote", "urgent", "send", "suspended", "claim", "refund"]


def hgb(): return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=SEED)
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fusion_features(X):
    """Fusion inputs in log-odds space. Missing message -> neutral 0 plus has_message flag (no fake score)."""
    has = X["has_message"].values
    txt = np.where(has == 1, _logit(np.nan_to_num(X["p_txt"].values.astype(float), nan=0.5)), 0.0)
    return np.column_stack([_logit(X["p_beh_net"].values), txt, has])


FUSION_NAMES = ["behaviour_network", "message", "has_message"]


def new_fusion():
    """Logistic regression on the behaviour+network score and the message score: few parameters, stable with few scams, readable weights."""
    return make_pipeline(FunctionTransformer(fusion_features), LogisticRegression(C=1.0, max_iter=2000))
def recall_at_threshold(y, p, thr): return float(((p >= thr) & (y == 1)).sum() / max(1, (y == 1).sum()))
def fpr_at_threshold(y, p, thr): return float(((p >= thr) & (y == 0)).sum() / max(1, (y == 0).sum()))
def thr_at_fpr(y, p, fpr=0.01): return float(np.quantile(p[y == 0], 1 - fpr))   # chosen on VALIDATION negatives only


def make_splits(df):
    sc = df[df.kind == "payment"].copy()     # forwards feed the graph features but are not scored
    tr = sc[sc.day < TRAIN_END]
    tr = tr[tr.family != HELDOUT]            # held-out family never seen in training
    va = sc[(sc.day >= TRAIN_END) & (sc.day < VAL_END)]
    te = sc[sc.day >= VAL_END]
    return sc, tr, va, te


def base_predict(models, d):
    """Scores from the three base models. Missing message -> NaN (not a fake score)."""
    out = pd.DataFrame(index=d.index)
    out["p_beh"] = models["beh"].predict_proba(d[BEH])[:, 1]
    out["p_net"] = models["net"].predict_proba(d[NET])[:, 1]
    out["p_beh_net"] = models["beh_net"].predict_proba(d[BEH + NET])[:, 1]
    p = np.full(len(d), np.nan); has = d.has_message.values == 1
    if has.any(): p[has] = models["txt"].predict_proba(list(d.message[has]))[:, 1]
    out["p_txt"] = p
    out["has_message"] = d.has_message.values
    return out


def rules_score(d):
    k = d.message.str.lower().apply(lambda m: int(any(w in m for w in KEYWORDS)))
    return (0.4 * (d.rcv_unique_senders_60m >= 3) + 0.3 * ((d.is_new_receiver == 1) & (d.amount_ratio.fillna(0) >= 3)) + 0.3 * k).astype(float).values


def fit_all(df, text_source="real"):
    """Fit base models on TRAIN, fusion on VALIDATION. Validation scores of the fusion are
    5-fold cross-validated so thresholds are not picked from scores the fusion was fitted on.
    text_source="real": message model trained on real public data (orion3_text.py, never sees simulator messages).
    text_source="simulated": message model trained on the simulator's own template messages (old behaviour)."""
    sc, tr, va, te = make_splits(df)
    msg_tr = tr[tr.has_message == 1]
    txt = OT.default_text_model() if text_source == "real" else None
    if txt is None:
        text_source = "simulated"
        txt = OT.build_text_model(msg_tr.message, msg_tr.is_scam)
    models = {"beh": hgb().fit(tr[BEH], tr.is_scam), "net": hgb().fit(tr[NET], tr.is_scam),
              "beh_net": hgb().fit(tr[BEH + NET], tr.is_scam), "txt": txt, "text_source": text_source}
    Pva, Pte = base_predict(models, va), base_predict(models, te)
    models["fusion"] = new_fusion().fit(Pva[FUS], va.is_scam)
    Pva["p_fusion"] = cross_val_predict(new_fusion(), Pva[FUS], va.is_scam,
                                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    Pte["p_fusion"] = models["fusion"].predict_proba(Pte[FUS])[:, 1]
    Pva["p_rules"], Pte["p_rules"] = rules_score(va), rules_score(te)
    return {"models": models, "sc": sc, "tr": tr, "va": va, "te": te, "Pva": Pva, "Pte": Pte}


def main(hard=False, text_source="real"):
    df = simulate(hard=hard)
    R = fit_all(df, text_source)
    sc, tr, va, te, Pva, Pte, models = R["sc"], R["tr"], R["va"], R["te"], R["Pva"], R["Pte"], R["models"]
    print(f"events={len(df):,} scored={len(sc):,} scam_rate={sc.is_scam.mean():.3%} | message model trained on: {models['text_source']} data")
    for n, d in [("train", tr), ("val", va), ("test", te)]:
        print(f"{n}: rows={len(d):,} scams={int(d.is_scam.sum())} families={sorted(d[d.is_scam == 1].family.unique())}")
    y_va, y_te = va.is_scam.values, te.is_scam.values

    rows = []
    for name, col in [("Rules (hand weights)", "p_rules"), ("Text only", "p_txt"), ("Sender behaviour only", "p_beh"),
                      ("Receiver/network only", "p_net"), ("Behaviour + network", "p_beh_net"), ("Learned fusion", "p_fusion")]:
        pv, pt = Pva[col].fillna(-1).values, Pte[col].fillna(-1).values   # missing text -> lowest score
        thr = thr_at_fpr(y_va, pv, 0.01)                                  # threshold from VALIDATION
        rows.append({"model": name, "PR_AUC": average_precision_score(y_te, pt), "ROC_AUC": roc_auc_score(y_te, pt),
                     "val_thr@1%FPR": round(thr, 4), "test_recall": recall_at_threshold(y_te, pt, thr),
                     "test_FPR": fpr_at_threshold(y_te, pt, thr)})
    print("\n=== TEST RESULTS (threshold chosen on validation) ==="); print(pd.DataFrame(rows).round(4).to_string(index=False))

    # per-family: family scams vs ALL test negatives, so both classes exist
    thr = thr_at_fpr(y_va, Pva.p_fusion.values, 0.01); neg = te.is_scam == 0
    fam_rows = []
    for f in sorted(te[te.is_scam == 1].family.unique()):
        m = (te.family == f) | neg; y = (te.family[m] == f).astype(int).values; p = Pte.p_fusion[m].values
        fam_rows.append({"family": f + (" (HELD OUT)" if f == HELDOUT else ""), "scams": int(y.sum()),
                         "PR_AUC": average_precision_score(y, p), "recall@val_thr": recall_at_threshold(y, p, thr)})
    print("\n=== FUSION BY SCAM FAMILY ==="); print(pd.DataFrame(fam_rows).round(4).to_string(index=False))

    for name, mask in [("no-message", te.has_message == 0), ("with-message", te.has_message == 1)]:
        y, p = y_te[mask.values], Pte.p_fusion[mask].values
        print(f"\n{name}: n={int(mask.sum())} scam_rate={y.mean():.3%} PR_AUC={average_precision_score(y, p):.4f} "
              f"recall={recall_at_threshold(y, p, thr):.3f} FPR={fpr_at_threshold(y, p, thr):.4f}")
    hn = te[(te.family == "normal") & (te.rcv_unique_senders_60m >= 3)]
    print(f"legit receiver bursts (>=3 senders/60m): n={len(hn)}  flagged by fusion={float((Pte.p_fusion[hn.index] >= thr).mean()):.3f}")

    rng = np.random.default_rng(0); b = []; pf = Pte.p_fusion.values
    for _ in range(200):
        i = rng.integers(0, len(y_te), len(y_te)); b.append(average_precision_score(y_te[i], pf[i]))
    print(f"\nFusion PR-AUC 95% CI (bootstrap): [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]")

    print("\nthreshold  recall   FPR    cost   (illustrative: missed scam = 1, false alarm = 0.1)")
    for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        fn = ((pf < t) & (y_te == 1)).sum(); fp = ((pf >= t) & (y_te == 0)).sum()
        print(f"{t:>8}  {recall_at_threshold(y_te, pf, t):.3f}  {fpr_at_threshold(y_te, pf, t):.4f}  {fn + 0.1 * fp:.1f}")

    row = te.iloc[[0]]; lat = []
    for _ in range(300):
        t0 = time.perf_counter(); b_ = base_predict(models, row); models["fusion"].predict_proba(b_[FUS]); lat.append((time.perf_counter() - t0) * 1000)
    print(f"\nscoring latency ms: p50={np.percentile(lat, 50):.1f}  p95={np.percentile(lat, 95):.1f}  (models only, features precomputed)")


if __name__ == "__main__":
    for h in (False, True):
        print("\n" + "#" * 70 + f"\n# SIMULATOR DIFFICULTY: {'HARD' if h else 'EASY'}\n" + "#" * 70)
        main(h)
