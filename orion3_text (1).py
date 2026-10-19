"""
ORION3 message model trained on REAL public data.

Training data (stored normalised in data/text_train.csv.gz, see data/ATTRIBUTION.md). The scam split is by template group,
so test scams come from message templates never seen in training:
  * real smishing (positives): "Fishing for Smishing" dataset, IMC 2025, CC BY 4.0
  * real legitimate SMS (negatives): UCI SMS Spam Collection ham, CC BY 4.0
  * AUTHOR-WRITTEN legitimate bank / UPI / bill notifications (negatives, template based, NOT real messages).
    Three disjoint template sets: A is used for training, C for the validation period, B only for the test period.
No simulator messages are used for training, so the simulator is a genuine transfer test for the message model.

Rebuild the corpus (needs Internet):   python orion3_text.py --rebuild
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

HELDOUT_TYPES = ["hey mum/dad", "wrong number"]       # the held-out "impersonation" family in the simulator
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "data", "text_train.csv.gz")
IMC_URL = "https://raw.githubusercontent.com/reportsmishing/Smishing-Dataset-IMC25/main/dataset/final_dataset_output.csv"
UCI_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

_URL = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|in|co|net|org|io|ly|me|xyz|top|link|app|info)\b\S*)")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s-]{7,}\d(?!\w)")


def normalize_text(s):
    """Lower-case; keep URL / phone as tokens; remove digits, anonymisation placeholders and punctuation
    (so that source-specific formatting cannot be used as a shortcut)."""
    s = str(s).lower()
    s = re.sub(r"<url>", " urltoken ", s)
    s = re.sub(r"<phone_number>", " phonetoken ", s)
    s = re.sub(r"<[a-z_ ]{2,30}>", " ", s)
    s = _URL.sub(" urltoken ", s)
    s = _PHONE.sub(" phonetoken ", s)
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_many(texts):
    return [normalize_text(t) for t in texts]


def build_text_model(texts, labels):
    """Raw text in, probability out (normalisation happens inside the pipeline)."""
    model = make_pipeline(FunctionTransformer(normalize_many, validate=False),
                          TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=60000),
                          LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced"))
    return model.fit(list(texts), np.asarray(labels))


# ------------------------------------------------------------------ author-written legitimate messages
_BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak", "PNB", "Canara Bank", "Bank of Baroda"]
_MERCH = ["Amazon", "Flipkart", "Swiggy", "Zomato", "BigBasket", "Reliance Smart", "Apollo Pharmacy", "Uber", "IRCTC", "BookMyShow"]
_NAMES = ["Ravi", "Priya", "Karthik", "Anita", "Suresh", "Meena", "Arun", "Divya"]
_TEL = ["Airtel", "Jio", "Vi", "BSNL"]

TEMPLATES_A = [
    "Rs {amt} debited from A/c XX{d4} on {date} to VPA {vpa}. UPI Ref {ref}. If not done by you, call {ph}.",
    "Rs {amt} credited to your A/c XX{d4} on {date} by UPI. Ref {ref}. Avl Bal Rs {bal}.",
    "{otp} is your OTP for the transaction of Rs {amt} at {merch}. Do not share it with anyone. -{bank}",
    "Your {bank} credit card statement of Rs {amt} is generated. Minimum due Rs {mn} by {date}. Pay via the {bank} app.",
    "Dear customer, your electricity bill of Rs {amt} is due on {date}. Pay at the official website or counter.",
    "Your recharge of Rs {amt} is successful. Validity {n} days. Thank you for choosing {tel}.",
    "Your order {ref} from {merch} has been delivered. Rate your experience in the app.",
    "Salary of Rs {amt} credited to A/c XX{d4} on {date}.",
    "EMI of Rs {amt} for loan A/c XX{d4} will be auto-debited on {date}. Please maintain sufficient balance.",
    "Payment of Rs {amt} to {merch} was successful. Txn ID {ref}.",
    "Your {bank} KYC is up to date. No action is required. For queries visit {url}",
    "Reminder: your insurance premium of Rs {amt} is due on {date}. Pay through the official app or at the branch.",
    "Your appointment with Dr {name} is confirmed for {date}. Reply C to cancel.",
    "Your {bank} debit card ending {d4} has been blocked as requested. Visit the branch or app to get a new card.",
    "Your parcel from {merch} is out for delivery today. Track it here {url}",
]
CASUAL_A = ["rent for the month", "chai and snacks", "groceries", "dinner split", "auto fare", "tuition fee", "movie tickets",
            "returning the money you lent me", "gift for amma", "cab share", "lunch today", "mobile recharge for appa"]

TEMPLATES_B = [   # held out: different phrasing, never used for training
    "Transaction alert: INR {amt} spent on card ending {d4} at {merch} on {date}. Not you? Block the card in the app.",
    "Money received: {name} sent you Rs {amt} on GPay. Check your balance in the app.",
    "Your {bank} fixed deposit of Rs {amt} matures on {date}. Visit the branch or the app to renew.",
    "OTP {otp} for login to {bank} NetBanking. Valid for 5 minutes. Bank staff will never ask for your OTP.",
    "Ticket booked. PNR {ref}, journey on {date}. Happy journey. -IRCTC",
    "Dear {name}, your gas cylinder booking is confirmed, delivery by {date}. Pay only the displayed amount on delivery.",
    "Your income tax refund of Rs {amt} has been processed. It will reach your bank account in a few days.",
    "Reminder: school fee of Rs {amt} is due on {date}. Pay at the school office or the official portal {url}",
    "Your account will be updated with the new IFSC code from {date}. No action is needed from you.",
    "Alert: your {bank} account has a low balance of Rs {bal}. Add money using the app.",
    "Your {tel} postpaid bill of Rs {amt} is due on {date}. Pay in the app to avoid late charges.",
    "Hi {name}, can you pay Rs {amt} for the cab, I'll send the bill photo in the group.",
]
TEMPLATES_C = [   # used only for the validation period of the simulator: also never seen by the message model
    "UPI payment of Rs {amt} to {merch} on {date} was successful. Balance Rs {bal}. -{bank}",
    "Your {bank} savings account statement for the month is available in the app.",
    "Dear {name}, your LPG subsidy of Rs {amt} was credited to your account on {date}.",
    "Do not share your card details or PIN with anyone. {bank} never asks for them by call or SMS.",
    "{otp} is the code to verify your mobile number on {merch}. It expires in 10 minutes.",
    "Your electricity meter reading is due. Submit it on the official portal {url} by {date}.",
    "Alert: Rs {amt} withdrawn at ATM on {date}. If this was not you, call {ph} immediately.",
    "Premium renewal reminder: Rs {amt} is due on {date}. Pay on the company website.",
    "Congratulations {name}, your loan application is approved. Visit the branch with documents to complete the process.",
    "Your refund of Rs {amt} from {merch} has been credited to the original payment method.",
    "Reminder: your dental appointment is tomorrow at 5 pm. Reply YES to confirm.",
    "The last date to link PAN with Aadhaar is {date}. Check the status on the official income tax website {url}",
]
CASUAL_C = ["pani puri", "hostel mess fee", "photocopy charges", "bus ticket"]
CASUAL_B = ["pooja flowers", "plumber charges", "school bus fee", "newspaper bill", "birthday cake", "weekend trip petrol"]


def make_legit(which, n, rng):
    T, C = {"A": (TEMPLATES_A, CASUAL_A), "B": (TEMPLATES_B, CASUAL_B), "C": (TEMPLATES_C, CASUAL_C)}[which]
    out = []
    for _ in range(n):
        if rng.random() < 0.2:
            out.append(f"{rng.choice(C)} sending {int(rng.integers(50, 5000))}")
            continue
        out.append(rng.choice(T).format(
            amt=f"{int(rng.integers(50, 90000)):,}", bal=f"{int(rng.integers(100, 200000)):,}", mn=int(rng.integers(100, 5000)),
            d4=int(rng.integers(1000, 9999)), date=f"{int(rng.integers(1, 28))}-Oct-26", ref=int(rng.integers(10**8, 10**9)),
            vpa=f"{rng.choice(_NAMES).lower()}@ybl", ph=f"1800{int(rng.integers(100000, 999999))}", otp=int(rng.integers(100000, 999999)),
            merch=rng.choice(_MERCH), bank=rng.choice(_BANKS), name=rng.choice(_NAMES), tel=rng.choice(_TEL), n=int(rng.choice([28, 56, 84])),
            url=f"https://{rng.choice(['sbi.co.in', 'hdfcbank.com', 'icicibank.com', 'axisbank.com'])}/help"))
    return out


# ------------------------------------------------------------------ corpus
def build_corpus(out_path=CORPUS_PATH, seed=42):
    import urllib.request
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imc_p, uci_p = "/tmp/orion3_imc.csv", "/tmp/orion3_uci.tsv"
    for url, p in [(IMC_URL, imc_p), (UCI_URL, uci_p)]:
        if not os.path.exists(p): urllib.request.urlretrieve(url, p)
    rng = np.random.default_rng(seed)

    imc = pd.read_csv(imc_p)
    imc = imc[(imc.language == "English") & imc.text.notna() & ~imc.scam_type.isin(["spam", "others"])].copy()
    imc["text_norm"] = imc.text.map(normalize_text)
    imc = imc[imc.text_norm.str.len() >= 15].drop_duplicates("text_norm")          # near-identical campaign texts count once
    # The dataset's "time" column is screenshot text ("Today 15:33"), so a real time split is impossible.
    # Instead split by TEMPLATE GROUP (first 5 normalised words): test scams come from templates the model never saw.
    import hashlib
    key = imc.text_norm.map(lambda t: " ".join(t.split()[:5]))
    bucket = key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 100)
    pos = pd.DataFrame({"text_norm": imc.text_norm.values, "label": 1, "source": "imc25_smishing",
                        "split": np.where(bucket < 70, "train", np.where(bucket < 85, "val", "test")),
                        "india": imc.original_network_country.astype(str).eq("IND").values, "scam_type": imc.scam_type.values})

    uci = pd.read_csv(uci_p, sep="\t", header=None, names=["label", "message"]).dropna().drop_duplicates("message")
    ham = uci[uci.label == "ham"].copy(); ham["text_norm"] = ham.message.map(normalize_text)
    ham = ham[ham.text_norm.str.len() >= 3].drop_duplicates("text_norm")
    u = rng.random(len(ham))
    neg_real = pd.DataFrame({"text_norm": ham.text_norm.values, "label": 0, "source": "uci_ham",
                             "split": np.where(u < 0.70, "train", np.where(u < 0.85, "val", "test")), "india": False, "scam_type": ""})

    def author(which, split, n):
        t = pd.Series(make_legit(which, n, rng)).map(normalize_text).drop_duplicates()
        return pd.DataFrame({"text_norm": t.values, "label": 0, "source": f"author_legit_{which}", "split": split, "india": True, "scam_type": ""})

    corpus = pd.concat([pos, neg_real, author("A", "train", 1500), author("C", "val", 600), author("B", "test", 600)], ignore_index=True)
    corpus.to_csv(out_path, index=False, compression="gzip")
    print(corpus.groupby(["source", "split"]).size().unstack(fill_value=0)); print("saved", out_path, f"{os.path.getsize(out_path) / 1e6:.2f} MB")
    return corpus


def load_corpus(path=None):
    """Looks for data/text_train.csv.gz, and also for text_train.csv.gz next to this file (if the data folder was flattened)."""
    for p in ([path] if path else [CORPUS_PATH, os.path.join(HERE, "text_train.csv.gz")]):
        if os.path.exists(p): return pd.read_csv(p)
    return None


def default_text_model():
    """Message model used by the demo: trained on the TRAIN split of the real corpus (None if the corpus file is missing)."""
    c = load_corpus()
    if c is None: return None
    tr = c[(c.split == "train") & ~c.scam_type.isin(HELDOUT_TYPES)]      # impersonation types stay unseen
    return build_text_model(tr.text_norm, tr.label)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rebuild", action="store_true"); a = ap.parse_args()
    if a.rebuild: build_corpus()
    else: print("corpus present:", load_corpus() is not None)
