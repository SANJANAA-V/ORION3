# ORION³: pre-payment risk scoring for authorized payment scams

**Live demo:** https://orion3.streamlit.app  
**Status:** research prototype. Payment behaviour is simulated; the message model and the fraud benchmark use real public data. It is not connected to any bank or UPI system and must not be used for real payment decisions.

## The problem

In authorized payment scams (fake KYC calls, fake prizes, fake customer care, remote-access tricks) the victim approves the payment themselves. Ordinary "unauthorized transaction" fraud models do not target this case, and money sent by UPI is hard to recover. ORION³ explores scoring a payment **before it completes**.

## What it does

For one payment, ORION³ combines signals and returns **ALLOW, WARN or BLOCK** with a plain-language reason.

| Signal | What it looks at |
|---|---|
| Sender behaviour | Amount compared with the sender's usual payments, first payment to this receiver, time of day |
| Receiver network | How many different people paid the receiver in the last hour, whether it already passed money on, account age |
| Message | Whether the message that led to the payment resembles scam wording (TF-IDF + logistic regression trained on real smishing) |

A gradient-boosted model scores behaviour and network together, and a logistic-regression layer adds the message score. Warn and block thresholds are chosen on validation data. The reasons come from the models' own weights.

**Main finding so far:** the payment signals (behaviour + network) carry most of the detection. The message signal adds little in my simulations, and my message model flags many legitimate formal messages (see Results A2 and Limitations).

## Results

### A. Real fraud data: IEEE-CIS (`ieee_cis_benchmark.py`)

590,540 card transactions, 3.50% fraud. Chronological split by transaction time: 413,378 train / 88,581 validation / 88,581 final holdout. Same 58 features for every model.

| Model | Validation PR-AUC | Validation ROC-AUC | Final holdout PR-AUC | Final holdout ROC-AUC |
|---|---|---|---|---|
| Dummy (base rate) | 0.034 | 0.500 | 0.035 | 0.500 |
| Logistic regression | 0.319 | 0.834 | 0.281 | 0.831 |
| Decision tree | 0.369 | 0.826 | 0.379 | 0.822 |
| **LightGBM** | **0.557** | **0.912** | **0.502** | **0.900** |

- LightGBM final holdout PR-AUC 0.502 (95% bootstrap CI 0.483 to 0.522). The holdout was scored once, after all choices were made on validation.
- The threshold giving 1% false positives on validation (0.219) gave recall 0.449, precision 0.538 and a false-positive rate of **1.39%** on the later holdout, so thresholds drift as data changes over time.
- Feature-group ablation (validation PR-AUC, mean of 3 seeds, full model 0.552):

| Removed group | PR-AUC | Change | Effect |
|---|---|---|---|
| C features (count features) | 0.396 | -0.156 | clear |
| Card attributes | 0.515 | -0.037 | clear |
| Address and distance | 0.541 | -0.011 | clear, small |
| M features | 0.542 | -0.011 | clear, small |
| D features | 0.543 | -0.010 | within noise |
| Time and amount | 0.545 | -0.008 | within noise |
| V features | 0.551 | -0.001 | within noise |

IEEE-CIS is credit-card fraud, not UPI and not authorized-payment scams. These numbers show the methods work on real data. They do not show that ORION³ catches real UPI scams.

### A2. Message models on real messages (`text_real_eval.py`)

Test set: 1,432 real smishing messages (192 from Indian networks) from the "Fishing for Smishing" dataset (IMC 2025, CC BY 4.0), split **by message template group** so the test templates were never seen in training; 679 real casual legitimate SMS (UCI SMS Spam Collection, held out); and 70 **author-written** formal legitimate bank/UPI/bill notifications from templates not used for training (these are not real messages). Every model uses the same threshold rule: the score giving 1% false alarms on real validation ham.

| Message model | Recall, all smishing | Recall, Indian-network | False alarms, real casual SMS | False alarms, formal legit (author-written) |
|---|---|---|---|---|
| **Real-data model (used in the demo)** | **0.967** | **0.984** | 0.009 | **0.514** |
| Simulated-template model (earlier demo model) | 0.223 | 0.469 | 0.012 | 0.014 |
| UCI spam model | 0.854 | 0.906 | 0.007 | 0.257 |

Recall of the real-data model by type: banking 0.99, delivery 0.99, government 0.97, telecom 1.00, wrong number 0.04 (this type is deliberately held out of training).

What this shows: training on real smishing fixes the weak recall of the template-trained model (0.22 to 0.97) but brings a serious problem: the model flags about half of the formal legitimate notifications I wrote, because it partly learned "formal message" as "scam". No public dataset of real legitimate bank and UPI alerts was found, so this cannot be fixed or tested properly yet.

### B. Simulated UPI-style system (`orion3_sim_v3.py`)

Payments, accounts and mule campaigns are simulated; message text is real (scam messages are real smishing from unseen templates; legitimate ones are real casual SMS plus author-written formal notifications). Test PR-AUC on a time-ordered held-out period. "Hard" mode uses low-and-slow campaigns, smaller scam amounts, fewer messages, more legitimate bursts and more formal legitimate messages.

| Model | Easy | Hard |
|---|---|---|
| Rules (hand-set weights) | 0.277 | 0.018 |
| Message only | 0.262 | 0.141 |
| Sender behaviour only | 0.279 | 0.175 |
| Receiver network only | 0.591 | 0.215 |
| **Behaviour + network** | **0.898** | **0.383** |
| Fusion (behaviour + network + message) | 0.873 (95% CI 0.820 to 0.907) | 0.386 (95% CI 0.324 to 0.468) |

- The message signal does **not** improve on behaviour + network in these tests (differences are inside the confidence interval).
- Held-out scam family ("impersonation": hey-mum/dad and wrong-number messages, never seen in training by any model): fusion PR-AUC 0.668 (easy), 0.253 (hard).
- Payments with no message: PR-AUC 0.883 (easy), 0.411 (hard).
- Legitimate fundraiser-style bursts wrongly flagged: 2.4% (easy), 3.1% (hard).
- Thresholds are chosen on validation data, then applied to the test period. The hard-mode validation period has few scams, so results are noisy.

### C. Serving speed

FastAPI endpoint, 300 HTTP requests on a Kaggle notebook (local network, models only, payment statistics supplied in the request): **p50 11.4 ms, p95 13.3 ms, p99 14.5 ms** (measured before the final message-model update; re-run `bench_api.py` to refresh). A real deployment would also need time to look up those statistics.

## What is real and what is simulated

| Part | Real data | Simulated |
|---|---|---|
| IEEE-CIS benchmark, baselines, ablation, thresholds | yes | no |
| Message model training and message test sets (smishing, casual SMS) | yes | no |
| Formal legitimate bank/UPI notifications | no | **author-written from templates** |
| Payments, accounts, mule campaigns, receiver statistics | no | **yes** |
| Fusion model, live demo, API | trained on simulated payments + real message text | **yes** |

## How the simulator avoids testing itself

An earlier version generated features directly from the label, so models only learned the generator. The current version generates payments and scam campaigns as events and then **computes every feature in time order from past events only**. It includes legitimate fundraiser bursts, large legitimate payments to new people, and campaigns that forward money through mule accounts. The scam family "impersonation" never appears in the training or validation days and is excluded from message-model training. Message text in each period comes from separate template groups, so no message model is tested on templates it trained on.

## Limitations

- Payment behaviour is synthetic. Real UPI behaviour will be harder.
- The message model flags about half of the author-written formal legitimate notifications (51%). Real legitimate bank and UPI alerts were not available, so false alarms on them are unmeasured. The formal legitimate messages in the simulator are also author-written.
- The message model misses "wrong number" style scams (4% recall; held out on purpose).
- The scam split is by template group, not by date, because the dataset's time column is not a usable timestamp.
- IEEE-CIS features are anonymized, so readable reasons come from the simulator's interpretable features.
- The hard-mode validation period has few scams, and some per-family test counts are small.
- The cost curve uses illustrative units, not real money.
- Not validated for privacy, fairness or safety. Not for real payment decisions.

## Roadmap

1. Collect real legitimate bank and UPI alerts (with consent) to measure and fix false alarms on formal messages.
2. Fine-tune a small transformer for message classification and compare it with TF-IDF.
3. Compare a graph neural network with the engineered receiver-network features.

## Run it

```
pip install -r requirements.txt
python orion3_sim_v3.py        # simulator + full evaluation (easy and hard mode)
python text_real_eval.py       # message models on real public smishing
python ieee_cis_benchmark.py --data path/to/train_transaction.csv --out results   # real-data fraud benchmark
streamlit run app.py           # demo app (trains the models on first start)

pip install fastapi uvicorn requests
uvicorn api:app --port 8000    # API; open http://localhost:8000/docs
python bench_api.py            # measures round-trip time over 300 requests
```

## Files

| File | Purpose |
|---|---|
| `orion3_sim_v3.py` | Simulator, features, models, evaluation |
| `orion3_text.py` | Message model, text normalisation, corpus builder |
| `data/text_train.csv.gz` | Normalised message corpus (see `data/ATTRIBUTION.md`) |
| `orion3_core.py` | Trains and scores one payment, builds the reasons |
| `app.py` | Streamlit demo |
| `api.py`, `bench_api.py` | FastAPI endpoint and latency benchmark |
| `ieee_cis_benchmark.py` | Real-data IEEE-CIS benchmark (models, holdout, ablation) |
| `text_real_eval.py` | Tests message models on real public smishing and legitimate SMS |
| `text_external_eval.py` | Optional: tests the message model on a CSV of messages you collect yourself |
| `RESULTS.md` | Simulator results |
| `01_*.ipynb`, `orion-3.ipynb` | Earlier exploration notebooks |

## Data sources and credit

- Real smishing messages: "Fishing for Smishing" dataset (Agarwal, Papasavva, Suarez-Tangil, Vasek; ACM IMC 2025), CC BY 4.0, github.com/reportsmishing/Smishing-Dataset-IMC25.
- Real legitimate SMS: UCI SMS Spam Collection (Almeida and Gomez Hidalgo), CC BY 4.0.
- Fraud transactions: IEEE-CIS Fraud Detection (Kaggle competition data, used under the competition rules, not redistributed here).
- Details of the changes made to the message data are in `data/ATTRIBUTION.md`.
