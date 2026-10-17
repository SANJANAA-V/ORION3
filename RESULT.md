# ORION³ — Simulator v3 results (synthetic data)

All results below come from a SYNTHETIC UPI-style simulator (orion3_sim_v3.py).
They test the system design. They do not show performance on real payments.

Setup: ~65k payments, ~1% scams. Features are computed in time order from past
events only. Split by day: train / validation / test. Thresholds are chosen on
validation, then applied to test. Scam family "impersonation" is never seen in training.

## Test PR-AUC by signal

| Model | Easy | Hard |
|---|---|---|
| Rules (hand weights) | 0.487 | 0.069 |
| Text only | 0.631 | 0.465 |
| Sender behaviour only | 0.626 | 0.366 |
| Receiver/network only | 0.798 | 0.489 |
| Behaviour + network | 0.935 | 0.679 |
| Learned fusion | 0.975 | 0.835 (95% CI 0.789–0.866) |

Hard mode = low-and-slow campaigns, smaller scam amounts, fewer messages,
more legitimate bursts and more alarming-looking legitimate messages.

## Held-out scam family (impersonation, never seen in training)
Fusion PR-AUC: 0.959 (easy), 0.864 (hard). Recall at the validation threshold: 0.963 / 0.869.

## Other findings
- Payments with no message (hard mode): PR-AUC 0.619, recall 0.636.
- Legitimate fundraiser-style bursts wrongly flagged: 4.7% (easy), 1.0% (hard).
- Model scoring time: p50 12.4 ms, p95 13.7 ms (features precomputed, models only).

## Limitations
- Synthetic data. Real UPI behaviour will be harder.
- Messages come from a few fixed templates, so text results here are NOT reliable.
  Text must be judged on independent and public data.
- Hard mode has only 94 validation scams and 4 test scams in one family.
- Illustrative cost units, not real money.
