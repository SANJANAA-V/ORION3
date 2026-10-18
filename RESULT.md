| Model | Easy | Hard |
|---|---|---|
| Rules (hand weights) | 0.500 | 0.044 |
| Text only | 0.549 | 0.459 |
| Sender behaviour only | 0.454 | 0.192 |
| Receiver/network only | 0.824 | 0.366 |
| Behaviour + network | 0.946 | 0.556 |
| Learned fusion | 0.968 (95% CI 0.951–0.982) | 0.788 (95% CI 0.731–0.842) |

Held-out family (impersonation), fusion PR-AUC: 0.948 (easy), 0.652 (hard).
Recall at the validation threshold: 0.939 / 0.624.
Payments with no message: PR-AUC 0.939 (easy), 0.577 (hard).
Legitimate fundraiser-style bursts wrongly flagged: 0.9% (easy), 0.5% (hard).
Scoring time: use the p50/p95 your own run prints (about 6 ms in my test, models only).
