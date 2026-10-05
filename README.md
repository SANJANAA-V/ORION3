# ORION³

### Pre-Transaction Intelligence for Authorized Payment Scams

ORION³ is an AI-based risk intelligence system designed to detect
authorized-payment scams before a victim completes a digital payment.

Unlike traditional unauthorized-fraud detection, authorized scams occur
when the victim personally approves the transaction after being
manipulated by a scammer.

---

## Problem

Digital payments are fast and often difficult to reverse.

Traditional fraud systems primarily focus on unauthorized transactions,
while authorized scams can look legitimate because the victim technically
initiates the payment.

ORION³ aims to identify risk using multiple signals:

1. Transaction behavior
2. Scam-message patterns
3. Receiver/network behavior

The final system will produce an interpretable:

**ALLOW / WARN / BLOCK**

decision before payment completion.

---

# M1 — Real-Data Fraud Detection Benchmark

M1 establishes the machine-learning foundation of ORION³ using the
IEEE-CIS Fraud Detection dataset.

The purpose of this stage is to benchmark fraud-ranking capability before
building the authorized-scam simulator.

## Dataset

- Transactions: 590,540
- Fraud cases: 20,663
- Fraud rate: 3.50%
- Dataset: IEEE-CIS Fraud Detection

## Validation Strategy

A chronological split was used rather than a random split.

The final evaluation uses:

- 70% oldest transactions → training
- 15% → tuning
- 15% newest transactions → final holdout

This reduces temporal leakage and better represents future deployment.

---

# Model Comparison

Initial chronological validation:

| Model | Features | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| Dummy Baseline | — | 0.0344 | 0.5000 |
| Logistic Regression | 52 | 0.2778 | 0.8148 |
| Decision Tree | 52 | 0.3193 | 0.7884 |
| LightGBM | 52 | **0.5267** | **0.9091** |

LightGBM was selected as the champion model.

---

# Final Holdout

The final model was evaluated on an untouched chronological 15% holdout.

| Metric | Score |
|---|---:|
| PR-AUC | **0.5046** |
| ROC-AUC | **0.8992** |

The difference between the initial validation result and final holdout
result demonstrates why a separate temporal holdout is important.

---

# Explainability

SHAP was used to understand model behavior.

Top influential features included:

- C5
- TransactionAmt
- D3
- C14
- card6

The IEEE-CIS dataset contains anonymized feature names, so these features
cannot directly provide human-readable explanations.

ORION³ therefore plans to generate readable explanations from its own
interpretable behavioral and network features.

---

# Operating Point

At approximately 1% false-positive rate:

- Threshold: 0.2953
- False-positive rate: ~1.00%
- Recall: 43.87%

This demonstrates the trade-off between fraud detection and false alarms.

---

# Cost-Sensitive Analysis

An illustrative cost model was evaluated where:

- Missed fraud cost = transaction amount
- False-positive cost = fixed cost

On the tuning/validation set:

| Strategy | Total Loss | Loss Saved |
|---|---:|---:|
| Flag nothing | 609,934.31 | 0% |
| Flag everything | 1,140,440.00 | -86.98% |
| Threshold 0.5 | 443,677.50 | 27.26% |
| Optimized threshold ≈ 0.06 | 271,528.75 | 55.48% |

These values are illustrative and depend on the assumed business cost.

### Sensitivity

| False-positive cost | Best threshold |
|---:|---:|
| 5 | 0.0311 |
| 10 | 0.0612 |
| 50 | 0.1966 |
| 100 | 0.4624 |

The optimal threshold changes substantially as the assumed cost of false
alarms changes.

---

# Temporal Drift

Six-week validation analysis:

| Week | PR-AUC |
|---:|---:|
| 1 | 0.4954 |
| 2 | 0.5623 |
| 3 | 0.5141 |
| 4 | 0.4828 |
| 5 | 0.5903 |
| 6 | 0.5186 |

There was moderate temporal variation with no severe decay over this
six-week validation window.

---

# Feature Ablation

Additional behavioral features were tested.

The experiment removed the dataset's C/D features and compared:

1. Remaining original features
2. Remaining features + engineered behavioral features

| Model | PR-AUC | ROC-AUC |
|---|---:|---:|
| Full 52-feature LightGBM | **0.5046** | **0.8992** |
| Without C/D | 0.2395 | 0.8390 |
| Without C/D + Engineered | 0.2116 | 0.8366 |

The experiment showed that the anonymized C/D features contain substantial
predictive information. The current engineered features did not provide
additional signal in this setting.

This negative result is retained rather than treated as a failed project
step because it provides information about where the predictive signal
already exists in the benchmark dataset.

---

# Architecture Direction

The project will eventually combine three signals:

### 1. Behavioral Risk
- unusual transaction amount
- unusual transaction timing
- new receiver
- deviation from normal user behavior

### 2. Scam Message Analysis
- scam-like SMS/chat content
- urgency
- fake KYC
- prize scams
- fake customer support
- remote-access scams

### 3. Receiver Network Risk
- number of distinct senders
- rapid forwarding of funds
- mule-account behavior
- transaction-network patterns

These signals will eventually be combined into a single risk decision:

**ALLOW → WARN → BLOCK**

---

# Limitations

- IEEE-CIS is a benchmark fraud dataset and does not directly represent
  authorized UPI scam behavior.
- Many benchmark features are anonymized.
- Cost analysis depends on assumed business costs.
- Temporal evaluation currently covers a limited validation window.
- The authorized-payment scam simulator is still under development.
- Current results should not be interpreted as production fraud-detection
  performance.

---

# Roadmap

### M1 — Real-Data Fraud Pipeline
**Status: Complete**

- Dataset exploration
- Chronological validation
- Baseline models
- LightGBM
- Feature experiments
- SHAP
- Threshold analysis
- Cost analysis
- Drift analysis
- Final holdout
- Ablation study

### M2 — Authorized Scam Simulator
**Next**

- Normal users
- Mule accounts
- Cash-out accounts
- Scam scenarios
- Payment network simulation
- Behavioral risk features

### M3 — Multimodal Scam Intelligence

- Scam message analysis
- Receiver-network intelligence
- Risk fusion
- Explainable ALLOW/WARN/BLOCK decision
- Interactive demonstration

---

## Project Status

**M1 Complete ✅**

Next milestone:

**Authorized-Payment Scam Simulator**
