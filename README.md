# Cash Advance Repayment Model — SageMaker Autopilot + Manual XGBoost

Binary credit-risk classifier for fintech cash advance products. Evaluates SageMaker Autopilot against a tuned manual pipeline (XGBoost + SMOTE + isotonic calibration) on an imbalanced (~88/12) repayment dataset.

## Architecture

```
Plaid transactions + PostgreSQL history
        |
        v
data_prep.py  (time-based split, feature engineering, no leakage)
        |
   +----+----+
   |         |
Baseline   SMOTE + XGBoost + Calibration
(XGBoost)  (manual_model.py)
   |         |
   +----+----+
        |
   Autopilot job (autopilot_runner.py)
        |
comparison.py  (AUC-ROC / PR-AUC / KS / FNR@FPR10 / threshold analysis)
```

## Features Engineered

| Category | Features |
|---|---|
| Account | age_days, available_balance, days_since_last_tx |
| Transactions (30d) | count, mean_amount, std_amount |
| Payroll | salary_detected, days_until_next_payroll |
| User | age, device_os, neobank_flag |
| History (returning) | prior_advances, prior_default_rate |

## Setup

```bash
pip install -r requirements.txt
# Demo mode (synthetic data, no AWS needed):
python main.py --mode demo --skip-autopilot

# Full mode (requires AWS credentials + S3 bucket):
python main.py --mode full --s3-bucket YOUR_BUCKET
```

## Key design decisions

- **Time-based split** (not random): prevents data leakage from future observations
- **SMOTE on train only**: oversampling applied after split to avoid contaminating test set
- **Isotonic calibration**: corrects probability outputs for downstream threshold analysis
- **FNR@FPR=10%** as primary metric: false negatives (approved defaulters) carry direct financial cost
- **Reject inference placeholder**: `data_prep.py` documents where propensity-score weighting would be applied on real data
- **Segment-aware evaluation**: new vs returning users evaluated separately (different default rates and feature availability)

## SageMaker Autopilot Notes

The `autopilot_runner.py` module:
1. Uploads train CSV to S3
2. Launches `AutoML` job with `F1` objective (not accuracy — class imbalance aware)
3. Deploys best candidate to a real-time endpoint
4. Scores the held-out test set and returns comparable metrics

Estimated Autopilot job cost: ~$2-8 USD for 20 candidates on this dataset size.

## License

MIT
