"""Shared metric helpers."""

import numpy as np
from sklearn.metrics import roc_curve


def ks_statistic(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def fnr_at_fpr(y_true, y_score, target_fpr=0.10):
    """False negative rate at a given false positive rate operating point."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr)
    if idx >= len(tpr):
        idx = len(tpr) - 1
    fnr = 1.0 - tpr[idx]
    return float(fnr)


def print_metrics_table(m):
    label = m.get("label", "Model")
    print(f"  {label}:")
    print(f"    AUC-ROC      : {m['auc_roc']:.4f}")
    print(f"    PR-AUC       : {m['pr_auc']:.4f}")
    print(f"    KS statistic : {m['ks']:.4f}")
    print(f"    FNR@FPR=10%  : {m['fnr_at_fpr10']:.4f}")
