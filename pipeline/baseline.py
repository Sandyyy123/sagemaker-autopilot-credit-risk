"""Baseline XGBoost model — replicates current production scorer."""

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from .utils import ks_statistic, fnr_at_fpr


def train_baseline(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=(1 - y_train.mean()) / y_train.mean(),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label="Model", segment_col=None, segments=None):
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "label": label,
        "auc_roc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "ks": ks_statistic(y_test, proba),
        "fnr_at_fpr10": fnr_at_fpr(y_test, proba, target_fpr=0.10),
        "proba": proba,
        "y_test": y_test,
    }
    return metrics
