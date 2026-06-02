"""
Manual tuned model: XGBoost + SMOTE oversampling + isotonic calibration.
Includes basic SHAP feature importance.
"""

import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin


class SMOTEXGBPipeline(BaseEstimator, ClassifierMixin):
    """SMOTE -> XGBoost -> Platt/isotonic calibration wrapper."""

    def __init__(self, xgb_params=None, sampling_strategy=0.25, calibration="isotonic"):
        self.xgb_params = xgb_params or {}
        self.sampling_strategy = sampling_strategy
        self.calibration = calibration

    def fit(self, X, y):
        sm = SMOTE(sampling_strategy=self.sampling_strategy, random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        xgb = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            **self.xgb_params,
        )
        self.cal_model_ = CalibratedClassifierCV(xgb, method=self.calibration, cv=5)
        self.cal_model_.fit(X_res, y_res)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.cal_model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_manual_model(X_train, y_train, feature_names=None):
    model = SMOTEXGBPipeline(sampling_strategy=0.30, calibration="isotonic")
    model.fit(X_train, y_train)
    return model
