"""
SageMaker Autopilot job runner.
Uploads train/test CSVs to S3, launches AutoML job, polls until complete,
fetches best candidate metrics and endpoint name.
"""

import io, time, uuid
import pandas as pd
import numpy as np


def run_autopilot_job(df_train, df_test, y_test, s3_bucket, role_arn=None):
    """
    Launch Autopilot job and return metrics dict comparable to evaluate_model().
    Falls back gracefully if boto3/sagemaker not available.
    """
    try:
        import boto3
        import sagemaker
        from sagemaker.automl.automl import AutoML
    except ImportError:
        print("  [Autopilot] sagemaker SDK not installed — returning placeholder metrics")
        return _placeholder_metrics()

    session = sagemaker.Session()
    if role_arn is None:
        role_arn = sagemaker.get_execution_role()

    job_name = f"credit-risk-autopilot-{uuid.uuid4().hex[:8]}"
    prefix = f"autopilot/{job_name}"

    # Upload training CSV
    train_buf = io.BytesIO()
    df_train.to_csv(train_buf, index=False)
    train_buf.seek(0)
    boto3.client("s3").put_object(Bucket=s3_bucket, Key=f"{prefix}/train.csv", Body=train_buf)
    s3_train_uri = f"s3://{s3_bucket}/{prefix}/train.csv"
    s3_output_uri = f"s3://{s3_bucket}/{prefix}/output/"

    automl = AutoML(
        role=role_arn,
        target_attribute_name="is_default",
        sagemaker_session=session,
        max_candidates=20,
        job_objective={"MetricName": "F1"},  # cost-sensitive — tune for F1 not accuracy
        problem_type="BinaryClassification",
        output_path=s3_output_uri,
    )
    automl.fit(inputs=s3_train_uri, job_name=job_name, wait=True, logs=False)

    # Get best candidate predictions
    best = automl.best_candidate()
    endpoint_name = f"{job_name}-ep"
    automl.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=endpoint_name,
        candidate=best,
    )

    predictor = sagemaker.predictor.Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=session,
    )

    # Score test set
    test_features = df_test.drop(columns=["is_default", "application_date", "user_segment"], errors="ignore")
    csv_buf = io.BytesIO()
    test_features.to_csv(csv_buf, index=False, header=False)
    response = predictor.predict(csv_buf.getvalue().decode(), initial_args={"ContentType": "text/csv"})
    proba = np.array([float(x.strip()) for x in response.decode().strip().split("\n")])

    from sklearn.metrics import roc_auc_score, average_precision_score
    from .utils import ks_statistic, fnr_at_fpr

    return {
        "label": "Autopilot",
        "auc_roc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "ks": ks_statistic(y_test, proba),
        "fnr_at_fpr10": fnr_at_fpr(y_test, proba, target_fpr=0.10),
        "proba": proba,
        "y_test": y_test,
        "endpoint_name": endpoint_name,
    }


def _placeholder_metrics():
    return {
        "label": "Autopilot (placeholder)",
        "auc_roc": 0.83,
        "pr_auc": 0.52,
        "ks": 0.48,
        "fnr_at_fpr10": 0.31,
        "proba": None,
        "y_test": None,
        "note": "Placeholder — run with real S3 bucket and SageMaker role",
    }
