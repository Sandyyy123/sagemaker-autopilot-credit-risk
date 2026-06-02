"""
Cash Advance Repayment Model — SageMaker Autopilot + Manual XGBoost
Demo entry point: runs end-to-end pipeline on synthetic data when real data unavailable.
"""

import argparse
import sys
from pipeline.data_prep import load_and_split, build_features
from pipeline.baseline import train_baseline, evaluate_model
from pipeline.autopilot_runner import run_autopilot_job
from pipeline.manual_model import train_manual_model
from pipeline.comparison import generate_comparison_report
from pipeline.utils import print_metrics_table


def main():
    parser = argparse.ArgumentParser(description="Credit risk model pipeline")
    parser.add_argument("--mode", choices=["demo", "full"], default="demo",
                        help="demo=synthetic data, full=real data from S3")
    parser.add_argument("--s3-bucket", default="", help="S3 bucket with production data")
    parser.add_argument("--skip-autopilot", action="store_true",
                        help="Skip Autopilot (AWS cost); run manual model only")
    parser.add_argument("--target-fpr", type=float, default=0.10,
                        help="Target false positive rate for threshold selection")
    args = parser.parse_args()

    print("=" * 60)
    print("Cash Advance Repayment Prediction Pipeline")
    print("=" * 60)

    # Data preparation
    print("\n[1/4] Loading and splitting data...")
    df_train, df_test = load_and_split(mode=args.mode, s3_bucket=args.s3_bucket)
    X_train, y_train, X_test, y_test, feature_names = build_features(df_train, df_test)
    print(f"  Train: {len(df_train):,} rows | Test: {len(df_test):,} rows")
    print(f"  Class balance (test): {y_test.mean():.1%} default rate")

    # Baseline
    print("\n[2/4] Training baseline XGBoost...")
    baseline_model = train_baseline(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test, label="Baseline")
    print_metrics_table(baseline_metrics)

    # Autopilot (optional)
    autopilot_metrics = None
    if not args.skip_autopilot and args.s3_bucket:
        print("\n[3/4] Running SageMaker Autopilot job...")
        autopilot_metrics = run_autopilot_job(df_train, df_test, y_test, args.s3_bucket)
        print_metrics_table(autopilot_metrics)
    else:
        print("\n[3/4] Skipping Autopilot (--skip-autopilot or no S3 bucket)")

    # Manual tuned model
    print("\n[4/4] Training manual tuned model (XGBoost + SMOTE + calibration)...")
    manual_model = train_manual_model(X_train, y_train, feature_names)
    manual_metrics = evaluate_model(manual_model, X_test, y_test, label="Manual Tuned")
    print_metrics_table(manual_metrics)

    # Comparison report
    report = generate_comparison_report(
        baseline_metrics, manual_metrics, autopilot_metrics,
        X_test, y_test, feature_names, target_fpr=args.target_fpr
    )
    print(f"\nReport saved to: {report['output_path']}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
