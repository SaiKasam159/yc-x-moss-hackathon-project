"""
Diagnostic script to analyze model overfitting and feature importance.

Usage:
    python ml/diagnose_model.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVR

from ml.train_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_uci_dataset,
)


def diagnose():
    print("Loading dataset...")
    df = load_uci_dataset()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n=== Dataset Analysis ===")
    print(f"Total samples: {len(df)}")
    print(f"Train/test split: {len(X_train)}/{len(X_test)}")
    print(f"Target (UPDRS) range: {y.min():.1f} - {y.max():.1f}, mean={y.mean():.1f}, std={y.std():.1f}")

    print("\n=== Feature Statistics ===")
    for col in FEATURE_COLUMNS:
        print(f"{col:20s}: mean={X[col].mean():.4f}, std={X[col].std():.4f}")

    print("\n=== RandomForest (default params) ===")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_r2 = rf.score(X_train, y_train)
    test_r2 = rf.score(X_test, y_test)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"5-fold CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"Overfitting gap: {train_r2 - test_r2:.4f}")

    print("\n=== Feature Importance ===")
    importances = rf.feature_importances_
    for feat, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True):
        print(f"{feat:20s}: {imp:.4f}")

    print("\n=== Permutation Importance ===")
    perm_imp = permutation_importance(rf, X_test, y_test, random_state=42, n_repeats=10)
    for feat, imp in sorted(
        zip(FEATURE_COLUMNS, perm_imp.importances_mean), key=lambda x: x[1], reverse=True
    ):
        print(f"{feat:20s}: {imp:.4f}")

    print("\n=== RandomForest with reduced complexity ===")
    rf_shallow = RandomForestRegressor(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
    )
    rf_shallow.fit(X_train, y_train)
    train_r2_s = rf_shallow.score(X_train, y_train)
    test_r2_s = rf_shallow.score(X_test, y_test)
    cv_scores_s = cross_val_score(rf_shallow, X_train, y_train, cv=5)
    print(f"Train R²: {train_r2_s:.4f}")
    print(f"Test R²: {test_r2_s:.4f}")
    print(f"5-fold CV R²: {cv_scores_s.mean():.4f} (±{cv_scores_s.std():.4f})")
    print(f"Overfitting gap: {train_r2_s - test_r2_s:.4f}")

    print("\n=== SVR (tuned) ===")
    svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    svr.fit(X_train, y_train)
    train_r2_svr = svr.score(X_train, y_train)
    test_r2_svr = svr.score(X_test, y_test)
    cv_scores_svr = cross_val_score(svr, X_train, y_train, cv=5)
    print(f"Train R²: {train_r2_svr:.4f}")
    print(f"Test R²: {test_r2_svr:.4f}")
    print(f"5-fold CV R²: {cv_scores_svr.mean():.4f} (±{cv_scores_svr.std():.4f})")
    print(f"Overfitting gap: {train_r2_svr - test_r2_svr:.4f}")

    print("\n=== Recommendations ===")
    if train_r2 - test_r2 > 0.3:
        print("⚠️  Significant overfitting detected (gap > 0.3)")
        print("   Options:")
        print("   1. Reduce model complexity (max_depth, n_estimators)")
        print("   2. Increase regularization (higher C for SVR)")
        print("   3. Feature selection (drop low-importance features)")
        print("   4. Cross-validation ensures generalization on new data")
        print(f"   5. Current CV R² ({cv_scores.mean():.3f}) is more realistic than test R²")


if __name__ == "__main__":
    diagnose()
