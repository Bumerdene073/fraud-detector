"""
Model Training - 3 Model Comparison + SMOTE
=============================================
Trains and compares 3 fraud detection models:
  1. Logistic Regression  (baseline)
  2. Random Forest        (ensemble)
  3. XGBoost              (gradient boosting - industry standard)

Uses SMOTE to handle class imbalance.
Picks winner based on F1 Score + ROC-AUC.
Saves winning model to serving/model/

Run: python training/train_model.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)
from imblearn.over_sampling  import SMOTE
from xgboost                 import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PROCESSED_FILE    = "data/processed/features.csv"
MODEL_DIR         = "serving/model"
FEATURE_COLS_FILE = "serving/model/feature_columns.pkl"

RANDOM_STATE = 42
TEST_SIZE    = 0.2    # 80% train, 20% test


# ─────────────────────────────────────────────────────────────────────────────
# LOAD FEATURE COLUMNS
# Must use EXACT same columns in exact same order as feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def load_feature_cols():
    with open(FEATURE_COLS_FILE, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE MODEL
# Calculates all metrics + prints a clean report
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(name, model, X_test, y_test, train_time):
    start    = time.time()
    y_pred   = model.predict(X_test)
    y_proba  = model.predict_proba(X_test)[:, 1]
    infer_ms = round((time.time() - start) / len(X_test) * 1000, 4)

    precision  = precision_score(y_test, y_pred,  zero_division=0)
    recall     = recall_score(y_test, y_pred,     zero_division=0)
    f1         = f1_score(y_test, y_pred,          zero_division=0)
    roc_auc    = roc_auc_score(y_test, y_proba)
    pr_auc     = average_precision_score(y_test, y_proba)
    cm         = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  MODEL: {name}")
    print(f"{'='*55}")
    print(f"  Precision      : {precision:.4f}  (of flagged fraud, how many are real)")
    print(f"  Recall         : {recall:.4f}  (of real fraud, how many caught)")
    print(f"  F1 Score       : {f1:.4f}  (PRIMARY metric - balance)")
    print(f"  ROC-AUC        : {roc_auc:.4f}  (overall discrimination)")
    print(f"  PR-AUC         : {pr_auc:.4f}  (best for imbalanced data)")
    print(f"  Train time     : {train_time:.2f}s")
    print(f"  Inference/call : {infer_ms:.4f}ms")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"               Legit    Fraud")
    print(f"  Actual Legit  {cm[0][0]:5d}    {cm[0][1]:5d}  <- TN / FP")
    print(f"  Actual Fraud  {cm[1][0]:5d}    {cm[1][1]:5d}  <- FN / TP")
    print(f"\n  Fraud caught    : {cm[1][1]} out of {cm[1][0]+cm[1][1]}")
    print(f"  False alarms    : {cm[0][1]}")

    return {
        "name"       : name,
        "model"      : model,
        "precision"  : precision,
        "recall"     : recall,
        "f1"         : f1,
        "roc_auc"    : roc_auc,
        "pr_auc"     : pr_auc,
        "train_time" : train_time,
        "infer_ms"   : infer_ms,
        "tp"         : cm[1][1],
        "fp"         : cm[0][1],
        "fn"         : cm[1][0],
        "tn"         : cm[0][0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def train():

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    print("Loading engineered features...")
    df          = pd.read_csv(PROCESSED_FILE)
    feature_cols = load_feature_cols()

    X = df[feature_cols].values
    y = df["is_fraud"].values

    print(f"  Dataset shape   : {X.shape}")
    print(f"  Legitimate      : {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"  Fraud           : {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    print(f"  Imbalance ratio : {(y==0).sum() / (y==1).sum():.0f}:1")

    # ── TRAIN / TEST SPLIT ────────────────────────────────────────────────────
    # StratifiedKFold ensures fraud % is same in train and test
    # Without stratify: test set might have 0 fraud cases!
    print(f"\nSplitting data (80% train / 20% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y          # CRITICAL for imbalanced data
    )
    print(f"  Train set: {X_train.shape[0]} rows, {y_train.sum()} fraud")
    print(f"  Test set : {X_test.shape[0]} rows, {y_test.sum()} fraud")

    # ── SCALE FEATURES ────────────────────────────────────────────────────────
    # StandardScaler: mean=0, std=1 for every feature
    # Fit ONLY on training data — never on test data!
    # WHY: If we scale on all data, test data leaks into training = cheating
    print(f"\nScaling features...")
    scaler   = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform train
    X_test_scaled  = scaler.transform(X_test)         # ONLY transform test

    # ── SMOTE — SOLVE CLASS IMBALANCE ─────────────────────────────────────────
    # Apply SMOTE ONLY on training data — never on test data!
    # WHY: If we SMOTE test data, evaluation becomes unreliable
    #      Test data must represent REAL world distribution
    print(f"\nApplying SMOTE to balance training data...")
    print(f"  Before SMOTE: {y_train.sum()} fraud / {(y_train==0).sum()} legit")

    smote = SMOTE(
        sampling_strategy = 0.3,  # make fraud = 30% of training data
                                  # not 50/50 — too aggressive for fraud
        random_state      = RANDOM_STATE,
        k_neighbors       = 5
    )
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    print(f"  After SMOTE : {y_train_balanced.sum()} fraud / {(y_train_balanced==0).sum()} legit")
    print(f"  New fraud % : {y_train_balanced.mean()*100:.1f}%")

    # ── MODEL 1: LOGISTIC REGRESSION (Baseline) ────────────────────────────────
    print(f"\nTraining Model 1: Logistic Regression (baseline)...")
    start = time.time()
    lr    = LogisticRegression(
        C            = 1.0,          # regularization strength
        max_iter     = 1000,         # enough iterations to converge
        random_state = RANDOM_STATE,
        n_jobs       = -1
    )
    lr.fit(X_train_balanced, y_train_balanced)
    lr_time = time.time() - start
    print(f"  Done in {lr_time:.2f}s")
    lr_results = evaluate_model("Logistic Regression", lr, X_test_scaled, y_test, lr_time)

    # ── MODEL 2: RANDOM FOREST ─────────────────────────────────────────────────
    print(f"\nTraining Model 2: Random Forest...")
    start = time.time()
    rf    = RandomForestClassifier(
        n_estimators = 100,          # 100 trees
        max_depth    = 10,           # prevent overfitting
        min_samples_leaf = 5,        # minimum samples at leaf node
        random_state = RANDOM_STATE,
        n_jobs       = -1,           # use all CPU cores
        class_weight = "balanced"    # additional imbalance handling
    )
    rf.fit(X_train_balanced, y_train_balanced)
    rf_time = time.time() - start
    print(f"  Done in {rf_time:.2f}s")
    rf_results = evaluate_model("Random Forest", rf, X_test_scaled, y_test, rf_time)

    # ── MODEL 3: XGBOOST (Industry Standard) ───────────────────────────────────
    print(f"\nTraining Model 3: XGBoost...")
    # scale_pos_weight: additional weight on fraud class
    # = number of legitimate / number of fraud in ORIGINAL data
    fraud_weight = int((y_train == 0).sum() / (y_train == 1).sum())
    start = time.time()
    xgb   = XGBClassifier(
        n_estimators      = 200,         # more trees = better (with early stopping)
        max_depth         = 6,           # sweet spot for tabular data
        learning_rate     = 0.05,        # slow learning = better generalization
        subsample         = 0.8,         # use 80% of data per tree
        colsample_bytree  = 0.8,         # use 80% of features per tree
        scale_pos_weight  = fraud_weight,# handles remaining imbalance
        random_state      = RANDOM_STATE,
        eval_metric       = "logloss",
        verbosity         = 0,
        n_jobs            = -1
    )
    xgb.fit(
        X_train_balanced, y_train_balanced,
        eval_set              = [(X_test_scaled, y_test)],
        verbose               = False
    )
    xgb_time = time.time() - start
    print(f"  Done in {xgb_time:.2f}s")
    xgb_results = evaluate_model("XGBoost", xgb, X_test_scaled, y_test, xgb_time)

    # ── MODEL COMPARISON TABLE ─────────────────────────────────────────────────
    all_results = [lr_results, rf_results, xgb_results]

    print(f"\n\n{'='*70}")
    print(f"  FINAL MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'F1':>7} {'ROC-AUC':>9} {'Precision':>10} {'Recall':>8} {'ms/call':>9}")
    print(f"  {'-'*65}")
    for r in all_results:
        winner_tag = " <- WINNER" if r == max(all_results, key=lambda x: x["f1"]) else ""
        print(f"  {r['name']:<25} {r['f1']:>7.4f} {r['roc_auc']:>9.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f} "
              f"{r['infer_ms']:>8.4f}ms{winner_tag}")

    # ── PICK WINNER ────────────────────────────────────────────────────────────
    # Primary: F1 Score | Tiebreaker: ROC-AUC
    winner = max(all_results, key=lambda x: (x["f1"], x["roc_auc"]))
    print(f"\n  WINNER: {winner['name']}")
    print(f"  F1={winner['f1']:.4f} | ROC-AUC={winner['roc_auc']:.4f}")
    print(f"  Fraud caught: {winner['tp']} / {winner['tp']+winner['fn']}")
    print(f"  False alarms: {winner['fp']}")

    # ── FEATURE IMPORTANCE (XGBoost + Random Forest) ───────────────────────────
    print(f"\n\nTop 10 Most Important Features (XGBoost):")
    print(f"  {'Feature':<35} {'Importance':>10}")
    print(f"  {'-'*47}")
    importances  = xgb.feature_importances_
    feat_imp     = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1], reverse=True
    )
    for feat, imp in feat_imp[:10]:
        bar = "|" * int(imp * 200)
        print(f"  {feat:<35} {imp:>8.4f}  {bar}")

    # ── SAVE WINNER ────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path  = os.path.join(MODEL_DIR, "fraud_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    with open(model_path,  "wb") as f: pickle.dump(winner["model"], f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler,          f)

    # Save winner metadata for the API to use
    metadata = {
        "model_name"  : winner["name"],
        "f1_score"    : winner["f1"],
        "roc_auc"     : winner["roc_auc"],
        "precision"   : winner["precision"],
        "recall"      : winner["recall"],
        "infer_ms"    : winner["infer_ms"],
        "feature_cols": feature_cols,
        "trained_on"  : str(pd.Timestamp.now()),
    }
    with open(os.path.join(MODEL_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n[DONE] Winner saved  -> {model_path}")
    print(f"[DONE] Scaler saved  -> {scaler_path}")
    print(f"[DONE] Metadata saved -> {MODEL_DIR}/metadata.pkl")
    print(f"\n[DONE] F1={winner['f1']:.4f} | ROC-AUC={winner['roc_auc']:.4f} | "
          f"Recall={winner['recall']:.4f} | Precision={winner['precision']:.4f}")


if __name__ == "__main__":
    train()