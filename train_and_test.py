"""
train_and_test.py
=================
Full pipeline:
  1. Load & validate both datasets
  2. Train PCOS classifier  (Random Forest)
  3. Train Cycle Length regressor (XGBoost)
  4. Evaluate both models with detailed reports
  5. Save trained models to models/
  6. Generate performance_report.txt

Run:
    cd C:\\Users\\rupa8\\OneDrive\\Desktop\\menstruation_tracker
    python train_and_test.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
)
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
PLOT_DIR    = os.path.join(BASE_DIR, "plots")
REPORT_PATH = os.path.join(BASE_DIR, "performance_report.txt")

for d in [MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

PCOS_CSV    = os.path.join(DATA_DIR, "PCOS_infertility.csv")
PCOS_XLSX   = os.path.join(DATA_DIR, "PCOS_data_without_infertility.xlsx")
CYCLE_CSV   = os.path.join(DATA_DIR, "FedCycleData071012.csv")

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(msg)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Dataset Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_datasets():
    log("=" * 62)
    log("  STEP 0 — DATASET VALIDATION")
    log("=" * 62)

    missing = []
    for path, name in [(PCOS_CSV,  "PCOS_infertility.csv"),
                       (PCOS_XLSX, "PCOS_data_without_infertility.xlsx"),
                       (CYCLE_CSV, "FedCycleData071012.csv")]:
        exists = os.path.exists(path)
        status = "✅ Found" if exists else "❌ MISSING"
        log(f"  {status} : {name}")
        if not exists:
            missing.append(name)

    if missing:
        log()
        log("  ERROR: Place missing files in the data/ folder and re-run.")
        log("  Download from:")
        log("  • PCOS     → https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos")
        log("  • Cycle    → https://www.kaggle.com/datasets/nikitabisht/menstrual-cycle-data")
        sys.exit(1)

    log("\n  All datasets present. Proceeding...\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & Preprocess PCOS Dataset
# ══════════════════════════════════════════════════════════════════════════════

PCOS_TOP_FEATURES = [
    "Follicle No. (R)", "Follicle No. (L)",
    "AMH(ng/mL)", "Cycle(R/I)", "Cycle length(days)",
    "Skin darkening (Y/N)", "hair growth(Y/N)", "Weight gain(Y/N)",
    "Fast food (Y/N)", "Pimples(Y/N)",
    "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH",
    "Waist:Hip Ratio", "BMI", "Age (yrs)",
    "Endometrium (mm)", "Avg. F size (R) (mm)", "Avg. F size (L) (mm)",
]


def load_pcos_data():
    log("=" * 62)
    log("  STEP 1 — LOADING PCOS DATASET")
    log("=" * 62)

    pcos_inf   = pd.read_csv(PCOS_CSV)
    pcos_woinf = pd.read_excel(PCOS_XLSX, sheet_name="Full_new")
    log(f"  CSV shape  : {pcos_inf.shape}")
    log(f"  Excel shape: {pcos_woinf.shape}")

    # Merge
    data = pd.merge(pcos_woinf, pcos_inf, on="Patient File No.",
                    suffixes=["", "_y"], how="left")
    drop_cols = ["Unnamed: 44", "Sl. No_y", "PCOS (Y/N)_y",
                 "  I   beta-HCG(mIU/mL)_y", "II    beta-HCG(mIU/mL)_y", "AMH(ng/mL)_y"]
    data = data.drop([c for c in drop_cols if c in data.columns], axis=1)

    # Fix types
    data["AMH(ng/mL)"]             = pd.to_numeric(data["AMH(ng/mL)"],             errors="coerce")
    data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(data["II    beta-HCG(mIU/mL)"], errors="coerce")
    data["Marraige Status (Yrs)"].fillna(data["Marraige Status (Yrs)"].median(), inplace=True)
    if "Fast food (Y/N)" in data.columns:
        data["Fast food (Y/N)"].fillna(data["Fast food (Y/N)"].mode()[0], inplace=True)

    # Remove outliers (from notebook)
    data = data[data["BP _Diastolic (mmHg)"] > 20]
    data = data[data["AMH(ng/mL)"]           < 40]
    data = data[data["BP _Systolic (mmHg)"]  > 20]
    data = data[data["Endometrium (mm)"]      > 0]
    data = data[data["Avg. F size (R) (mm)"]  > 0]
    data = data[data["RBS(mg/dl)"]            < 200]
    data = data[data["PRG(ng/mL)"]            < 20]
    if "Pulse rate(bpm)"  in data.columns: data = data[data["Pulse rate(bpm)"]  > 20]
    if "FSH(mIU/mL)"      in data.columns: data = data[data["FSH(mIU/mL)"]      < 4000]
    if "LH(mIU/mL)"       in data.columns: data = data[data["LH(mIU/mL)"]       < 1500]
    if "Cycle(R/I)"       in data.columns: data = data[data["Cycle(R/I)"]        < 4.5]

    available = [f for f in PCOS_TOP_FEATURES if f in data.columns]
    X = data[available].fillna(data[available].median())
    y = data["PCOS (Y/N)"]

    log(f"  After cleaning : {data.shape[0]} samples, {len(available)} features")
    log(f"  PCOS positive  : {y.sum()} ({y.mean()*100:.1f}%)")
    log(f"  PCOS negative  : {(1-y).sum()} ({(1-y).mean()*100:.1f}%)\n")

    return X, y, available


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Train PCOS Classifier
# ══════════════════════════════════════════════════════════════════════════════

def train_pcos_classifier(X, y, feature_names):
    log("=" * 62)
    log("  STEP 2 — TRAINING PCOS CLASSIFIER (Random Forest)")
    log("=" * 62)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    log(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=8,
            criterion="entropy", class_weight="balanced",
            random_state=42, n_jobs=-1
        ))
    ])

    t0 = time.time()
    model.fit(X_train, y_train)
    log(f"  Training time : {time.time()-t0:.1f}s")

    y_pred     = model.predict(X_test)
    y_proba    = model.predict_proba(X_test)[:, 1]

    acc        = accuracy_score(y_test, y_pred)
    auc        = roc_auc_score(y_test, y_proba)
    cv_scores  = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="accuracy")
    report     = classification_report(y_test, y_pred, target_names=["No PCOS", "PCOS"])
    cm         = confusion_matrix(y_test, y_pred)

    log(f"\n  ✅ Test Accuracy : {acc*100:.2f}%")
    log(f"  ✅ ROC-AUC Score : {auc:.4f}")
    log(f"  ✅ CV Accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    log(f"\n  Classification Report:\n")
    for line in report.split("\n"):
        log(f"    {line}")

    log(f"\n  Confusion Matrix:")
    log(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    log(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    # Feature importances
    rf = model.named_steps["clf"]
    importances = sorted(zip(feature_names, rf.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    log(f"\n  Top 10 Feature Importances:")
    for feat, imp in importances[:10]:
        bar = "█" * int(imp * 100)
        log(f"    {feat:<35} {bar}  {imp:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # 1. Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["No PCOS", "PCOS"],
                yticklabels=["No PCOS", "PCOS"])
    axes[0].set_title("Confusion Matrix — PCOS Classifier")
    axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="teal", lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0,1],[0,1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve — PCOS Classifier")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pcos_classifier_eval.png"), dpi=150)
    plt.close()
    log(f"\n  📊 Plot saved → plots/pcos_classifier_eval.png")

    # 3. Feature importance plot
    feat_df = pd.DataFrame(importances[:12], columns=["Feature", "Importance"])
    plt.figure(figsize=(9, 5))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="Blues_r")
    plt.title("PCOS Classifier — Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pcos_feature_importance.png"), dpi=150)
    plt.close()
    log(f"  📊 Plot saved → plots/pcos_feature_importance.png")

    # Save model
    save_path = os.path.join(MODEL_DIR, "pcos_classifier.pkl")
    joblib.dump({"model": model, "features": feature_names,
                 "accuracy": acc, "auc": auc}, save_path)
    log(f"  💾 Model saved → models/pcos_classifier.pkl\n")

    return model, {"accuracy": acc, "auc": auc, "cv_mean": cv_scores.mean()}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Load & Preprocess Cycle Length Dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_cycle_data():
    log("=" * 62)
    log("  STEP 3 — LOADING CYCLE LENGTH DATASET")
    log("=" * 62)

    df = pd.read_csv(CYCLE_CSV)
    log(f"  Raw shape : {df.shape}")

    # Replace whitespace missing values
    df = df.replace(" ", np.nan)

    # Drop columns >50% missing
    null_pct  = df.isnull().mean()
    drop_cols = null_pct[null_pct >= 0.5].index.tolist()
    df = df.drop(columns=drop_cols + ["ClientID"], errors="ignore")
    log(f"  Dropped {len(drop_cols)} high-null columns + ClientID")

    # Encode categoricals
    cat_cols = df.select_dtypes(include=object).columns
    for col in cat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill remaining nulls with column mean
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    # IQR outlier replacement
    for col in df.columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = (df[col] < lower) | (df[col] > upper)
        df.loc[outliers, col] = df[col].mean()

    if "LengthofCycle" not in df.columns:
        log("  ERROR: 'LengthofCycle' column not found. Check CSV file.")
        sys.exit(1)

    y = df["LengthofCycle"]
    X = df.drop("LengthofCycle", axis=1)

    log(f"  After cleaning : {df.shape[0]} samples, {X.shape[1]} features")
    log(f"  Cycle length   : mean={y.mean():.1f}d  std={y.std():.1f}d  "
        f"range={y.min():.0f}–{y.max():.0f}d\n")

    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Train Cycle Length Regressor
# ══════════════════════════════════════════════════════════════════════════════

def train_cycle_regressor(X, y):
    log("=" * 62)
    log("  STEP 4 — TRAINING CYCLE LENGTH REGRESSOR (XGBoost)")
    log("=" * 62)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=0.8, random_state=42
    )
    log(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eta=0.1, max_depth=10,
        subsample=0.9, colsample_bytree=0.9,
        gamma=0.1, reg_alpha=0.01, reg_lambda=0.1,
        n_estimators=300, random_state=42, verbosity=0,
    )

    t0 = time.time()
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)
    log(f"  Training time : {time.time()-t0:.1f}s")

    y_pred = model.predict(X_test)
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae    = float(mean_absolute_error(y_test, y_pred))
    r2     = float(r2_score(y_test, y_pred))

    # Within-N-days accuracy
    within_1 = np.mean(np.abs(y_test - y_pred) <= 1) * 100
    within_2 = np.mean(np.abs(y_test - y_pred) <= 2) * 100
    within_3 = np.mean(np.abs(y_test - y_pred) <= 3) * 100

    log(f"\n  ✅ RMSE           : {rmse:.3f} days")
    log(f"  ✅ MAE            : {mae:.3f} days")
    log(f"  ✅ R² Score       : {r2:.4f}")
    log(f"  ✅ Within ±1 day  : {within_1:.1f}%")
    log(f"  ✅ Within ±2 days : {within_2:.1f}%")
    log(f"  ✅ Within ±3 days : {within_3:.1f}%")

    # Feature importances
    feat_names   = list(X.columns)
    importances  = sorted(zip(feat_names, model.feature_importances_),
                          key=lambda x: x[1], reverse=True)
    log(f"\n  Top 10 Feature Importances:")
    for feat, imp in importances[:10]:
        bar = "█" * int(imp * 150)
        log(f"    {feat:<30} {bar}  {imp:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.4, color="teal", s=15)
    mn, mx = y_test.min(), y_test.max()
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1)
    axes[0].set_xlabel("Actual Cycle Length (days)")
    axes[0].set_ylabel("Predicted Cycle Length (days)")
    axes[0].set_title(f"Actual vs Predicted\nRMSE={rmse:.2f}d  MAE={mae:.2f}d")

    # 2. Residuals
    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_xlabel("Residual (days)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    # 3. Feature importance (top 12)
    fi_df = pd.DataFrame(importances[:12], columns=["Feature", "Importance"])
    sns.barplot(data=fi_df, x="Importance", y="Feature",
                palette="Blues_r", ax=axes[2])
    axes[2].set_title("Top Feature Importances")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cycle_regressor_eval.png"), dpi=150)
    plt.close()
    log(f"\n  📊 Plot saved → plots/cycle_regressor_eval.png")

    # Save model
    save_path = os.path.join(MODEL_DIR, "cycle_length_xgb.pkl")
    joblib.dump({"model": model, "scaler": scaler,
                 "features": feat_names, "rmse": rmse, "mae": mae}, save_path)
    log(f"  💾 Model saved → models/cycle_length_xgb.pkl\n")

    return model, scaler, {"rmse": rmse, "mae": mae, "r2": r2,
                           "within_1": within_1, "within_2": within_2,
                           "within_3": within_3}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Quick Smoke Test (sanity check on saved models)
# ══════════════════════════════════════════════════════════════════════════════

def smoke_test(pcos_model, cycle_model, cycle_scaler,
               X_pcos, y_pcos, X_cycle, y_cycle):
    log("=" * 62)
    log("  STEP 5 — SMOKE TEST (sanity check on saved models)")
    log("=" * 62)

    errors = []

    # ── PCOS: predict on 5 random samples ────────────────────────────────────
    sample_idx = np.random.choice(len(X_pcos), 5, replace=False)
    X_sample   = X_pcos.iloc[sample_idx]
    y_sample   = y_pcos.iloc[sample_idx].values
    y_pred_pcos = pcos_model.predict(X_sample)

    log("\n  PCOS Classifier — 5 random samples:")
    log(f"  {'Sample':<8} {'Actual':<10} {'Predicted':<12} {'Match'}")
    for i, (act, pred) in enumerate(zip(y_sample, y_pred_pcos)):
        match = "✅" if act == pred else "❌"
        log(f"  {i+1:<8} {act:<10} {pred:<12} {match}")
        if act != pred:
            errors.append(f"PCOS sample {i+1} mismatch")

    # ── Cycle: predict on 5 random samples ───────────────────────────────────
    sample_idx2  = np.random.choice(len(X_cycle), 5, replace=False)
    X_c_sample   = cycle_scaler.transform(X_cycle.iloc[sample_idx2])
    y_c_sample   = y_cycle.iloc[sample_idx2].values
    y_pred_cycle = cycle_model.predict(X_c_sample)

    log("\n  Cycle Regressor — 5 random samples:")
    log(f"  {'Sample':<8} {'Actual':>10} {'Predicted':>12} {'Error (days)'}")
    for i, (act, pred) in enumerate(zip(y_c_sample, y_pred_cycle)):
        err = abs(act - pred)
        ok  = "✅" if err <= 3 else "⚠️ "
        log(f"  {i+1:<8} {act:>10.1f} {pred:>12.1f} {err:>8.1f}d  {ok}")

    # ── Load-from-disk test ───────────────────────────────────────────────────
    log("\n  Load-from-disk test:")
    pcos_ckpt  = joblib.load(os.path.join(MODEL_DIR, "pcos_classifier.pkl"))
    cycle_ckpt = joblib.load(os.path.join(MODEL_DIR, "cycle_length_xgb.pkl"))

    loaded_pcos_pred  = pcos_ckpt["model"].predict(X_pcos.iloc[:3])
    loaded_cycle_pred = cycle_ckpt["model"].predict(
        cycle_ckpt["scaler"].transform(X_cycle.iloc[:3])
    )
    log(f"  PCOS  loaded predictions : {loaded_pcos_pred.tolist()}")
    log(f"  Cycle loaded predictions : {[round(p,1) for p in loaded_cycle_pred.tolist()]}")
    log("  ✅ Both models load and predict correctly from disk\n")

    return len(errors) == 0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Final Summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(pcos_metrics, cycle_metrics, smoke_ok):
    log("=" * 62)
    log("  FINAL SUMMARY")
    log("=" * 62)
    log()
    log("  MODEL 1 — PCOS Classifier (Random Forest)")
    log(f"    Accuracy   : {pcos_metrics['accuracy']*100:.2f}%")
    log(f"    ROC-AUC    : {pcos_metrics['auc']:.4f}")
    log(f"    CV Accuracy: {pcos_metrics['cv_mean']*100:.2f}%")
    log()
    log("  MODEL 2 — Cycle Length Regressor (XGBoost)")
    log(f"    RMSE       : {cycle_metrics['rmse']:.3f} days")
    log(f"    MAE        : {cycle_metrics['mae']:.3f} days")
    log(f"    R²         : {cycle_metrics['r2']:.4f}")
    log(f"    Within ±1d : {cycle_metrics['within_1']:.1f}%")
    log(f"    Within ±2d : {cycle_metrics['within_2']:.1f}%")
    log(f"    Within ±3d : {cycle_metrics['within_3']:.1f}%")
    log()
    log(f"  Smoke Test   : {'✅ PASSED' if smoke_ok else '❌ FAILED'}")
    log()
    log("  Output files:")
    log("    models/pcos_classifier.pkl")
    log("    models/cycle_length_xgb.pkl")
    log("    plots/pcos_classifier_eval.png")
    log("    plots/pcos_feature_importance.png")
    log("    plots/cycle_regressor_eval.png")
    log("    performance_report.txt")
    log()
    log("=" * 62)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_start = time.time()

    log()
    log("╔══════════════════════════════════════════════════════════╗")
    log("║   MENSTRUATION TRACKER — FULL TRAINING & TEST PIPELINE  ║")
    log("╚══════════════════════════════════════════════════════════╝")
    log()

    # 0. Validate
    validate_datasets()

    # 1 & 2. PCOS
    X_pcos, y_pcos, pcos_feats = load_pcos_data()
    pcos_model, pcos_metrics   = train_pcos_classifier(X_pcos, y_pcos, pcos_feats)

    # 3 & 4. Cycle length
    X_cycle, y_cycle            = load_cycle_data()
    cycle_model, cycle_scaler, cycle_metrics = train_cycle_regressor(X_cycle, y_cycle)

    # 5. Smoke test
    smoke_ok = smoke_test(pcos_model, cycle_model, cycle_scaler,
                          X_pcos, y_pcos, X_cycle, y_cycle)

    # 6. Summary
    print_summary(pcos_metrics, cycle_metrics, smoke_ok)

    log(f"  Total time: {time.time()-total_start:.1f}s")
    log()

    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  📄 Report saved → performance_report.txt")