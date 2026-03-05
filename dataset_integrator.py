"""
Integrates the two collected datasets into our tracker's ML pipeline.

Dataset 1: PCOS diagnosis (PCOS_infertility.csv + PCOS_data_without_infertility.xlsx)
Dataset 2: Menstrual cycle length prediction (FedCycleData071012.csv)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, mean_squared_error, mean_absolute_error)
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET 1 — PCOS CLASSIFIER (from pcos-diagnosis.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

class PCOSDatasetLoader:
    """
    Loads and preprocesses the Kerala hospital PCOS dataset.
    Mirrors the exact preprocessing from pcos-diagnosis.ipynb.
    """

    # Top features most correlated with PCOS (from notebook's correlation analysis)
    TOP_FEATURES = [
        "Follicle No. (R)", "Follicle No. (L)",
        "AMH(ng/mL)", "Cycle(R/I)", "Cycle length(days)",
        "Skin darkening (Y/N)", "hair growth(Y/N)", "Weight gain(Y/N)",
        "Fast food (Y/N)", "Pimples(Y/N)",
        "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH",
        "Waist:Hip Ratio", "BMI", "Age (yrs)",
        "Endometrium (mm)", "Avg. F size (R) (mm)", "Avg. F size (L) (mm)",
    ]

    def load(self, path_csv: str, path_xlsx: str) -> pd.DataFrame:
        pcos_inf   = pd.read_csv(path_csv)
        pcos_woinf = pd.read_excel(path_xlsx, sheet_name="Full_new")

        # Merge exactly as in notebook
        data = pd.merge(pcos_woinf, pcos_inf, on="Patient File No.",
                        suffixes=["", "_y"], how="left")
        data = data.drop([
            "Unnamed: 44", "Sl. No_y", "PCOS (Y/N)_y",
            "  I   beta-HCG(mIU/mL)_y", "II    beta-HCG(mIU/mL)_y",
            "AMH(ng/mL)_y"
        ], axis=1, errors="ignore")

        # Fix string-encoded numerics
        data["AMH(ng/mL)"]             = pd.to_numeric(data["AMH(ng/mL)"], errors="coerce")
        data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(data["II    beta-HCG(mIU/mL)"], errors="coerce")
        data["Marraige Status (Yrs)"].fillna(data["Marraige Status (Yrs)"].median(), inplace=True)
        data["Fast food (Y/N)"].fillna(data["Fast food (Y/N)"].mode()[0], inplace=True)

        # Remove outliers (from notebook)
        data = data[data["BP _Diastolic (mmHg)"] > 20]
        data = data[data["AMH(ng/mL)"] < 40]
        data = data[data["BP _Systolic (mmHg)"] > 20]
        data = data[data["Endometrium (mm)"] > 0]
        data = data[data["Avg. F size (R) (mm)"] > 0]
        data = data[data["RBS(mg/dl)"] < 200]
        data = data[data["PRG(ng/mL)"] < 20]

        return data

    def get_features_target(self, data: pd.DataFrame):
        available = [f for f in self.TOP_FEATURES if f in data.columns]
        X = data[available].fillna(data[available].median())
        y = data["PCOS (Y/N)"]
        return X, y, available


class PCOSClassifierTrainer:
    """
    Trains RandomForest PCOS classifier on the Kerala dataset.
    Replaces our rule-based ConditionClassifier with a data-driven model.
    """

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                criterion="entropy",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])
        self.feature_names = []
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self.feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="accuracy")

        results = {
            "test_accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "cv_mean":        round(cv_scores.mean(), 4),
            "cv_std":         round(cv_scores.std(), 4),
            "report":         classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "n_train":        len(X_train),
            "n_test":         len(X_test),
        }
        self.is_trained = True
        return results

    def predict_proba(self, patient_data: dict) -> float:
        """
        Returns PCOS probability (0–1) for a single patient.
        patient_data keys must match self.feature_names.
        """
        row = pd.DataFrame([patient_data]).reindex(
            columns=self.feature_names, fill_value=0
        )
        return float(self.model.predict_proba(row)[0][1])

    def feature_importances(self) -> dict:
        rf = self.model.named_steps["clf"]
        return dict(sorted(
            zip(self.feature_names, rf.feature_importances_),
            key=lambda x: x[1], reverse=True
        ))

    def save(self, path: str):
        joblib.dump({"model": self.model, "features": self.feature_names}, path)

    def load(self, path: str):
        ckpt = joblib.load(path)
        self.model         = ckpt["model"]
        self.feature_names = ckpt["features"]
        self.is_trained    = True


# ══════════════════════════════════════════════════════════════════════════════
# DATASET 2 — CYCLE LENGTH PREDICTOR (from eda-menstrual-cycle-length-prediction.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

class CycleLengthDatasetLoader:
    """
    Loads FedCycleData071012.csv with exact preprocessing from EDA notebook.
    """

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        # Replace whitespace missing values (from notebook)
        df = df.replace(" ", np.nan)

        # Drop columns with >50% missing
        null_pct = df.isnull().mean()
        drop_cols = null_pct[null_pct >= 0.5].index.tolist()
        df = df.drop(columns=drop_cols + ["ClientID"], errors="ignore")

        # Encode categoricals
        cat_cols = df.select_dtypes(include=object).columns
        for col in cat_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill remaining missing values with mean
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())

        # IQR outlier replacement (from notebook)
        for col in df.columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = (df[col] < lower) | (df[col] > upper)
            df.loc[outliers, col] = df[col].mean()

        return df

    def get_features_target(self, df: pd.DataFrame):
        y = df["LengthofCycle"]
        X = df.drop("LengthofCycle", axis=1)
        return X, y


class CycleLengthXGBTrainer:
    """
    XGBoost cycle length regressor — trained on FedCycleData.
    Replaces / supplements our LSTM+GBM ensemble with a real-data-trained model.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, train_size=0.8, random_state=42
        )

        # Best params derived from notebook's GridSearchCV
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eta=0.1,
            max_depth=10,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.1,
            reg_alpha=0.01,
            reg_lambda=0.1,
            n_estimators=300,
            random_state=42,
            verbosity=0,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = self.model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae  = float(mean_absolute_error(y_test, y_pred))

        self.is_trained = True
        return {
            "rmse":    round(rmse, 3),
            "mae":     round(mae, 3),
            "n_train": len(X_train),
            "n_test":  len(X_test),
        }

    def predict(self, feature_dict: dict) -> float:
        row = pd.DataFrame([feature_dict]).reindex(
            columns=self.feature_names, fill_value=0
        )
        scaled = self.scaler.transform(row)
        return float(self.model.predict(scaled)[0])

    def feature_importances(self, top_n=15) -> dict:
        scores = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, scores),
            key=lambda x: x[1], reverse=True
        )[:top_n])

    def save(self, path: str):
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "features": self.feature_names
        }, path)

    def load(self, path: str):
        ckpt = joblib.load(path)
        self.model         = ckpt["model"]
        self.scaler        = ckpt["scaler"]
        self.feature_names = ckpt["features"]
        self.is_trained    = True


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED TRAINER — trains both models and prints full report
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(
    pcos_csv_path: str,
    pcos_xlsx_path: str,
    cycle_csv_path: str,
    save_dir: str = "models/"
):
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  TRAINING FROM YOUR COLLECTED DATASETS")
    print("="*60)

    # ── Model 1: PCOS Classifier ─────────────────────────────────────────────
    print("\n📂 Loading PCOS dataset...")
    pcos_loader  = PCOSDatasetLoader()
    pcos_data    = pcos_loader.load(pcos_csv_path, pcos_xlsx_path)
    X_pcos, y_pcos, feats = pcos_loader.get_features_target(pcos_data)
    print(f"   Samples: {len(pcos_data)} | Features used: {len(feats)}")
    print(f"   PCOS positive: {y_pcos.sum()} ({y_pcos.mean()*100:.1f}%)")

    print("\n🤖 Training PCOS Random Forest Classifier...")
    pcos_trainer = PCOSClassifierTrainer()
    pcos_results = pcos_trainer.train(X_pcos, y_pcos)

    print(f"\n   ✅ Test Accuracy : {pcos_results['test_accuracy']*100:.2f}%")
    print(f"   ✅ CV Accuracy   : {pcos_results['cv_mean']*100:.2f}% ± {pcos_results['cv_std']*100:.2f}%")
    print(f"\n   Classification Report:")
    rpt = pcos_results["report"]
    print(f"   {'':20} Precision  Recall  F1")
    for label in ["0", "1"]:
        name = "No PCOS" if label == "0" else "PCOS   "
        m = rpt[label]
        print(f"   {name:20} {m['precision']:.2f}      {m['recall']:.2f}    {m['f1-score']:.2f}")

    print(f"\n   Top 8 Predictive Features:")
    for feat, imp in list(pcos_trainer.feature_importances().items())[:8]:
        bar = "█" * int(imp * 100)
        print(f"   {feat:<35} {bar}  {imp:.4f}")

    pcos_trainer.save(f"{save_dir}/pcos_classifier.pkl")
    print(f"\n   💾 Saved → {save_dir}/pcos_classifier.pkl")

    # ── Model 2: Cycle Length XGBoost ───────────────────────────────────────
    print("\n📂 Loading Menstrual Cycle dataset...")
    cycle_loader = CycleLengthDatasetLoader()
    cycle_data   = cycle_loader.load(cycle_csv_path)
    X_cycle, y_cycle = cycle_loader.get_features_target(cycle_data)
    print(f"   Samples: {len(cycle_data)} | Features: {X_cycle.shape[1]}")
    print(f"   Cycle length range: {y_cycle.min():.0f}–{y_cycle.max():.0f} days "
          f"(mean={y_cycle.mean():.1f})")

    print("\n🤖 Training XGBoost Cycle Length Regressor...")
    cycle_trainer = CycleLengthXGBTrainer()
    cycle_results = cycle_trainer.train(X_cycle, y_cycle)

    print(f"\n   ✅ Test RMSE : {cycle_results['rmse']} days")
    print(f"   ✅ Test MAE  : {cycle_results['mae']} days")
    print(f"\n   Top 10 Predictive Features:")
    for feat, imp in cycle_trainer.feature_importances(top_n=10).items():
        bar = "█" * int(imp * 150)
        print(f"   {feat:<30} {bar}  {imp:.4f}")

    cycle_trainer.save(f"{save_dir}/cycle_length_xgb.pkl")
    print(f"\n   💾 Saved → {save_dir}/cycle_length_xgb.pkl")

    print("\n" + "="*60)
    print("  TRAINING COMPLETE — MODELS READY")
    print("="*60)

    return pcos_trainer, cycle_trainer