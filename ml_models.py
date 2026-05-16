"""
ml_models.py
============
Machine learning models for cycle length prediction.

Classes:
    LSTMTrainer    — PyTorch LSTM with early stopping + dropout
    GBMCycleModel  — XGBoost gradient boosted trees

Both models predict the next cycle length (in days) from engineered features.
"""

from typing import Optional, Tuple
import numpy as np

# ── Optional imports (graceful degradation) ───────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not found — LSTMTrainer will be unavailable. pip install torch")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not found — GBMCycleModel will be unavailable. pip install xgboost")


# ══════════════════════════════════════════════════════════════════════════════
# LSTM MODEL (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:

    class _CycleLSTM(nn.Module):
        """Single-layer LSTM with dropout for cycle length regression."""

        def __init__(self, input_size: int, hidden_size: int = 64,
                     num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size  = input_size,
                hidden_size = hidden_size,
                num_layers  = num_layers,
                dropout     = dropout if num_layers > 1 else 0.0,
                batch_first = True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq_len, features)
            out, _ = self.lstm(x)
            out    = self.dropout(out[:, -1, :])   # take last time step
            return self.fc(out).squeeze(-1)


class LSTMTrainer:
    """
    Wraps _CycleLSTM with training loop, early stopping, and predict().

    Input to fit():   X_seq shape (n, seq_len, features),  y shape (n,)
    Input to predict(): X_seq shape (1, seq_len, features)  → float
    """

    def __init__(self,
                 input_size:   int   = 8,
                 hidden_size:  int   = 64,
                 num_layers:   int   = 2,
                 dropout:      float = 0.2,
                 lr:           float = 1e-3,
                 patience:     int   = 15):

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMTrainer. pip install torch")

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.lr          = lr
        self.patience    = patience
        self.model: Optional["_CycleLSTM"] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self) -> "_CycleLSTM":
        return _CycleLSTM(
            input_size  = self.input_size,
            hidden_size = self.hidden_size,
            num_layers  = self.num_layers,
            dropout     = self.dropout,
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 150, batch_size: int = 16,
            val_split: float = 0.15) -> "LSTMTrainer":
        """
        Train the LSTM with early stopping.

        Args:
            X:          shape (n, seq_len, features)
            y:          shape (n,) — cycle lengths
            epochs:     max training epochs
            batch_size: mini-batch size
            val_split:  fraction of data held out for early stopping
        """
        n_val   = max(1, int(len(X) * val_split))
        X_tr, X_val = X[:-n_val], X[-n_val:]
        y_tr, y_val = y[:-n_val], y[-n_val:]

        X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32).to(self.device)
        y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        loader   = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True)

        self.model = self._build_model()
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion  = nn.MSELoss()

        best_val_loss  = float("inf")
        best_state     = None
        no_improve     = 0

        self.model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = float(criterion(val_pred, y_val_t).item())
            self.model.train()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        self.model.eval()
        return self

    def predict(self, X: np.ndarray) -> float:
        """
        Predict cycle length for one or more sequences.

        Args:
            X: shape (1, seq_len, features) — already batch-first
               (X_seq[-1:] from build_sequences — no np.newaxis needed)

        Returns:
            float — predicted cycle length in days
        """
        if self.model is None:
            raise RuntimeError("LSTMTrainer.fit() must be called before predict()")

        t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred = self.model(t)
        val = float(pred.cpu().numpy()[0])
        # Clip to physiological range
        return float(np.clip(val, 21, 60))

    def save(self, path: str):
        """Save model weights to disk."""
        if self.model is None:
            raise RuntimeError("No model to save — call fit() first.")
        torch.save({
            "state_dict":  self.model.state_dict(),
            "input_size":  self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers":  self.num_layers,
            "dropout":     self.dropout,
        }, path)

    def load(self, path: str) -> "LSTMTrainer":
        """Load model weights from disk."""
        ckpt = torch.load(path, map_location=self.device)
        self.input_size  = ckpt["input_size"]
        self.hidden_size = ckpt["hidden_size"]
        self.num_layers  = ckpt["num_layers"]
        self.dropout     = ckpt["dropout"]
        self.model       = self._build_model()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self


else:
    # Stub when PyTorch is not installed
    class LSTMTrainer:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LSTMTrainer. pip install torch")


# ══════════════════════════════════════════════════════════════════════════════
# GRADIENT BOOSTED MODEL (XGBoost)
# ══════════════════════════════════════════════════════════════════════════════

if XGB_AVAILABLE:

    class GBMCycleModel:
        """
        XGBoost regressor for cycle length prediction from flat features.

        Input:  2D array (n_samples, n_features) — from CycleDataPreprocessor
        Output: 1D array (n_samples,) — predicted cycle lengths
        """

        DEFAULT_PARAMS = {
            "n_estimators":      300,
            "max_depth":         4,
            "learning_rate":     0.05,
            "subsample":         0.8,
            "colsample_bytree":  0.8,
            "min_child_weight":  3,
            "reg_alpha":         0.1,   # L1
            "reg_lambda":        1.0,   # L2
            "objective":         "reg:squarederror",
            "eval_metric":       "rmse",
            "random_state":      42,
            "n_jobs":            -1,
        }

        def __init__(self, params: Optional[dict] = None):
            merged = {**self.DEFAULT_PARAMS, **(params or {})}
            self.model = xgb.XGBRegressor(**merged)
            self._is_trained = False

        def fit(self, X: np.ndarray, y: np.ndarray,
                eval_set: Optional[list] = None,
                early_stopping_rounds: int = 20) -> "GBMCycleModel":
            """
            Train the XGBoost model.

            Args:
                X:                     shape (n, features)
                y:                     shape (n,) — cycle lengths
                eval_set:              optional [(X_val, y_val)] for early stopping
                early_stopping_rounds: patience for early stopping
            """
            fit_kwargs: dict = {}
            if eval_set:
                fit_kwargs["eval_set"]              = eval_set
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
                fit_kwargs["verbose"]               = False

            self.model.fit(X, y, **fit_kwargs)
            self._is_trained = True
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            """
            Predict cycle lengths.

            Args:
                X: shape (n, features)

            Returns:
                np.ndarray shape (n,) — clipped to [21, 60]
            """
            if not self._is_trained:
                raise RuntimeError("GBMCycleModel.fit() must be called before predict()")
            preds = self.model.predict(X)
            return np.clip(preds, 21, 60).astype(np.float32)

        def feature_importance(self) -> dict:
            """Return feature importances as {feature_index: importance}."""
            if not self._is_trained:
                raise RuntimeError("Model not trained yet.")
            imp = self.model.feature_importances_
            return {i: float(v) for i, v in enumerate(imp)}

        def save(self, path: str):
            """Save model to disk via joblib."""
            import joblib
            joblib.dump({"model": self.model, "is_trained": self._is_trained}, path)

        def load(self, path: str) -> "GBMCycleModel":
            """Load model from disk."""
            import joblib
            ckpt = joblib.load(path)
            self.model       = ckpt["model"]
            self._is_trained = ckpt["is_trained"]
            return self

else:
    # Stub when XGBoost is not installed
    class GBMCycleModel:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("XGBoost is required for GBMCycleModel. pip install xgboost")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from datetime import date
    from data_models import UserProfile, CycleRecord, CycleDataPreprocessor

    # Build a small synthetic dataset
    starts = [date(2025, 1, 1) + __import__("datetime").timedelta(days=i * 28)
              for i in range(12)]
    cycles = [CycleRecord(s, s + __import__("datetime").timedelta(days=4)) for s in starts]
    profile = UserProfile(cycles=cycles)
    df      = profile.to_dataframe()
    prep    = CycleDataPreprocessor(sequence_length=4)
    X, y    = prep.build_sequences(df)
    print(f"Sequences: X={X.shape} y={y.shape}")

    if TORCH_AVAILABLE and len(X) >= 4:
        trainer = LSTMTrainer(input_size=X.shape[2])
        trainer.fit(X, y, epochs=30)
        pred = trainer.predict(X[-1:])
        print(f"LSTM prediction: {pred:.1f} days")
    else:
        print("Skipping LSTM demo (PyTorch not available or insufficient data)")

    if XGB_AVAILABLE:
        df_feat = prep.build_features(df)
        X_flat  = df_feat[prep.feature_cols].dropna().values
        y_flat  = df_feat["cycle_length"].dropna().values
        if len(X_flat) >= 4:
            gbm = GBMCycleModel()
            gbm.fit(X_flat, y_flat)
            pred_gbm = gbm.predict(X_flat[-1:])
            print(f"GBM prediction: {pred_gbm[0]:.1f} days")
    else:
        print("Skipping GBM demo (XGBoost not available)")