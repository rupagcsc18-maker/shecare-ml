"""
ML models: LSTM (PyTorch) + Random Forest ensemble.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib


# ── LSTM ──────────────────────────────────────────────────────────────────────
class CycleLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)                        # (B, T, H)
        attn_w = self.attention(out)                 # (B, T, 1)
        context = (attn_w * out).sum(dim=1)          # (B, H)
        return self.head(context).squeeze(-1)        # (B,)


class LSTMTrainer:
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, lr: float = 1e-3, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = CycleLSTM(input_size, hidden_size, num_layers).to(self.device)
        self.opt    = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, patience=5)
        self.loss_fn = nn.HuberLoss()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def _to_tensor(self, arr):
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 150, batch_size: int = 16, val_split: float = 0.15):
        # scale
        B, T, F = X.shape
        X_flat = X.reshape(-1, F)
        X_flat = self.scaler_X.fit_transform(X_flat)
        X = X_flat.reshape(B, T, F)
        y = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        split = int(len(X) * (1 - val_split))
        Xtr, Xva = X[:split], X[split:]
        ytr, yva = y[:split], y[split:]

        best_val, best_state, patience_cnt = np.inf, None, 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            idx = np.random.permutation(len(Xtr))
            train_loss = 0.0
            for start in range(0, len(Xtr), batch_size):
                batch_idx = idx[start:start+batch_size]
                xb = self._to_tensor(Xtr[batch_idx])
                yb = self._to_tensor(ytr[batch_idx])
                self.opt.zero_grad()
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                train_loss += loss.item()

            # validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(self._to_tensor(Xva)).cpu().numpy()
            val_pred = self.scaler_y.inverse_transform(val_pred.reshape(-1,1)).ravel()
            val_true = self.scaler_y.inverse_transform(yva.reshape(-1,1)).ravel()
            val_mae  = mean_absolute_error(val_true, val_pred)
            self.sched.step(val_mae)

            if val_mae < best_val:
                best_val  = val_mae
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 15:
                    print(f"Early stop at epoch {epoch}")
                    break

            if epoch % 20 == 0:
                print(f"Epoch {epoch:>4} | Train loss {train_loss/max(1,len(Xtr)//batch_size):.4f}"
                      f" | Val MAE {val_mae:.2f} days")

        if best_state:
            self.model.load_state_dict(best_state)
        print(f"Best validation MAE: {best_val:.2f} days")

    def predict(self, X: np.ndarray) -> np.ndarray:
        B, T, F = X.shape
        X_flat = self.scaler_X.transform(X.reshape(-1, F)).reshape(B, T, F)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self._to_tensor(X_flat)).cpu().numpy()
        return self.scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()

    def save(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.scaler_X = ckpt["scaler_X"]
        self.scaler_y = ckpt["scaler_y"]


# ── Gradient Boosting (tree-based ensemble) ───────────────────────────────────
class GBMCycleModel:
    """Flat feature (no sequence) gradient boosting model."""

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("gbm",    GradientBoostingRegressor(
                n_estimators=300, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                min_samples_leaf=3, random_state=42,
            )),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        train_mae = mean_absolute_error(y, self.model.predict(X))
        print(f"GBM train MAE: {train_mae:.2f} days")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importances(self, feature_names):
        importances = self.model.named_steps["gbm"].feature_importances_
        return dict(sorted(zip(feature_names, importances),
                            key=lambda x: x[1], reverse=True))

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)