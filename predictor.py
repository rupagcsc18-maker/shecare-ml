"""
High-level predictor: combines LSTM + GBM, computes cycle windows.
"""
from datetime import date, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from data_models import UserProfile, CycleDataPreprocessor
from ml_models import LSTMTrainer, GBMCycleModel


class CyclePredictor:
    """
    Ensemble predictor (LSTM + GBM) with full cycle window output.
    Falls back to personalised average when data is scarce.
    """

    MIN_CYCLES_FOR_ML = 6
    SEQUENCE_LENGTH   = 6

    def __init__(self):
        self.preprocessor = CycleDataPreprocessor(self.SEQUENCE_LENGTH)
        self.lstm: Optional[LSTMTrainer] = None
        self.gbm:  Optional[GBMCycleModel] = None
        self.is_trained = False

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, profile: UserProfile):
        df = profile.to_dataframe()
        if len(df) < self.MIN_CYCLES_FOR_ML:
            print(f"Only {len(df)} cycles — need {self.MIN_CYCLES_FOR_ML}. "
                  "Using statistical fallback.")
            return

        df_feat = self.preprocessor.build_features(df)
        X_seq, y_seq = self.preprocessor.build_sequences(df)

        # LSTM
        if len(X_seq) >= 4:
            input_size = X_seq.shape[2]
            self.lstm = LSTMTrainer(input_size=input_size)
            self.lstm.fit(X_seq, y_seq, epochs=150)

        # GBM (uses flattened last-sequence features)
        flat_feat_cols = self.preprocessor.feature_cols
        df_feat_clean  = df_feat.dropna(subset=["cycle_length"])
        X_flat = df_feat_clean[flat_feat_cols].values
        y_flat = df_feat_clean["cycle_length"].values
        if len(X_flat) >= 4:
            self.gbm = GBMCycleModel()
            self.gbm.fit(X_flat, y_flat)

        self.is_trained = True
        print("Training complete.")

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict_next_cycle_length(self, profile: UserProfile) -> Dict:
        df = profile.to_dataframe()
        df_feat = self.preprocessor.build_features(df)

        preds = []

        if self.lstm and len(df) >= self.SEQUENCE_LENGTH:
            X_seq, _ = self.preprocessor.build_sequences(df)
            if len(X_seq) > 0:
                lstm_pred = float(self.lstm.predict(X_seq[-1:][np.newaxis] if X_seq.ndim == 2
                                                    else X_seq[-1:]))
                preds.append(("LSTM", lstm_pred, 0.55))

        if self.gbm:
            last_row = df_feat[self.preprocessor.feature_cols].iloc[-1:].values
            gbm_pred = float(self.gbm.predict(last_row)[0])
            preds.append(("GBM", gbm_pred, 0.45))

        if not preds:
            # statistical fallback
            known = [getattr(c, "_cycle_length", None) for c in profile.cycles]
            known = [l for l in known if l]
            stat_pred = np.mean(known[-6:]) if known else profile.avg_cycle_length
            return {"method": "statistical", "predicted_cycle_length": round(stat_pred, 1),
                    "confidence": "low"}

        # weighted ensemble
        total_w = sum(w for _, _, w in preds)
        ensemble = sum(p * w for _, p, w in preds) / total_w
        ensemble = float(np.clip(ensemble, 21, 45))

        return {
            "method":                "ensemble (LSTM + GBM)",
            "predicted_cycle_length": round(ensemble, 1),
            "component_predictions":  {name: round(p, 1) for name, p, _ in preds},
            "confidence":             "high" if len(profile.cycles) >= 12 else "medium",
        }

    def predict_windows(self, profile: UserProfile,
                        from_date: Optional[date] = None,
                        num_cycles: int = 3) -> List[Dict]:
        """
        Returns next `num_cycles` predicted windows including:
        - period window
        - fertile window
        - ovulation day
        - luteal phase
        """
        from_date = from_date or date.today()
        result = self.predict_next_cycle_length(profile)
        cycle_len = result["predicted_cycle_length"]

        # find latest period start
        if profile.cycles:
            last_start = max(c.period_start for c in profile.cycles)
        else:
            last_start = from_date

        avg_duration = profile.avg_period_duration
        windows = []
        current_start = last_start

        for i in range(num_cycles):
            current_start = current_start + timedelta(days=round(cycle_len))
            ovulation_day = current_start + timedelta(days=round(cycle_len - 14))
            windows.append({
                "cycle_number":    i + 1,
                "period_start":    current_start,
                "period_end":      current_start + timedelta(days=round(avg_duration) - 1),
                "fertile_start":   ovulation_day - timedelta(days=5),
                "fertile_end":     ovulation_day + timedelta(days=1),
                "ovulation_day":   ovulation_day,
                "luteal_start":    ovulation_day + timedelta(days=1),
                "luteal_end":      current_start + timedelta(days=round(cycle_len) - 1),
                "predicted_cycle_length": round(cycle_len, 1),
            })

        return windows


def print_windows(windows: List[Dict]):
    for w in windows:
        print(f"\n{'='*55}")
        print(f" Cycle {w['cycle_number']}  (predicted length: {w['predicted_cycle_length']} days)")
        print(f"{'='*55}")
        print(f"  🩸 Period       : {w['period_start']}  →  {w['period_end']}")
        print(f"  🌱 Fertile      : {w['fertile_start']}  →  {w['fertile_end']}")
        print(f"  🥚 Ovulation    : {w['ovulation_day']}")
        print(f"  🌙 Luteal phase : {w['luteal_start']}  →  {w['luteal_end']}")