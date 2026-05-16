"""
data_models.py
==============
Data structures for the CyclePredictor ensemble model.
Contains UserProfile, CycleRecord, and CycleDataPreprocessor.
"""

from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# CYCLE RECORD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CycleRecord:
    """A single historical cycle entry."""
    period_start:    date
    period_end:      date
    _cycle_length:   Optional[float] = field(default=None, repr=False)

    @property
    def period_duration(self) -> int:
        return (self.period_end - self.period_start).days + 1

    def __post_init__(self):
        if self.period_end < self.period_start:
            raise ValueError(
                f"period_end ({self.period_end}) must not be before period_start ({self.period_start})"
            )
        if self.period_duration > 15:
            raise ValueError(
                f"Period duration {self.period_duration} days > 15. Check your dates."
            )


# ══════════════════════════════════════════════════════════════════════════════
# USER PROFILE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    """
    Holds a user's complete cycle history and derived statistics.

    Usage:
        profile = UserProfile(cycles=[
            CycleRecord(date(2025,1,1), date(2025,1,5)),
            CycleRecord(date(2025,1,29), date(2025,2,3)),
            ...
        ])
        # cycle_lengths are auto-computed from consecutive period starts
    """
    cycles:               List[CycleRecord] = field(default_factory=list)
    default_cycle_length: float             = 28.0
    default_period_days:  float             = 5.0

    def __post_init__(self):
        if self.cycles:
            self.cycles = sorted(self.cycles, key=lambda c: c.period_start)
            self._compute_cycle_lengths()

    # ── Auto-compute cycle lengths from consecutive period starts ─────────────
    def _compute_cycle_lengths(self):
        for i in range(len(self.cycles) - 1):
            cl = (self.cycles[i + 1].period_start - self.cycles[i].period_start).days
            self.cycles[i]._cycle_length = float(cl)
        # Last cycle: no next period yet — leave as None

    # ── Derived statistics ────────────────────────────────────────────────────
    @property
    def avg_cycle_length(self) -> float:
        lengths = [c._cycle_length for c in self.cycles if c._cycle_length is not None]
        return float(np.mean(lengths)) if lengths else self.default_cycle_length

    @property
    def avg_period_duration(self) -> float:
        durations = [c.period_duration for c in self.cycles]
        return float(np.mean(durations)) if durations else self.default_period_days

    @property
    def std_cycle_length(self) -> float:
        lengths = [c._cycle_length for c in self.cycles if c._cycle_length is not None]
        return float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 2.0

    # ── Convert to DataFrame for ML pipeline ─────────────────────────────────
    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per cycle that has a known length.
        Columns: cycle_length, period_duration, period_start (date), cycle_index
        """
        rows = []
        for i, c in enumerate(self.cycles):
            if c._cycle_length is not None:
                rows.append({
                    "cycle_index":     i,
                    "cycle_length":    c._cycle_length,
                    "period_duration": float(c.period_duration),
                    "period_start":    c.period_start,
                })
        return pd.DataFrame(rows)

    def __repr__(self):
        return (
            f"UserProfile(cycles={len(self.cycles)}, "
            f"avg_cycle={self.avg_cycle_length:.1f}d, "
            f"avg_period={self.avg_period_duration:.1f}d)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CYCLE DATA PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class CycleDataPreprocessor:
    """
    Transforms a UserProfile DataFrame into feature arrays for LSTM and GBM.

    Features engineered:
        - cycle_length (target)
        - period_duration
        - rolling_mean_3  (3-cycle rolling mean of cycle lengths)
        - rolling_std_3   (3-cycle rolling std)
        - rolling_mean_6
        - cycle_index     (trend proxy)
        - prev_cl_1       (lag-1 cycle length)
        - prev_cl_2       (lag-2 cycle length)
        - prev_cl_3       (lag-3 cycle length)
    """

    FEATURE_COLS = [
        "period_duration",
        "rolling_mean_3",
        "rolling_std_3",
        "rolling_mean_6",
        "cycle_index",
        "prev_cl_1",
        "prev_cl_2",
        "prev_cl_3",
    ]

    def __init__(self, sequence_length: int = 6):
        self.sequence_length = sequence_length
        self.feature_cols    = self.FEATURE_COLS

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the cycle DataFrame."""
        if df.empty:
            return df

        df = df.copy().reset_index(drop=True)
        cl = df["cycle_length"]

        df["rolling_mean_3"] = cl.rolling(3, min_periods=1).mean()
        df["rolling_std_3"]  = cl.rolling(3, min_periods=2).std().fillna(2.0)
        df["rolling_mean_6"] = cl.rolling(6, min_periods=1).mean()
        df["prev_cl_1"]      = cl.shift(1).fillna(cl.mean())
        df["prev_cl_2"]      = cl.shift(2).fillna(cl.mean())
        df["prev_cl_3"]      = cl.shift(3).fillna(cl.mean())

        return df

    def build_sequences(self, df: pd.DataFrame,
                        sequence_length: Optional[int] = None) -> tuple:
        """
        Build (X_seq, y_seq) for LSTM training.

        X_seq: shape (n_samples, sequence_length, n_features)
        y_seq: shape (n_samples,)
        """
        seq_len = sequence_length or self.sequence_length
        df_feat = self.build_features(df)

        # Only use rows where all features are non-null
        df_clean = df_feat.dropna(subset=["cycle_length"] + self.feature_cols)

        if len(df_clean) <= seq_len:
            return np.empty((0, seq_len, len(self.feature_cols))), np.empty(0)

        X_all = df_clean[self.feature_cols].values
        y_all = df_clean["cycle_length"].values

        X_seqs, y_seqs = [], []
        for i in range(len(X_all) - seq_len):
            X_seqs.append(X_all[i: i + seq_len])
            y_seqs.append(y_all[i + seq_len])

        return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def profile_from_period_dates(period_dates: List[tuple]) -> UserProfile:
    """
    Convenience factory.

    Args:
        period_dates: list of (start_date, end_date) tuples

    Returns:
        UserProfile with cycle lengths auto-computed

    Example:
        from datetime import date
        profile = profile_from_period_dates([
            (date(2025, 1, 1),  date(2025, 1, 5)),
            (date(2025, 1, 29), date(2025, 2, 3)),
            (date(2025, 2, 27), date(2025, 3, 3)),
        ])
    """
    cycles = [CycleRecord(period_start=s, period_end=e) for s, e in period_dates]
    return UserProfile(cycles=cycles)


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from datetime import date

    print("=" * 50)
    print("UserProfile demo")
    print("=" * 50)

    profile = profile_from_period_dates([
        (date(2025, 1,  1), date(2025, 1,  5)),
        (date(2025, 1, 29), date(2025, 2,  3)),
        (date(2025, 2, 27), date(2025, 3,  3)),
        (date(2025, 3, 27), date(2025, 4,  1)),
        (date(2025, 4, 24), date(2025, 4, 29)),
        (date(2025, 5, 22), date(2025, 5, 27)),
    ])

    print(profile)
    print(f"avg_cycle_length : {profile.avg_cycle_length:.1f} days")
    print(f"std_cycle_length : {profile.std_cycle_length:.1f} days")
    print(f"avg_period_dur   : {profile.avg_period_duration:.1f} days")

    df = profile.to_dataframe()
    print("\nDataFrame:")
    print(df.to_string(index=False))

    prep = CycleDataPreprocessor(sequence_length=4)
    df_feat = prep.build_features(df)
    print("\nFeatures:")
    print(df_feat[["cycle_length"] + prep.feature_cols].to_string(index=False))

    X, y = prep.build_sequences(df)
    print(f"\nSequences — X: {X.shape}  y: {y.shape}")