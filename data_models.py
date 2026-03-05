"""
Data models and preprocessing for menstruation cycle tracker.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Dict
import numpy as np
import pandas as pd


@dataclass
class CycleEntry:
    period_start: date
    period_end: date
    symptoms: Dict[str, int] = field(default_factory=dict)  # name -> severity 1-5
    notes: str = ""

    @property
    def period_duration(self) -> int:
        return (self.period_end - self.period_start).days + 1


@dataclass
class UserProfile:
    user_id: str
    age: int
    avg_cycle_length: float = 28.0
    avg_period_duration: float = 5.0
    cycles: List[CycleEntry] = field(default_factory=list)

    def add_cycle(self, entry: CycleEntry):
        self.cycles.append(entry)
        self._compute_cycle_lengths()
        self._update_averages()

    def _compute_cycle_lengths(self):
        sorted_cycles = sorted(self.cycles, key=lambda c: c.period_start)
        for i in range(len(sorted_cycles) - 1):
            cl = (sorted_cycles[i+1].period_start - sorted_cycles[i].period_start).days
            sorted_cycles[i]._cycle_length = cl
        self.cycles = sorted_cycles

    def _update_averages(self):
        lengths = [getattr(c, "_cycle_length", None) for c in self.cycles]
        lengths = [l for l in lengths if l]
        if lengths:
            self.avg_cycle_length = np.mean(lengths)
        self.avg_period_duration = np.mean([c.period_duration for c in self.cycles])

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for c in self.cycles:
            row = {
                "period_start":    c.period_start,
                "period_end":      c.period_end,
                "period_duration": c.period_duration,
                "cycle_length":    getattr(c, "_cycle_length", None),
                "day_of_year":     c.period_start.timetuple().tm_yday,
                "month":           c.period_start.month,
            }
            row.update({f"symptom_{k}": v for k, v in c.symptoms.items()})
            rows.append(row)
        return pd.DataFrame(rows)


COMMON_SYMPTOMS = [
    "cramps", "bloating", "headache", "mood_swings",
    "fatigue", "breast_tenderness", "acne", "back_pain",
    "nausea", "food_cravings", "spotting", "heavy_flow",
]


class CycleDataPreprocessor:
    SYMPTOM_COLS = [f"symptom_{s}" for s in COMMON_SYMPTOMS]

    def __init__(self, sequence_length: int = 6):
        self.sequence_length = sequence_length
        self.feature_cols: List[str] = []

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.SYMPTOM_COLS:
            if col not in df.columns:
                df[col] = 0

        df["cycle_length_roll3"] = df["cycle_length"].rolling(3, min_periods=1).mean()
        df["cycle_length_roll6"] = df["cycle_length"].rolling(6, min_periods=1).mean()
        df["cycle_length_std3"]  = df["cycle_length"].rolling(3, min_periods=1).std().fillna(0)
        df["duration_roll3"]     = df["period_duration"].rolling(3, min_periods=1).mean()
        df["symptom_total"]      = df[self.SYMPTOM_COLS].sum(axis=1)
        df["symptom_severe_cnt"] = (df[self.SYMPTOM_COLS] >= 4).sum(axis=1)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)

        self.feature_cols = [
            "period_duration", "cycle_length_roll3", "cycle_length_roll6",
            "cycle_length_std3", "duration_roll3",
            "symptom_total", "symptom_severe_cnt",
            "month_sin", "month_cos", "doy_sin", "doy_cos",
        ] + self.SYMPTOM_COLS
        return df

    def build_sequences(self, df: pd.DataFrame):
        df = self.build_features(df)
        df = df.dropna(subset=["cycle_length"])
        values  = df[self.feature_cols].values
        targets = df["cycle_length"].values
        X, y = [], []
        for i in range(self.sequence_length, len(values)):
            X.append(values[i - self.sequence_length : i])
            y.append(targets[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)