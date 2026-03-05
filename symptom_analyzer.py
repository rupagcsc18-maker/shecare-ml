"""
Symptom correlation analysis and phase-based pattern detection.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data_models import UserProfile, COMMON_SYMPTOMS


class SymptomAnalyzer:

    SYMPTOM_COLS = [f"symptom_{s}" for s in COMMON_SYMPTOMS]

    def __init__(self, profile: UserProfile):
        self.profile = profile
        self.df = profile.to_dataframe()
        for col in self.SYMPTOM_COLS:
            if col not in self.df.columns:
                self.df[col] = 0

    # ── Correlations ──────────────────────────────────────────────────────────
    def symptom_cycle_length_correlation(self) -> Dict[str, Tuple[float, float]]:
        """
        Pearson correlation of each symptom severity with cycle length.
        Returns {symptom: (r, p_value)}.
        """
        df = self.df.dropna(subset=["cycle_length"])
        results = {}
        for col in self.SYMPTOM_COLS:
            if df[col].std() == 0:
                continue
            r, p = stats.pearsonr(df[col], df["cycle_length"])
            results[col.replace("symptom_", "")] = (round(r, 3), round(p, 4))
        return dict(sorted(results.items(), key=lambda x: abs(x[1][0]), reverse=True))

    def symptom_co_occurrence(self) -> pd.DataFrame:
        """Symptom co-occurrence matrix (normalised)."""
        binary = (self.df[self.SYMPTOM_COLS] > 0).astype(int)
        co = binary.T.dot(binary)
        diag = np.diag(co.values)
        norm = co / (diag[:, None] + 1e-9)
        norm.index   = [c.replace("symptom_", "") for c in norm.index]
        norm.columns = [c.replace("symptom_", "") for c in norm.columns]
        return norm.round(3)

    # ── Cycle clustering ──────────────────────────────────────────────────────
    def cluster_cycles(self, n_clusters: int = 3) -> pd.DataFrame:
        """
        K-means clustering of cycles using symptom + duration features.
        Returns original df with a 'cluster' label column.
        """
        feat_cols = self.SYMPTOM_COLS + ["period_duration"]
        X = self.df[feat_cols].fillna(0).values
        X_scaled = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df["cluster"] = km.fit_predict(X_scaled)
        return self.df[["period_start", "cycle_length", "period_duration",
                         "cluster"] + self.SYMPTOM_COLS]

    # ── Reports ───────────────────────────────────────────────────────────────
    def top_symptoms_by_severity(self, top_n: int = 5) -> List[Tuple[str, float]]:
        means = self.df[self.SYMPTOM_COLS].mean().sort_values(ascending=False)
        return [(col.replace("symptom_", ""), round(v, 2))
                for col, v in means.head(top_n).items()]

    def cycle_regularity_score(self) -> Dict:
        lengths = self.df["cycle_length"].dropna().values
        if len(lengths) < 3:
            return {"score": None, "interpretation": "Insufficient data"}
        cv  = (np.std(lengths) / np.mean(lengths)) * 100
        score = max(0, 100 - cv * 3)
        if score >= 80:
            interp = "Very regular"
        elif score >= 60:
            interp = "Mostly regular"
        elif score >= 40:
            interp = "Somewhat irregular"
        else:
            interp = "Irregular — consider consulting a healthcare provider"
        return {
            "score":           round(score, 1),
            "cv_percent":      round(cv, 2),
            "mean_cycle":      round(float(np.mean(lengths)), 1),
            "std_cycle":       round(float(np.std(lengths)), 1),
            "interpretation":  interp,
        }

    def print_report(self):
        print("\n" + "="*55)
        print("          SYMPTOM ANALYSIS REPORT")
        print("="*55)

        print("\n📊 Cycle Regularity:")
        reg = self.cycle_regularity_score()
        for k, v in reg.items():
            print(f"   {k}: {v}")

        print("\n💊 Top Symptoms by Average Severity:")
        for sym, sev in self.top_symptoms_by_severity():
            bar = "█" * int(sev * 4)
            print(f"   {sym:<20} {bar}  {sev:.1f}/5")

        print("\n🔗 Symptom ↔ Cycle Length Correlations:")
        for sym, (r, p) in self.symptom_cycle_length_correlation().items():
            sig = "✓ significant" if p < 0.05 else ""
            print(f"   {sym:<20} r={r:+.3f}  p={p:.4f}  {sig}")