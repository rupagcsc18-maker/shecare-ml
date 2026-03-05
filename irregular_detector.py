"""
Detects irregular cycle patterns using statistical + ML methods.
Flags: short cycles, long cycles, skipped periods, high variability,
       sudden shifts, and trending changes.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ── Irregularity types ────────────────────────────────────────────────────────
IRREGULARITY_TYPES = {
    "OLIGOMENORRHEA":   "Cycle > 35 days (infrequent periods)",
    "POLYMENORRHEA":    "Cycle < 21 days (too-frequent periods)",
    "AMENORRHEA":       "Cycle > 90 days or missing period (absent period)",
    "HIGH_VARIABILITY": "Cycle SD > 8 days (unpredictable cycle)",
    "SUDDEN_SHIFT":     "Cycle changed > 10 days from personal baseline",
    "TRENDING_LONGER":  "Cycles consistently getting longer over time",
    "TRENDING_SHORTER": "Cycles consistently getting shorter over time",
    "ANOVULATORY":      "Suspected anovulatory cycle (no temp rise / LH surge)",
    "SPOTTING_PATTERN": "Frequent mid-cycle spotting detected",
    "ANOMALY":          "Statistical outlier detected by Isolation Forest",
}


@dataclass
class IrregularityFlag:
    flag_type: str
    description: str
    severity: str          # "mild" | "moderate" | "severe"
    affected_cycles: List[int] = field(default_factory=list)
    value: Optional[float] = None
    recommendation: str = ""


@dataclass
class IrregularityReport:
    flags: List[IrregularityFlag] = field(default_factory=list)
    overall_severity: str = "none"
    irregularity_score: float = 0.0  # 0–100
    is_irregular: bool = False
    summary: str = ""

    def add_flag(self, flag: IrregularityFlag):
        self.flags.append(flag)

    def compute_overall(self):
        if not self.flags:
            self.overall_severity  = "none"
            self.irregularity_score = 0.0
            self.is_irregular = False
            return

        severity_weights = {"mild": 15, "moderate": 35, "severe": 60}
        score = min(100.0, sum(severity_weights[f.severity] for f in self.flags))
        self.irregularity_score = round(score, 1)
        self.is_irregular = score > 20

        if score >= 60:
            self.overall_severity = "severe"
        elif score >= 35:
            self.overall_severity = "moderate"
        elif score > 0:
            self.overall_severity = "mild"
        else:
            self.overall_severity = "none"

        self.summary = self._build_summary()

    def _build_summary(self) -> str:
        flag_names = [f.flag_type for f in self.flags]
        parts = []
        if "OLIGOMENORRHEA" in flag_names or "AMENORRHEA" in flag_names:
            parts.append("infrequent/absent periods")
        if "POLYMENORRHEA" in flag_names:
            parts.append("too-frequent periods")
        if "HIGH_VARIABILITY" in flag_names or "ANOMALY" in flag_names:
            parts.append("unpredictable cycle lengths")
        if "TRENDING_LONGER" in flag_names:
            parts.append("cycles progressively lengthening")
        if "TRENDING_SHORTER" in flag_names:
            parts.append("cycles progressively shortening")
        if "SUDDEN_SHIFT" in flag_names:
            parts.append("sudden pattern change")
        return "Detected: " + "; ".join(parts) if parts else "Irregular cycle pattern"


class IrregularCycleDetector:
    """
    Multi-method irregularity detector:
      1. Rule-based clinical thresholds (WHO / ACOG guidelines)
      2. Statistical process control (z-score, CUSUM)
      3. Trend detection (Mann-Kendall + linear regression)
      4. Anomaly detection (Isolation Forest)
    """

    # Clinical thresholds
    OLIGO_THRESHOLD     = 35   # days
    POLY_THRESHOLD      = 21   # days
    AMENORRHEA_THRESHOLD= 90   # days
    HIGH_VAR_SD         = 8    # days
    SUDDEN_SHIFT_DELTA  = 10   # days
    MIN_CYCLES          = 4    # minimum to run detection

    def __init__(self):
        self.iso_forest = IsolationForest(
            contamination=0.05, random_state=42, n_estimators=100
        )

    def detect(self, cycle_lengths: List[float],
               period_durations: List[float] = None,
               bbt_data: List[float] = None,
               spotting_flags: List[bool] = None) -> IrregularityReport:

        report = IrregularityReport()
        cl = np.array(cycle_lengths, dtype=float)

        if len(cl) < self.MIN_CYCLES:
            report.summary = f"Need {self.MIN_CYCLES}+ cycles for analysis (have {len(cl)})"
            return report

        # 1. Clinical rule-based checks
        self._check_oligomenorrhea(cl, report)
        self._check_polymenorrhea(cl, report)
        self._check_amenorrhea(cl, report)
        self._check_high_variability(cl, report)
        self._check_sudden_shift(cl, report)

        # 2. Trend detection
        self._check_trends(cl, report)

        # 3. Anomaly detection (Isolation Forest)
        self._check_anomalies(cl, period_durations, report)

        # 4. Optional signals
        if spotting_flags:
            self._check_spotting(spotting_flags, report)
        if bbt_data:
            self._check_anovulatory(bbt_data, report)

        report.compute_overall()
        return report

    # ── Rule-based checks ─────────────────────────────────────────────────────
    def _check_oligomenorrhea(self, cl, report):
        long_idx = np.where(cl > self.OLIGO_THRESHOLD)[0].tolist()
        if len(long_idx) >= 2 or (len(long_idx) == 1 and len(cl) <= 6):
            severity = "severe" if len(long_idx) >= 3 else "moderate"
            report.add_flag(IrregularityFlag(
                flag_type="OLIGOMENORRHEA",
                description=f"{len(long_idx)} cycle(s) > {self.OLIGO_THRESHOLD} days",
                severity=severity,
                affected_cycles=long_idx,
                value=float(np.mean(cl[long_idx])),
                recommendation="Consult a gynaecologist. May indicate PCOS, thyroid issues, or stress."
            ))

    def _check_polymenorrhea(self, cl, report):
        short_idx = np.where(cl < self.POLY_THRESHOLD)[0].tolist()
        if len(short_idx) >= 2:
            report.add_flag(IrregularityFlag(
                flag_type="POLYMENORRHEA",
                description=f"{len(short_idx)} cycle(s) < {self.POLY_THRESHOLD} days",
                severity="moderate",
                affected_cycles=short_idx,
                value=float(np.mean(cl[short_idx])),
                recommendation="May indicate hormonal imbalance or perimenopause transition."
            ))

    def _check_amenorrhea(self, cl, report):
        absent_idx = np.where(cl > self.AMENORRHEA_THRESHOLD)[0].tolist()
        if absent_idx:
            report.add_flag(IrregularityFlag(
                flag_type="AMENORRHEA",
                description=f"Cycle(s) exceeding {self.AMENORRHEA_THRESHOLD} days",
                severity="severe",
                affected_cycles=absent_idx,
                value=float(np.max(cl[absent_idx])),
                recommendation="Seek medical evaluation. Rule out pregnancy, POI, or hypothalamic amenorrhea."
            ))

    def _check_high_variability(self, cl, report):
        sd = float(np.std(cl))
        cv = sd / np.mean(cl) * 100
        if sd > self.HIGH_VAR_SD:
            severity = "severe" if sd > 14 else "moderate" if sd > 10 else "mild"
            report.add_flag(IrregularityFlag(
                flag_type="HIGH_VARIABILITY",
                description=f"Cycle SD = {sd:.1f} days (CV = {cv:.1f}%)",
                severity=severity,
                value=sd,
                recommendation="High variability may indicate PCOS, perimenopause, or chronic stress."
            ))

    def _check_sudden_shift(self, cl, report):
        """Detect sudden mean shift using CUSUM-style scan."""
        baseline = np.mean(cl[:max(3, len(cl)//2)])
        for i in range(3, len(cl)):
            recent_mean = np.mean(cl[max(0, i-3):i])
            delta = abs(recent_mean - baseline)
            if delta >= self.SUDDEN_SHIFT_DELTA:
                direction = "longer" if recent_mean > baseline else "shorter"
                report.add_flag(IrregularityFlag(
                    flag_type="SUDDEN_SHIFT",
                    description=(f"Cycles shifted {direction} by ~{delta:.1f} days "
                                 f"(baseline {baseline:.1f}d → recent {recent_mean:.1f}d)"),
                    severity="moderate",
                    affected_cycles=list(range(max(0, i-3), i)),
                    value=delta,
                    recommendation="Sudden shifts often follow major life events, illness, or medication changes."
                ))
                break  # report only first major shift

    # ── Trend detection ───────────────────────────────────────────────────────
    def _check_trends(self, cl, report):
        """Mann-Kendall trend test + linear regression."""
        n   = len(cl)
        tau, p_value = self._mann_kendall(cl)
        slope = np.polyfit(range(n), cl, 1)[0]

        if p_value < 0.05 and abs(slope) > 0.5:
            if slope > 0:
                report.add_flag(IrregularityFlag(
                    flag_type="TRENDING_LONGER",
                    description=(f"Cycles lengthening by ~{slope:.2f} days/cycle "
                                 f"(τ={tau:.2f}, p={p_value:.3f})"),
                    severity="moderate" if slope > 1.5 else "mild",
                    value=slope,
                    recommendation="Progressive lengthening may indicate perimenopause transition or hypothyroidism."
                ))
            else:
                report.add_flag(IrregularityFlag(
                    flag_type="TRENDING_SHORTER",
                    description=(f"Cycles shortening by ~{abs(slope):.2f} days/cycle "
                                 f"(τ={tau:.2f}, p={p_value:.3f})"),
                    severity="mild",
                    value=slope,
                    recommendation="Cycle shortening can occur in perimenopause or with increased exercise/stress."
                ))

    @staticmethod
    def _mann_kendall(x) -> Tuple[float, float]:
        """Two-sided Mann-Kendall trend test."""
        n = len(x)
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(x[j] - x[i])
        var_s = n * (n - 1) * (2 * n + 5) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        tau = s / (0.5 * n * (n - 1))
        return float(tau), float(p)

    # ── Isolation Forest anomaly detection ───────────────────────────────────
    def _check_anomalies(self, cl, durations, report):
        features = [cl.reshape(-1, 1)]
        if durations and len(durations) == len(cl):
            features.append(np.array(durations).reshape(-1, 1))
        X = np.hstack(features)
        X_scaled = StandardScaler().fit_transform(X)

        preds = self.iso_forest.fit_predict(X_scaled)
        anomaly_idx = np.where(preds == -1)[0].tolist()

        # AFTER — only flag if 2+ anomalies detected (avoids false positives on clean data)
        if len(anomaly_idx) >= 2:
            report.add_flag(IrregularityFlag(
                flag_type="ANOMALY",
                description=f"{len(anomaly_idx)} statistically anomalous cycle(s) detected",
                severity="mild" if len(anomaly_idx) == 2 else "moderate",
                affected_cycles=anomaly_idx,
                recommendation="Review these cycles for illness, travel, or significant stress events."
            ))

    # ── Optional signal checks ────────────────────────────────────────────────
    def _check_spotting(self, spotting_flags, report):
        spotting_count = sum(spotting_flags)
        if spotting_count / len(spotting_flags) > 0.3:
            report.add_flag(IrregularityFlag(
                flag_type="SPOTTING_PATTERN",
                description=f"Spotting in {spotting_count}/{len(spotting_flags)} cycles",
                severity="moderate",
                recommendation="Frequent spotting warrants gynaecological evaluation (fibroids, polyps, or hormonal imbalance)."
            ))

    def _check_anovulatory(self, bbt_data, report):
        """
        Detect anovulatory cycles via BBT: no biphasic shift means no ovulation.
        BBT data: list of daily temperatures across the cycle.
        """
        bbt = np.array(bbt_data)
        mid = len(bbt) // 2
        pre  = np.mean(bbt[:mid])
        post = np.mean(bbt[mid:])
        if (post - pre) < 0.2:  # less than 0.2°C rise = likely anovulatory
            report.add_flag(IrregularityFlag(
                flag_type="ANOVULATORY",
                description=f"No biphasic BBT shift detected (Δ={post-pre:.2f}°C)",
                severity="moderate",
                recommendation="Anovulatory cycles are common in PCOS and perimenopause. Confirm with LH testing."
            ))