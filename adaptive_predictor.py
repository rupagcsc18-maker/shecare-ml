"""
Adaptive predictor: switches strategy based on irregularity type.
Uses Bayesian updating + robust statistics for irregular users.
"""
import numpy as np
from datetime import date, timedelta
from typing import List, Dict, Optional, Tuple
from scipy import stats


class AdaptiveCyclePredictor:
    """
    Strategy router:
    - Regular cycles      → LSTM ensemble (standard predictor)
    - PCOS / high var     → Robust median + IQR-based intervals
    - Perimenopause       → Kalman filter (tracks drifting mean)
    - Stress-induced      → EWMA (exponentially weighted, recent bias)
    - Amenorrhea          → Conservative wide-interval estimate
    """

    def __init__(self, irregularity_report, condition_results):
        self.report     = irregularity_report
        self.conditions = {r.condition: r for r in condition_results}
        self.strategy   = self._select_strategy()

    def _select_strategy(self) -> str:
        flags = {f.flag_type for f in self.report.flags}
        top_condition = max(self.conditions.values(),
                            key=lambda r: r.probability,
                            default=None)

        if not self.report.is_irregular:
            return "standard"
        if "AMENORRHEA" in flags:
            return "amenorrhea"
        if top_condition and top_condition.condition == "PCOS" and top_condition.probability > 0.5:
            return "pcos_robust"
        if top_condition and top_condition.condition == "PERIMENOPAUSE" and top_condition.probability > 0.4:
            return "kalman"
        if top_condition and top_condition.condition == "STRESS_INDUCED" and top_condition.probability > 0.4:
            return "ewma"
        if "HIGH_VARIABILITY" in flags:
            return "robust"
        return "robust"

    def predict(self, cycle_lengths: List[float],
                avg_duration: float = 5.0) -> Dict:
        cl  = np.array(cycle_lengths, dtype=float)
        fn  = {
            "standard":     self._predict_standard,
            "pcos_robust":  self._predict_robust,
            "robust":       self._predict_robust,
            "kalman":       self._predict_kalman,
            "ewma":         self._predict_ewma,
            "amenorrhea":   self._predict_amenorrhea,
        }
        return fn[self.strategy](cl, avg_duration)

    # ── Strategy implementations ──────────────────────────────────────────────
    def _predict_standard(self, cl, avg_dur) -> Dict:
        mean = float(np.mean(cl[-6:]))
        std  = float(np.std(cl[-6:]))
        return self._build_result("standard", mean, std, cl, avg_dur)

    def _predict_robust(self, cl, avg_dur) -> Dict:
        """Median + IQR — resistant to outliers (PCOS long cycles)."""
        median = float(np.median(cl))
        iqr    = float(stats.iqr(cl))
        # Robust std estimate
        robust_std = iqr / 1.349
        return self._build_result("robust (median/IQR)", median, robust_std, cl, avg_dur,
                                  note="Using robust statistics due to high variability / PCOS pattern.")

    def _predict_kalman(self, cl, avg_dur) -> Dict:
        """
        1D Kalman filter — tracks drifting mean (perimenopause).
        State: cycle length estimate.
        """
        Q = 2.0   # process noise (cycle length drifts)
        R = 6.0   # measurement noise

        x_est = cl[0]
        p_est = 10.0

        for z in cl[1:]:
            # predict
            p_pred = p_est + Q
            # update
            K      = p_pred / (p_pred + R)
            x_est  = x_est + K * (z - x_est)
            p_est  = (1 - K) * p_pred

        uncertainty = float(np.sqrt(p_est + R))
        return self._build_result("Kalman filter", float(x_est), uncertainty, cl, avg_dur,
                                  note="Kalman filter applied — tracking gradual cycle drift (perimenopause pattern).")

    def _predict_ewma(self, cl, avg_dur) -> Dict:
        """EWMA — weights recent cycles more (stress: quick recovery)."""
        alpha = 0.4   # higher = more responsive to recent changes
        ewma  = cl[0]
        for c in cl[1:]:
            ewma = alpha * c + (1 - alpha) * ewma

        # Residual std
        residuals = []
        e = cl[0]
        for c in cl[1:]:
            residuals.append(abs(c - e))
            e = alpha * c + (1 - alpha) * e
        std = float(np.std(residuals)) if residuals else 3.0

        return self._build_result("EWMA", float(ewma), std, cl, avg_dur,
                                  note="EWMA applied — recent cycles weighted more (stress-induced pattern).")

    def _predict_amenorrhea(self, cl, avg_dur) -> Dict:
        valid = cl[cl < 90]
        base  = float(np.median(valid)) if len(valid) > 0 else 35.0
        return self._build_result("conservative (amenorrhea)", base, 15.0, cl, avg_dur,
                                  note="Wide uncertainty interval due to amenorrhea pattern. Seek medical evaluation.")

    # ── Build result ──────────────────────────────────────────────────────────
    def _build_result(self, strategy: str, predicted_length: float,
                      uncertainty: float, cl: np.ndarray,
                      avg_dur: float, note: str = "") -> Dict:
        predicted_length = float(np.clip(predicted_length, 15, 90))
        ci_low  = max(15.0, predicted_length - 1.96 * uncertainty)
        ci_high = min(90.0, predicted_length + 1.96 * uncertainty)

        return {
            "strategy":               strategy,
            "predicted_cycle_length": round(predicted_length, 1),
            "ci_95_low":              round(ci_low, 1),
            "ci_95_high":             round(ci_high, 1),
            "uncertainty_days":       round(uncertainty, 1),
            "confidence_interval":    f"{round(ci_low)}–{round(ci_high)} days",
            "note":                   note,
        }

    def predict_windows(self, cycle_lengths: List[float],
                        last_period_start: date,
                        avg_duration: float = 5.0,
                        num_cycles: int = 3) -> List[Dict]:
        pred   = self.predict(cycle_lengths, avg_duration)
        length = pred["predicted_cycle_length"]
        windows = []
        current = last_period_start

        for i in range(num_cycles):
            current     = current + timedelta(days=round(length))
            ovulation   = current + timedelta(days=round(length - 14))
            windows.append({
                "cycle_number":    i + 1,
                "period_start":    current,
                "period_end":      current + timedelta(days=round(avg_duration) - 1),
                "fertile_start":   ovulation - timedelta(days=5),
                "fertile_end":     ovulation + timedelta(days=1),
                "ovulation_day":   ovulation,
                "predicted_length": round(length, 1),
                "uncertainty":     f"±{round(pred['uncertainty_days'], 1)} days",
                "ci":              pred["confidence_interval"],
                "strategy":        pred["strategy"],
            })
        return windows