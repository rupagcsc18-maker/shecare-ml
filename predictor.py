"""
predictor.py  —  Fixed v2
==============================
Fixes:
  1. Ovulation formula corrected:
       ovulation = period_start + (cycle_length - LUTEAL_DAYS)
       28-day cycle → Day 14 ✓   35-day cycle → Day 21 ✓

  2. Phase order enforced throughout:
       Menstruation → Follicular → Fertile Window → Ovulation → Luteal → PMS

  3. Fertile window NEVER overlaps menstruation:
       fertile_start >= follicular_start (day after period ends)
       fertile_start >= period_start + MIN_DAYS_AFTER_PERIOD_START (7 days)

  4. Statistical fallback uses winsorized mean + Bessel-corrected std + t-CI

  5. predict_windows propagates uncertainty per cycle (widens for cycle 2, 3)

  6. Clip upper bound raised to 60 (handles PCOS/perimenopause cycles)

  7. LSTM indexing bug fixed: X_seq[-1:] shape (1, T, F) — no newaxis needed
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from data_models import UserProfile, CycleDataPreprocessor
from ml_models import LSTMTrainer, GBMCycleModel


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

LUTEAL_DAYS                  = 14   # clinical constant; ~13-15 in literature
FERTILE_PRE                  = 5    # sperm survival before ovulation
FERTILE_POST                 = 1    # egg survival after ovulation
PMS_BEFORE                   = 5    # PMS window starts N days before period
IQR_MULTIPLIER               = 1.75 # outlier clipping (vs Tukey's 1.5)
MIN_DAYS_AFTER_PERIOD_START  = 7    # fertile window cannot start within first 7 days


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _winsorize(cl: np.ndarray) -> np.ndarray:
    """IQR-based winsorization. Returns capped array (same length)."""
    if len(cl) < 4:
        return cl.copy()
    q1, q3 = np.percentile(cl, [25, 75])
    iqr = q3 - q1
    return np.clip(cl, q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr)


def _stat_predict(cycle_lengths: List[float],
                  cycles_ahead: int = 1) -> Tuple[float, float, float, float]:
    """
    Returns (point, sd, ci_low, ci_high).
    Mirrors ImprovedPeriodPredictor logic so both predictors agree.
    """
    cl_raw = np.array(cycle_lengths, dtype=float)
    cl     = _winsorize(cl_raw)
    n      = len(cl)

    mean = float(np.mean(cl))
    std  = float(np.std(cl, ddof=1)) if n > 1 else 2.0
    cv   = std / mean if mean > 0 else 0.0

    # Trend detection
    trend_slope = 0.0
    trend_p     = 1.0
    if n >= 4:
        slope, _, _, p, _ = scipy_stats.linregress(range(n), cl)
        trend_slope, trend_p = float(slope), float(p)

    # Point estimate
    if n == 1:
        point = float(cl[0])
    elif cv < 0.05:
        weights = np.arange(1, n + 1, dtype=float)
        point   = float(np.average(cl, weights=weights))
    elif cv < 0.15:
        alpha = min(0.6, 0.25 + cv * 2.0)
        ewma  = cl[0]
        for c in cl[1:]:
            ewma = alpha * c + (1.0 - alpha) * ewma
        point = float(ewma)
    else:
        point = float(np.median(cl))

    # Trend correction
    if trend_p < 0.10 and abs(trend_slope) > 0.3:
        point += trend_slope * cycles_ahead

    point = float(np.clip(point, 21, 60))

    # CI (t-distribution, Bessel-corrected, propagated)
    propagated_sd = std * np.sqrt(cycles_ahead)
    t_crit = float(scipy_stats.t.ppf(0.975, df=max(1, n - 1))) if n > 2 else 4.303
    margin = t_crit * propagated_sd
    ci_low  = max(15.0, point - margin)
    ci_high = min(90.0, point + margin)

    return point, propagated_sd, ci_low, ci_high


# ══════════════════════════════════════════════════════════════════════════════
# WINDOW BUILDER  (shared by ML and statistical paths)
# ══════════════════════════════════════════════════════════════════════════════

def build_cycle_window(period_start: date,
                       cycle_len: float,
                       avg_duration: float,
                       cycle_number: int,
                       sd: float = 0.0,
                       ci_low: float = 0.0,
                       ci_high: float = 0.0) -> Dict:
    """
    Compute all cycle phases from a period start date + cycle length.

    PHASE ORDER: Menstruation → Follicular → Fertile Window → Ovulation → Luteal → PMS

    OVULATION FORMULA (corrected):
        ovulation_day = period_start + (cycle_len - LUTEAL_DAYS)
        28-day cycle: ovulation on Day 14 ✓
        35-day cycle: ovulation on Day 21 ✓

    FERTILE WINDOW CONSTRAINTS:
        1. fertile_start >= day after period ends  (no overlap with menstruation)
        2. fertile_start >= period_start + 7 days  (minimum 7 days from cycle start)
        3. fertile_start = ovulation_day - 5       (sperm survival)
        4. fertile_end   = ovulation_day + 1       (egg survival)
    """
    pd_d  = round(avg_duration)
    cl_d  = round(cycle_len)
    ov_d  = cl_d - LUTEAL_DAYS                    # days from period_start

    period_end       = period_start + timedelta(days=pd_d - 1)
    ovulation_day    = period_start + timedelta(days=ov_d)
    follicular_start = period_end + timedelta(days=1)

    # Fertile window — raw, then clamped
    fertile_start_raw = ovulation_day - timedelta(days=FERTILE_PRE)
    fertile_end       = ovulation_day + timedelta(days=FERTILE_POST)

    # Enforce constraints: no overlap with menstruation, min 7 days from period start
    earliest_fertile = max(
        follicular_start,
        period_start + timedelta(days=MIN_DAYS_AFTER_PERIOD_START)
    )
    fertile_start = max(fertile_start_raw, earliest_fertile)

    # Follicular: day after period ends → day before fertile window starts
    follicular_end = fertile_start - timedelta(days=1)
    if follicular_end < follicular_start:
        follicular_end = follicular_start   # graceful collapse on very short cycles

    # Luteal and PMS
    luteal_start = ovulation_day + timedelta(days=1)
    next_period  = period_start + timedelta(days=cl_d)
    luteal_end   = next_period - timedelta(days=1)
    pms_start    = next_period - timedelta(days=PMS_BEFORE)

    def s(d: date) -> str:
        return str(d)

    result = {
        "cycle_number":           cycle_number,
        "period_start":           s(period_start),
        "period_end":             s(period_end),
        "fertile_start":          s(fertile_start),
        "fertile_end":            s(fertile_end),
        "ovulation_day":          s(ovulation_day),
        "luteal_start":           s(luteal_start),
        "luteal_end":             s(luteal_end),
        "pms_start":              s(pms_start),
        "predicted_cycle_length": round(cycle_len, 1),
        "phases": {
            "menstruation": {
                "start":         s(period_start),
                "end":           s(period_end),
                "duration_days": pd_d,
                "description":   "Uterine lining sheds — Day 1 of cycle",
            },
            "follicular": {
                "start":         s(follicular_start),
                "end":           s(follicular_end),
                "duration_days": max(0, (follicular_end - follicular_start).days + 1),
                "description":   "Follicles develop, estrogen rises",
            },
            "fertile_window": {
                "start":         s(fertile_start),
                "end":           s(fertile_end),
                "duration_days": (fertile_end - fertile_start).days + 1,
                "description":   "Highest chance of conception — always after menstruation ends",
            },
            "ovulation": {
                "day":           s(ovulation_day),
                "description":   "Egg released — peak fertility (cycle_length - 14 from Day 1)",
            },
            "luteal": {
                "start":         s(luteal_start),
                "end":           s(luteal_end),
                "duration_days": (luteal_end - luteal_start).days + 1,
                "description":   "Progesterone rises; body prepares for next period",
            },
            "pms_window": {
                "start":         s(pms_start),
                "end":           s(luteal_end),
                "description":   "Possible PMS symptoms",
            },
        },
    }

    if sd:
        result["uncertainty"] = f"±{round(sd, 1)} days"
        result["ci_95"]       = f"{round(ci_low)}–{round(ci_high)} days"

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PREDICTOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class CyclePredictor:
    """
    Ensemble predictor (LSTM + GBM) with full cycle window output.
    Falls back to ImprovedPeriodPredictor-style statistics when data is scarce.

    Phase order: Menstruation → Follicular → Fertile Window → Ovulation → Luteal → PMS
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

        df_feat      = self.preprocessor.build_features(df)
        X_seq, y_seq = self.preprocessor.build_sequences(df)

        # LSTM
        if len(X_seq) >= 4:
            input_size = X_seq.shape[2]
            self.lstm  = LSTMTrainer(input_size=input_size)
            self.lstm.fit(X_seq, y_seq, epochs=150)

        # GBM (flat features)
        flat_feat_cols = self.preprocessor.feature_cols
        df_feat_clean  = df_feat.dropna(subset=["cycle_length"])
        X_flat = df_feat_clean[flat_feat_cols].values
        y_flat = df_feat_clean["cycle_length"].values
        if len(X_flat) >= 4:
            self.gbm = GBMCycleModel()
            self.gbm.fit(X_flat, y_flat)

        self.is_trained = True
        print("Training complete.")

    # ── Cycle length prediction ───────────────────────────────────────────────
    def predict_next_cycle_length(self, profile: UserProfile,
                                  cycles_ahead: int = 1) -> Dict:
        """
        Returns dict: method, predicted_cycle_length, confidence, sd, ci_low, ci_high.
        """
        df      = profile.to_dataframe()
        df_feat = self.preprocessor.build_features(df)
        known   = [getattr(c, "_cycle_length", None) for c in profile.cycles]
        known   = [l for l in known if l is not None]

        preds = []

        # LSTM path
        if self.lstm and len(df) >= self.SEQUENCE_LENGTH:
            X_seq, _ = self.preprocessor.build_sequences(df)
            if len(X_seq) > 0:
                # X_seq[-1:] already has shape (1, T, F) — no newaxis needed
                lstm_pred = float(self.lstm.predict(X_seq[-1:]))
                preds.append(("LSTM", lstm_pred, 0.55))

        # GBM path
        if self.gbm:
            last_row = df_feat[self.preprocessor.feature_cols].iloc[-1:].values
            gbm_pred = float(self.gbm.predict(last_row)[0])
            preds.append(("GBM", gbm_pred, 0.45))

        # Statistical fallback
        if not preds:
            if not known:
                point = profile.avg_cycle_length
                sd, ci_low, ci_high = 3.0, point - 9.0, point + 9.0
            else:
                point, sd, ci_low, ci_high = _stat_predict(known, cycles_ahead)

            return {
                "method":                 "statistical",
                "predicted_cycle_length": round(point, 1),
                "sd":                     round(sd, 1),
                "ci_95":                  f"{round(ci_low)}–{round(ci_high)} days",
                "confidence":             "low",
            }

        # Weighted ensemble
        total_w  = sum(w for _, _, w in preds)
        ensemble = sum(p * w for _, p, w in preds) / total_w
        ensemble = float(np.clip(ensemble, 21, 60))

        # Statistical SD for CI
        sd_for_ci = 3.0
        if known and len(known) >= 2:
            cl_w = _winsorize(np.array(known, dtype=float))
            sd_for_ci = float(np.std(cl_w, ddof=1))

        propagated_sd = sd_for_ci * np.sqrt(cycles_ahead)
        n = len(known)
        t_crit = float(scipy_stats.t.ppf(0.975, df=max(1, n - 1))) if n > 2 else 4.303
        margin = t_crit * propagated_sd
        ci_low  = max(15.0, ensemble - margin)
        ci_high = min(90.0, ensemble + margin)

        return {
            "method":                 "ensemble (LSTM + GBM)",
            "predicted_cycle_length": round(ensemble, 1),
            "sd":                     round(propagated_sd, 1),
            "ci_95":                  f"{round(ci_low)}–{round(ci_high)} days",
            "component_predictions":  {name: round(p, 1) for name, p, _ in preds},
            "confidence":             "high" if len(profile.cycles) >= 12 else "medium",
        }

    # ── Window prediction ─────────────────────────────────────────────────────
    def predict_windows(self, profile: UserProfile,
                        from_date: Optional[date] = None,
                        num_cycles: int = 3) -> List[Dict]:
        """
        Returns next `num_cycles` cycle windows with all 6 phases.
        Uncertainty widens for each additional cycle predicted.
        Phase order: Menstruation → Follicular → Fertile Window → Ovulation → Luteal → PMS
        """
        from_date    = from_date or date.today()
        avg_duration = profile.avg_period_duration

        last_start = (max(c.period_start for c in profile.cycles)
                      if profile.cycles else from_date)

        windows       = []
        current_start = last_start

        for i in range(num_cycles):
            cycles_ahead = i + 1
            result       = self.predict_next_cycle_length(profile, cycles_ahead)
            cycle_len    = result["predicted_cycle_length"]
            sd           = result.get("sd", 3.0)

            ci_parts = result.get("ci_95", "").replace(" days", "").split("–")
            ci_low   = float(ci_parts[0]) if len(ci_parts) == 2 else cycle_len - sd * 2
            ci_high  = float(ci_parts[1]) if len(ci_parts) == 2 else cycle_len + sd * 2

            current_start = current_start + timedelta(days=round(cycle_len))
            window        = build_cycle_window(
                period_start = current_start,
                cycle_len    = cycle_len,
                avg_duration = avg_duration,
                cycle_number = cycles_ahead,
                sd           = sd,
                ci_low       = ci_low,
                ci_high      = ci_high,
            )
            windows.append(window)

        return windows


# ══════════════════════════════════════════════════════════════════════════════
# PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_windows(windows: List[Dict]):
    print(f"\n📅 PHASE ORDER: Menstruation → Follicular → Fertile Window → Ovulation → Luteal → PMS\n")
    for w in windows:
        print(f"\n{'=' * 58}")
        print(f" Cycle {w['cycle_number']}  "
              f"(~{w['predicted_cycle_length']}d"
              + (f"  {w.get('uncertainty', '')}"
                 f"  CI: {w.get('ci_95', '')}" if w.get("uncertainty") else "")
              + ")")
        print(f"{'=' * 58}")
        p = w["phases"]
        print(f"  🩸 Period         : {p['menstruation']['start']} → {p['menstruation']['end']}"
              f"  ({p['menstruation']['duration_days']}d)")
        print(f"  🌱 Follicular     : {p['follicular']['start']} → {p['follicular']['end']}"
              f"  ({p['follicular']['duration_days']}d)")
        print(f"  💚 Fertile window : {p['fertile_window']['start']} → {p['fertile_window']['end']}"
              f"  ({p['fertile_window']['duration_days']}d)")
        print(f"  🥚 Ovulation      : {p['ovulation']['day']}  ← peak fertility")
        print(f"  🌙 Luteal phase   : {p['luteal']['start']} → {p['luteal']['end']}"
              f"  ({p['luteal']['duration_days']}d)")
        print(f"  😔 PMS window     : {p['pms_window']['start']} → {p['pms_window']['end']}")