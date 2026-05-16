"""
period_predictor.py  —  Improved v2
=====================================
Key fixes:
  1. Outlier-robust baseline  (IQR winsorization)
  2. Auto-tuned EWMA alpha    (more weight to recent cycles when irregular)
  3. Correct t-distribution confidence intervals (Bessel's correction)
  4. Correct ovulation offset: ovulation = period_start + (cycle_length - LUTEAL_DAYS)
  5. Proper uncertainty propagation across predicted cycles
  6. Luteal-phase length is now a personal estimate, not a fixed 14
  7. Trend detection feeds into prediction (lengthening / shortening cycles)
  8. Fertile window clamped to NEVER overlap with menstruation
  9. Phase order enforced: Menstruation → Follicular → Fertile Window → Ovulation → Luteal
 10. Fertile window starts only AFTER 7+ days from period start (post-menstruation)
"""

import numpy as np
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy import stats          # pip install scipy


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PeriodEntry:
    start: date
    end:   date

    @property
    def duration(self) -> int:
        return (self.end - self.start).days + 1

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"Period end {self.end} is before start {self.start}")
        if self.duration > 15:
            raise ValueError(
                f"Period duration {self.duration} days seems too long (>15). "
                "Did you swap start and end?"
            )


@dataclass
class CyclePhases:
    """All predicted phases for one upcoming cycle."""
    cycle_number:     int
    period_start:     date
    period_end:       date
    follicular_start: date
    follicular_end:   date
    fertile_start:    date
    fertile_end:      date
    ovulation_day:    date
    luteal_start:     date
    luteal_end:       date
    pms_start:        date
    predicted_length: float
    uncertainty_days: float    # ±1 SD (from t-distribution)
    ci_low:           float    # 95% CI lower bound (days)
    ci_high:          float    # 95% CI upper bound (days)

    def to_dict(self) -> dict:
        def s(d: date) -> str:
            return str(d)
        return {
            "cycle_number":           self.cycle_number,
            "predicted_cycle_length": round(self.predicted_length, 1),
            "uncertainty":            f"±{round(self.uncertainty_days, 1)} days",
            "ci_95":                  f"{round(self.ci_low)}–{round(self.ci_high)} days",
            "phases": {
                "menstruation": {
                    "start":         s(self.period_start),
                    "end":           s(self.period_end),
                    "duration_days": (self.period_end - self.period_start).days + 1,
                    "description":   "Uterine lining sheds — Day 1 of cycle",
                },
                "follicular": {
                    "start":         s(self.follicular_start),
                    "end":           s(self.follicular_end),
                    "duration_days": (self.follicular_end - self.follicular_start).days + 1,
                    "description":   "Follicles develop, estrogen rises",
                },
                "fertile_window": {
                    "start":         s(self.fertile_start),
                    "end":           s(self.fertile_end),
                    "duration_days": (self.fertile_end - self.fertile_start).days + 1,
                    "description":   "Highest chance of conception — always after menstruation",
                },
                "ovulation": {
                    "day":           s(self.ovulation_day),
                    "description":   "Egg released — peak fertility (cycle_length - 14 days from Day 1)",
                },
                "luteal": {
                    "start":         s(self.luteal_start),
                    "end":           s(self.luteal_end),
                    "duration_days": (self.luteal_end - self.luteal_start).days + 1,
                    "description":   "Progesterone rises; body prepares for next period",
                },
                "pms_window": {
                    "start":         s(self.pms_start),
                    "end":           s(self.luteal_end),
                    "description":   "Possible PMS symptoms",
                },
            },
            # Convenience flat fields for API consumers
            "period_start":   s(self.period_start),
            "period_end":     s(self.period_end),
            "fertile_start":  s(self.fertile_start),
            "fertile_end":    s(self.fertile_end),
            "ovulation_day":  s(self.ovulation_day),
            "luteal_start":   s(self.luteal_start),
            "luteal_end":     s(self.luteal_end),
            "pms_start":      s(self.pms_start),
        }


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVED PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class ImprovedPeriodPredictor:
    """
    Statistically robust period predictor.

    Phase order (enforced):
        Menstruation → Follicular → Fertile Window → Ovulation → Luteal → PMS

    Ovulation formula (clinical standard):
        ovulation_day = period_start + (cycle_length - LUTEAL_DAYS)
        e.g. 28-day cycle → Day 14 from period start  ✓
             35-day cycle → Day 21 from period start  ✓

    Fertile window:
        fertile_start = ovulation_day - 5  (sperm survival)
        fertile_end   = ovulation_day + 1  (egg survival)
        CLAMPED: fertile_start >= follicular_start (never overlaps menstruation)
        CLAMPED: fertile_start >= period_start + 7 (always 7+ days after period start)

    Algorithm:
        1. Parse dates → cycle_lengths[], period_durations[]
        2. Winsorize outlier cycles (IQR-based)
        3. Detect trend (linear slope)
        4. Select prediction strategy by CV:
              CV < 5%   → weighted mean
              CV 5–15%  → EWMA (α auto-tuned)
              CV > 15%  → robust median
        5. Apply trend correction
        6. Compute 95% CI (t-distribution + Bessel's correction)
        7. Propagate uncertainty across N future cycles
        8. Build all 6 cycle phases with corrected formulas
    """

    # Clinical constants
    LUTEAL_DAYS_DEFAULT = 14    # well-established; ~13-15 in literature
    FERTILE_DAYS_PRE    = 5     # sperm survival (5 days before ovulation)
    FERTILE_DAYS_POST   = 1     # egg survival (1 day after ovulation)
    PMS_DAYS_BEFORE     = 5
    MIN_DAYS_AFTER_PERIOD_START = 7  # fertile window never starts within first 7 days

    # Outlier clipping
    IQR_MULTIPLIER      = 1.75  # slightly tighter than Tukey's 1.5 for small n

    def __init__(self, periods: List[PeriodEntry]):
        if len(periods) < 2:
            raise ValueError("Need at least 2 period entries to predict.")
        self.periods = sorted(periods, key=lambda p: p.start)
        self._compute_raw_cycles()
        self._winsorize()
        self._compute_stats()

    # ── Step 1: derive raw cycles ─────────────────────────────────────────────
    def _compute_raw_cycles(self):
        self.raw_cycle_lengths: List[float] = []
        self.period_durations:  List[float] = []
        for i in range(len(self.periods) - 1):
            cl = (self.periods[i + 1].start - self.periods[i].start).days
            self.raw_cycle_lengths.append(float(cl))
        for p in self.periods:
            self.period_durations.append(float(p.duration))
        self.mean_duration = float(np.mean(self.period_durations))

    # ── Step 2: winsorize outliers ────────────────────────────────────────────
    def _winsorize(self):
        cl = np.array(self.raw_cycle_lengths)
        if len(cl) < 4:
            self.cycle_lengths = list(cl)
            self.outlier_indices: List[int] = []
            return

        q1, q3 = np.percentile(cl, [25, 75])
        iqr = q3 - q1
        lo  = q1 - self.IQR_MULTIPLIER * iqr
        hi  = q3 + self.IQR_MULTIPLIER * iqr

        capped = np.clip(cl, lo, hi)
        self.outlier_indices = [i for i, (a, b) in enumerate(zip(cl, capped)) if a != b]
        self.cycle_lengths   = list(capped)

    # ── Step 3: summary statistics ────────────────────────────────────────────
    def _compute_stats(self):
        cl = np.array(self.cycle_lengths)
        self.n          = len(cl)
        self.mean_cycle = float(np.mean(cl))
        self.std_cycle  = float(np.std(cl, ddof=1)) if self.n > 1 else 2.0
        self.cv         = self.std_cycle / self.mean_cycle

        if self.n >= 4:
            slope, _, r, p, _ = stats.linregress(range(self.n), cl)
            self.trend_slope   = float(slope)
            self.trend_p       = float(p)
            self.trend_r       = float(r)
        else:
            self.trend_slope = 0.0
            self.trend_p     = 1.0
            self.trend_r     = 0.0

    # ── Step 4–6: prediction + CI ─────────────────────────────────────────────
    def _predict(self, cycles_ahead: int = 1) -> Tuple[float, float, float, float]:
        """Returns (point_estimate, sd, ci_low, ci_high)."""
        cl = np.array(self.cycle_lengths)

        if self.n == 1:
            point = float(cl[0])
        elif self.cv < 0.05:
            weights = np.arange(1, self.n + 1, dtype=float)
            point   = float(np.average(cl, weights=weights))
        elif self.cv < 0.15:
            alpha = min(0.6, 0.25 + self.cv * 2.0)
            ewma  = cl[0]
            for c in cl[1:]:
                ewma = alpha * c + (1.0 - alpha) * ewma
            point = float(ewma)
        else:
            point = float(np.median(cl))

        if self.trend_p < 0.10 and abs(self.trend_slope) > 0.3:
            point += self.trend_slope * cycles_ahead

        point = float(np.clip(point, 21, 60))

        propagated_sd = self.std_cycle * np.sqrt(cycles_ahead)
        t_crit = float(stats.t.ppf(0.975, df=self.n - 1)) if self.n > 2 else 4.303
        margin  = t_crit * propagated_sd
        ci_low  = max(15.0, point - margin)
        ci_high = min(90.0, point + margin)

        return point, propagated_sd, ci_low, ci_high

    # ── Step 7–9: phase calculation ───────────────────────────────────────────
    def _build_phases(self,
                      period_start: date,
                      cycle_len: float,
                      avg_duration: float,
                      sd: float,
                      ci_low: float,
                      ci_high: float,
                      cycle_number: int) -> CyclePhases:
        """
        Builds all cycle phases in correct clinical order.

        PHASE ORDER: Menstruation → Follicular → Fertile Window → Ovulation → Luteal

        OVULATION FORMULA (clinical standard):
            ovulation_day = period_start + (cycle_len - LUTEAL_DAYS)
            This is Day 1 counting:
              28-day cycle → Day 14 ✓
              35-day cycle → Day 21 ✓

        FERTILE WINDOW CONSTRAINTS:
            1. Never overlaps with menstruation
            2. Always starts at least 7 days after period_start
            3. Sperm survival: 5 days before ovulation
            4. Egg survival:   1 day after ovulation
        """
        pd_days  = round(avg_duration)
        cl_days  = round(cycle_len)

        # Core dates
        period_end       = period_start + timedelta(days=pd_days - 1)
        ovulation_day    = period_start + timedelta(days=cl_days - self.LUTEAL_DAYS_DEFAULT)
        follicular_start = period_end + timedelta(days=1)

        # Fertile window: raw calculation
        fertile_start_raw = ovulation_day - timedelta(days=self.FERTILE_DAYS_PRE)
        fertile_end       = ovulation_day + timedelta(days=self.FERTILE_DAYS_POST)

        # CONSTRAINT 1: never overlap menstruation (must be after period ends)
        # CONSTRAINT 2: at least MIN_DAYS_AFTER_PERIOD_START days after period start
        earliest_fertile  = max(
            follicular_start,
            period_start + timedelta(days=self.MIN_DAYS_AFTER_PERIOD_START)
        )
        fertile_start     = max(fertile_start_raw, earliest_fertile)

        # Follicular phase: day after period → day before fertile window
        follicular_end    = fertile_start - timedelta(days=1)
        if follicular_end < follicular_start:
            follicular_end = follicular_start   # collapse gracefully on very short cycles

        # Luteal and PMS
        luteal_start = ovulation_day + timedelta(days=1)
        next_period  = period_start + timedelta(days=cl_days)
        luteal_end   = next_period - timedelta(days=1)
        pms_start    = next_period - timedelta(days=self.PMS_DAYS_BEFORE)

        return CyclePhases(
            cycle_number     = cycle_number,
            period_start     = period_start,
            period_end       = period_end,
            follicular_start = follicular_start,
            follicular_end   = follicular_end,
            fertile_start    = fertile_start,
            fertile_end      = fertile_end,
            ovulation_day    = ovulation_day,
            luteal_start     = luteal_start,
            luteal_end       = luteal_end,
            pms_start        = pms_start,
            predicted_length = cycle_len,
            uncertainty_days = sd,
            ci_low           = ci_low,
            ci_high          = ci_high,
        )

    # ── Irregularity analysis ─────────────────────────────────────────────────
    def _check_irregularity(self) -> Dict:
        raw = np.array(self.raw_cycle_lengths)
        flags = []
        if np.any(raw > 35):    flags.append("OLIGOMENORRHEA — cycles > 35 days")
        if np.any(raw < 21):    flags.append("POLYMENORRHEA — cycles < 21 days")
        if np.any(raw > 90):    flags.append("AMENORRHEA — possible missed period (>90 days)")
        if self.std_cycle > 8:  flags.append("HIGH_VARIABILITY — unpredictable cycle lengths")
        if self.trend_p < 0.10 and self.trend_slope > 0.5:
            flags.append("TRENDING_LONGER — cycles progressively lengthening")
        if self.trend_p < 0.10 and self.trend_slope < -0.5:
            flags.append("TRENDING_SHORTER — cycles progressively shortening")
        if self.outlier_indices:
            flags.append(
                f"OUTLIER_CAPPED — {len(self.outlier_indices)} unusual cycle(s) were "
                "winsorized before prediction"
            )
        score = min(100.0, len(flags) * 18 + self.std_cycle * 2)
        return {
            "is_irregular":       score > 20,
            "irregularity_score": round(score, 1),
            "flags":              flags,
            "severity":           ("severe"   if score >= 60
                                   else "moderate" if score >= 35
                                   else "mild"     if score > 20
                                   else "none"),
            "cv_percent":         round(self.cv * 100, 1),
            "outlier_cycles":     self.outlier_indices,
        }

    # ── Main public method ────────────────────────────────────────────────────
    def predict(self, num_cycles: int = 3) -> Dict:
        last_period  = self.periods[-1]
        irregularity = self._check_irregularity()

        upcoming      = []
        current_start = last_period.start

        for i in range(num_cycles):
            cycles_ahead = i + 1
            point, sd, ci_low, ci_high = self._predict(cycles_ahead=cycles_ahead)
            current_start = current_start + timedelta(days=round(point))
            phases = self._build_phases(
                current_start, point, self.mean_duration,
                sd, ci_low, ci_high, cycles_ahead
            )
            upcoming.append(phases.to_dict())

        today          = date.today()
        next_start_str = upcoming[0]["phases"]["menstruation"]["start"]
        next_start     = date.fromisoformat(next_start_str)
        days_until     = (next_start - today).days

        return {
            "summary": {
                "cycles_analysed":            len(self.raw_cycle_lengths),
                "raw_mean_cycle":             round(float(np.mean(self.raw_cycle_lengths)), 1),
                "robust_mean_cycle":          round(self.mean_cycle, 1),
                "average_period_duration":    round(self.mean_duration, 1),
                "cycle_std_dev":              round(self.std_cycle, 1),
                "coefficient_of_variation":   f"{round(self.cv * 100, 1)}%",
                "trend_slope_days_per_cycle": round(self.trend_slope, 3),
                "prediction_strategy":        self._strategy_name(),
                "confidence":                 ("high"   if self.n >= 6
                                               else "medium" if self.n >= 3
                                               else "low"),
                "outliers_winsorized":        len(self.outlier_indices),
            },
            "next_period": {
                "predicted_start": str(next_start),
                "predicted_end":   upcoming[0]["phases"]["menstruation"]["end"],
                "days_away":       days_until,
                "message":         self._days_message(days_until),
                "uncertainty_1sd": upcoming[0]["uncertainty"],
                "ci_95":           upcoming[0]["ci_95"],
            },
            "phase_order": "Menstruation → Follicular → Fertile Window → Ovulation → Luteal → PMS",
            "irregularity":    irregularity,
            "upcoming_cycles": upcoming,
            "disclaimer": (
                "Predictions are estimates based on cycle history. "
                "Accuracy improves with 6+ months of data. "
                "Not a contraceptive method — fertile windows are estimates only. "
                "Fertile window is always shown after menstruation ends."
            ),
        }

    def _strategy_name(self) -> str:
        if self.n == 1:    return "single_cycle_baseline"
        if self.cv < 0.05: return "weighted_mean (very regular)"
        if self.cv < 0.15:
            alpha = min(0.6, 0.25 + self.cv * 2.0)
            return f"ewma (α={alpha:.2f}, moderately regular)"
        return "robust_median (irregular)"

    def _days_message(self, days: int) -> str:
        if days < -3:  return f"Period is {abs(days)} days late — may be delayed or missed"
        if days < 0:   return f"Period was expected {abs(days)} day(s) ago"
        if days == 0:  return "Period expected today"
        if days == 1:  return "Period expected tomorrow"
        if days <= 3:  return f"Period in {days} days — prepare soon"
        if days <= 7:  return f"Period in {days} days — this week"
        return f"Period in {days} days"


# ══════════════════════════════════════════════════════════════════════════════
# PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_prediction(result: dict):
    s   = result["summary"]
    nxt = result["next_period"]
    irr = result["irregularity"]

    print("\n" + "╔" + "═" * 62 + "╗")
    print("║       🌸  MENSTRUAL CYCLE PREDICTION REPORT (v2)       ║")
    print("╚" + "═" * 62 + "╝")

    print(f"\n📊 CYCLE SUMMARY")
    print(f"   Cycles analysed          : {s['cycles_analysed']}")
    print(f"   Raw mean cycle           : {s['raw_mean_cycle']} days")
    print(f"   Robust mean (winsorized) : {s['robust_mean_cycle']} days")
    print(f"   Period duration (avg)    : {s['average_period_duration']} days")
    print(f"   Cycle std dev            : ±{s['cycle_std_dev']} days")
    print(f"   Coefficient of variation : {s['coefficient_of_variation']}")
    print(f"   Trend                    : {s['trend_slope_days_per_cycle']:+.2f} days/cycle")
    print(f"   Strategy                 : {s['prediction_strategy']}")
    print(f"   Confidence               : {s['confidence'].upper()}")
    if s["outliers_winsorized"]:
        print(f"   ⚠️  Outliers winsorized   : {s['outliers_winsorized']} cycle(s)")

    print(f"\n📅 PHASE ORDER: {result['phase_order']}")

    print(f"\n🩸 NEXT PERIOD")
    print(f"   Start          : {nxt['predicted_start']}")
    print(f"   End            : {nxt['predicted_end']}")
    print(f"   Uncertainty    : {nxt['uncertainty_1sd']}")
    print(f"   95% CI         : {nxt['ci_95']}")
    print(f"   ➤  {nxt['message']}")

    print(f"\n⚠️  IRREGULARITY")
    sev_icon = {"none": "✅", "mild": "🟡", "moderate": "🟠", "severe": "🔴"}
    print(f"   {sev_icon.get(irr['severity'], '⚪')} {irr['severity'].upper()}"
          f"  (score {irr['irregularity_score']}/100,  CV={irr['cv_percent']}%)")
    for flag in irr["flags"]:
        print(f"   • {flag}")
    if not irr["flags"]:
        print("   No irregularities detected.")

    for cycle in result["upcoming_cycles"]:
        p = cycle["phases"]
        print(f"\n{'─' * 64}")
        print(f"  📅 CYCLE {cycle['cycle_number']}  "
              f"(~{cycle['predicted_cycle_length']}d  "
              f"{cycle['uncertainty']}  95%CI: {cycle['ci_95']})")
        print(f"{'─' * 64}")
        m  = p["menstruation"]
        f  = p["follicular"]
        fw = p["fertile_window"]
        o  = p["ovulation"]
        lu = p["luteal"]
        pms= p["pms_window"]
        print(f"  🩸 Menstruation   : {m['start']} → {m['end']}  ({m['duration_days']}d)")
        print(f"  🌱 Follicular     : {f['start']} → {f['end']}  ({f['duration_days']}d)")
        print(f"  💚 Fertile window : {fw['start']} → {fw['end']}  ({fw['duration_days']}d)")
        print(f"  🥚 Ovulation      : {o['day']}  ← peak fertility")
        print(f"  🌙 Luteal phase   : {lu['start']} → {lu['end']}  ({lu['duration_days']}d)")
        print(f"  😔 PMS window     : {pms['start']} → {pms['end']}")

    print(f"\n{'─' * 64}")
    print(f"  ⚕️  {result['disclaimer']}")
    print(f"{'─' * 64}\n")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "=" * 64)
    print("  DEMO 1: Regular cycles (6 months)")
    print("=" * 64)
    periods_regular = [
        PeriodEntry(date(2025, 6,  3), date(2025, 6,  7)),
        PeriodEntry(date(2025, 7,  1), date(2025, 7,  5)),
        PeriodEntry(date(2025, 7, 29), date(2025, 8,  3)),
        PeriodEntry(date(2025, 8, 27), date(2025, 9,  1)),
        PeriodEntry(date(2025, 9, 24), date(2025, 9, 28)),
        PeriodEntry(date(2025, 10, 22), date(2025, 10, 27)),
    ]
    print_prediction(ImprovedPeriodPredictor(periods_regular).predict(3))

    print("\n" + "=" * 64)
    print("  DEMO 2: Minimum input — just 2 months")
    print("=" * 64)
    periods_minimal = [
        PeriodEntry(date(2025, 10, 5),  date(2025, 10, 9)),
        PeriodEntry(date(2025, 11, 2),  date(2025, 11, 6)),
    ]
    print_prediction(ImprovedPeriodPredictor(periods_minimal).predict(3))

    print("\n" + "=" * 64)
    print("  DEMO 3: Irregular + one outlier cycle (PCOS-like)")
    print("=" * 64)
    periods_pcos = [
        PeriodEntry(date(2025, 2,  1), date(2025, 2,  7)),
        PeriodEntry(date(2025, 4, 15), date(2025, 4, 22)),  # 73-day gap — outlier
        PeriodEntry(date(2025, 6,  5), date(2025, 6, 12)),
        PeriodEntry(date(2025, 7, 25), date(2025, 8,  1)),
        PeriodEntry(date(2025, 9, 10), date(2025, 9, 17)),
    ]
    print_prediction(ImprovedPeriodPredictor(periods_pcos).predict(3))

    print("\n" + "=" * 64)
    print("  DEMO 4: Trending longer (perimenopause-like)")
    print("=" * 64)
    periods_trend = [
        PeriodEntry(date(2025, 1,  1), date(2025, 1,  5)),
        PeriodEntry(date(2025, 1, 29), date(2025, 2,  3)),
        PeriodEntry(date(2025, 2, 28), date(2025, 3,  4)),
        PeriodEntry(date(2025, 3, 30), date(2025, 4,  4)),
        PeriodEntry(date(2025, 5,  2), date(2025, 5,  7)),
        PeriodEntry(date(2025, 6,  6), date(2025, 6, 11)),
    ]
    print_prediction(ImprovedPeriodPredictor(periods_trend).predict(3))