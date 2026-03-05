"""
period_predictor.py
===================
User provides only period start + end dates (2+ months).
Model predicts:
  - Next period date
  - Ovulation day
  - Fertile window
  - Luteal phase
  - PMS window
  - Cycle regularity score
  - Irregularity flags

Run standalone:
    python period_predictor.py
"""

import numpy as np
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


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
            raise ValueError(f"Period duration {self.duration} days seems too long (>15)")


@dataclass
class CyclePhases:
    """All predicted phases for one upcoming cycle."""
    cycle_number:         int
    period_start:         date
    period_end:           date
    follicular_start:     date   # day after period ends
    follicular_end:       date   # day before fertile window
    fertile_start:        date
    fertile_end:          date
    ovulation_day:        date
    luteal_start:         date
    luteal_end:           date
    pms_start:            date   # ~5 days before next period
    predicted_length:     float
    uncertainty_days:     float

    def to_dict(self) -> dict:
        return {
            "cycle_number":           self.cycle_number,
            "predicted_cycle_length": round(self.predicted_length, 1),
            "uncertainty":            f"±{round(self.uncertainty_days, 1)} days",
            "phases": {
                "menstruation": {
                    "start": str(self.period_start),
                    "end":   str(self.period_end),
                    "duration_days": (self.period_end - self.period_start).days + 1,
                    "description": "Menstrual bleeding phase",
                },
                "follicular": {
                    "start": str(self.follicular_start),
                    "end":   str(self.follicular_end),
                    "duration_days": (self.follicular_end - self.follicular_start).days + 1,
                    "description": "Follicles develop, estrogen rises",
                },
                "fertile_window": {
                    "start": str(self.fertile_start),
                    "end":   str(self.fertile_end),
                    "duration_days": (self.fertile_end - self.fertile_start).days + 1,
                    "description": "Highest chance of conception",
                },
                "ovulation": {
                    "day":         str(self.ovulation_day),
                    "description": "Egg released — peak fertility",
                },
                "luteal": {
                    "start": str(self.luteal_start),
                    "end":   str(self.luteal_end),
                    "duration_days": (self.luteal_end - self.luteal_start).days + 1,
                    "description": "Progesterone rises, body prepares for period",
                },
                "pms_window": {
                    "start": str(self.pms_start),
                    "end":   str(self.luteal_end),   # luteal_end = day before next period
                    "description": "Possible PMS symptoms (bloating, mood changes)",
                },
            }
        }


# ══════════════════════════════════════════════════════════════════════════════
# CORE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class PeriodDatePredictor:
    """
    Takes raw period start/end dates → predicts all future cycle phases.

    Algorithm:
      1. Compute cycle lengths from consecutive period starts
      2. Compute average period duration
      3. Detect irregularity pattern → choose prediction strategy
      4. Predict next cycle length with confidence interval
      5. Compute all 6 phases for next N cycles
    """

    LUTEAL_PHASE_DAYS = 14    # Luteal phase is remarkably constant (~14d)
    PMS_DAYS_BEFORE   = 5     # PMS typically starts 5d before period
    FERTILE_DAYS_PRE  = 5     # Fertile window: 5 days before ovulation
    FERTILE_DAYS_POST = 1     # + 1 day after ovulation

    def __init__(self, periods: List[PeriodEntry]):
        if len(periods) < 2:
            raise ValueError("Need at least 2 period entries to predict.")

        # Sort by start date
        self.periods = sorted(periods, key=lambda p: p.start)
        self._compute_cycles()

    def _compute_cycles(self):
        """Derive cycle lengths and period durations from raw dates."""
        self.cycle_lengths: List[float] = []
        self.period_durations: List[float] = []

        for i in range(len(self.periods) - 1):
            cl = (self.periods[i+1].start - self.periods[i].start).days
            self.cycle_lengths.append(float(cl))

        for p in self.periods:
            self.period_durations.append(float(p.duration))

        self.mean_cycle    = float(np.mean(self.cycle_lengths))
        self.std_cycle     = float(np.std(self.cycle_lengths)) if len(self.cycle_lengths) > 1 else 2.0
        self.mean_duration = float(np.mean(self.period_durations))

    # ── Prediction strategy ────────────────────────────────────────────────
    def _predict_cycle_length(self) -> Tuple[float, float]:
        """
        Returns (predicted_length, uncertainty_days).
        Strategy chosen based on cycle variability.
        """
        cl = np.array(self.cycle_lengths)

        if len(cl) == 1:
            # Only 2 periods given → use that one cycle length
            return float(cl[0]), 3.0

        cv = self.std_cycle / self.mean_cycle

        if cv < 0.05:
            # Very regular — simple weighted average
            weights  = np.arange(1, len(cl) + 1, dtype=float)
            pred     = float(np.average(cl, weights=weights))
            uncertainty = max(1.5, self.std_cycle)

        elif cv < 0.15:
            # Moderately regular — EWMA (recent cycles matter more)
            alpha = 0.4
            ewma  = cl[0]
            for c in cl[1:]:
                ewma = alpha * c + (1 - alpha) * ewma
            pred        = float(ewma)
            uncertainty = max(2.0, self.std_cycle)

        else:
            # Irregular — robust median + wider interval
            pred        = float(np.median(cl))
            uncertainty = max(3.5, self.std_cycle * 1.5)

        pred = float(np.clip(pred, 21, 45))
        return pred, uncertainty

    # ── Phase calculation ──────────────────────────────────────────────────
    def _compute_phases(self, period_start: date,
                        cycle_length: float,
                        avg_duration: float,
                        cycle_number: int,
                        uncertainty: float) -> CyclePhases:

        period_end      = period_start + timedelta(days=round(avg_duration) - 1)
        ovulation_day   = period_start + timedelta(days=round(cycle_length - self.LUTEAL_PHASE_DAYS))
        fertile_start   = ovulation_day - timedelta(days=self.FERTILE_DAYS_PRE)
        fertile_end     = ovulation_day + timedelta(days=self.FERTILE_DAYS_POST)
        follicular_start= period_end + timedelta(days=1)
        follicular_end  = fertile_start - timedelta(days=1)
        luteal_start    = ovulation_day + timedelta(days=1)
        next_period     = period_start + timedelta(days=round(cycle_length))
        luteal_end      = next_period - timedelta(days=1)
        pms_start       = next_period - timedelta(days=self.PMS_DAYS_BEFORE)

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
            predicted_length = cycle_length,
            uncertainty_days = uncertainty,
        )

    # ── Irregularity check ─────────────────────────────────────────────────
    def _check_irregularity(self) -> Dict:
        cl    = np.array(self.cycle_lengths)
        flags = []

        if np.any(cl > 35):    flags.append("OLIGOMENORRHEA — cycles longer than 35 days")
        if np.any(cl < 21):    flags.append("POLYMENORRHEA — cycles shorter than 21 days")
        if np.any(cl > 90):    flags.append("AMENORRHEA — possible missed period (>90 days)")
        if self.std_cycle > 8: flags.append("HIGH_VARIABILITY — unpredictable cycle lengths")
        if len(cl) >= 3:
            slope = float(np.polyfit(range(len(cl)), cl, 1)[0])
            if slope > 1.0:    flags.append("TRENDING_LONGER — cycles getting progressively longer")
            if slope < -1.0:   flags.append("TRENDING_SHORTER — cycles getting progressively shorter")

        cv    = self.std_cycle / self.mean_cycle * 100
        score = min(100.0, len(flags) * 20 + self.std_cycle * 2)

        return {
            "is_irregular":       score > 20,
            "irregularity_score": round(score, 1),
            "flags":              flags,
            "severity":           "severe"   if score >= 60
                                  else "moderate" if score >= 35
                                  else "mild"     if score >  20
                                  else "none",
            "cv_percent":         round(cv, 1),
        }

    # ── Main predict method ────────────────────────────────────────────────
    def predict(self, num_cycles: int = 3) -> Dict:
        pred_length, uncertainty = self._predict_cycle_length()
        last_period              = self.periods[-1]
        irregularity             = self._check_irregularity()

        # Generate upcoming cycles
        upcoming = []
        current_start = last_period.start

        for i in range(num_cycles):
            current_start = current_start + timedelta(days=round(pred_length))
            phases        = self._compute_phases(
                current_start, pred_length,
                self.mean_duration, i + 1, uncertainty
            )
            upcoming.append(phases.to_dict())

        # Days until next period
        today          = date.today()
        next_start     = date.fromisoformat(upcoming[0]["phases"]["menstruation"]["start"])
        days_until     = (next_start - today).days

        return {
            "summary": {
                "cycles_analysed":       len(self.cycle_lengths),
                "average_cycle_length":  round(self.mean_cycle, 1),
                "average_period_duration": round(self.mean_duration, 1),
                "cycle_std_dev":         round(self.std_cycle, 1),
                "prediction_strategy":   self._strategy_name(),
                "confidence":            "high"   if len(self.cycle_lengths) >= 6
                                         else "medium" if len(self.cycle_lengths) >= 3
                                         else "low",
            },
            "next_period": {
                "predicted_start": str(next_start),
                "predicted_end":   upcoming[0]["phases"]["menstruation"]["end"],
                "days_away":       days_until,
                "message":         self._days_message(days_until),
                "uncertainty":     f"±{round(uncertainty)} days",
            },
            "irregularity":  irregularity,
            "upcoming_cycles": upcoming,
            "disclaimer": (
                "Predictions are estimates based on your cycle history. "
                "Accuracy improves with more months of data. "
                "Not suitable for use as contraception."
            )
        }

    def _strategy_name(self) -> str:
        cl = np.array(self.cycle_lengths)
        cv = self.std_cycle / self.mean_cycle
        if len(cl) == 1:   return "single_cycle_baseline"
        if cv < 0.05:      return "weighted_average (very regular)"
        if cv < 0.15:      return "ewma (moderately regular)"
        return "robust_median (irregular)"

    def _days_message(self, days: int) -> str:
        if days < 0:     return f"Period was expected {abs(days)} days ago — may be late"
        if days == 0:    return "Period expected today"
        if days == 1:    return "Period expected tomorrow"
        if days <= 3:    return f"Period in {days} days — prepare soon"
        if days <= 7:    return f"Period in {days} days — within this week"
        return f"Period in {days} days"


# ══════════════════════════════════════════════════════════════════════════════
# PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_prediction(result: dict):
    s    = result["summary"]
    nxt  = result["next_period"]
    irr  = result["irregularity"]

    print("\n" + "╔" + "═"*58 + "╗")
    print("║        🌸  MENSTRUAL CYCLE PREDICTION REPORT         ║")
    print("╚" + "═"*58 + "╝")

    print(f"\n📊 YOUR CYCLE SUMMARY")
    print(f"   Cycles analysed      : {s['cycles_analysed']}")
    print(f"   Average cycle length : {s['average_cycle_length']} days")
    print(f"   Average period length: {s['average_period_duration']} days")
    print(f"   Cycle variability    : ±{s['cycle_std_dev']} days")
    print(f"   Strategy used        : {s['prediction_strategy']}")
    print(f"   Prediction confidence: {s['confidence'].upper()}")

    print(f"\n🩸 NEXT PERIOD")
    print(f"   Start date  : {nxt['predicted_start']}")
    print(f"   End date    : {nxt['predicted_end']}")
    print(f"   Uncertainty : {nxt['uncertainty']}")
    print(f"   ➤  {nxt['message']}")

    print(f"\n⚠️  IRREGULARITY CHECK")
    sev_icon = {"none": "✅", "mild": "🟡", "moderate": "🟠", "severe": "🔴"}
    print(f"   Status : {sev_icon.get(irr['severity'], '⚪')} {irr['severity'].upper()}"
          f"  (score {irr['irregularity_score']}/100)")
    if irr["flags"]:
        for flag in irr["flags"]:
            print(f"   • {flag}")
    else:
        print("   No irregularities detected.")

    for cycle in result["upcoming_cycles"]:
        p = cycle["phases"]
        print(f"\n{'─'*60}")
        print(f"  📅 CYCLE {cycle['cycle_number']}  "
              f"(~{cycle['predicted_cycle_length']}d  {cycle['uncertainty']})")
        print(f"{'─'*60}")
        print(f"  🩸 Menstruation  : {p['menstruation']['start']} → {p['menstruation']['end']}"
              f"  ({p['menstruation']['duration_days']}d)")
        print(f"  🌱 Follicular    : {p['follicular']['start']} → {p['follicular']['end']}"
              f"  ({p['follicular']['duration_days']}d)")
        print(f"  💚 Fertile window: {p['fertile_window']['start']} → {p['fertile_window']['end']}"
              f"  ({p['fertile_window']['duration_days']}d)")
        print(f"  🥚 Ovulation     : {p['ovulation']['day']}  ← peak fertility")
        print(f"  🌙 Luteal phase  : {p['luteal']['start']} → {p['luteal']['end']}"
              f"  ({p['luteal']['duration_days']}d)")
        print(f"  😔 PMS window    : {p['pms_window']['start']} → {p['pms_window']['end']}")

    print(f"\n{'─'*60}")
    print(f"  ⚕️  {result['disclaimer']}")
    print(f"{'─'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO — run directly
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  DEMO 1: Regular cycles (6 months)")
    print("="*60)
    periods_regular = [
        PeriodEntry(date(2025, 6,  3), date(2025, 6,  7)),
        PeriodEntry(date(2025, 7,  1), date(2025, 7,  5)),
        PeriodEntry(date(2025, 7, 29), date(2025, 8,  3)),
        PeriodEntry(date(2025, 8, 27), date(2025, 9,  1)),
        PeriodEntry(date(2025, 9, 24), date(2025, 9, 28)),
        PeriodEntry(date(2025,10, 22), date(2025,10, 27)),
    ]
    predictor = PeriodDatePredictor(periods_regular)
    result    = predictor.predict(num_cycles=3)
    print_prediction(result)

    print("\n" + "="*60)
    print("  DEMO 2: Minimum input — just 2 months")
    print("="*60)
    periods_minimal = [
        PeriodEntry(date(2025, 10, 5),  date(2025, 10, 9)),
        PeriodEntry(date(2025, 11, 2),  date(2025, 11, 6)),
    ]
    predictor2 = PeriodDatePredictor(periods_minimal)
    result2    = predictor2.predict(num_cycles=3)
    print_prediction(result2)

    print("\n" + "="*60)
    print("  DEMO 3: Irregular cycles (PCOS-like)")
    print("="*60)
    periods_irregular = [
        PeriodEntry(date(2025, 3,  1), date(2025, 3,  7)),
        PeriodEntry(date(2025, 4, 15), date(2025, 4, 22)),
        PeriodEntry(date(2025, 6,  5), date(2025, 6, 12)),
        PeriodEntry(date(2025, 7, 25), date(2025, 8,  1)),
    ]
    predictor3 = PeriodDatePredictor(periods_irregular)
    result3    = predictor3.predict(num_cycles=3)
    print_prediction(result3)