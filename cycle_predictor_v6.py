"""
cycle_predictor_v6.py — Category-leader final layer.

Upgrades on top of v5 (zero regression):

  1. Probabilistic ovulation distribution
  2. Actionable uncertainty
  3. Medical risk scoring
  4. Reliability horizon
  5. Active confidence alerts
  BONUS: Cycle Health Score
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from cycle_predictor_v2 import (
    CycleEngineResult, DailyLog, PeriodEntry,
    SIGMA_CALENDAR, _MUCUS_FERTILITY, CycleEngine as _BaseEngine,
)
from cycle_predictor_v4 import (
    contextual_confidence_v4, build_insights_v4, build_ui_labels_v4,
)
from cycle_predictor_v5 import (
    CycleEngineV5, build_education, compute_stability,
    confidence_trend, detect_medical_edge,
)


# ════════════════════════════════════════════════════════════════════════════
# 1. PROBABILISTIC OVULATION DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════

def ovulation_distribution(ov_day: date, ov_sigma: float, window: int = 4) -> Dict[str, float]:
    """
    Discretised Gaussian: {date_iso: probability}, sums to 1.0.
    sigma=1 → tight peak. sigma=3 → spread across 7+ days.
    """
    sigma = max(0.5, ov_sigma)
    raw = {(ov_day + timedelta(days=d)): math.exp(-0.5 * (d / sigma) ** 2)
           for d in range(-window, window + 1)}
    total = sum(raw.values())
    return {str(k): round(v / total, 4) for k, v in raw.items()}


def fertility_score_from_distribution(
    day: date,
    ov_distribution: Dict[str, float],
    cycle_sd: float,
    mucus: Optional[str] = None,
) -> Tuple[int, float]:
    """
    E[fertility_score] = Σ P(ov=d) * score(day | ov=d)
    Integrating over the distribution gives the correct expected score
    and properly propagates uncertainty.
    """
    from cycle_predictor_v5 import fertility_score_v5

    expected_score = 0.0
    expected_unc   = 0.0
    for d_str, prob in ov_distribution.items():
        ov_d = date.fromisoformat(d_str)
        s, u = fertility_score_v5(day, ov_d, cycle_sd, ov_sigma=0.0, mucus=mucus)
        expected_score += prob * s
        expected_unc   += prob * u

    # Extra uncertainty from the spread of the distribution itself
    days_spread = [
        abs((date.fromisoformat(d) - day).days) * p
        for d, p in ov_distribution.items()
    ]
    dist_boost = min(0.4, float(np.std(days_spread)) * 0.08)
    return int(round(expected_score)), round(min(1.0, expected_unc + dist_boost), 3)


# ════════════════════════════════════════════════════════════════════════════
# 2. ACTIONABLE UNCERTAINTY
# ════════════════════════════════════════════════════════════════════════════

def actionable_uncertainty(
    uncertainty: float,
    biosignal_types: List[str],
    ov_sigma: float,
    cycle_sd: float,
) -> Dict:
    if uncertainty <= 0.15:
        level  = "low"
        action = "Your predictions are well-anchored. Keep logging daily."
    elif uncertainty <= 0.35:
        level = "medium"
        if "lh" not in biosignal_types and "bbt" not in biosignal_types:
            action = "Log an LH test today to sharpen your ovulation timing."
        elif "bbt" not in biosignal_types:
            action = "Add daily BBT readings to confirm ovulation and reduce uncertainty."
        else:
            action = "Continue daily logging — uncertainty will reduce over the next 2–3 days."
    else:
        level = "high"
        if ov_sigma >= 3.0 and "lh" not in biosignal_types:
            action = "Ovulation timing is uncertain. Start LH testing daily from today."
        elif cycle_sd >= 5.0:
            action = "High cycle variability. Log every day and consider tracking BBT."
        else:
            action = "Predictions are shifting. Log BBT + LH for the next 5 days to stabilise."

    return {"value": round(uncertainty, 3), "level": level, "action": action}


# ════════════════════════════════════════════════════════════════════════════
# 3. MEDICAL RISK SCORING
# ════════════════════════════════════════════════════════════════════════════

def medical_risk_scores(
    luteal_lengths: List[float],
    past_ovulations_confirmed: List[date],
    biosignal_logged_cycles: int,
    n_cycles: int,
    cycle_sd: float,
    cycle_lengths: List[float],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}

    if luteal_lengths:
        avg_l       = float(np.mean(luteal_lengths))
        short_frac  = sum(1 for l in luteal_lengths if l < 10) / len(luteal_lengths)
        defect_score= max(0.0, (14.0 - avg_l) / 7.0) * 0.7 + short_frac * 0.3
        scores["luteal_phase_defect"] = round(min(1.0, defect_score), 3)

    if n_cycles >= 2:
        if biosignal_logged_cycles >= 2 and len(past_ovulations_confirmed) == 0:
            scores["possible_anovulation"] = round(
                min(1.0, 0.4 + 0.15 * min(4, biosignal_logged_cycles)), 3)
        elif n_cycles >= 4 and len(past_ovulations_confirmed) == 0:
            scores["possible_anovulation"] = 0.25
        else:
            scores["possible_anovulation"] = 0.0

    if len(cycle_lengths) >= 3:
        cl = np.array(cycle_lengths)
        pcos_score = min(1.0,
            float(np.mean(cl > 35)) * 0.5
            + min(1.0, cycle_sd / 10.0) * 0.3
            + (0.2 if float(np.mean(cl)) > 35 else 0.0))
        scores["cycle_irregularity_pcos_adjacent"] = round(pcos_score, 3)

    if len(luteal_lengths) >= 3:
        scores["luteal_length_variability"] = round(
            min(1.0, float(np.std(luteal_lengths)) / 5.0), 3)

    return scores


def risk_score_to_insight(code: str, score: float) -> Optional[Dict]:
    if score < 0.2:
        return None
    registry = {
        "luteal_phase_defect":              (0.5,  "warning", "Possible luteal phase defect. Consider consulting a clinician, especially if trying to conceive."),
        "possible_anovulation":             (0.35, "warning", "Possible anovulatory cycles. Confirm with LH testing."),
        "cycle_irregularity_pcos_adjacent": (0.5,  "warning", "Cycle pattern consistent with hormonal irregularity. Consider PCOS screening."),
        "luteal_length_variability":        (0.4,  "info",    "Luteal phase length varies — affects PMS timing predictability."),
    }
    if code not in registry:
        return None
    threshold, level, message = registry[code]
    if score >= threshold:
        return {"level": level, "code": code, "risk_score": score, "message": message}
    return None


# ════════════════════════════════════════════════════════════════════════════
# 4. RELIABILITY HORIZON
# ════════════════════════════════════════════════════════════════════════════

def reliability_horizon(
    ov_sigma: float,
    cycle_sd: float,
    n_biosignals: int,
    today: date,
    ov_day: date,
) -> Dict:
    days_to_ov    = max(0, (ov_day - today).days)
    base          = days_to_ov + 1
    sigma_penalty = int(ov_sigma * 1.5)
    sd_penalty    = int(cycle_sd * 0.8)
    bio_bonus     = n_biosignals * 2
    horizon       = max(1, min(21, base - sigma_penalty - sd_penalty + bio_bonus))

    if horizon >= 10:
        status, note = "reliable", f"Predictions solid for the next {horizon} days."
    elif horizon >= 5:
        status, note = "moderate", f"Predictions reasonable for ~{horizon} days. Log biosignals to extend."
    else:
        status, note = "limited",  f"Only {horizon} day(s) of reliable prediction. Start LH testing now."

    return {"days": horizon, "status": status, "note": note}


# ════════════════════════════════════════════════════════════════════════════
# 5. ACTIVE CONFIDENCE ALERTS
# ════════════════════════════════════════════════════════════════════════════

def active_confidence_alert(
    history: List[float],
    current_stability: str,
    n_cycles: int,
) -> Optional[Dict]:
    if len(history) < 3:
        return None

    trend = confidence_trend(history)
    drop  = history[-3] - history[-1] if len(history) >= 3 else 0

    if trend == "declining" and drop >= 0.10:
        return {
            "level":   "warning",
            "trigger": "declining_confidence",
            "message": (
                f"Prediction accuracy has declined over your last {min(5, len(history))} cycles. "
                "This often means your cycle is shifting — common during stress, "
                "weight changes, or hormonal transitions."
            ),
            "action": "Log BBT daily and consider reviewing recent lifestyle changes.",
        }
    if current_stability == "volatile" and n_cycles >= 4:
        return {
            "level":   "info",
            "trigger": "volatile_stability",
            "message": "Predictions have been shifting. Your cycle may be in transition.",
            "action":  "Daily LH testing during your expected fertile window will anchor predictions.",
        }
    return None


# ════════════════════════════════════════════════════════════════════════════
# BONUS: CYCLE HEALTH SCORE
# ════════════════════════════════════════════════════════════════════════════

def cycle_health_score(
    cycle_sd: float,
    n_confirmed_ovulations: int,
    n_cycles: int,
    luteal_lengths: List[float],
    n_biosignals: int,
    irregularity_flags: List[str],
) -> Dict:
    """
    0–100 composite. Four pillars of 25pts each:
      Regularity | Ovulation | Luteal health | Data completeness
    """
    # Regularity
    regularity  = max(0.0, 1.0 - cycle_sd / 8.0) * 25.0
    bad_flags   = {"OLIGOMENORRHEA", "POLYMENORRHEA", "AMENORRHEA", "HIGH_VARIABILITY"}
    regularity  = max(0.0, regularity - sum(4 for f in irregularity_flags if f in bad_flags))

    # Ovulation
    if n_cycles == 0:
        ovulation = 0.0
    else:
        confirm_rate = min(1.0, n_confirmed_ovulations / max(1, n_cycles))
        ovulation    = min(25.0, confirm_rate * 20.0 + min(5.0, n_biosignals * 1.5))

    # Luteal health
    if not luteal_lengths:
        luteal = 12.5
    else:
        avg_l  = float(np.mean(luteal_lengths))
        l_sd   = float(np.std(luteal_lengths)) if len(luteal_lengths) >= 2 else 0.0
        luteal = min(25.0,
            max(0.0, 1.0 - abs(avg_l - 13.0) / 6.0) * 20.0
            + max(0.0, 1.0 - l_sd / 4.0) * 5.0)

    # Data completeness
    data = min(25.0,
        min(1.0, n_cycles / 6.0) * 15.0
        + min(1.0, n_biosignals / 3.0) * 10.0)

    total = max(0, min(100, int(round(regularity + ovulation + luteal + data))))

    if   total >= 85: grade, desc = "Excellent",        "Strong regularity and good biosignal coverage."
    elif total >= 70: grade, desc = "Good",             "Solid cycle health — more biosignal logging will improve this."
    elif total >= 50: grade, desc = "Fair",             "Some irregularity detected. Consistent logging will improve this score."
    else:             grade, desc = "Needs attention",  "Significant irregularity or limited data. Consider a clinician review."

    return {
        "score": total, "grade": grade, "description": desc,
        "breakdown": {
            "regularity":        round(regularity, 1),
            "ovulation":         round(ovulation, 1),
            "luteal_health":     round(luteal, 1),
            "data_completeness": round(data, 1),
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# ENGINE v6
# ════════════════════════════════════════════════════════════════════════════

class CycleEngineV6(CycleEngineV5):

    def predict(self, num_cycles: int = 3) -> CycleEngineResult:
        result = _BaseEngine.predict(self, num_cycles=num_cycles)

        cur          = result.current_cycle
        ov_day       = date.fromisoformat(cur["phases"]["ovulation"]["day"])
        ov_sigma     = float(cur["phases"]["ovulation"]["sigma_days"])
        ov_source    = cur["phases"]["ovulation"].get("source", "calendar")
        period_start = date.fromisoformat(cur["phases"]["menstruation"]["start"])
        period_end   = date.fromisoformat(cur["phases"]["menstruation"]["end"])
        cycle_len    = float(cur["predicted_cycle_length"])
        next_start   = period_start + timedelta(days=int(round(cycle_len)))
        cycle_sd     = float(result.summary["cycle_sd_days"])
        cycle_lengths= [float((self.periods[i+1].start - self.periods[i].start).days)
                        for i in range(len(self.periods) - 1)]
        mucus_by_day = {l.day: l.mucus for l in self.logs if l.mucus}

        # 1. Ovulation distribution
        ov_dist = ovulation_distribution(ov_day, ov_sigma)
        cur["phases"]["ovulation"]["distribution"] = ov_dist

        # Fertility curve — integrates over distribution
        curve: List[Dict] = []
        d = period_start
        while d < next_start:
            score, unc = fertility_score_from_distribution(d, ov_dist, cycle_sd, mucus_by_day.get(d))
            if period_start <= d <= period_end:
                score, unc = 0, 0.0
            curve.append({"date": str(d), "score": score, "uncertainty": unc})
            d += timedelta(days=1)

        cur["fertility_curve"]    = curve
        cur["peak_fertility_day"] = max(curve, key=lambda x: x["score"])["date"]

        # Biosignals
        biosignal_types: List[str] = []
        for ev in cur["phases"]["ovulation"].get("evidence_chain", []):
            src = ev.get("source", "")
            if src in ("bbt", "lh", "mucus") and src not in biosignal_types:
                biosignal_types.append(src)
        n_bio = len(biosignal_types)

        has_bio = any(
            l.bbt_c is not None or l.lh_test
            or (l.mucus in ("egg_white","watery") if l.mucus else False)
            for l in self.logs)
        self._biosignal_logged_cycles = max(self._biosignal_logged_cycles, int(has_bio))

        # Confidence
        ctx = contextual_confidence_v4(
            cycle_sd=cycle_sd, ov_sigma=ov_sigma,
            n_cycles=int(result.summary["cycles_analysed"]),
            n_biosignal_sources=n_bio, biosignal_types=biosignal_types,
        )
        self.confidence_history.append(ctx["final"])
        result.summary["contextual_confidence"] = ctx
        result.summary["user_bias_days"] = round(self.user_bias_days, 2)

        # Stability
        if self._prev_ov_day_v5 is not None and ov_day != self._prev_ov_day_v5:
            self._prediction_delta_history.append(abs((ov_day - self._prev_ov_day_v5).days))
        self._prev_ov_day_v5 = ov_day
        stability = compute_stability(self._prediction_delta_history, ov_sigma, n_bio)
        result.summary["prediction_stability"] = stability

        result.summary["confidence_history"] = {
            "scores": [round(s, 3) for s in self.confidence_history[-5:]],
            "trend":  confidence_trend(self.confidence_history),
        }

        luteal_lengths = self._compute_luteal_lengths()

        # 3. Medical risk scoring
        risk_scores = medical_risk_scores(
            luteal_lengths=luteal_lengths,
            past_ovulations_confirmed=self._past_ovulations,
            biosignal_logged_cycles=self._biosignal_logged_cycles,
            n_cycles=int(result.summary["cycles_analysed"]),
            cycle_sd=cycle_sd, cycle_lengths=cycle_lengths,
        )
        result.summary["medical_risk_scores"] = risk_scores

        # Merge all insight sources
        base_insights = build_insights_v4(
            irregularity=result.irregularity,
            n_confirmed_ov=len(self._past_ovulations),
            n_cycles=int(result.summary["cycles_analysed"]),
            cycle_sd=cycle_sd,
        )
        medical_edges = detect_medical_edge(
            luteal_lengths=luteal_lengths,
            past_ovulations_confirmed=self._past_ovulations,
            biosignal_logged_cycles=self._biosignal_logged_cycles,
            n_cycles=int(result.summary["cycles_analysed"]),
        )
        risk_insights = [
            ins for code, score in risk_scores.items()
            for ins in [risk_score_to_insight(code, score)] if ins
        ]
        result.summary["insights"] = risk_insights + medical_edges + base_insights

        # 2. Actionable uncertainty
        today_entry  = next((c for c in curve if c["date"] == str(self.today)), None)
        today_score  = today_entry["score"] if today_entry else 0
        today_unc    = today_entry["uncertainty"] if today_entry else 0.0
        result.summary["uncertainty"] = actionable_uncertainty(
            today_unc, biosignal_types, ov_sigma, cycle_sd)

        # 4. Reliability horizon
        result.summary["reliability_horizon"] = reliability_horizon(
            ov_sigma, cycle_sd, n_bio, self.today, ov_day)

        # 5. Active confidence alert
        result.summary["confidence_alert"] = active_confidence_alert(
            self.confidence_history, stability,
            int(result.summary["cycles_analysed"]))

        # Cycle health score
        result.summary["cycle_health_score"] = cycle_health_score(
            cycle_sd=cycle_sd,
            n_confirmed_ovulations=len(self._past_ovulations),
            n_cycles=int(result.summary["cycles_analysed"]),
            luteal_lengths=luteal_lengths, n_biosignals=n_bio,
            irregularity_flags=result.irregularity.get("flags", []),
        )

        # Prediction update
        changed = False
        update_reason = "No change"
        if self._prev_ov_day is not None and ov_day != self._prev_ov_day:
            delta     = (ov_day - self._prev_ov_day).days
            direction = "later" if delta > 0 else "earlier"
            changed   = True
            update_reason = f"Ovulation moved {abs(delta)}d {direction} based on new {ov_source} signal"
        self._prev_ov_day = ov_day
        result.prediction_update = {"changed": changed, "reason": update_reason}

        # UI labels + education
        result.summary["ui_labels"] = build_ui_labels_v4(
            phase=result.current_phase["phase"], today_score=today_score)
        result.summary["ui_labels"]["today_uncertainty"] = today_unc
        result.summary["education"] = build_education(
            phase=result.current_phase["phase"],
            ov_source=ov_source, prediction_changed=changed, stability=stability,
        )

        self._cached_result = result
        return result


# ════════════════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    periods = [
        PeriodEntry(date(2025, 6,  3), date(2025, 6,  7)),
        PeriodEntry(date(2025, 7,  1), date(2025, 7,  5)),
        PeriodEntry(date(2025, 7, 29), date(2025, 8,  3)),
        PeriodEntry(date(2025, 8, 27), date(2025, 9,  1)),
        PeriodEntry(date(2025, 9, 24), date(2025, 9, 28)),
        PeriodEntry(date(2025, 10, 22), date(2025, 10, 27)),
    ]
    logs = [
        DailyLog(date(2025, 10, 2), bbt_c=36.40),
        DailyLog(date(2025, 10, 3), bbt_c=36.42),
        DailyLog(date(2025, 10, 6), bbt_c=36.39),
        DailyLog(date(2025, 10, 8), bbt_c=36.70, mucus="egg_white", lh_test=True),
        DailyLog(date(2025, 10, 9), bbt_c=36.72),
    ]

    eng = CycleEngineV6(periods, logs=logs, today=date(2025, 11, 4))
    r   = eng.predict(2)

    print("\n" + "═" * 70)
    print("  v6 DEMO")
    print("═" * 70)
    out = {
        "cycle_health_score":      r.summary["cycle_health_score"],
        "reliability_horizon":     r.summary["reliability_horizon"],
        "uncertainty":             r.summary["uncertainty"],
        "medical_risk_scores":     r.summary["medical_risk_scores"],
        "confidence_alert":        r.summary["confidence_alert"],
        "ovulation_distribution":  r.current_cycle["phases"]["ovulation"]["distribution"],
        "peak_fertility_day":      r.current_cycle["peak_fertility_day"],
        "top_curve":               sorted(r.current_cycle["fertility_curve"],
                                          key=lambda x: x["score"], reverse=True)[:5],
        "prediction_stability":    r.summary["prediction_stability"],
        "top_insights":            r.summary["insights"][:3],
        "ui_labels":               r.summary["ui_labels"],
    }
    print(json.dumps(out, indent=2, default=str))