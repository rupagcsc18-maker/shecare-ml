"""
Generates a comprehensive clinical-style irregularity + condition report.
"""
from typing import List, Dict
from irregular_detector import IrregularCycleDetector, IrregularityReport
from condition_classifier import ConditionClassifier, ConditionResult
from adaptive_predictor import AdaptiveCyclePredictor


class MenstrualHealthAnalyzer:
    """One-stop analyzer: irregularity + condition + adaptive prediction."""

    def __init__(self):
        self.detector    = IrregularCycleDetector()
        self.classifier  = ConditionClassifier()

    def analyze(self,
                cycle_lengths: List[float],
                period_durations: List[float],
                symptoms: Dict[str, List[float]],
                age: int,
                last_period_start=None,
                bbt_data: List[float] = None,
                spotting_flags: List[bool] = None) -> Dict:

        # 1. Detect irregularities
        irr_report: IrregularityReport = self.detector.detect(
            cycle_lengths, period_durations, bbt_data, spotting_flags
        )

        # 2. Classify conditions
        conditions: List[ConditionResult] = self.classifier.classify_rule_based(
            cycle_lengths, period_durations, symptoms, age, bbt_data
        )

        # 3. Adaptive prediction
        from datetime import date
        last_start = last_period_start or date.today()
        avg_dur    = sum(period_durations) / len(period_durations)
        predictor  = AdaptiveCyclePredictor(irr_report, conditions)
        windows    = predictor.predict_windows(
            cycle_lengths, last_start, avg_dur, num_cycles=3
        )
        pred_result = predictor.predict(cycle_lengths, avg_dur)

        return {
            "irregularity":  irr_report,
            "conditions":    conditions,
            "prediction":    pred_result,
            "windows":       windows,
        }

    def print_full_report(self, result: Dict):
        irr   = result["irregularity"]
        conds = result["conditions"]
        pred  = result["prediction"]
        wins  = result["windows"]

        print("\n" + "╔" + "═"*55 + "╗")
        print("║       MENSTRUAL HEALTH ANALYSIS REPORT          ║")
        print("╚" + "═"*55 + "╝")

        # ── Irregularity summary ──────────────────────────────────────────
        print(f"\n{'─'*55}")
        print(f" IRREGULARITY SCORE: {irr.irregularity_score:.0f}/100  "
              f"[{irr.overall_severity.upper()}]")
        print(f"{'─'*55}")

        if not irr.flags:
            print("  ✅ No significant irregularities detected.")
        else:
            for flag in irr.flags:
                icon = {"mild": "🟡", "moderate": "🟠", "severe": "🔴"}[flag.severity]
                print(f"\n  {icon} {flag.flag_type}")
                print(f"     {flag.description}")
                print(f"     💡 {flag.recommendation}")

        # ── Condition probabilities ───────────────────────────────────────
        print(f"\n{'─'*55}")
        print(" POSSIBLE CONDITIONS")
        print(f"{'─'*55}")
        for r in conds:
            if r.probability < 0.10:
                continue
            bar = "█" * int(r.probability * 20)
            print(f"\n  {r.label}")
            print(f"  Probability: {bar} {r.probability*100:.0f}%")
            if r.supporting_evidence:
                for ev in r.supporting_evidence:
                    print(f"    • {ev}")
            print(f"  📋 {r.recommendation}")

        # ── Adaptive prediction ───────────────────────────────────────────
        print(f"\n{'─'*55}")
        print(" NEXT CYCLE PREDICTION  (adaptive)")
        print(f"{'─'*55}")
        print(f"  Strategy          : {pred['strategy']}")
        print(f"  Predicted length  : {pred['predicted_cycle_length']} days")
        print(f"  95% CI            : {pred['confidence_interval']}")
        if pred.get("note"):
            print(f"  ℹ️  {pred['note']}")

        print(f"\n{'─'*55}")
        print(" PREDICTED WINDOWS (next 3 cycles)")
        print(f"{'─'*55}")
        for w in wins:
            print(f"\n  Cycle {w['cycle_number']}  "
                  f"(~{w['predicted_length']}d  {w['uncertainty']}  CI: {w['ci']})")
            print(f"    🩸 Period    : {w['period_start']} → {w['period_end']}")
            print(f"    🌱 Fertile  : {w['fertile_start']} → {w['fertile_end']}")
            print(f"    🥚 Ovulation: {w['ovulation_day']}")

        print(f"\n{'═'*55}")
        print("  ⚠️  This tool is for informational purposes only.")
        print("  Always consult a healthcare professional for diagnosis.")
        print(f"{'═'*55}\n")