"""
Tests for AdaptiveCyclePredictor — verifies correct strategy selection
and that predictions stay within medically plausible bounds.
"""
import unittest
from datetime import date
from irregular_detector import IrregularCycleDetector
from condition_classifier import ConditionClassifier
from adaptive_predictor import AdaptiveCyclePredictor


def build_predictor(cycles, durations, symptoms, age):
    detector = IrregularCycleDetector()
    clf      = ConditionClassifier()
    report   = detector.detect(cycles, durations)
    conds    = clf.classify_rule_based(cycles, durations, symptoms, age)
    return AdaptiveCyclePredictor(report, conds)


class TestAdaptivePredictor(unittest.TestCase):

    # ── 1. PCOS → robust strategy ─────────────────────────────────────────────
    def test_pcos_uses_robust_strategy(self):
        cycles   = [42, 55, 38, 60, 45, 50, 48, 62]
        durations= [6]*8
        pred = build_predictor(cycles, durations,
                               {"acne": [4]*8, "hirsutism": [3]*8}, age=26)
        self.assertEqual(pred.strategy, "pcos_robust",
            f"PCOS profile should use robust strategy, got '{pred.strategy}'")
        print(f"✅ PASS: PCOS → strategy='{pred.strategy}'")

    # ── 2. Perimenopause → Kalman strategy ───────────────────────────────────
    def test_perimenopause_uses_kalman(self):
        cycles   = [28, 30, 33, 37, 41, 45, 48, 50, 52, 55]
        durations= [5, 5, 6, 6, 7, 7, 7, 8, 8, 8]
        pred = build_predictor(cycles, durations,
                               {"hot_flashes": [4]*10, "night_sweats": [3]*10}, age=49)
        self.assertEqual(pred.strategy, "kalman",
            f"Perimenopause should use kalman strategy, got '{pred.strategy}'")
        print(f"✅ PASS: Perimenopause → strategy='{pred.strategy}'")

    # ── 3. Stress → EWMA strategy ────────────────────────────────────────────
    def test_stress_uses_ewma(self):
        cycles   = [28, 27, 29, 28, 43, 45, 38, 30, 29, 28]
        durations= [5]*10
        pred = build_predictor(cycles, durations,
                               {"stress_level": [2,2,2,2,5,5,4,3,2,2],
                                "anxiety":      [1,1,1,2,5,5,4,2,1,1]}, age=30)
        self.assertEqual(pred.strategy, "ewma",
            f"Stress profile should use ewma strategy, got '{pred.strategy}'")
        print(f"✅ PASS: Stress → strategy='{pred.strategy}'")

    # ── 4. Regular cycles → standard strategy ────────────────────────────────
    def test_regular_uses_standard_strategy(self):
        cycles   = [28, 27, 29, 28, 28, 29, 27, 28]
        durations= [5]*8
        pred = build_predictor(cycles, durations, {}, age=25)
        self.assertEqual(pred.strategy, "standard",
            f"Regular cycles should use standard strategy, got '{pred.strategy}'")
        print(f"✅ PASS: Regular → strategy='{pred.strategy}'")

    # ── 5. All predictions within medical bounds (15–90 days) ────────────────
    def test_prediction_within_medical_bounds(self):
        test_cases = [
            ([28]*8,             "regular"),
            ([42, 55, 38, 60]*2, "PCOS"),
            ([28,30,35,40,45,50],"perimenopause"),
            ([28,27,28,45,44,29],"stress"),
        ]
        for cycles, label in test_cases:
            durations = [5] * len(cycles)
            pred = build_predictor(cycles, durations, {}, age=30)
            result = pred.predict(cycles)
            length = result["predicted_cycle_length"]
            self.assertGreaterEqual(length, 15,
                f"{label}: prediction {length} below minimum 15 days")
            self.assertLessEqual(length, 90,
                f"{label}: prediction {length} exceeds maximum 90 days")
            print(f"✅ PASS: {label} prediction = {length}d (within 15–90)")

    # ── 6. Confidence intervals are valid ─────────────────────────────────────
    def test_confidence_intervals_valid(self):
        cycles   = [42, 55, 38, 60, 45, 50]
        durations= [6]*6
        pred = build_predictor(cycles, durations, {"acne":[4]*6}, age=26)
        result = pred.predict(cycles)
        self.assertLessEqual(result["ci_95_low"], result["predicted_cycle_length"],
            "CI lower bound should be ≤ prediction")
        self.assertGreaterEqual(result["ci_95_high"], result["predicted_cycle_length"],
            "CI upper bound should be ≥ prediction")
        print(f"✅ PASS: CI valid → {result['ci_95_low']}–"
              f"{result['predicted_cycle_length']}–{result['ci_95_high']}")

    # ── 7. Window outputs are chronologically ordered ─────────────────────────
    def test_windows_chronological_order(self):
        cycles   = [28]*8
        durations= [5]*8
        pred = build_predictor(cycles, durations, {}, age=28)
        windows = pred.predict_windows(cycles, date(2025, 12, 1))
        for i in range(1, len(windows)):
            self.assertGreater(
                windows[i]["period_start"],
                windows[i-1]["period_start"],
                "Windows should be in chronological order"
            )
        print("✅ PASS: Windows are chronologically ordered")

    # ── 8. Amenorrhea gets wide uncertainty ───────────────────────────────────
    def test_amenorrhea_wide_uncertainty(self):
        cycles   = [28, 29, 95, 28, 100]
        durations= [5]*5
        pred = build_predictor(cycles, durations, {}, age=30)
        result = pred.predict(cycles)
        self.assertGreaterEqual(result["uncertainty_days"], 10,
            f"Amenorrhea should have wide uncertainty, got {result['uncertainty_days']}")
        print(f"✅ PASS: Amenorrhea uncertainty = ±{result['uncertainty_days']}d")


if __name__ == "__main__":
    unittest.main(verbosity=2)