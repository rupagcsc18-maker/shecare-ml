"""
Tests for ConditionClassifier.
Each test verifies the TOP predicted condition matches expected diagnosis.
"""
import unittest
from condition_classifier import ConditionClassifier


class TestConditionClassifier(unittest.TestCase):

    def setUp(self):
        self.clf = ConditionClassifier()

    def _top_condition(self, results):
        return max(results, key=lambda r: r.probability).condition

    # ── 1. Clear PCOS profile ─────────────────────────────────────────────────
    def test_pcos_detected(self):
        results = self.clf.classify_rule_based(
            cycle_lengths    = [42, 55, 38, 60, 45, 50, 48, 44, 52, 46],
            period_durations = [6, 7, 5, 8, 6, 7, 5, 6, 7, 6],
            symptoms={
                "acne":        [4, 4, 3, 5, 4, 4, 3, 5, 4, 4],
                "weight_gain": [3, 3, 3, 4, 3, 3, 3, 4, 3, 3],
                "hirsutism":   [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            },
            age=26,
        )
        top = self._top_condition(results)
        pcos_prob = next(r.probability for r in results if r.condition == "PCOS")
        self.assertEqual(top, "PCOS",
            f"Expected PCOS as top condition, got {top}")
        self.assertGreater(pcos_prob, 0.5,
            f"PCOS probability should be >50%, got {pcos_prob:.0%}")
        print(f"✅ PASS: PCOS detected (prob={pcos_prob:.0%})")

    # ── 2. Clear perimenopause profile ───────────────────────────────────────
    def test_perimenopause_detected(self):
        results = self.clf.classify_rule_based(
            cycle_lengths    = [28, 30, 33, 36, 40, 44, 48, 42, 50, 45],
            period_durations = [5, 5, 6, 6, 7, 7, 8, 7, 8, 7],
            symptoms={
                "hot_flashes":     [3, 3, 4, 4, 5, 5, 5, 4, 5, 5],
                "night_sweats":    [2, 2, 3, 3, 4, 4, 5, 4, 5, 4],
                "vaginal_dryness": [1, 2, 2, 3, 3, 4, 4, 3, 4, 4],
            },
            age=49,
        )
        top = self._top_condition(results)
        peri_prob = next(r.probability for r in results if r.condition == "PERIMENOPAUSE")
        self.assertEqual(top, "PERIMENOPAUSE",
            f"Expected PERIMENOPAUSE, got {top}")
        self.assertGreater(peri_prob, 0.5,
            f"Perimenopause probability should be >50%, got {peri_prob:.0%}")
        print(f"✅ PASS: Perimenopause detected (prob={peri_prob:.0%})")

    # ── 3. Stress-induced profile ─────────────────────────────────────────────
    def test_stress_detected(self):
        results = self.clf.classify_rule_based(
            cycle_lengths    = [28, 27, 29, 28, 42, 45, 38, 30, 29, 28],
            period_durations = [5, 5, 5, 5, 6, 6, 5, 5, 5, 5],
            symptoms={
                "stress_level": [2, 2, 2, 2, 5, 5, 4, 3, 2, 2],
                "anxiety":      [1, 1, 1, 2, 5, 5, 4, 2, 1, 1],
                "mood_swings":  [2, 2, 2, 3, 5, 5, 4, 3, 2, 2],
            },
            age=30,
        )
        stress_prob = next(r.probability for r in results if r.condition == "STRESS_INDUCED")
        self.assertGreater(stress_prob, 0.35,
            f"Stress probability should be >35%, got {stress_prob:.0%}")
        print(f"✅ PASS: Stress-induced detected (prob={stress_prob:.0%})")

    # ── 4. Endometriosis profile ──────────────────────────────────────────────
    def test_endometriosis_detected(self):
        results = self.clf.classify_rule_based(
            cycle_lengths    = [27, 28, 26, 27, 28, 26, 27, 28, 26, 27],
            period_durations = [7, 8, 7, 8, 7, 8, 7, 8, 7, 8],
            symptoms={
                "cramps":          [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                "pelvic_pain":     [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                "pain_during_sex": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                "back_pain":       [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            },
            age=32,
        )
        endo_prob = next(r.probability for r in results if r.condition == "ENDOMETRIOSIS")
        self.assertGreater(endo_prob, 0.5,
            f"Endometriosis probability should be >50%, got {endo_prob:.0%}")
        print(f"✅ PASS: Endometriosis detected (prob={endo_prob:.0%})")

    # ── 5. Healthy young woman — nothing above threshold ─────────────────────
    def test_healthy_profile_no_high_probability(self):
        results = self.clf.classify_rule_based(
            cycle_lengths    = [28, 27, 29, 28, 28, 29, 27, 28, 29, 28],
            period_durations = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            symptoms={
                "cramps":      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                "mood_swings": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            age=24,
        )
        for r in results:
            self.assertLess(r.probability, 0.40,
                f"{r.condition} should be low probability for healthy profile, "
                f"got {r.probability:.0%}")
        print("✅ PASS: Healthy profile → no high-probability conditions")

    # ── 6. All results have required fields ───────────────────────────────────
    def test_result_structure(self):
        results = self.clf.classify_rule_based(
            cycle_lengths=[28]*10, period_durations=[5]*10,
            symptoms={}, age=28
        )
        for r in results:
            self.assertIsNotNone(r.condition)
            self.assertIsNotNone(r.probability)
            self.assertIsNotNone(r.recommendation)
            self.assertBetween(r.probability, 0.0, 1.0)
        print("✅ PASS: All result objects have valid structure")

    def assertBetween(self, value, low, high):
        self.assertGreaterEqual(value, low)
        self.assertLessEqual(value, high)


if __name__ == "__main__":
    unittest.main(verbosity=2)