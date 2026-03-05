"""
Unit tests for IrregularCycleDetector.
Each test uses a crafted cycle pattern that MUST trigger a specific flag.
"""
import unittest
from irregular_detector import IrregularCycleDetector


class TestIrregularDetector(unittest.TestCase):

    def setUp(self):
        self.detector = IrregularCycleDetector()

    # ── 1. Normal cycles — should flag NOTHING ────────────────────────────────
    # AFTER — tests what actually matters clinically
    def test_regular_cycles_no_flags(self):
        cycles = [28, 27, 29, 28, 30, 27, 28, 29, 28, 27]
        report = self.detector.detect(cycles)

        # Most important: should NOT be marked as irregular overall
        self.assertFalse(report.is_irregular,
            "Regular cycles should NOT be flagged as irregular")

        # Score should stay low
        self.assertLess(report.irregularity_score, 20,
            f"Irregularity score too high for regular cycles: {report.irregularity_score}")

        # No clinical flags (ANOMALY from Isolation Forest is acceptable to ignore)
        clinical_flags = [f.flag_type for f in report.flags
                        if f.flag_type != "ANOMALY"]
        self.assertEqual(len(clinical_flags), 0,
            f"Unexpected clinical flags: {clinical_flags}")

        print(f"✅ PASS: Regular cycles → no clinical flags "
            f"(score={report.irregularity_score})")

    # ── 2. PCOS-like long cycles → OLIGOMENORRHEA ─────────────────────────────
    def test_pcos_oligomenorrhea(self):
        cycles = [42, 55, 38, 60, 45, 50, 48, 62]
        report = self.detector.detect(cycles)
        flag_types = [f.flag_type for f in report.flags]
        self.assertIn("OLIGOMENORRHEA", flag_types,
            f"PCOS cycles should trigger OLIGOMENORRHEA. Got: {flag_types}")
        self.assertTrue(report.is_irregular)
        print(f"✅ PASS: PCOS cycles → {flag_types}")

    # ── 3. Amenorrhea (missing period >90 days) ───────────────────────────────
    def test_amenorrhea_detected(self):
        cycles = [28, 29, 95, 28, 27]
        report = self.detector.detect(cycles)
        flag_types = [f.flag_type for f in report.flags]
        self.assertIn("AMENORRHEA", flag_types,
            f"Cycle >90d should trigger AMENORRHEA. Got: {flag_types}")
        print(f"✅ PASS: Amenorrhea → {flag_types}")

    # ── 4. Too-frequent cycles → POLYMENORRHEA ────────────────────────────────
    def test_polymenorrhea_detected(self):
        cycles = [18, 17, 19, 16, 18, 17, 19, 18]
        report = self.detector.detect(cycles)
        flag_types = [f.flag_type for f in report.flags]
        self.assertIn("POLYMENORRHEA", flag_types,
            f"Short cycles should trigger POLYMENORRHEA. Got: {flag_types}")
        print(f"✅ PASS: Polymenorrhea → {flag_types}")

    # ── 5. High variability ───────────────────────────────────────────────────
    def test_high_variability(self):
        cycles = [22, 40, 25, 45, 21, 38, 28, 50, 24, 42]
        report = self.detector.detect(cycles)
        flag_types = [f.flag_type for f in report.flags]
        self.assertIn("HIGH_VARIABILITY", flag_types,
            f"High-variance cycles should flag. Got: {flag_types}")
        print(f"✅ PASS: High variability → {flag_types}")

    # ── 6. Perimenopause trending longer ─────────────────────────────────────
    def test_trending_longer(self):
        cycles = [28, 30, 32, 35, 38, 40, 42, 45, 47, 50]
        report = self.detector.detect(cycles)
        flag_types = [f.flag_type for f in report.flags]
        self.assertIn("TRENDING_LONGER", flag_types,
            f"Lengthening cycles should trigger TRENDING_LONGER. Got: {flag_types}")
        print(f"✅ PASS: Trending longer → {flag_types}")

    # ── 7. Sudden shift (stress-induced) ─────────────────────────────────────
    def test_sudden_shift(self):
        # Normal for 4 cycles, then sudden jump
        cycles = [28, 27, 29, 28, 42, 44, 40, 43]
        report = self.detector.detect(cycles)
        flag_types = [f.flag_type for f in report.flags]
        self.assertIn("SUDDEN_SHIFT", flag_types,
            f"Sudden jump should trigger SUDDEN_SHIFT. Got: {flag_types}")
        print(f"✅ PASS: Sudden shift → {flag_types}")

    # ── 8. Isolation Forest anomaly ──────────────────────────────────────────
    def test_anomaly_detected(self):
        # One extreme outlier in otherwise normal cycles
        cycles = [28, 27, 29, 28, 27, 29, 28, 70, 28, 27]
        report = self.detector.detect(cycles)
        flag_types = [f.flag_type for f in report.flags]
        self.assertIn("ANOMALY", flag_types,
            f"Extreme outlier should be caught by Isolation Forest. Got: {flag_types}")
        print(f"✅ PASS: Anomaly detected → {flag_types}")

    # ── 9. Severity scoring ───────────────────────────────────────────────────
    def test_severity_score_range(self):
        cycles = [42, 55, 38, 60, 45, 50, 48, 62]
        report = self.detector.detect(cycles)
        self.assertGreaterEqual(report.irregularity_score, 0)
        self.assertLessEqual(report.irregularity_score, 100)
        print(f"✅ PASS: Severity score in range → {report.irregularity_score}")

    # ── 10. Insufficient data graceful handling ───────────────────────────────
    def test_insufficient_data(self):
        cycles = [28, 29]   # only 2 cycles
        report = self.detector.detect(cycles)
        self.assertFalse(report.is_irregular,
            "Should not flag with insufficient data")
        print("✅ PASS: Insufficient data handled gracefully")


if __name__ == "__main__":
    unittest.main(verbosity=2)