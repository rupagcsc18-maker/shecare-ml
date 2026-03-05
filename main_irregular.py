"""Demo: simulates PCOS, perimenopause, and stress-induced profiles."""
from datetime import date
from irregularity_report import MenstrualHealthAnalyzer


def demo_pcos():
    print("\n\n" + "🔬 "*10)
    print("DEMO: PCOS PROFILE")
    analyzer = MenstrualHealthAnalyzer()
    result = analyzer.analyze(
        cycle_lengths    = [42, 55, 38, 60, 45, 50, 35, 62, 48, 44],
        period_durations = [6, 7, 5, 8, 6, 7, 5, 8, 6, 6],
        symptoms={
            "acne":       [4, 4, 3, 5, 4, 4, 3, 5, 4, 4],
            "weight_gain":[3, 3, 3, 4, 3, 3, 3, 4, 3, 3],
            "hirsutism":  [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "fatigue":    [3, 4, 3, 4, 3, 4, 3, 4, 3, 3],
        },
        age=26,
        last_period_start=date(2025, 10, 1),
    )
    analyzer.print_full_report(result)


def demo_perimenopause():
    print("\n\n" + "🔬 "*10)
    print("DEMO: PERIMENOPAUSE PROFILE")
    analyzer = MenstrualHealthAnalyzer()
    result = analyzer.analyze(
        cycle_lengths    = [28, 29, 30, 31, 33, 35, 38, 40, 38, 45, 50, 42],
        period_durations = [5, 5, 6, 6, 6, 7, 7, 7, 6, 8, 8, 7],
        symptoms={
            "hot_flashes":    [1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
            "night_sweats":   [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4],
            "mood_swings":    [2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5],
            "sleep_disruption":[1,1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5],
            "vaginal_dryness":[0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4],
        },
        age=48,
        last_period_start=date(2025, 8, 15),
    )
    analyzer.print_full_report(result)


def demo_stress():
    print("\n\n" + "🔬 "*10)
    print("DEMO: STRESS-INDUCED PROFILE")
    analyzer = MenstrualHealthAnalyzer()
    result = analyzer.analyze(
        cycle_lengths    = [28, 27, 29, 28, 38, 42, 35, 30, 29, 28],
        period_durations = [5, 5, 5, 5, 6, 6, 5, 5, 5, 5],
        symptoms={
            "stress_level": [2, 2, 2, 2, 5, 5, 4, 3, 2, 2],
            "anxiety":      [1, 1, 1, 2, 5, 5, 4, 2, 1, 1],
            "mood_swings":  [2, 2, 2, 3, 5, 5, 4, 3, 2, 2],
            "fatigue":      [2, 2, 2, 2, 4, 5, 4, 3, 2, 2],
        },
        age=30,
        last_period_start=date(2025, 11, 1),
    )
    analyzer.print_full_report(result)


if __name__ == "__main__":
    demo_pcos()
    demo_perimenopause()
    demo_stress()