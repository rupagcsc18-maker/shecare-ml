"""
End-to-end demo: synthetic data → train → predict → analyze.
"""
import random
from datetime import date, timedelta
from data_models import UserProfile, CycleEntry, COMMON_SYMPTOMS
from predictor import CyclePredictor, print_windows
from symptom_analyzer import SymptomAnalyzer


def generate_synthetic_profile(n_cycles: int = 18) -> UserProfile:
    """Generate a realistic synthetic user profile."""
    profile = UserProfile(user_id="user_001", age=28, avg_cycle_length=28.0)
    start   = date(2022, 1, 5)
    base_cl = 28

    for i in range(n_cycles):
        cycle_len    = int(np.clip(np.random.normal(base_cl, 2.5), 22, 38))
        period_dur   = int(np.clip(np.random.normal(5, 1), 3, 8))
        period_start = start
        period_end   = start + timedelta(days=period_dur - 1)

        symptoms = {
            s: random.randint(0, 5)
            for s in random.sample(COMMON_SYMPTOMS, k=random.randint(3, 8))
        }

        profile.add_cycle(CycleEntry(
            period_start=period_start,
            period_end=period_end,
            symptoms=symptoms,
        ))
        start = start + timedelta(days=cycle_len)

    return profile


if __name__ == "__main__":
    import numpy as np

    print("🌸 Menstruation Cycle ML Tracker")
    print("-" * 40)

    # 1. Build profile
    profile = generate_synthetic_profile(n_cycles=18)
    print(f"Profile: {len(profile.cycles)} cycles loaded")
    print(f"Average cycle: {profile.avg_cycle_length:.1f} days")

    # 2. Train models
    predictor = CyclePredictor()
    predictor.train(profile)

    # 3. Predict next cycles
    windows = predictor.predict_windows(profile, num_cycles=3)
    print_windows(windows)

    # 4. Symptom analysis
    analyzer = SymptomAnalyzer(profile)
    analyzer.print_report()

    # 5. Feature importance from GBM
    if predictor.gbm:
        print("\n\n🔍 Top Predictive Features (GBM):")
        fi = predictor.gbm.feature_importances(predictor.preprocessor.feature_cols)
        for feat, imp in list(fi.items())[:8]:
            bar = "█" * int(imp * 200)
            print(f"   {feat:<30} {bar}  {imp:.4f}")