"""
Multi-label condition classifier.
Identifies: PCOS, Perimenopause, Stress-induced irregularity,
            Hypothyroidism pattern, Endometriosis pattern.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


CONDITIONS = ["PCOS", "PERIMENOPAUSE", "STRESS_INDUCED",
              "HYPOTHYROIDISM", "ENDOMETRIOSIS"]


@dataclass
class ConditionResult:
    condition: str
    probability: float       # 0.0 – 1.0
    confidence: str          # "low" | "medium" | "high"
    supporting_evidence: List[str] = field(default_factory=list)
    recommendation: str = ""

    @property
    def label(self) -> str:
        if self.probability >= 0.65:
            return f"⚠️  Possible {self.condition}"
        elif self.probability >= 0.40:
            return f"🔍 Watch: {self.condition}"
        else:
            return f"✅ Unlikely: {self.condition}"


class ConditionFeatureExtractor:
    """Extracts condition-specific features from cycle history."""

    def extract(self, cycle_lengths: List[float],
                period_durations: List[float],
                symptoms: Dict[str, List[float]],
                age: int,
                bbt_data: List[float] = None) -> np.ndarray:
        cl = np.array(cycle_lengths, dtype=float)
        pd_ = np.array(period_durations, dtype=float)

        feats = {}

        # ── Cycle statistics ────────────────────────────────────────────────
        feats["mean_cycle"]      = np.mean(cl)
        feats["std_cycle"]       = np.std(cl)
        feats["cv_cycle"]        = np.std(cl) / np.mean(cl)
        feats["max_cycle"]       = np.max(cl)
        feats["min_cycle"]       = np.min(cl)
        feats["pct_long"]        = np.mean(cl > 35)    # oligomenorrhea rate
        feats["pct_short"]       = np.mean(cl < 21)    # polymenorrhea rate
        feats["pct_very_long"]   = np.mean(cl > 45)
        feats["range_cycle"]     = np.max(cl) - np.min(cl)

        # rolling variability (last 6 vs first 6)
        if len(cl) >= 12:
            feats["var_change"] = np.std(cl[-6:]) - np.std(cl[:6])
            feats["mean_change"] = np.mean(cl[-6:]) - np.mean(cl[:6])
        else:
            feats["var_change"]  = 0.0
            feats["mean_change"] = 0.0

        # ── Period duration features ────────────────────────────────────────
        feats["mean_duration"]    = np.mean(pd_)
        feats["heavy_flow_pct"]   = np.mean(pd_ > 7)
        feats["light_flow_pct"]   = np.mean(pd_ < 3)

        # ── Symptom features ────────────────────────────────────────────────
        def sym(name): return np.mean(symptoms.get(name, [0]))

        # PCOS indicators
        feats["acne_severity"]      = sym("acne")
        feats["weight_gain"]        = sym("weight_gain")
        feats["hirsutism"]          = sym("hirsutism")      # excess hair
        feats["hair_loss"]          = sym("hair_loss")

        # Perimenopause indicators
        feats["hot_flashes"]        = sym("hot_flashes")
        feats["night_sweats"]       = sym("night_sweats")
        feats["vaginal_dryness"]    = sym("vaginal_dryness")
        feats["sleep_disruption"]   = sym("sleep_disruption")

        # Stress indicators
        feats["stress_level"]       = sym("stress_level")
        feats["mood_swings"]        = sym("mood_swings")
        feats["anxiety"]            = sym("anxiety")
        feats["fatigue"]            = sym("fatigue")

        # Hypothyroidism indicators
        feats["cold_intolerance"]   = sym("cold_intolerance")
        feats["constipation"]       = sym("constipation")
        feats["brain_fog"]          = sym("brain_fog")

        # Endometriosis indicators
        feats["severe_cramps"]      = sym("cramps")
        feats["pelvic_pain"]        = sym("pelvic_pain")
        feats["pain_during_sex"]    = sym("pain_during_sex")
        feats["back_pain"]          = sym("back_pain")

        # ── Age-based features ──────────────────────────────────────────────
        feats["age"]                = float(age)
        feats["age_peri_risk"]      = float(age >= 40)      # perimenopause risk
        feats["age_pcos_range"]     = float(15 <= age <= 44)

        # ── BBT features ────────────────────────────────────────────────────
        if bbt_data and len(bbt_data) > 10:
            bbt = np.array(bbt_data)
            mid = len(bbt) // 2
            feats["bbt_shift"]      = np.mean(bbt[mid:]) - np.mean(bbt[:mid])
            feats["bbt_variability"]= np.std(bbt)
        else:
            feats["bbt_shift"]      = 0.0
            feats["bbt_variability"]= 0.0

        return np.array(list(feats.values()), dtype=np.float32), list(feats.keys())


class ConditionClassifier:
    """
    Rule-based + ML multi-label condition classifier.
    Uses rule-based scoring as primary (works with 0 training data),
    with optional ML override when trained on labelled data.
    """

    def __init__(self):
        self.extractor   = ConditionFeatureExtractor()
        self.ml_model    = None   # set after supervised training
        self.is_trained  = False

    # ── Rule-based scoring (always available) ─────────────────────────────────
    def classify_rule_based(
        self,
        cycle_lengths: List[float],
        period_durations: List[float],
        symptoms: Dict[str, List[float]],
        age: int,
        bbt_data: List[float] = None,
    ) -> List[ConditionResult]:

        cl = np.array(cycle_lengths, dtype=float)
        feats, feat_names = self.extractor.extract(
            cycle_lengths, period_durations, symptoms, age, bbt_data
        )
        feat_dict = dict(zip(feat_names, feats))
        results = []

        # ── PCOS ─────────────────────────────────────────────────────────────
        pcos_score = 0.0
        pcos_evidence = []

        if feat_dict["mean_cycle"] > 35:
            pcos_score += 0.30; pcos_evidence.append(f"Long average cycle ({feat_dict['mean_cycle']:.0f}d)")
        if feat_dict["pct_long"] > 0.4:
            pcos_score += 0.20; pcos_evidence.append(f"{feat_dict['pct_long']*100:.0f}% of cycles > 35 days")
        if feat_dict["std_cycle"] > 8:
            pcos_score += 0.15; pcos_evidence.append(f"High cycle variability (SD={feat_dict['std_cycle']:.1f}d)")
        if feat_dict["acne_severity"] >= 3:
            pcos_score += 0.10; pcos_evidence.append("Moderate-severe acne")
        if feat_dict["weight_gain"] >= 3:
            pcos_score += 0.10; pcos_evidence.append("Reported weight gain")
        if feat_dict["hirsutism"] >= 2:
            pcos_score += 0.10; pcos_evidence.append("Hirsutism reported")
        if feat_dict["pct_very_long"] > 0.2:
            pcos_score += 0.15; pcos_evidence.append("Multiple cycles > 45 days")
        if feat_dict["bbt_shift"] < 0.15:
            pcos_score += 0.10; pcos_evidence.append("Weak/absent BBT biphasic shift (possible anovulation)")

        results.append(ConditionResult(
            condition="PCOS",
            probability=min(0.95, pcos_score),
            confidence="high" if len(pcos_evidence) >= 3 else "medium" if pcos_evidence else "low",
            supporting_evidence=pcos_evidence,
            recommendation=(
                "Schedule ultrasound (ovarian morphology) + hormone panel: "
                "LH, FSH, testosterone, AMH, fasting insulin."
            )
        ))

        # ── Perimenopause ─────────────────────────────────────────────────────
        peri_score = 0.0
        peri_evidence = []

        if age >= 45:
            peri_score += 0.35; peri_evidence.append(f"Age {age} (perimenopause range)")
        elif age >= 40:
            peri_score += 0.15; peri_evidence.append(f"Age {age} (early perimenopause possible)")
        if feat_dict["var_change"] > 4:
            peri_score += 0.20; peri_evidence.append("Increasing cycle variability over time")
        if feat_dict["mean_change"] > 5:
            peri_score += 0.15; peri_evidence.append("Progressive cycle lengthening")
        if feat_dict["hot_flashes"] >= 3:
            peri_score += 0.15; peri_evidence.append("Hot flashes reported")
        if feat_dict["night_sweats"] >= 2:
            peri_score += 0.10; peri_evidence.append("Night sweats reported")
        if feat_dict["vaginal_dryness"] >= 2:
            peri_score += 0.10; peri_evidence.append("Vaginal dryness reported")
        if feat_dict["sleep_disruption"] >= 3:
            peri_score += 0.05; peri_evidence.append("Sleep disruption reported")
        if feat_dict["pct_long"] > 0.3 and age >= 38:
            peri_score += 0.10; peri_evidence.append("Frequent long cycles after age 38")

        results.append(ConditionResult(
            condition="PERIMENOPAUSE",
            probability=min(0.95, peri_score),
            confidence="high" if len(peri_evidence) >= 3 else "medium" if peri_evidence else "low",
            supporting_evidence=peri_evidence,
            recommendation=(
                "Check FSH + estradiol levels (day 2-3 of cycle). "
                "FSH > 10 IU/L on multiple tests suggests perimenopause transition."
            )
        ))

        # ── Stress-induced ────────────────────────────────────────────────────
        stress_score = 0.0
        stress_evidence = []

        if feat_dict["stress_level"] >= 4:
            stress_score += 0.35; stress_evidence.append("High self-reported stress")
        if feat_dict["std_cycle"] > 7 and feat_dict["mean_cycle"] < 35:
            stress_score += 0.20; stress_evidence.append("Variable cycles without oligomenorrhea pattern")
        if feat_dict["sudden_shift"] if "sudden_shift" in feat_dict else False:
            stress_score += 0.20; stress_evidence.append("Sudden cycle shift detected")
        if feat_dict["mood_swings"] >= 3:
            stress_score += 0.10; stress_evidence.append("Significant mood swings")
        if feat_dict["anxiety"] >= 3:
            stress_score += 0.10; stress_evidence.append("Anxiety reported")
        if feat_dict["fatigue"] >= 3:
            stress_score += 0.05; stress_evidence.append("Chronic fatigue reported")
        # stress doesn't usually cause extreme lengthening like PCOS
        if feat_dict["mean_cycle"] <= 38 and feat_dict["std_cycle"] > 5:
            stress_score += 0.10

        results.append(ConditionResult(
            condition="STRESS_INDUCED",
            probability=min(0.90, stress_score),
            confidence="medium",
            supporting_evidence=stress_evidence,
            recommendation=(
                "Track stress levels daily. Practice stress reduction (mindfulness, sleep hygiene). "
                "Cycles often normalise within 2-3 months after stress reduction."
            )
        ))

        # ── Hypothyroidism ────────────────────────────────────────────────────
        hypo_score = 0.0
        hypo_evidence = []

        if feat_dict["mean_duration"] > 7:
            hypo_score += 0.20; hypo_evidence.append("Heavy/prolonged periods")
        if feat_dict["cold_intolerance"] >= 3:
            hypo_score += 0.25; hypo_evidence.append("Cold intolerance reported")
        if feat_dict["brain_fog"] >= 3:
            hypo_score += 0.15; hypo_evidence.append("Brain fog reported")
        if feat_dict["constipation"] >= 2:
            hypo_score += 0.10; hypo_evidence.append("Constipation reported")
        if feat_dict["fatigue"] >= 4:
            hypo_score += 0.10; hypo_evidence.append("Severe fatigue")
        if feat_dict["mean_cycle"] > 35:
            hypo_score += 0.10; hypo_evidence.append("Long cycles (overlap with hypothyroid pattern)")

        results.append(ConditionResult(
            condition="HYPOTHYROIDISM",
            probability=min(0.90, hypo_score),
            confidence="medium" if hypo_evidence else "low",
            supporting_evidence=hypo_evidence,
            recommendation="Request TSH + free T4 blood test. Hypothyroidism is highly treatable with medication."
        ))

        # ── Endometriosis ─────────────────────────────────────────────────────
        endo_score = 0.0
        endo_evidence = []

        if feat_dict["severe_cramps"] >= 4:
            endo_score += 0.30; endo_evidence.append("Severe menstrual cramps (dysmenorrhoea)")
        if feat_dict["pelvic_pain"] >= 3:
            endo_score += 0.25; endo_evidence.append("Chronic pelvic pain")
        if feat_dict["pain_during_sex"] >= 2:
            endo_score += 0.20; endo_evidence.append("Pain during intercourse (dyspareunia)")
        if feat_dict["heavy_flow_pct"] > 0.4:
            endo_score += 0.10; endo_evidence.append("Frequent heavy periods")
        if feat_dict["back_pain"] >= 3:
            endo_score += 0.05; endo_evidence.append("Significant back pain during period")

        results.append(ConditionResult(
            condition="ENDOMETRIOSIS",
            probability=min(0.90, endo_score),
            confidence="medium" if endo_evidence else "low",
            supporting_evidence=endo_evidence,
            recommendation=(
                "Consult a gynaecologist for pelvic exam + transvaginal ultrasound. "
                "Definitive diagnosis requires laparoscopy."
            )
        ))

        # Sort by probability
        return sorted(results, key=lambda r: r.probability, reverse=True)

    # ── Optional: ML-based classifier (train on labelled dataset) ────────────
    def train_ml(self, X: np.ndarray, y: np.ndarray):
        """
        y shape: (n_samples, 5) — binary labels for each condition.
        Requires labelled clinical dataset.
        """
        self.ml_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MultiOutputClassifier(
                RandomForestClassifier(
                    n_estimators=300, max_depth=8,
                    class_weight="balanced", random_state=42
                )
            ))
        ])
        self.ml_model.fit(X, y)
        self.is_trained = True
        print("ML condition classifier trained.")

    def save(self, path: str):
        if self.ml_model:
            joblib.dump(self.ml_model, path)

    def load(self, path: str):
        self.ml_model = joblib.load(path)
        self.is_trained = True