"""
api.py — FastAPI REST API
=========================
Endpoints:
  POST /predict/pcos          → PCOS probability from clinical data
  POST /predict/cycle         → Next cycle length prediction
  POST /predict/windows       → Period / fertile / ovulation windows
  POST /analyze/irregularity  → Detect irregular patterns + conditions
  GET  /health                → API health check

Run:
    pip install fastapi uvicorn joblib scikit-learn xgboost pydantic
    cd C:\\Users\\rupa8\\OneDrive\\Desktop\\menstruation_tracker
    uvicorn api:app --reload --port 8000

Test in browser: http://127.0.0.1:8000/docs
"""

import os
import numpy as np
import pandas as pd
import joblib
from datetime import date, timedelta
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from period_predictor import PeriodDatePredictor, PeriodEntry

# ── Load trained models ───────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model '{filename}' not found. Run train_and_test.py first."
        )
    return joblib.load(path)

try:
    pcos_ckpt  = load_model("pcos_classifier.pkl")
    cycle_ckpt = load_model("cycle_length_xgb.pkl")
    PCOS_MODEL    = pcos_ckpt["model"]
    PCOS_FEATURES = pcos_ckpt["features"]
    CYCLE_MODEL   = cycle_ckpt["model"]
    CYCLE_SCALER  = cycle_ckpt["scaler"]
    CYCLE_FEATURES= cycle_ckpt["features"]
    MODELS_LOADED = True
except FileNotFoundError as e:
    print(f"⚠️  {e}")
    MODELS_LOADED = False


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🌸 Menstruation Tracker API",
    description="AI-powered menstrual health predictions — PCOS detection, cycle length, fertile windows & irregularity analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PCOSInput(BaseModel):
    """Clinical data for PCOS prediction."""
    age:                    float = Field(...,  ge=10,  le=60,  example=26,   description="Age in years")
    bmi:                    float = Field(...,  ge=10,  le=60,  example=24.5, description="Body Mass Index")
    cycle_length:           float = Field(...,  ge=15,  le=90,  example=35,   description="Menstrual cycle length in days")
    cycle_regularity:       int   = Field(...,  ge=2,   le=4,   example=4,    description="2=regular, 4=irregular")
    follicle_no_right:      float = Field(...,  ge=0,   le=40,  example=12,   description="Number of follicles in right ovary")
    follicle_no_left:       float = Field(...,  ge=0,   le=40,  example=11,   description="Number of follicles in left ovary")
    amh:                    float = Field(...,  ge=0,   le=20,  example=4.5,  description="AMH level (ng/mL)")
    fsh:                    float = Field(...,  ge=0,   le=200, example=5.2,  description="FSH level (mIU/mL)")
    lh:                     float = Field(...,  ge=0,   le=200, example=8.1,  description="LH level (mIU/mL)")
    fsh_lh_ratio:           float = Field(...,  ge=0,   le=10,  example=0.64, description="FSH/LH ratio")
    waist_hip_ratio:        float = Field(...,  ge=0.5, le=1.5, example=0.82, description="Waist to Hip ratio")
    endometrium_mm:         float = Field(...,  ge=1,   le=30,  example=8.0,  description="Endometrium thickness (mm)")
    avg_follicle_size_r:    float = Field(...,  ge=1,   le=40,  example=18.0, description="Average follicle size right (mm)")
    avg_follicle_size_l:    float = Field(...,  ge=1,   le=40,  example=17.5, description="Average follicle size left (mm)")
    weight_gain:            int   = Field(...,  ge=0,   le=1,   example=1,    description="Weight gain: 1=Yes, 0=No")
    hair_growth:            int   = Field(...,  ge=0,   le=1,   example=1,    description="Excess hair growth: 1=Yes, 0=No")
    skin_darkening:         int   = Field(...,  ge=0,   le=1,   example=0,    description="Skin darkening: 1=Yes, 0=No")
    pimples:                int   = Field(...,  ge=0,   le=1,   example=1,    description="Pimples/acne: 1=Yes, 0=No")


class CycleInput(BaseModel):
    """Recent cycle history for cycle length prediction."""
    cycle_lengths:      List[float] = Field(..., min_items=3, example=[28, 30, 27, 29, 31],
                                            description="List of recent cycle lengths in days (oldest first, min 3)")
    period_durations:   List[float] = Field(..., min_items=3, example=[5, 5, 6, 5, 5],
                                            description="Duration of each period in days")
    age:                int         = Field(..., ge=10, le=60, example=28)
    last_period_start:  str         = Field(..., example="2025-12-01",
                                            description="Start date of most recent period (YYYY-MM-DD)")

    @validator("last_period_start")
    def validate_date(cls, v):
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v

    @validator("period_durations")
    def validate_lengths_match(cls, v, values):
        if "cycle_lengths" in values and len(v) != len(values["cycle_lengths"]):
            raise ValueError("cycle_lengths and period_durations must have the same number of entries")
        return v


class WindowInput(BaseModel):
    """Input for predicting period/fertile/ovulation windows."""
    cycle_lengths:      List[float] = Field(..., min_items=3, example=[28, 30, 27, 29, 31])
    period_durations:   List[float] = Field(..., min_items=3, example=[5, 5, 6, 5, 5])
    last_period_start:  str         = Field(..., example="2025-12-01")
    num_cycles:         int         = Field(3, ge=1, le=6, description="How many future cycles to predict")

    @validator("last_period_start")
    def validate_date(cls, v):
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("Date must be YYYY-MM-DD")
        return v


class IrregularityInput(BaseModel):
    """Input for irregularity + condition analysis."""
    cycle_lengths:      List[float]       = Field(..., min_items=4, example=[28, 42, 55, 38, 60, 45])
    period_durations:   List[float]       = Field(..., min_items=4, example=[5, 7, 8, 6, 8, 7])
    age:                int               = Field(..., ge=10, le=60, example=28)
    symptoms:           Dict[str, float]  = Field(default={},
                                                  example={"acne": 4, "weight_gain": 3,
                                                           "hirsutism": 3, "fatigue": 3})
    last_period_start:  Optional[str]     = Field(None, example="2025-11-01")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

PCOS_FEATURE_MAP = {
    "Follicle No. (R)":       "follicle_no_right",
    "Follicle No. (L)":       "follicle_no_left",
    "AMH(ng/mL)":             "amh",
    "Cycle(R/I)":             "cycle_regularity",
    "Cycle length(days)":     "cycle_length",
    "Skin darkening (Y/N)":   "skin_darkening",
    "hair growth(Y/N)":       "hair_growth",
    "Weight gain(Y/N)":       "weight_gain",
    "Fast food (Y/N)":        None,   # not collected → default 0
    "Pimples(Y/N)":           "pimples",
    "FSH(mIU/mL)":            "fsh",
    "LH(mIU/mL)":             "lh",
    "FSH/LH":                 "fsh_lh_ratio",
    "Waist:Hip Ratio":        "waist_hip_ratio",
    "BMI":                    "bmi",
    "Age (yrs)":              "age",
    "Endometrium (mm)":       "endometrium_mm",
    "Avg. F size (R) (mm)":   "avg_follicle_size_r",
    "Avg. F size (L) (mm)":   "avg_follicle_size_l",
}


def build_pcos_row(data: PCOSInput) -> pd.DataFrame:
    """Map user input → model feature DataFrame."""
    input_dict = data.dict()
    row = {}
    for feat in PCOS_FEATURES:
        mapped = PCOS_FEATURE_MAP.get(feat)
        row[feat] = input_dict.get(mapped, 0) if mapped else 0
    return pd.DataFrame([row])


def predict_next_cycle(cycle_lengths: List[float]) -> float:
    """Weighted average of recent cycles — used when XGBoost features unavailable."""
    arr = np.array(cycle_lengths[-6:])
    weights = np.arange(1, len(arr) + 1, dtype=float)
    return float(np.average(arr, weights=weights))


def build_windows(last_start: date, cycle_len: float,
                  avg_duration: float, num_cycles: int) -> List[dict]:
    windows = []
    current = last_start
    for i in range(num_cycles):
        current   = current + timedelta(days=round(cycle_len))
        ovulation = current + timedelta(days=round(cycle_len - 14))
        windows.append({
            "cycle_number":           i + 1,
            "period_start":           str(current),
            "period_end":             str(current + timedelta(days=round(avg_duration) - 1)),
            "fertile_window_start":   str(ovulation - timedelta(days=5)),
            "fertile_window_end":     str(ovulation + timedelta(days=1)),
            "ovulation_day":          str(ovulation),
            "luteal_phase_start":     str(ovulation + timedelta(days=1)),
            "luteal_phase_end":       str(current + timedelta(days=round(cycle_len) - 1)),
            "predicted_cycle_length": round(cycle_len, 1),
        })
    return windows


def analyze_irregularity(cycle_lengths: List[float]) -> dict:
    """Rule-based irregularity analysis."""
    cl   = np.array(cycle_lengths)
    mean = float(np.mean(cl))
    std  = float(np.std(cl))
    flags = []

    if np.any(cl > 35):   flags.append("OLIGOMENORRHEA")
    if np.any(cl < 21):   flags.append("POLYMENORRHEA")
    if np.any(cl > 90):   flags.append("AMENORRHEA")
    if std > 8:           flags.append("HIGH_VARIABILITY")

    # Trend
    slope = np.polyfit(range(len(cl)), cl, 1)[0]
    if slope > 0.8:       flags.append("TRENDING_LONGER")
    elif slope < -0.8:    flags.append("TRENDING_SHORTER")

    # Sudden shift
    if len(cl) >= 6:
        baseline = float(np.mean(cl[:len(cl)//2]))
        recent   = float(np.mean(cl[len(cl)//2:]))
        if abs(recent - baseline) >= 10:
            flags.append("SUDDEN_SHIFT")

    score = min(100, len(flags) * 20 + (std * 2))
    return {
        "flags":              flags,
        "irregularity_score": round(score, 1),
        "is_irregular":       score > 20,
        "mean_cycle":         round(mean, 1),
        "std_cycle":          round(std, 1),
        "severity":           "severe" if score >= 60 else
                              "moderate" if score >= 35 else
                              "mild" if score > 0 else "none",
    }


def classify_conditions(cycle_lengths, period_durations, symptoms, age) -> List[dict]:
    """Rule-based condition scoring."""
    cl  = np.array(cycle_lengths)
    pd_ = np.array(period_durations)

    def sym(name): return symptoms.get(name, 0)

    results = []

    # PCOS
    pcos = 0.0
    if np.mean(cl) > 35:           pcos += 0.30
    if np.mean(cl > 35) > 0.4:     pcos += 0.20
    if np.std(cl) > 8:             pcos += 0.15
    if sym("acne") >= 3:           pcos += 0.10
    if sym("weight_gain") >= 3:    pcos += 0.10
    if sym("hirsutism") >= 2:      pcos += 0.10
    if sym("hair_loss") >= 2:      pcos += 0.05
    results.append({"condition": "PCOS", "probability": round(min(0.95, pcos), 2)})

    # Perimenopause
    peri = 0.0
    if age >= 45:                   peri += 0.35
    elif age >= 40:                 peri += 0.15
    if sym("hot_flashes") >= 3:    peri += 0.20
    if sym("night_sweats") >= 2:   peri += 0.10
    if sym("vaginal_dryness") >= 2:peri += 0.10
    slope = float(np.polyfit(range(len(cl)), cl, 1)[0])
    if slope > 0.5:                peri += 0.15
    results.append({"condition": "PERIMENOPAUSE", "probability": round(min(0.95, peri), 2)})

    # Stress-induced
    stress = 0.0
    if sym("stress_level") >= 4:   stress += 0.35
    if sym("anxiety") >= 3:        stress += 0.15
    if sym("mood_swings") >= 3:    stress += 0.10
    if np.std(cl) > 5 and np.mean(cl) < 40: stress += 0.20
    results.append({"condition": "STRESS_INDUCED", "probability": round(min(0.90, stress), 2)})

    # Endometriosis
    endo = 0.0
    if sym("cramps") >= 4:         endo += 0.30
    if sym("pelvic_pain") >= 3:    endo += 0.25
    if sym("pain_during_sex") >= 2:endo += 0.20
    if np.mean(pd_) > 7:           endo += 0.10
    results.append({"condition": "ENDOMETRIOSIS", "probability": round(min(0.90, endo), 2)})

    return sorted(results, key=lambda x: x["probability"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status":        "ok",
        "models_loaded": MODELS_LOADED,
        "pcos_model":    "pcos_classifier.pkl" if MODELS_LOADED else "not loaded",
        "cycle_model":   "cycle_length_xgb.pkl" if MODELS_LOADED else "not loaded",
        "version":       "1.0.0",
    }


# ── 1. PCOS Prediction ────────────────────────────────────────────────────────
@app.post("/predict/pcos", tags=["Predictions"])
def predict_pcos(data: PCOSInput):
    """
    Predict PCOS probability from clinical measurements.
    Provide hormone levels, follicle counts, and symptoms.
    """
    if not MODELS_LOADED:
        raise HTTPException(503, "Models not loaded. Run train_and_test.py first.")

    row     = build_pcos_row(data)
    prob    = float(PCOS_MODEL.predict_proba(row)[0][1])
    label   = PCOS_MODEL.predict(row)[0]

    if prob >= 0.65:
        risk_level    = "HIGH"
        interpretation = "Strong indicators of PCOS. Recommend clinical evaluation."
    elif prob >= 0.40:
        risk_level    = "MODERATE"
        interpretation = "Some PCOS indicators present. Monitor symptoms and consult a doctor."
    else:
        risk_level    = "LOW"
        interpretation = "Low likelihood of PCOS based on provided data."

    return {
        "prediction":       "PCOS" if label == 1 else "No PCOS",
        "pcos_probability": round(prob, 4),
        "risk_level":       risk_level,
        "interpretation":   interpretation,
        "top_risk_factors": _get_top_risk_factors(data),
        "recommendation":   (
            "Schedule ultrasound + hormone panel (LH, FSH, AMH, testosterone)."
            if prob >= 0.40 else
            "Continue regular health checkups."
        ),
        "disclaimer": "This is an AI screening tool, not a clinical diagnosis."
    }


def _get_top_risk_factors(data: PCOSInput) -> List[str]:
    factors = []
    if data.follicle_no_right > 10 or data.follicle_no_left > 10:
        factors.append(f"High follicle count (R:{data.follicle_no_right}, L:{data.follicle_no_left})")
    if data.cycle_length > 35:
        factors.append(f"Long cycle ({data.cycle_length} days)")
    if data.amh > 3.5:
        factors.append(f"Elevated AMH ({data.amh} ng/mL)")
    if data.fsh_lh_ratio < 1.0:
        factors.append(f"Low FSH/LH ratio ({data.fsh_lh_ratio})")
    if data.hair_growth:
        factors.append("Excess hair growth (hirsutism)")
    if data.weight_gain:
        factors.append("Weight gain reported")
    if data.bmi > 27:
        factors.append(f"Elevated BMI ({data.bmi})")
    return factors[:5]


# ── 2. Cycle Length Prediction ────────────────────────────────────────────────
@app.post("/predict/cycle", tags=["Predictions"])
def predict_cycle_length(data: CycleInput):
    """
    Predict next cycle length from recent cycle history.
    Provide at least 3 past cycle lengths and their period durations.
    """
    cl          = data.cycle_lengths
    pd_         = data.period_durations
    avg_dur     = float(np.mean(pd_))
    predicted   = predict_next_cycle(cl)
    std_dev     = float(np.std(cl[-6:]) if len(cl) >= 3 else 3.0)
    ci_low      = max(15.0, predicted - 1.96 * std_dev)
    ci_high     = min(90.0, predicted + 1.96 * std_dev)

    next_date   = date.fromisoformat(data.last_period_start) + timedelta(days=round(predicted))

    return {
        "predicted_cycle_length_days": round(predicted, 1),
        "confidence_interval_95":      f"{round(ci_low)}–{round(ci_high)} days",
        "uncertainty_days":            round(std_dev, 1),
        "predicted_next_period_start": str(next_date),
        "average_period_duration":     round(avg_dur, 1),
        "based_on_cycles":             len(cl),
        "confidence":                  "high" if len(cl) >= 12 else
                                       "medium" if len(cl) >= 6 else "low",
    }


# ── 3. Cycle Windows ──────────────────────────────────────────────────────────
@app.post("/predict/windows", tags=["Predictions"])
def predict_windows(data: WindowInput):
    """
    Predict period, fertile window, ovulation, and luteal phase
    for the next N cycles.
    """
    cl          = data.cycle_lengths
    pd_         = data.period_durations
    predicted   = predict_next_cycle(cl)
    avg_dur     = float(np.mean(pd_))
    last_start  = date.fromisoformat(data.last_period_start)
    windows     = build_windows(last_start, predicted, avg_dur, data.num_cycles)

    return {
        "predicted_cycle_length":  round(predicted, 1),
        "average_period_duration": round(avg_dur, 1),
        "cycles":                  windows,
        "note": (
            "Fertile window = 5 days before ovulation + ovulation day. "
            "Ovulation estimated as cycle_length - 14 days from period start."
        ),
        "disclaimer": "Not suitable for use as contraception. Consult a healthcare provider."
    }


# ── 4. Irregularity Analysis ──────────────────────────────────────────────────
@app.post("/analyze/irregularity", tags=["Analysis"])
def analyze_cycle_irregularity(data: IrregularityInput):
    """
    Detect cycle irregularities and possible conditions
    (PCOS, perimenopause, stress-induced, endometriosis).
    Provide at least 4 cycle lengths for meaningful analysis.
    """
    irr        = analyze_irregularity(data.cycle_lengths)
    conditions = classify_conditions(
        data.cycle_lengths, data.period_durations,
        data.symptoms, data.age
    )
    top        = conditions[0] if conditions else {}
    avg_dur    = float(np.mean(data.period_durations))

    # Adaptive prediction strategy
    strategy = "standard"
    if "AMENORRHEA" in irr["flags"]:          strategy = "conservative"
    elif top.get("condition") == "PCOS" and top.get("probability", 0) > 0.5:
        strategy = "robust_median"
    elif top.get("condition") == "PERIMENOPAUSE" and top.get("probability", 0) > 0.4:
        strategy = "kalman_filter"
    elif top.get("condition") == "STRESS_INDUCED" and top.get("probability", 0) > 0.4:
        strategy = "ewma"

    # Prediction using strategy
    cl = np.array(data.cycle_lengths, dtype=float)
    if strategy == "robust_median":
        predicted = float(np.median(cl))
    elif strategy == "ewma":
        ewma = cl[0]
        for c in cl[1:]: ewma = 0.4 * c + 0.6 * ewma
        predicted = float(ewma)
    elif strategy == "conservative":
        valid = cl[cl < 90]
        predicted = float(np.median(valid)) if len(valid) else 35.0
    else:
        predicted = predict_next_cycle(data.cycle_lengths)

    predicted = float(np.clip(predicted, 15, 90))

    # Upcoming windows
    last_start = (date.fromisoformat(data.last_period_start)
                  if data.last_period_start else date.today())
    windows    = build_windows(last_start, predicted, avg_dur, 2)

    return {
        "irregularity_analysis": {
            "is_irregular":       irr["is_irregular"],
            "severity":           irr["severity"],
            "score":              irr["irregularity_score"],
            "flags":              irr["flags"],
            "mean_cycle_days":    irr["mean_cycle"],
            "std_cycle_days":     irr["std_cycle"],
        },
        "condition_screening":    conditions,
        "top_suspected_condition": {
            "condition":   top.get("condition"),
            "probability": top.get("probability"),
            "recommendation": _condition_recommendation(top.get("condition")),
        },
        "adaptive_prediction": {
            "strategy":               strategy,
            "predicted_cycle_length": round(predicted, 1),
            "next_2_cycles":          windows,
        },
        "disclaimer": "Screening only. Always consult a healthcare professional for diagnosis."
    }


def _condition_recommendation(condition: str) -> str:
    recs = {
        "PCOS":           "Schedule ultrasound + hormone panel (LH, FSH, AMH, testosterone, fasting insulin).",
        "PERIMENOPAUSE":  "Check FSH + estradiol on day 2–3 of cycle. FSH >10 IU/L suggests perimenopause.",
        "STRESS_INDUCED": "Track stress daily. Cycles often normalise within 2–3 months of stress reduction.",
        "ENDOMETRIOSIS":  "Consult gynaecologist for pelvic exam + transvaginal ultrasound.",
    }
    return recs.get(condition, "Consult a healthcare provider for further evaluation.")


# ── 5. Predict from Period Dates ──────────────────────────────────────────────

class PeriodDateEntry(BaseModel):
    start: str = Field(..., example="2025-09-05", description="Period start date YYYY-MM-DD")
    end:   str = Field(..., example="2025-09-09", description="Period end date YYYY-MM-DD")


class PeriodDatesInput(BaseModel):
    periods:    List[PeriodDateEntry] = Field(
        ...,
        min_items=2,
        description="List of period start+end dates. Minimum 2 months. More = better accuracy.",
        example=[
            {"start": "2025-09-05", "end": "2025-09-09"},
            {"start": "2025-10-03", "end": "2025-10-07"},
            {"start": "2025-11-01", "end": "2025-11-05"},
        ]
    )
    num_cycles: int = Field(3, ge=1, le=6, description="How many future cycles to predict")


@app.post("/predict/from-dates", tags=["Predictions"])
def predict_from_period_dates(data: PeriodDatesInput):
    """
    Simplest endpoint — user provides ONLY their period start and end dates.

    From just the dates, the model predicts:
    - Next period start & end date
    - Fertile window (best days for conception)
    - Ovulation day (peak fertility)
    - Follicular phase
    - Luteal phase
    - PMS window
    - Irregularity flags (PCOS, stress patterns etc.)

    Minimum: 2 months of data.
    Recommended: 6+ months for high confidence.
    """
    # Validate and parse dates
    try:
        entries = []
        for i, p in enumerate(data.periods):
            try:
                start = date.fromisoformat(p.start)
                end   = date.fromisoformat(p.end)
            except ValueError:
                raise HTTPException(
                    400,
                    f"Period {i+1}: Invalid date format '{p.start}' or '{p.end}'. Use YYYY-MM-DD."
                )
            try:
                entries.append(PeriodEntry(start=start, end=end))
            except ValueError as e:
                raise HTTPException(400, f"Period {i+1}: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Date parsing error: {str(e)}")

    # Run predictor
    try:
        predictor = PeriodDatePredictor(entries)
        result    = predictor.predict(num_cycles=data.num_cycles)
    except ValueError as e:
        raise HTTPException(422, str(e))

    return result
# ── 6. Cycle Phase Health Insights ───────────────────────────────────────────

class InsightInput(BaseModel):
    phase: str = Field(
        ...,
        example="luteal",
        description="Current cycle phase: menstrual | follicular | ovulation | luteal"
    )
    age:          Optional[int]   = Field(None, ge=10, le=60, example=28)
    symptoms:     Optional[Dict[str, float]] = Field(
        default={},
        example={"cramps": 3, "fatigue": 4, "mood_swings": 3}
    )
    health_goals: Optional[List[str]] = Field(
        default=[],
        example=["lose weight", "improve sleep", "reduce stress"]
    )
    conditions:   Optional[List[str]] = Field(
        default=[],
        example=["PCOS", "anemia"]
    )

PHASE_INSIGHTS = {
    "menstrual": {
        "icon": "🩸",
        "color": "#e85d75",
        "duration": "Days 1–5",
        "what_happens": "Uterine lining sheds. Estrogen and progesterone are at their lowest.",
        "energy_level": "Low",
        "nutrition": [
            "Eat iron-rich foods: spinach, lentils, red meat to replenish blood loss",
            "Include vitamin C (oranges, bell peppers) to boost iron absorption",
            "Stay hydrated — aim for 8–10 glasses of water daily",
            "Dark chocolate (70%+) helps with cramps and boosts mood",
            "Avoid excess salt to reduce bloating",
            "Ginger or chamomile tea can ease cramps naturally",
        ],
        "exercise": [
            "Gentle yoga or stretching — child's pose relieves cramps",
            "Short walks (20–30 min) boost mood without exhaustion",
            "Rest is productive — your body is working hard",
            "Light swimming if comfortable",
            "Avoid high-intensity workouts on heavy days",
        ],
        "mental_wellness": [
            "Practice self-compassion — fatigue is normal and valid",
            "Journaling helps process emotions during hormonal shifts",
            "Use a heating pad for physical and emotional comfort",
            "Reduce social obligations if energy is low",
            "Watch comforting content — this is a natural rest phase",
        ],
        "sleep": [
            "Sleep on your side with a pillow between knees for cramp relief",
            "Aim for 8–9 hours — your body repairs during sleep",
            "Avoid screens 1 hour before bed to improve sleep quality",
            "Magnesium glycinate before bed eases cramps and improves sleep",
        ],
        "avoid": [
            "Caffeine — worsens cramps and anxiety",
            "Alcohol — increases prostaglandins (cramp chemicals)",
            "Processed foods high in sugar and salt",
        ],
    },
    "follicular": {
        "icon": "🌱",
        "color": "#4caf82",
        "duration": "Days 6–13",
        "what_happens": "Estrogen rises, follicles develop. Energy and mood naturally improve.",
        "energy_level": "Rising → High",
        "nutrition": [
            "Focus on lean proteins: eggs, chicken, fish to support follicle development",
            "Eat fermented foods (yogurt, kefir) for gut health and estrogen metabolism",
            "Include flaxseeds — support healthy estrogen levels",
            "Fresh fruits and vegetables for antioxidants",
            "Complex carbs: oats, quinoa for sustained energy",
            "Zinc-rich foods (pumpkin seeds, chickpeas) support ovarian health",
        ],
        "exercise": [
            "Best phase for high-intensity workouts — energy peaks here",
            "Try new fitness classes or challenges",
            "Strength training yields best results this week",
            "Running, cycling, HIIT — your body handles intensity well",
            "Great time to set fitness goals and start new routines",
        ],
        "mental_wellness": [
            "Ideal time for brainstorming and creative projects",
            "Start new habits — motivation is naturally higher",
            "Social energy is up — great time for meetings and networking",
            "Learn new skills: your brain is primed for information",
            "Set goals for the cycle ahead",
        ],
        "sleep": [
            "Sleep improves naturally this phase",
            "Maintain consistent sleep/wake times",
            "Morning exercise can enhance alertness all day",
            "7–8 hours is optimal to fuel rising energy levels",
        ],
        "avoid": [
            "Skipping meals — energy demands are rising",
            "Overworking — save capacity for ovulation peak",
        ],
    },
    "ovulation": {
        "icon": "🥚",
        "color": "#f4a742",
        "duration": "Days 14–16",
        "what_happens": "LH surge releases an egg. Estrogen peaks. Peak fertility window.",
        "energy_level": "Peak",
        "nutrition": [
            "Antioxidant-rich foods protect the egg: berries, nuts, leafy greens",
            "Vitamin E (almonds, sunflower seeds) supports reproductive health",
            "Stay hydrated — cervical mucus is mostly water",
            "Light meals: digestive sensitivity can increase",
            "Omega-3 fatty acids (salmon, walnuts) reduce inflammation",
            "Avoid heavy processed foods that spike inflammation",
        ],
        "exercise": [
            "Peak performance phase — ideal for personal bests",
            "High-intensity intervals, strength records, competitive sports",
            "Confidence is high — try advanced yoga or dance",
            "Group fitness classes: social energy is at maximum",
            "Outdoor activities: your body temperature regulation is optimal",
        ],
        "mental_wellness": [
            "Best time for important conversations and negotiations",
            "Confidence and communication skills are naturally elevated",
            "Great for presentations, public speaking, interviews",
            "Channel peak energy into creative work",
            "Social connection is most fulfilling this phase",
        ],
        "sleep": [
            "Body temperature rises slightly — keep bedroom cool",
            "Short sleep disruption is normal due to LH surge",
            "Light evening walks can improve sleep quality",
            "Avoid overexertion to protect sleep quality",
        ],
        "avoid": [
            "Skipping self-care — peak energy is temporary",
            "Overcommitting — the dip after ovulation is real",
        ],
    },
    "luteal": {
        "icon": "🌙",
        "color": "#8b6fcb",
        "duration": "Days 17–28",
        "what_happens": "Progesterone rises then falls. PMS symptoms may appear in the final days.",
        "energy_level": "Declining",
        "nutrition": [
            "Magnesium-rich foods reduce PMS: dark chocolate, avocado, bananas",
            "Complex carbs stabilize mood: sweet potatoes, oats, brown rice",
            "Calcium (dairy, leafy greens) reduces bloating and cramps",
            "Reduce caffeine — worsens anxiety and breast tenderness",
            "Vitamin B6 (chickpeas, tuna) helps with mood regulation",
            "Small frequent meals to maintain blood sugar and reduce irritability",
        ],
        "exercise": [
            "Shift to moderate intensity: pilates, yoga, walking",
            "Strength training still effective in early luteal phase",
            "Avoid extreme HIIT — body needs more recovery time",
            "Restorative yoga reduces PMS symptoms significantly",
            "Swimming is gentle and relieves bloating",
        ],
        "mental_wellness": [
            "Ideal for deep focused work and detail-oriented tasks",
            "Journaling helps identify PMS patterns over time",
            "Set boundaries — social energy naturally decreases",
            "Use this inward phase for reflection and planning",
            "Practice mindfulness to manage mood shifts",
            "Reduce decision-making load in final days",
        ],
        "sleep": [
            "Prioritize 8–9 hours — progesterone disrupts deep sleep",
            "Avoid caffeine after 2pm",
            "Magnesium glycinate supplement improves sleep quality",
            "Cool, dark room helps with progesterone-related temperature rises",
            "Wind-down routine is more important this phase",
        ],
        "avoid": [
            "Alcohol — dramatically worsens PMS symptoms",
            "High-sodium foods — cause water retention and bloating",
            "Sugar spikes — worsen mood crashes",
            "Overcommitting socially when energy is declining",
        ],
    },
}


def get_personalized_insights(phase: str, age: int, symptoms: dict,
                               health_goals: list, conditions: list) -> dict:
    base = PHASE_INSIGHTS.get(phase.lower())
    if not base:
        return {}

    insights = {
        "nutrition":        list(base["nutrition"]),
        "exercise":         list(base["exercise"]),
        "mental_wellness":  list(base["mental_wellness"]),
        "sleep":            list(base["sleep"]),
        "avoid":            list(base["avoid"]),
    }

    # Personalize based on symptoms
    if symptoms.get("cramps", 0) >= 3:
        insights["nutrition"].insert(0, "🔥 High cramps: increase omega-3s and magnesium urgently")
        insights["exercise"].insert(0, "🔥 High cramps: stick to gentle stretching only today")

    if symptoms.get("fatigue", 0) >= 4:
        insights["nutrition"].insert(0, "⚡ High fatigue: iron + B12 check recommended")
        insights["sleep"].insert(0, "⚡ High fatigue: add a 20-min afternoon nap if possible")

    if symptoms.get("mood_swings", 0) >= 3:
        insights["mental_wellness"].insert(0, "💙 Mood swings: omega-3 and B6 supplements can help")
        insights["nutrition"].insert(0, "💙 Mood swings: avoid sugar spikes — eat every 3–4 hours")

    if symptoms.get("bloating", 0) >= 3:
        insights["nutrition"].insert(0, "🌊 Bloating: reduce salt, increase potassium (banana, avocado)")

    # Personalize based on conditions
    if "PCOS" in (conditions or []):
        insights["nutrition"].insert(0, "🌸 PCOS: low-GI foods only — avoid refined carbs and sugar")
        insights["exercise"].insert(0, "🌸 PCOS: resistance training 3x/week improves insulin sensitivity")

    if "anemia" in (conditions or []):
        insights["nutrition"].insert(0, "🩸 Anemia: pair iron foods with vitamin C every meal")

    # Personalize based on health goals
    if "lose weight" in (health_goals or []):
        insights["exercise"].insert(0, "🎯 Weight goal: this phase — focus on consistency over intensity")

    if "improve sleep" in (health_goals or []):
        insights["sleep"].insert(0, "🎯 Sleep goal: no screens 90 min before bed + consistent wake time")

    if "reduce stress" in (health_goals or []):
        insights["mental_wellness"].insert(0, "🎯 Stress goal: 10-min morning breathwork reduces cortisol significantly")

    # Age-based adjustments
    if age and age >= 40:
        insights["nutrition"].insert(0, "👩 40+: increase calcium and vitamin D for bone health")
    if age and age <= 20:
        insights["nutrition"].insert(0, "🌟 Teen: ensure adequate iron and calcium for development")

    return insights


@app.post("/insights/phase", tags=["Health Insights"])
def get_phase_insights(data: InsightInput):
    """
    Returns personalized lifestyle suggestions for the current cycle phase.
    Covers nutrition, exercise, mental wellness, sleep, and what to avoid.
    Suggestions are personalized based on symptoms, health goals, and conditions.
    """
    phase = data.phase.lower().strip()
    valid_phases = ["menstrual", "follicular", "ovulation", "luteal"]

    if phase not in valid_phases:
        raise HTTPException(
            400,
            f"Invalid phase '{phase}'. Must be one of: {valid_phases}"
        )

    base    = PHASE_INSIGHTS[phase]
    details = get_personalized_insights(
        phase, data.age, data.symptoms or {},
        data.health_goals or [], data.conditions or []
    )

    return {
        "phase":          phase.capitalize(),
        "icon":           base["icon"],
        "duration":       base["duration"],
        "what_happens":   base["what_happens"],
        "energy_level":   base["energy_level"],
        "personalized":   bool(data.symptoms or data.health_goals or data.conditions),
        "insights":       details,
        "quick_summary":  {
            "top_nutrition":  details["nutrition"][:3],
            "top_exercise":   details["exercise"][:2],
            "top_wellness":   details["mental_wellness"][:2],
            "top_sleep":      details["sleep"][:2],
        },
        "tip_of_the_day": _phase_tip(phase),
    }


def _phase_tip(phase: str) -> str:
    tips = {
        "menstrual":  "Your body is renewing itself. Rest is not laziness — it's wisdom.",
        "follicular": "Your rising estrogen is your superpower this week. Use it boldly.",
        "ovulation":  "You are at your most magnetic and capable. Make your most important moves now.",
        "luteal":     "Turn inward. Your intuition and depth of focus are gifts of this phase.",
    }
    return tips.get(phase, "Listen to your body — it knows what it needs.")


# ══════════════════════════════════════════════════════════════════════════════
# TOXICITY DETECTION — Women's Safety
# ══════════════════════════════════════════════════════════════════════════════
from toxicity_detector import ToxicityDetector

# Load detector once at startup (set use_transformer=True after pip install transformers torch)
_toxicity_detector = ToxicityDetector(use_transformer=False)
_toxicity_detector.load()


class ToxicityInput(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=5000,
        example="You are worthless and nobody cares about you.",
        description="Text to analyze for toxicity"
    )
    context: Optional[str] = Field(
        None, example="social_media",
        description="Where the message was received: social_media | chat | email | comment"
    )
    report_user: Optional[bool] = Field(
        False, description="Whether to flag this for user reporting"
    )


class BatchToxicityInput(BaseModel):
    texts: List[str] = Field(
        ..., min_items=1, max_items=50,
        description="List of texts to analyze (max 50 at once)"
    )


@app.post("/safety/check-toxicity", tags=["Women's Safety"])
def check_toxicity(data: ToxicityInput):
    """
    🛡️ AI Toxicity Detection for Women's Safety.

    Detects:
    - Harassment & cyberbullying
    - Sexual harassment
    - Threatening messages
    - Hate speech (misogyny, sexism)
    - Body shaming
    - Gaslighting

    Returns severity, category, recommended action, and support resources.
    Uses 3-layer ensemble: rules + ML + transformer.
    """
    result = _toxicity_detector.detect(data.text)
    r      = result.to_dict()

    # Add context-specific advice
    context_advice = {
        "social_media": "Screenshot and report via the platform's report button.",
        "chat":         "Block the user immediately and save the conversation.",
        "email":        "Do not reply. Mark as spam and report.",
        "comment":      "Report the comment and block the commenter.",
    }
    if data.context and data.context in context_advice:
        r["context_advice"] = context_advice[data.context]

    r["safety_resources"] = {
        "cybercrime_portal": "https://cybercrime.gov.in",
        "ncw_helpline":      "7827170170",
        "icall_counseling":  "9152987821",
        "emergency":         "112",
    }

    return r


@app.post("/safety/check-toxicity/batch", tags=["Women's Safety"])
def check_toxicity_batch(data: BatchToxicityInput):
    """
    Check multiple texts for toxicity at once (max 50).
    Useful for moderating a thread or conversation history.
    """
    results = _toxicity_detector.detect_batch(data.texts)

    toxic_count = sum(1 for r in results if r.is_toxic)
    max_severity_order = ["safe","low","medium","high","critical"]
    highest = max(results, key=lambda r: max_severity_order.index(r.severity))

    return {
        "total_texts":      len(results),
        "toxic_count":      toxic_count,
        "clean_count":      len(results) - toxic_count,
        "highest_severity": highest.severity,
        "results": [
            {
                "index":         i,
                "text_preview":  r.text[:60] + ("…" if len(r.text) > 60 else ""),
                **r.to_dict()
            }
            for i, r in enumerate(results)
        ]
    }


@app.get("/safety/categories", tags=["Women's Safety"])
def get_toxicity_categories():
    """List all toxicity categories detected by SheCare."""
    return {
        "categories": [
            {
                "name":        cat,
                "icon":        info["icon"],
                "color":       info["color"],
                "description": {
                    "harassment":        "Verbal abuse, insults, unwanted aggressive contact",
                    "sexual_harassment": "Unwanted sexual advances, explicit content, sexual coercion",
                    "hate_speech":       "Misogyny, sexism, discrimination based on gender",
                    "bullying":          "Repeated intimidation, humiliation, social exclusion",
                    "threatening":       "Threats of physical harm, doxxing, stalking",
                    "body_shaming":      "Negative comments about appearance or weight",
                    "gaslighting":       "Psychological manipulation, reality distortion",
                    "clean":             "Safe, respectful content",
                }.get(cat, ""),
            }
            for cat, info in CATEGORIES.items()    
        ]
    }