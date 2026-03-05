"""
test_api.py — Tests all 4 POST endpoints with real user data.
Run AFTER starting the API:
    uvicorn api:app --reload --port 8000
    python test_api.py
"""
import requests
import json

BASE = "http://127.0.0.1:8000"


def pretty(title, response):
    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"  Status: {response.status_code}")
    print(f"{'='*58}")
    print(json.dumps(response.json(), indent=2))


# ── 0. Health check ───────────────────────────────────────────────────────────
r = requests.get(f"{BASE}/health")
pretty("GET /health", r)


# ── 1. PCOS Prediction ────────────────────────────────────────────────────────
pcos_payload = {
    "age":                 26,
    "bmi":                 27.5,
    "cycle_length":        38,
    "cycle_regularity":    4,      # 4 = irregular
    "follicle_no_right":   12,
    "follicle_no_left":    11,
    "amh":                 4.8,
    "fsh":                 5.2,
    "lh":                  9.1,
    "fsh_lh_ratio":        0.57,
    "waist_hip_ratio":     0.84,
    "endometrium_mm":      7.5,
    "avg_follicle_size_r": 18.0,
    "avg_follicle_size_l": 17.2,
    "weight_gain":         1,
    "hair_growth":         1,
    "skin_darkening":      0,
    "pimples":             1
}
r = requests.post(f"{BASE}/predict/pcos", json=pcos_payload)
pretty("POST /predict/pcos  (PCOS-like profile)", r)


# ── 2. Cycle Length Prediction ────────────────────────────────────────────────
cycle_payload = {
    "cycle_lengths":     [28, 30, 27, 29, 31, 28],
    "period_durations":  [5, 5, 6, 5, 5, 5],
    "age":               28,
    "last_period_start": "2025-12-01"
}
r = requests.post(f"{BASE}/predict/cycle", json=cycle_payload)
pretty("POST /predict/cycle  (regular cycles)", r)


# ── 3. Period / Fertile / Ovulation Windows ───────────────────────────────────
windows_payload = {
    "cycle_lengths":     [28, 30, 27, 29, 31, 28],
    "period_durations":  [5, 5, 6, 5, 5, 5],
    "last_period_start": "2025-12-01",
    "num_cycles":        3
}
r = requests.post(f"{BASE}/predict/windows", json=windows_payload)
pretty("POST /predict/windows  (next 3 cycles)", r)


# ── 4. Irregularity Analysis — PCOS pattern ───────────────────────────────────
irr_pcos_payload = {
    "cycle_lengths":     [42, 55, 38, 60, 45, 50, 48, 44],
    "period_durations":  [6, 7, 5, 8, 6, 7, 5, 6],
    "age":               26,
    "last_period_start": "2025-11-01",
    "symptoms": {
        "acne":        4,
        "weight_gain": 3,
        "hirsutism":   3,
        "fatigue":     3
    }
}
r = requests.post(f"{BASE}/analyze/irregularity", json=irr_pcos_payload)
pretty("POST /analyze/irregularity  (PCOS pattern)", r)


# ── 5. Irregularity Analysis — Stress pattern ─────────────────────────────────
irr_stress_payload = {
    "cycle_lengths":     [28, 27, 29, 28, 43, 45, 38, 30],
    "period_durations":  [5, 5, 5, 5, 6, 6, 5, 5],
    "age":               30,
    "last_period_start": "2025-11-15",
    "symptoms": {
        "stress_level": 5,
        "anxiety":      4,
        "mood_swings":  4,
        "fatigue":      3
    }
}
r = requests.post(f"{BASE}/analyze/irregularity", json=irr_stress_payload)
pretty("POST /analyze/irregularity  (Stress pattern)", r)
# ── 6. Predict from period dates ──────────────────────────────────────────────
dates_payload = {
    "periods": [
        {"start": "2025-09-25", "end": "2025-09-29"},
        {"start": "2025-10-23", "end": "2025-10-27"},
        {"start": "2025-11-20", "end": "2025-11-24"},
        {"start": "2025-12-18", "end": "2025-12-22"},
        {"start": "2026-01-15", "end": "2026-01-19"},
        {"start": "2026-02-12", "end": "2026-02-16"},   # ← most recent
    ],
    "num_cycles": 3
}
r = requests.post(f"{BASE}/predict/from-dates", json=dates_payload)
pretty("POST /predict/from-dates  (6 months of period dates)", r)


# ── 7. Phase Health Insights ──────────────────────────────────────────────────

# Basic — just phase
r = requests.post(f"{BASE}/insights/phase", json={"phase": "luteal"})
pretty("POST /insights/phase  (luteal — basic)", r)

# Personalized — with symptoms + goals + condition
r = requests.post(f"{BASE}/insights/phase", json={
    "phase":        "menstrual",
    "age":          26,
    "symptoms":     {"cramps": 4, "fatigue": 4, "bloating": 3},
    "health_goals": ["reduce stress", "improve sleep"],
    "conditions":   ["PCOS"]
})
pretty("POST /insights/phase  (menstrual — personalized PCOS)", r)

# Follicular — weight loss goal
r = requests.post(f"{BASE}/insights/phase", json={
    "phase":        "follicular",
    "age":          30,
    "symptoms":     {"fatigue": 2},
    "health_goals": ["lose weight"],
    "conditions":   []
})
pretty("POST /insights/phase  (follicular — weight goal)", r)

# Ovulation — teen
r = requests.post(f"{BASE}/insights/phase", json={
    "phase": "ovulation",
    "age":   17,
    "symptoms": {},
    "health_goals": [],
    "conditions": []
})
pretty("POST /insights/phase  (ovulation — teen)", r)

# ── 9. Toxicity Detection ─────────────────────────────────────────────────────

test_messages = [
    ("Clean",            "I hope you have a wonderful day! Your work is amazing.", "chat"),
    ("Harassment",       "You are worthless and nobody likes you, just disappear.", "social_media"),
    ("Sexual Harassment","Send me your nudes right now.", "chat"),
    ("Threatening",      "I know where you live and you will regret this.", "email"),
    ("Body Shaming",     "You are so fat it's disgusting. No wonder you're alone.", "comment"),
    ("Gaslighting",      "You're too sensitive and overreacting. You imagined it all.", "chat"),
]

for label, text, context in test_messages:
    r = requests.post(f"{BASE}/safety/check-toxicity", json={
        "text": text, "context": context, "report_user": False
    })
    data = r.json()
    sev_icons = {"safe":"✅","low":"🟡","medium":"🟠","high":"🔴","critical":"🆘"}
    print(f"\n{'='*55}")
    print(f"  [{label}] Status: {r.status_code}")
    print(f"  Toxic    : {data['is_toxic']}")
    print(f"  Severity : {sev_icons.get(data['severity'],'')} {data['severity'].upper()}")
    print(f"  Category : {data['top_category']}")
    print(f"  Score    : {data['overall_score']}")
    print(f"  Action   : {data['action']}")
    if data.get("support_message"):
        print(f"  Support  : {data['support_message'][:80]}…")

# Batch check
print(f"\n{'='*55}")
print("  BATCH CHECK — Conversation thread")
r = requests.post(f"{BASE}/safety/check-toxicity/batch", json={"texts": [
    "Thanks for sharing that with me.",
    "You are so ugly no one will ever love you.",
    "I really appreciate your help today.",
    "Send me your nudes or I'll post your pictures.",
    "Have a great rest of your day!",
]})
pretty("POST /safety/check-toxicity/batch", r)