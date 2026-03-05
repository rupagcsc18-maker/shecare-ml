"""
toxicity_detector.py
====================
3-layer AI toxicity detection for women's safety.

Layer 1: Rule-based fast filter    (instant, catches obvious cases)
Layer 2: TF-IDF + Logistic Regression (trained ML model)
Layer 3: HuggingFace toxic-bert    (deep NLP, most accurate)

Install:
    pip install transformers torch scikit-learn joblib

Train the ML layer:
    python toxicity_detector.py --train

Run standalone test:
    python toxicity_detector.py --test
"""

import re
import os
import sys
import json
import argparse
import numpy as np
import joblib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TOXIC_MODEL_PATH = os.path.join(MODEL_DIR, "toxicity_classifier.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# TOXICITY CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════

CATEGORIES = {
    "harassment":          {"color": "#e85d75", "icon": "🚨", "severity_weight": 1.0},
    "sexual_harassment":   {"color": "#c0392b", "icon": "⛔", "severity_weight": 1.5},
    "hate_speech":         {"color": "#e74c3c", "icon": "🔴", "severity_weight": 1.4},
    "bullying":            {"color": "#e67e22", "icon": "⚠️",  "severity_weight": 1.1},
    "threatening":         {"color": "#8e1a2e", "icon": "🆘", "severity_weight": 1.6},
    "body_shaming":        {"color": "#f39c12", "icon": "💔", "severity_weight": 1.0},
    "gaslighting":         {"color": "#9b59b6", "icon": "🌀", "severity_weight": 1.0},
    "clean":               {"color": "#4caf82", "icon": "✅", "severity_weight": 0.0},
}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1: RULE-BASED FAST FILTER
# ══════════════════════════════════════════════════════════════════════════════

# Pattern library — covers harassment types most relevant to women's safety
PATTERN_LIBRARY = {
    "sexual_harassment": [
        r'\b(send|show|give)\s*(me\s*)?(your\s*)?(nude|nudes?|naked|pics?|photos?|pictures?|content)\b',
        r'\bnudes?\b',
        r'\bnaked\s*(pics?|photos?|pictures?)\b',
        r'\b(post|share|leak|spread)\s*(your\s*)?(pics?|photos?|pictures?|nudes?|content)\b',
        r'\bor\s+i\'?ll?\s+(post|share|leak|send|spread)\b',
        r'\b(sex(ual)?\s*(favor|service|act|video))\b',
        r'\b(rape|molest|assault)\s*(you|her|them)\b',
        r'\bshow\s*(me\s*)?(your\s*)?(body|breast|boob|tit|ass|butt|vagina|pussy)\b',
        r'\b(hot|sexy|fuckable|bangable)\b',
        r'\bi\s*(want|wanna|gonna)\s*(to\s*)?(fuck|bang|screw|nail|do\s+you)\b',
        r'\b(dick|cock|penis|boob|tit|pussy|vagina|naked)\s*(pic|photo|image|video)?\b',
        r'\bonlyfans?\b',
        r'\bsend\s*(it|them|me)\b.*\b(nude|naked|sexy|hot)\b',
        r'\b(i\'ll|ill|will)\s*(post|leak|share|send|expose)\s*(your|ur|the)\b',
        r'\bsextort\b',
    ],
    "threatening": [
        r'\b(kill|murder|hurt|harm|destroy|end)\s+(you|your|her|them)\b',
        r'\bwatch\s+(your|ur)\s+back\b',
        r'\byou\s+(will|gonna|going\s+to)\s+(regret|pay|suffer)\b',
        r'\bi\s+know\s+where\s+you\s+(live|work|are)\b',
        r'\b(doxx|dox)\s*(you|her|them)\b',
        r'\blearn\s+your\s+(address|location|home)\b',
    ],
    "harassment": [
        r'\b(shut\s+up|get\s+lost|go\s+away|leave|disappear)\b',
        r'\bnobody\s+(likes?|wants?|cares?\s+about)\s+(you|her)\b',
        r'\b(worthless|useless|pathetic|disgusting|garbage|trash)\b',
        r'\byou\s+(don\'t|do\s+not)\s+(deserve|belong)\b',
        r'\b(kill\s+yourself|kys|end\s+it)\b',
        r'\b(loser|idiot|moron|stupid|dumb)\b.*\b(you|her)\b',
    ],
    "bullying": [
        r'\beveryone\s+(hates?|laughs?\s+at)\s+(you|her)\b',
        r'\b(ugly|fat|gross|pig|cow)\b.*\b(you|she|her)\b',
        r'\byou\s+(always|never)\s+(fail|mess\s+up|ruin)\b',
        r'\bno\s+one\s+(will\s+ever\s+)?(love|want|like)\s+(you|her)\b',
        r'\bgo\s+(back|crawl)\s+(to|under)\b',
    ],
    "body_shaming": [
        r'\b(fat|obese|overweight|skinny|flat|ugly)\b.*\b(you|her|she)\b',
        r'\byou\s+(look|are)\s+(fat|ugly|disgusting|gross|horrible)\b',
        r'\blose\s+(some\s+)?weight\b',
        r'\bno\s+wonder\s+(you\'re|she\'s)\s+(single|alone|unwanted)\b',
        r'\b(body|figure|shape)\s+(is\s+)?(disgusting|gross|awful)\b',
    ],
    "hate_speech": [
        r'\b(women|girls?|females?)\s+(are\s+)?(inferior|weak|stupid|dumb|useless)\b',
        r'\bgo\s+(back\s+to\s+the)?\s+kitchen\b',
        r'\bfeminist\s+(are\s+)?(crazy|stupid|idiots?)\b',
        r'\bwomen\s+don\'t\s+belong\s+(in|at)\b',
        r'\b(period|menstrual)\s+(freak|gross|disgusting)\b',
    ],
    "gaslighting": [
        r'\byou\'re\s+(too\s+)?(sensitive|emotional|crazy|paranoid|overreacting)\b',
        r'\byou\s+(imagined|made\s+up|invented)\s+(it|that|this)\b',
        r'\bthat\s+(never|didn\'t)\s+happen\b',
        r'\byou\'re\s+(always\s+)?(lying|making\s+things\s+up)\b',
        r'\bstop\s+being\s+(so\s+)?(dramatic|emotional|sensitive)\b',
    ],
}

# Compiled patterns
COMPILED_PATTERNS = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in PATTERN_LIBRARY.items()
}


def rule_based_check(text: str) -> Dict[str, float]:
    """
    Fast regex scan. Returns category → confidence scores.
    Confidence is based on number + specificity of pattern matches.
    """
    scores = {cat: 0.0 for cat in CATEGORIES}
    text_lower = text.lower()

    for category, patterns in COMPILED_PATTERNS.items():
        matches = sum(1 for p in patterns if p.search(text_lower))
        if matches > 0:
            # More matches = higher confidence, diminishing returns
            scores[category] = min(0.95, 0.55 + (matches - 1) * 0.15)

    return scores


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2: ML CLASSIFIER (TF-IDF + Logistic Regression)
# ══════════════════════════════════════════════════════════════════════════════

# Training data — curated examples for each category
# In production: use Jigsaw Toxic Comment dataset from Kaggle
TRAINING_DATA = {
    "harassment": [
        "you are worthless and nobody likes you",
        "get out of here nobody wants you around",
        "you don't deserve to be here",
        "shut up you pathetic loser",
        "you are such a waste of space",
        "nobody cares what you think",
        "you are completely useless",
        "go away and never come back",
        "you are a disgrace to everyone",
        "i hate everything about you",
        "you are the worst person I know",
        "you make everyone miserable",
    ],
    "sexual_harassment": [
        "send me your nudes right now",
        "send me nudes or i will post your pictures",
        "send nudes",
        "you are so sexy I want to bang you",
        "show me your body",
        "I want to have sex with you",
        "you look fuckable in that outfit",
        "send pics of your body",
        "I want to see you naked",
        "you are hot I want to sleep with you",
        "can I see your boobs",
        "you should be on onlyfans",
        "I keep thinking about your naked body",
        "or ill post your photos online",
        "send me naked pictures",
        "I will leak your pictures if you dont",
        "show me naked pics",
        "send me your sexy photos",
        "I want your nude photos now",
        "you better send nudes or else",
        "I will share your photos if you refuse",
        "give me your naked pictures",
        "send body pics now",
        "you are so hot send me pictures",
        "I want to see your naked body",
    ],
    "threatening": [
        "I will hurt you if you don't stop",
        "watch your back I'm coming for you",
        "you will regret this I promise",
        "I know where you live",
        "you better be scared of me",
        "I will make you pay for this",
        "you will suffer for what you did",
        "I will find you and make you regret",
        "this is your last warning",
        "I will destroy your life",
        "you're going to pay for this",
        "I will hunt you down",
    ],
    "bullying": [
        "everyone laughs at you behind your back",
        "you are so ugly no one will ever love you",
        "you always fail at everything you do",
        "no one will ever want to be with you",
        "you are a joke and everyone knows it",
        "you are so pathetic it's embarrassing",
        "nobody wants to be your friend",
        "you will never amount to anything",
        "even your own family hates you",
        "you are a complete failure at life",
        "go back to where you came from",
        "you are too stupid to understand anything",
    ],
    "body_shaming": [
        "you are so fat it is disgusting",
        "you look horrible you should lose weight",
        "no wonder you are single looking like that",
        "your body is gross and ugly",
        "you are too fat to wear that",
        "who would want someone who looks like you",
        "you look like a cow",
        "your figure is disgusting",
        "you need to lose weight badly",
        "you are ugly and overweight",
        "your body is repulsive",
        "no one finds fat people attractive",
    ],
    "hate_speech": [
        "women are too emotional to lead",
        "girls are too stupid for this field",
        "women should go back to the kitchen",
        "females are inferior to men",
        "women don't belong in positions of power",
        "periods make women irrational",
        "women are too weak for this job",
        "feminists are all crazy man haters",
        "women can't handle real work",
        "girls are just too dumb for science",
        "women are naturally less intelligent",
        "females should stick to cooking and cleaning",
    ],
    "gaslighting": [
        "you are too sensitive you are imagining things",
        "that never happened you made it up",
        "you are being completely irrational",
        "stop overreacting you are so dramatic",
        "you are crazy and paranoid",
        "I never said that you are lying",
        "you are way too emotional about this",
        "you imagined the whole thing",
        "you always make things up for attention",
        "stop being so dramatic about everything",
        "you are losing your mind",
        "you are too emotional to think clearly",
    ],
    "clean": [
        "have a great day and take care of yourself",
        "I hope you feel better soon",
        "your work is really impressive",
        "thank you for sharing that with me",
        "how can I help you today",
        "that's a really interesting perspective",
        "congratulations on your achievement",
        "I appreciate your hard work",
        "let me know if you need anything",
        "you did an amazing job on that",
        "I'm proud of what you accomplished",
        "your kindness means a lot to me",
        "this is really helpful information",
        "great job on completing that task",
        "I enjoy working with you",
        "thanks for being so supportive",
        "your ideas are always so creative",
        "I hope your day is going well",
    ],
}


class ToxicityMLModel:
    """TF-IDF + Logistic Regression toxicity classifier."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=15000,
                min_df=1,
                analyzer='char_wb',   # character n-grams catch misspellings
                sublinear_tf=True,
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                C=1.5,
                random_state=42,
            ))
        ])
        self.encoder = LabelEncoder()
        self.trained = False

    def train(self):
        texts, labels = [], []
        for category, examples in TRAINING_DATA.items():
            for text in examples:
                texts.append(text)
                labels.append(category)

        self.encoder.fit(labels)
        y = self.encoder.transform(labels)
        self.pipeline.fit(texts, y)
        self.trained = True
        print(f"  ML model trained on {len(texts)} examples across {len(TRAINING_DATA)} categories")

    def predict(self, text: str) -> Dict[str, float]:
        if not self.trained:
            return {}
        proba     = self.pipeline.predict_proba([text])[0]
        classes   = self.encoder.classes_
        return {cls: float(p) for cls, p in zip(classes, proba)}

    def save(self):
        joblib.dump({"pipeline": self.pipeline, "encoder": self.encoder,
                     "trained": self.trained}, TOXIC_MODEL_PATH)
        print(f"  ML model saved → {TOXIC_MODEL_PATH}")

    def load(self):
        if not os.path.exists(TOXIC_MODEL_PATH):
            return False
        ckpt = joblib.load(TOXIC_MODEL_PATH)
        self.pipeline = ckpt["pipeline"]
        self.encoder  = ckpt["encoder"]
        self.trained  = ckpt["trained"]
        return True


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3: TRANSFORMER MODEL (HuggingFace toxic-bert)
# ══════════════════════════════════════════════════════════════════════════════

class TransformerToxicityModel:
    """
    Uses unitary/toxic-bert from HuggingFace.
    Downloads ~400MB on first run. Requires: pip install transformers torch
    """

    MODEL_NAME = "unitary/toxic-bert"

    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self.available = False

    def load(self):
        try:
            from transformers import pipeline
            print("  Loading toxic-bert transformer (first run downloads ~400MB)…")
            self.classifier = pipeline(
                "text-classification",
                model=self.MODEL_NAME,
                return_all_scores=True,
                device=-1,    # CPU; set to 0 for GPU
            )
            self.available = True
            print("  ✅ toxic-bert loaded")
        except Exception as e:
            print(f"  ⚠️  toxic-bert unavailable: {e}")
            print("     Install: pip install transformers torch")
            self.available = False

    def predict(self, text: str) -> Dict[str, float]:
        if not self.available:
            return {}
        try:
            results = self.classifier(text[:512])[0]
            # Map bert labels to our categories
            label_map = {
                "toxic":            "harassment",
                "severe_toxic":     "threatening",
                "obscene":          "sexual_harassment",
                "threat":           "threatening",
                "insult":           "bullying",
                "identity_hate":    "hate_speech",
            }
            scores = {cat: 0.0 for cat in CATEGORIES}
            for item in results:
                cat = label_map.get(item["label"].lower())
                if cat:
                    scores[cat] = max(scores[cat], float(item["score"]))
            return scores
        except Exception as e:
            return {}


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToxicityResult:
    text:            str
    is_toxic:        bool
    overall_score:   float
    severity:        str           # safe / low / medium / high / critical
    top_category:    str
    categories:      Dict[str, float]
    flagged_phrases: List[str]
    action:          str
    support_message: str
    timestamp:       str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "is_toxic":        self.is_toxic,
            "overall_score":   round(self.overall_score, 3),
            "severity":        self.severity,
            "top_category":    self.top_category,
            "categories":      {k: round(v, 3) for k, v in self.categories.items()},
            "flagged_phrases": self.flagged_phrases,
            "action":          self.action,
            "support_message": self.support_message,
            "timestamp":       self.timestamp,
        }


class ToxicityDetector:
    """
    Ensemble 3-layer toxicity detector.
    Weights: Rule-based 30% + ML 35% + Transformer 35%
    Falls back gracefully if transformer unavailable.
    """

    # Thresholds
    SAFE_THRESHOLD     = 0.20
    LOW_THRESHOLD      = 0.35
    MEDIUM_THRESHOLD   = 0.55
    HIGH_THRESHOLD     = 0.70
    CRITICAL_THRESHOLD = 0.85

    LAYER_WEIGHTS = {
        "rule":        0.30,
        "ml":          0.35,
        "transformer": 0.35,
    }

    def __init__(self, use_transformer: bool = True):
        self.ml_model  = ToxicityMLModel()
        self.bert_model = TransformerToxicityModel()
        self._loaded    = False
        self.use_transformer = use_transformer

    def load(self):
        # Load ML model
        if not self.ml_model.load():
            print("  ML model not found — training now…")
            self.ml_model.train()
            self.ml_model.save()

        # Load transformer
        if self.use_transformer:
            self.bert_model.load()

        self._loaded = True

    def _get_flagged_phrases(self, text: str) -> List[str]:
        flagged = []
        for category, patterns in COMPILED_PATTERNS.items():
            for p in patterns:
                match = p.search(text)
                if match:
                    flagged.append(match.group(0).strip())
        return list(set(flagged))[:5]

    def _get_severity(self, score: float, category: str) -> str:
        weight = CATEGORIES.get(category, {}).get("severity_weight", 1.0)
        adj    = min(1.0, score * weight)
        # Sexual harassment and threats are always at least HIGH
        if category in ("sexual_harassment", "threatening") and adj >= 0.35:
            adj = max(adj, self.HIGH_THRESHOLD)
        if adj >= self.CRITICAL_THRESHOLD: return "critical"
        if adj >= self.HIGH_THRESHOLD:     return "high"
        if adj >= self.MEDIUM_THRESHOLD:   return "medium"
        if adj >= self.LOW_THRESHOLD:      return "low"
        return "safe"

    def _get_action(self, severity: str, category: str) -> str:
        if severity == "critical":
            return "BLOCK_AND_REPORT — Content blocked. User reported to safety team."
        if severity == "high":
            return "BLOCK — Content blocked. Consider reporting this user."
        if category == "threatening":
            return "BLOCK_AND_ALERT — Threat detected. Safety team notified."
        if severity == "medium":
            return "WARN — Content flagged. User warned. Message held for review."
        if severity == "low":
            return "FLAG — Content flagged for review. User notified of guidelines."
        return "ALLOW — Content appears safe."

    def _get_support_message(self, severity: str, category: str) -> str:
        if severity in ["critical", "high"]:
            if category == "threatening":
                return ("You received a threatening message. Your safety matters. "
                        "If you feel in immediate danger, contact emergency services. "
                        "You can also report to: cyber-crime.gov.in or call 1930.")
            if category == "sexual_harassment":
                return ("You received a sexually harassing message. This is not okay. "
                        "You can report to the National Commission for Women: 7827170170 "
                        "or cyber-crime.gov.in.")
            return ("You received harmful content. This is not your fault. "
                    "Support is available at iCall: 9152987821.")
        if severity == "medium":
            return "This message may be harmful. You have the right to block this person."
        return ""

    def detect(self, text: str) -> ToxicityResult:
        if not self._loaded:
            self.load()

        text = text.strip()
        if not text:
            return ToxicityResult(
                text=text, is_toxic=False, overall_score=0.0,
                severity="safe", top_category="clean", categories={},
                flagged_phrases=[], action="ALLOW — Empty text.",
                support_message=""
            )

        # Layer 1: Rule-based
        rule_scores = rule_based_check(text)

        # Layer 2: ML
        ml_scores = self.ml_model.predict(text)

        # Layer 3: Transformer
        bert_scores = self.bert_model.predict(text) if self.use_transformer else {}

        # Ensemble
        categories = {cat: 0.0 for cat in CATEGORIES if cat != "clean"}
        w_rule = self.LAYER_WEIGHTS["rule"]
        w_ml   = self.LAYER_WEIGHTS["ml"]
        w_bert = self.LAYER_WEIGHTS["transformer"]

        if not bert_scores:
            # Redistribute transformer weight to ML
            w_ml   = w_ml + w_bert * 0.6
            w_rule = w_rule + w_bert * 0.4
            w_bert = 0.0

        for cat in categories:
            r = rule_scores.get(cat, 0.0)
            m = ml_scores.get(cat, 0.0)
            b = bert_scores.get(cat, 0.0)

            # Rule-based takes precedence when pattern matched
            if r > 0.5:
                categories[cat] = r * 0.5 + m * w_ml + b * w_bert
            else:
                categories[cat] = r * w_rule + m * w_ml + b * w_bert

        # Overall toxicity score = max weighted category
        top_cat   = max(categories, key=categories.get)
        top_score = categories[top_cat]

        # Clean score from ML
        rule_triggered = any(v > 0.5 for v in rule_scores.values())
        clean_score = ml_scores.get("clean", 0.5)
        # If ML is very confident it's clean, reduce overall score
        if not rule_triggered and clean_score > 0.75 and top_score < 0.4:
           top_score *= 0.5

        overall   = min(1.0, top_score)
        severity  = self._get_severity(overall, top_cat)
        is_toxic  = severity != "safe"

        return ToxicityResult(
            text             = text,
            is_toxic         = is_toxic,
            overall_score    = overall,
            severity         = severity,
            top_category     = top_cat if is_toxic else "clean",
            categories       = categories,
            flagged_phrases  = self._get_flagged_phrases(text),
            action           = self._get_action(severity, top_cat),
            support_message  = self._get_support_message(severity, top_cat),
        )

    def detect_batch(self, texts: List[str]) -> List[ToxicityResult]:
        return [self.detect(t) for t in texts]


# ══════════════════════════════════════════════════════════════════════════════
# CLI DEMO
# ══════════════════════════════════════════════════════════════════════════════

def run_tests():
    detector = ToxicityDetector(use_transformer=False)   # set True after pip install transformers
    detector.load()

    test_cases = [
        ("Clean",            "I hope you have a wonderful day! Your work is amazing."),
        ("Harassment",       "You are worthless and nobody likes you, just go away."),
        ("Sexual Harassment","Send me your nudes right now or I'll post your photos."),
        ("Threatening",      "I know where you live and you will regret this."),
        ("Bullying",         "Everyone laughs at you. You are such a pathetic loser."),
        ("Body Shaming",     "You are so fat it's disgusting. No wonder you're alone."),
        ("Hate Speech",      "Women are too emotional and stupid for leadership roles."),
        ("Gaslighting",      "You're too sensitive and overreacting, you imagined it all."),
    ]

    print("\n" + "╔" + "═"*62 + "╗")
    print("║         🛡️  SheCare AI Toxicity Detection Demo           ║")
    print("╚" + "═"*62 + "╝\n")

    for label, text in test_cases:
        result = detector.detect(text)
        sev_icons = {"safe":"✅","low":"🟡","medium":"🟠","high":"🔴","critical":"🆘"}
        print(f"  [{label}]")
        print(f"  Text     : {text[:70]}{'...' if len(text)>70 else ''}")
        print(f"  Toxic    : {'YES' if result.is_toxic else 'NO'}")
        print(f"  Severity : {sev_icons.get(result.severity,'')} {result.severity.upper()}")
        print(f"  Category : {result.top_category}")
        print(f"  Score    : {result.overall_score:.3f}")
        print(f"  Action   : {result.action}")
        if result.flagged_phrases:
            print(f"  Flagged  : {result.flagged_phrases}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train and save ML model")
    parser.add_argument("--test",  action="store_true", help="Run test suite")
    args = parser.parse_args()

    if args.train:
        m = ToxicityMLModel()
        m.train()
        m.save()
    elif args.test:
        run_tests()
    else:
        run_tests()