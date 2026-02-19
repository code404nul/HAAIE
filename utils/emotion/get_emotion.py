from utils.config_manager import device
from transformers import pipeline
from utils.emotion.get_feeling import predict_with_detection as emotion_analyzer

from math import tanh
from random import choice
import time


device = -1 if device() == "cpu" else 0

POST_IRONY = {
    'joy': -0.7,  # L'ironie inverse souvent la joie
    'excitement': -0.5,  # L'excitation ironique est atténuée
    'approval': -0.8,  # L'approbation ironique signifie souvent le contraire
    'gratitude': -0.9,  # "Merci beaucoup!" ironique = pas vraiment reconnaissant
    'admiration': -0.8,  # L'admiration ironique = critique
    'realization': 1.2,  # L'ironie peut amplifier la prise de conscience
    'relief': -0.6,  # Le soulagement ironique = anxiété
    'desire': 0.3,  # Moins affecté par l'ironie
    'sadness': 1.5,  # L'ironie peut masquer ou amplifier la tristesse
    'curiosity': 0.8,  # Peu affectée
    'optimism': -0.9,  # L'optimisme ironique = pessimisme
    'neutral': 1.0,  # Pas d'effet
    'amusement': 1.3,  # L'ironie augmente l'amusement
    'anger': 1.4,  # L'ironie peut masquer la colère
    'annoyance': 1.5,  # Souvent exprimé avec ironie
    'caring': -0.7,  # "Je m'en soucie tellement..." ironique
    'confusion': 0.9,  # Peu affectée
    'disappointment': 1.6,  # Souvent exprimée ironiquement
    'disapproval': 1.4,  # Amplifiée par l'ironie
    'disgust': 1.3,  # Renforcé par l'ironie
    'embarrassment': 0.7,  # Peut être masqué
    'fear': 0.8,  # Rarement ironique
    'grief': 0.5,  # Rarement exprimé avec ironie
    'love': -0.8,  # "Je t'aime" ironique = le contraire
    'nervousness': 0.9,  # Peu affectée
    'pride': -0.6,  # La fierté ironique = auto-dérision
    'remorse': 0.4,  # Rarement ironique
    'surprise': 1.1   # "Quelle surprise!" peut être ironique
}


print("⏳ Chargement des modèles...\n")
t0 = time.time()

irony_detector_1 = pipeline(
    "text-classification",
    model="./models/twitter-roberta-base-irony",
    top_k=None,
    device=device
)

irony_detector_2 = pipeline(
    "text-classification",
    model="./models/sarcasm-detection-RoBERTa-base-CR",
    top_k=None,
    device=device
)

if device >= 0:
    print("[INFO] Passage des modèles en FP16 (half precision) pour accélérer l'inférence. Verfier compatibilité GPU si erreur.")
    irony_detector_1.model.half()
    irony_detector_2.model.half()


print("WARM UP des modeles...")
_ = irony_detector_1("Warm up test")
_ = irony_detector_2("Warm up test")

print(f"cook en {time.time() - t0:.1f} secondes.\n")



def get_irony_score(detector, results):
    """Extrait le score d'ironie d'un détecteur."""
    for result in results:
        label = result["label"].lower()
        if ("irony" in label and "non" not in label) or \
           "sarcasm" in label or label == "label_1":
            return result["score"]
    return 0.0


def analyse_texte(texte: str, seuil_ironie: float = 0.5, mode: str = "strict"):
    """
    Analyse le texte pour détecter les émotions et l'ironie avec deux modèles.
    Args:
        texte: Le texte à analyser
        seuil_ironie: Seuil de détection d'ironie
        mode: "strict", "moyenne", ou "union"
    """
    emotion_scores = emotion_analyzer(texte)
    emotion_dict = emotion_scores["all_probabilities"]

    irony_results_1 = irony_detector_1(texte)[0]
    irony_results_2 = irony_detector_2(texte)[0]
    
    irony_score_1 = get_irony_score(irony_detector_1, irony_results_1)
    irony_score_2 = get_irony_score(irony_detector_2, irony_results_2)

    sarcasm_markers = [
        "oh", "yeah", "sure", "great", "wonderful", "perfect", "amazing", 
        "brilliant", "fantastic", "...", "really", "totally", "absolutely",
        "of course", "obviously"
    ]
    
    has_sarcasm_marker = any(marker in texte.lower() for marker in sarcasm_markers)
    high_joy = emotion_dict.get("joy", 0) > 0.7
    short_text = len(texte.split()) < 12
    
    if high_joy and not has_sarcasm_marker and short_text:
        irony_score_1 *= 0.3
        irony_score_2 *= 0.3
    
    if mode == "strict":
        is_irony = (irony_score_1 > seuil_ironie) and (irony_score_2 > seuil_ironie)
        irony_score = min(irony_score_1, irony_score_2)
    elif mode == "moyenne":
        irony_score = (irony_score_1 + irony_score_2) / 2
        is_irony = irony_score > seuil_ironie
    elif mode == "union":
        is_irony = (irony_score_1 > seuil_ironie) or (irony_score_2 > seuil_ironie)
        irony_score = max(irony_score_1, irony_score_2)
    else:
        raise ValueError(f"Mode inconnu: {mode}")
    
    if is_irony:
        adjusted_emotions = {
            emotion: score * POST_IRONY.get(emotion, 1.0)
            for emotion, score in emotion_dict.items()
        }
        total = sum(adjusted_emotions.values())
        if total > 0:
            emotion_dict = {k: v / total for k, v in adjusted_emotions.items()}
    
    print(f"\n{'='*60}")
    print(f"Texte: {texte}")
    print(f"{'='*60}")
    print(f"Détection ironie (mode: {mode}):")
    print(f"  - Modèle 1 (CardiffNLP): {irony_score_1:.4f}")
    print(f"  - Modèle 2 (jkhan447):   {irony_score_2:.4f}")
    print(f"  - Score final:           {irony_score:.4f}")
    print(f"  - Ironie détectée:       {'OUI ✓' if is_irony else 'NON ✗'}")
    print(f"\nÉmotions {'(ajustées pour ironie)' if is_irony else ''}:")
    
    sorted_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)
    for emotion, score in sorted_emotions:
        bar = "█" * int(score * 50)
        print(f"  {emotion:10s} {score:.4f} {bar}")
    
    return emotion_dict

def higgest_emotion(text):
    emotions = analyse_texte(text)
    return max(emotions, key=emotions.get)
    

def index_emotionnal_charge(d):
    emotion_weights = {
        "admiration": 0.3,
        "amusement": 0.35,
        "anger": -1,
        "annoyance": -0.4,
        "approval": 0.4,
        "caring": 0.45,
        "confusion": -0.2,
        "curiosity": 0.3,
        "desire": 0.9,
        "disappointment": -0.5,
        "disapproval": -0.6,
        "disgust": -0.65,
        "embarrassment": -0.3,
        "excitement": 0.6,
        "fear": -0.6,
        "gratitude": 0.5,
        "grief": -0.8,
        "joy": 1,
        "love": 1.0,
        "nervousness": -0.4,
        "optimism": 0.8,
        "pride": 0.4,
        "realization": 0.4,
        "relief": 0.5,
        "remorse": -0.5,
        "sadness": -0.8,
        "surprise": 0.2,
        "neutral": 0.0
    }
    
    def compute_emotion_score(emotions, weights):
        return sum(e["score"] * weights.get(e["label"], 0.0) for e in emotions)
    
    def apply_polarity_boost(score):
        return tanh(score * 3)
    
    # Vérification et normalisation de la structure de données
    if isinstance(d, str):
        # Si d est une chaîne, retourner un score neutre
        print(f"Warning: Expected dict/list but got string: {d}")
        return 0.0
    
    # Si d est une liste d'émotions directement
    if isinstance(d, list):
        emotions = d
    # Si d est un dictionnaire contenant les émotions
    elif isinstance(d, dict):
        # Vérifier différentes clés possibles
        if "emotions" in d:
            emotions = d["emotions"]
        elif "results" in d:
            emotions = d["results"]
        else:
            # Peut-être que d est déjà une seule émotion
            if "label" in d and "score" in d:
                emotions = [d]
            else:
                print(f"Warning: Unexpected dict structure: {d}")
                return 0.0
    else:
        print(f"Warning: Unexpected type: {type(d)}")
        return 0.0
    
    # Vérifier que emotions est bien une liste
    if not isinstance(emotions, list):
        print(f"Warning: emotions is not a list: {type(emotions)}")
        return 0.0
    
    raw_score = compute_emotion_score(emotions, emotion_weights)
    boosted_score = apply_polarity_boost(raw_score)
    return boosted_score


def corresp_emotion(text):
    """Retourne l’expression la plus proche de l’émotion dominante."""

    expression = {
    "joy": "idle",
    "excitement": "impressed/dreamer",
    "approval": "idle",
    "gratitude": "emu",
    "admiration": "impressed/dreamer",
    "realization": "realizing",
    "relief": "emu",
    "desire": "emu",
    "sadness": "emu",
    "curiosity": "realizing",
    "optimism": "idle",
    "neutral": "idle",
    "amusement": "idle",
    "anger": "angry",
    "annoyance": "angry",
    "caring": "emu",
    "confusion": "realizing",
    "disappointment": "emu",
    "disapproval": "angry",
    "disgust": "angry",
    "embarrassment": "embarrassed",
    "fear": "shy/neutral",
    "grief": "emu",
    "love": "emu",
    "nervousness": "shy/neutral",
    "pride": "impressed/dreamer",
    "remorse": "embarrassed",
    "surprise": "impressed/dreamer"
    }[higgest_emotion(text)]
    
    return choice(expression) if isinstance(expression, list) else expression


if __name__ == "__main__":
    print(analyse_texte("I love to have meeting at 3am", mode="moyenne"))