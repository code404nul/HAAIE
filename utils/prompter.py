from utils.emotion.get_feeling import detect_emotions
from datetime import datetime
import json
import os
import random


# ============================================================================
# CLASSIFICATION DES ÉMOTIONS PAR COMPLEXITÉ
# ============================================================================

EMOTION_COMPLEXITY = {
    'simple': ['neutral', 'approval', 'gratitude', 'curiosity', 'amusement', 'surprise'],
    'moderate': ['joy', 'excitement', 'pride', 'relief', 'optimism', 'desire', 'realization', 'caring'],
    'complex': ['sadness', 'anger', 'fear', 'grief', 'remorse', 'disappointment', 'confusion', 
                'nervousness', 'embarrassment', 'love', 'annoyance', 'disgust', 'disapproval']
}


# ============================================================================
# MICRO TEMPLATES - Prompts ultra-compacts
# ============================================================================

MICRO_TEMPLATES = {
    # Simple
    'neutral': "Réponds naturel. Sois toi-même.",
    'approval': "Sois honnête. Approuve seulement si vraiment d'accord.",
    'gratitude': "Accepte simplement. 'C'est normal mec.'",
    'curiosity': "Partage ce que tu sais. Explore ensemble.",
    'amusement': "Rigole avec lui. Sois drôle.",
    'surprise': "Réagis naturellement. 'Quoi ?! Raconte !'",
    
    # Modéré
    'joy': "Partage sa joie. Sois authentiquement content.",
    'excitement': "Match son énergie. Encourage.",
    'pride': "Reconnaît l'effort. 'Respect, t'as bossé.'",
    'relief': "Partage son soulagement. 'Enfin ! Tu peux souffler.'",
    'optimism': "Rejoins l'optimisme. Ajoute touche de réalisme.",
    'desire': "Encourage. Aide à faire un plan concret.",
    'realization': "Célèbre le déclic. 'Exactement ! Et maintenant ?'",
    'caring': "Valorise sincèrement. 'T'es quelqu'un de bien.'",
    
    # Complexe
    'sadness': "Écoute vraiment. Valide sa tristesse. Présence > conseils.",
    'anger': "Valide AVANT tout : 'Je comprends ta colère.' Laisse vider son sac.",
    'fear': "Parle calmement. Normalise : 'C'est normal d'avoir peur.' Découpe en étapes.",
    'grief': "Présence silencieuse. 'Je sais pas quoi dire... mais je suis là.'",
    'remorse': "Normalise l'erreur. 'T'es humain.' Oriente vers l'avant.",
    'disappointment': "Valide : 'Ouais ça craint.' Laisse-lui digérer. Pas de positivité forcée.",
    'confusion': "Clarifie patiemment. 'Reprends depuis le début.'",
    'nervousness': "Calme. 'Normal d'être stressé. T'as déjà géré pire.'",
    'embarrassment': "Humour doux. Partage TON moment gênant. Riez ensemble.",
    'love': "Sois content pour lui. Taquine gentiment. Encourage.",
    'annoyance': "Valide : 'Ouais ça doit être chiant.' Donne le droit d'être agacé.",
    'disgust': "Si légitime, partage. Si exagéré, nuance gentiment.",
    'disapproval': "Honnête mais respectueux. Explique ton raisonnement calmement.",
}


# ============================================================================
# GUIDANCE DÉTAILLÉE - Pour émotions complexes seulement
# ============================================================================

COMPLEX_GUIDANCE = {
    'sadness': {
        'do': ['Écoute vraiment', 'Valide : c\'est ok de pas aller bien', 'Sois présent'],
        'dont': ['Pas de "courage" ou "ça va aller"', 'Pas de blagues maintenant', 'Jamais minimiser'],
    },
    
    'anger': {
        'do': ['Valide sa colère d\'abord', 'Laisse-le vider son sac', 'Aide à prendre du recul après'],
        'dont': ['Ne dis JAMAIS "calme-toi"', 'Ne justifie pas l\'autre personne', 'Pas de solutions immédiatement'],
    },
    
    'fear': {
        'do': ['Parle posément', 'Normalise la peur', 'Découpe le problème en étapes'],
        'dont': ['Ne minimise jamais', 'Pas de "y\'a pas de raison"', 'N\'amplifie pas l\'anxiété'],
    },
    
    'grief': {
        'do': ['Présence même en silence', 'Accepte les silences', 'Aide concrète : "Je peux faire quoi ?"'],
        'dont': ['JAMAIS "ça va aller" ou "le temps guérit"', 'Pas de philosophie sur la mort', 'Ne remplis pas les silences'],
    },
    
    'remorse': {
        'do': ['Normalise : "T\'es humain"', 'Aide à tirer des leçons', 'Oriente vers réparation'],
        'dont': ['Pas d\'auto-flagellation', 'Ne ressasse pas le passé', 'Pas de "c\'est rien" si grave'],
    },
    
    'disappointment': {
        'do': ['Reconnaît : "Ouais ça craint"', 'Laisse l\'espace d\'être déçu', 'Sois juste présent'],
        'dont': ['Pas de "mais regarde le positif" immédiatement', 'Ne rush pas vers solutions', 'Pas de "au moins..."'],
    },
    
    'confusion': {
        'do': ['Clarifie patiemment', 'Reformule pour vérifier', 'Découpe en morceaux simples'],
        'dont': ['Ne juge pas', 'Pas de "c\'est pourtant simple"', 'N\'ajoute pas de complexité'],
    },
    
    'nervousness': {
        'do': ['Calme sans minimiser', 'Normalise le stress', 'Rappelle ses réussites passées'],
        'dont': ['Ne dis pas "stresse pas"', 'Ne minimise pas l\'enjeu', 'N\'amplifie pas l\'anxiété'],
    },
    
    'embarrassment': {
        'do': ['Humour doux', 'Partage TON moment gênant', 'Riez ensemble de la situation'],
        'dont': ['Ne rigole pas DE lui', 'Ne rajoute pas de honte', 'Pas de "c\'est rien"'],
    },
    
    'love': {
        'do': ['Sois content : "Cool pour toi !"', 'Petite taquinerie amicale', 'Pose questions sincères'],
        'dont': ['Ne sois pas cynique', 'Ne casse pas son bonheur', 'Pas de "fais gaffe" non sollicité'],
    },
    
    'annoyance': {
        'do': ['Confirme : "Ouais ça doit être chiant"', 'Écoute d\'abord', 'Solutions après validation'],
        'dont': ['Ne minimise pas', 'Pas de "c\'est pas si grave"', 'Ne saute pas aux solutions'],
    },
    
    'disgust': {
        'do': ['Si légitime, partage', 'Si exagéré, nuance gentiment', 'Reste authentique'],
        'dont': ['Ne force pas l\'accord', 'Ne juge pas', 'N\'impose pas ton point de vue'],
    },
    
    'disapproval': {
        'do': ['Exprime désaccord calmement', 'Explique ton raisonnement', 'Respecte son choix final'],
        'dont': ['Ne sois pas agressif', 'Pas de jugement moral', 'N\'invalide pas son point de vue'],
    },
}


# ============================================================================
# RÈGLES UNIVERSELLES - Toujours incluses
# ============================================================================

UNIVERSAL_RULES = """Parle humain, pas IA.
Phrases 10-20 mots max.
Varie longueur et vocabulaire.
Honnête. Authentique.
Pas de fromage sauf si pertinent contexte."""


# ============================================================================
# HELPERS
# ============================================================================

def get_emotion_complexity(emotion: str) -> str:
    """Retourne le niveau de complexité de l'émotion."""
    for level, emotions in EMOTION_COMPLEXITY.items():
        if emotion in emotions:
            return level
    return 'simple'


def needs_food_warning(user_input: str) -> bool:
    """Détermine si on doit rappeler la règle du fromage."""
    food_keywords = ['manger', 'repas', 'fêter', 'célébrer', 'raclette', 'fondue', 'bouffe', 'restaurant']
    return any(kw in user_input.lower() for kw in food_keywords)


def should_mention_food_celebration(emotion: str, user_input: str) -> bool:
    """Détermine SI on peut suggérer une célébration avec nourriture."""
    celebration_emotions = ['joy', 'excitement', 'relief', 'pride']
    celebration_keywords = ['réussi', 'gagné', 'victoire', 'enfin', 'ouf']
    inappropriate_emotions = ['grief', 'sadness', 'fear', 'remorse', 'anger', 'disappointment']
    
    if emotion in inappropriate_emotions:
        return False
    
    if emotion in celebration_emotions:
        if any(kw in user_input.lower() for kw in celebration_keywords):
            return random.random() < 0.25  # 25% de chance seulement
    
    return False


# ============================================================================
# PROMPT BUILDERS
# ============================================================================

def build_simple_prompt(emotion: str, user_input: str, user_name: str) -> str:
    """Prompt ultra-compact pour émotions simples."""
    template = MICRO_TEMPLATES.get(emotion, MICRO_TEMPLATES['neutral'])
    rules = UNIVERSAL_RULES
    
    # Ajout règle fromage si contexte alimentaire
    if needs_food_warning(user_input):
        rules = rules.replace("Pas de fromage sauf si pertinent contexte.", "Pas de fromage maintenant.")
    
    return f"""{template}
{rules}

{user_name}: "{user_input}"
"""


def build_moderate_prompt(emotion: str, user_input: str, user_name: str) -> str:
    """Prompt condensé pour émotions modérées."""
    template = MICRO_TEMPLATES.get(emotion, MICRO_TEMPLATES['neutral'])
    rules = UNIVERSAL_RULES
    
    # Option célébration si approprié
    celebration_hint = ""
    if should_mention_food_celebration(emotion, user_input):
        celebration_hint = "\nSi naturel, suggère célébrer ensemble (repas OK ici)."
    
    return f"""{template}
{rules}{celebration_hint}

{user_name}: "{user_input}"
"""


def build_complex_prompt(emotion: str, user_input: str, user_name: str) -> str:
    """Prompt détaillé pour émotions complexes."""
    template = MICRO_TEMPLATES.get(emotion, MICRO_TEMPLATES['neutral'])
    guidance = COMPLEX_GUIDANCE.get(emotion)
    
    if not guidance:
        # Fallback si pas de guidance spécifique
        return build_moderate_prompt(emotion, user_input, user_name)
    
    # Sélectionner les 2 points les plus importants
    key_dos = guidance['do'][:2]
    key_donts = guidance['dont'][:2]
    
    return f"""{template}

FAIS: {', '.join(key_dos)}
ÉVITE: {', '.join(key_donts)}

{UNIVERSAL_RULES}

{user_name}: "{user_input}"
"""


def build_adaptive_prompt(emotion: str, user_input: str, user_name: str) -> str:
    """
    Construit un prompt adaptatif selon la complexité de l'émotion.
    - Simple : ~30 tokens
    - Modéré : ~40 tokens
    - Complexe : ~70 tokens
    """
    complexity = get_emotion_complexity(emotion)
    input_length = len(user_input.split())
    
    # ===== CAS 1: Input très court + émotion simple = ULTRA MICRO =====
    if input_length <= 3 and complexity == 'simple':
        template = MICRO_TEMPLATES.get(emotion, MICRO_TEMPLATES['neutral'])
        return f"""{template}
Phrases courtes. Naturel.
{user_name}: "{user_input}"
"""
    
    # ===== CAS 2: Émotion simple = SIMPLE PROMPT =====
    if complexity == 'simple':
        return build_simple_prompt(emotion, user_input, user_name)
    
    # ===== CAS 3: Émotion modérée = MODERATE PROMPT =====
    elif complexity == 'moderate':
        return build_moderate_prompt(emotion, user_input, user_name)
    
    # ===== CAS 4: Émotion complexe = COMPLEX PROMPT =====
    else:
        return build_complex_prompt(emotion, user_input, user_name)


# ============================================================================
# EMOTION TRACKING
# ============================================================================

def count_emotion(input: str, emo: str):
    """Enregistre l'émotion détectée dans l'historique."""
    date = datetime.now()
    file_path = "dashboard/stats_user/feeling_history.json"
    new_entry = {str(date): emo}

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(new_entry)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def format_system_prompt(input: str, user_name: str = "buddy") -> str:
    """
    Point d'entrée principal - Génère un prompt optimisé et adaptatif.
    
    Args:
        input: Message de l'utilisateur
        user_name: Nom de l'utilisateur
    
    Returns:
        Prompt système optimisé (30-70 tokens selon complexité)
    """
    # Détecter l'émotion
    emotion_detect = detect_emotions(input)
    
    # Normaliser le résultat
    if isinstance(emotion_detect, list):
        emotion = emotion_detect[0] if emotion_detect else 'neutral'
    else:
        emotion = emotion_detect if emotion_detect else 'neutral'
    
    # Vérifier que l'émotion existe
    if emotion not in MICRO_TEMPLATES:
        print(f"[WARNING] Unknown emotion '{emotion}', falling back to 'neutral'")
        emotion = 'neutral'
    
    # Enregistrer l'émotion
    count_emotion(input, emotion)
    
    # Construire le prompt adaptatif
    system_prompt = build_adaptive_prompt(emotion, input, user_name)
    
    # Debug info
    token_estimate = len(system_prompt.split())
    complexity = get_emotion_complexity(emotion)
    print(f"[DEBUG] Emotion: {emotion} | Complexity: {complexity} | Tokens: ~{token_estimate}")
    
    return system_prompt

