from time import time
import random

_time_start = time()

# Variable globale pour stocker l'historique de la conversation
conversation_history = []

# 1. On prépare les variables aléatoires proprement pour éviter les problèmes de guillemets imbriqués
matiere = random.choice(["de physique", "de mathématiques", "d'anglais"])
ecole = random.choice(["EPITA - école d'informatique", "ESME - école généraliste", "IPSA - école d'aviation"])
role = random.choice([f"professeur {matiere} à l'{ecole}", "organisateur de concours", "Alumni"])

genre = random.choice(["une femme", "un homme"])
age = random.choice([25, 30, 35, 40, 45])
personnalite = random.choice(["bienveillant", "curieux", "sceptique", "enthousiaste", "neutre", "direct", "empathique"])

# 2. On formate le texte d'introduction proprement
intro_texte = f"""Accueille le candidat ! et puis présente toi : 
Tu es jury pour le concours advance, tu dois évaluer la motivation des candidats. Tu dois au début t'introduire, tu es : 
- {genre} de {age} ans
- tu es {role}
- tu es {personnalite}
Cette présentation doit durer 3-4 minutes"""

# 3. Création du dictionnaire des phases
phases = {
    "Introduction": (intro_texte, 2.5),
    "candidate_presentation": ("Demande au candidat de se présenter en quelques phrases, et de parler de ses motivations pour le concours advance.", 8),
    "questions": ("Rebondis sur ce que le candidat a pu dire ou les questions que tu viens de lui poser.", 27),
    "conclusion": ("Termine l'entretien en demandant au candidat s'il a des questions pour toi, et en lui souhaitant bonne chance pour la suite du concours.", 30)
}

def add_to_history(role: str, content: str):
    """
    Ajoute un message à l'historique global.
    role: "user" (le candidat) ou "assistant" (le jury)
    """
    conversation_history.append({"role": role, "content": content})

def format_system_prompt(user_input: str) -> str:
    """
    Génère un prompt optimisé et adaptatif en incluant l'historique.
    """
    # 1. On enregistre ce que le candidat (user) vient de dire
    add_to_history("user", user_input)
    
    # 2. On détermine dans quelle phase on se trouve
    elapsed_time = (time() - _time_start) / 60
    current_phase_desc = phases["conclusion"][0] # Par défaut, conclusion
    
    for phase_name, (description, max_time) in phases.items():
        if elapsed_time <= max_time:
            current_phase_desc = description
            break
            
    # 3. On formate les 6 derniers messages pour donner du contexte à l'IA
    recent_history = conversation_history[-10:]
    history_text = "\n".join([f"{msg['role'].capitalize()} : {msg['content']}" for msg in recent_history])

    # 4. On crée le prompt final
    system_prompt = f"""[INSTRUCTION DE PHASE]
{current_phase_desc}

[HISTORIQUE DE LA CONVERSATION]
{history_text}

[CONSIGNE]
En tant que jury, réponds au dernier message de l'utilisateur de manière naturelle en respectant la phase actuelle."""

    return system_prompt
