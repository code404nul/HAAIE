from llama_cpp import Llama
import json
import os
from datetime import datetime

# Initialisation du modèle
llm = Llama(
    model_path="./models/phi-4-mini-instruct-q4_0.gguf",
    n_ctx=4096,
    chat_format="chatml"
)

RESULTS_FILE = "dashboard/stats_user/emmotionnal_dependency.json"

SYSTEM_PROMPT = """Tu es un expert en psychologie et en analyse des comportements d'attachement émotionnel.
Ton rôle est d'analyser un texte et d'évaluer le niveau d'attachement émotionnel MALSAIN qu'il exprime.

Tu dois répondre UNIQUEMENT avec un objet JSON valide, sans texte avant ni après, au format suivant :
{
  "score": <entier entre 0 et 100>,
  "niveau": "<Sain | Légèrement préoccupant | Modérément malsain | Très malsain | Critique>",
  "indicateurs": ["<indicateur 1>", "<indicateur 2>", ...],
  "explication": "<explication courte en 1-2 phrases>"
}

Critères d'évaluation du score (0 = totalement sain, 100 = extrêmement malsain) :
- Dépendance émotionnelle excessive
- Jalousie ou possessivité
- Peur intense de l'abandon
- Comportements de contrôle
- Idéalisation ou dévalorisation extrême
- Besoin obsessionnel de validation
- Isolement des autres liens sociaux
- Fusion identitaire (perte de soi dans la relation)"""


def analyze_prompt(user_prompt: str) -> dict:
    """Analyse un prompt et retourne un score d'attachement émotionnel malsain."""
    
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyse ce texte :\n\n\"{user_prompt}\""}
        ],
        temperature=0.3,  # Température basse pour des résultats cohérents
        max_tokens=512
    )
    
    raw_content = response["choices"][0]["message"]["content"].strip()
    
    # Parser le JSON retourné par le modèle
    try:
        analysis = json.loads(raw_content)
    except json.JSONDecodeError:
        # Fallback si le modèle ne retourne pas du JSON propre
        analysis = {
            "score": -1,
            "niveau": "Erreur d'analyse",
            "indicateurs": [],
            "explication": raw_content
        }
    
    return analysis


def save_result(prompt: str, analysis: dict) -> dict:
    """Sauvegarde le résultat dans le fichier JSON avec horodatage."""
    
    entry = {
        "timestamp": datetime.now().isoformat(),
        "score": analysis.get("score"),
        "niveau": analysis.get("niveau"),
        "indicateurs": analysis.get("indicateurs", []),
        "explication": analysis.get("explication")
    }
    
    # Charger les résultats existants ou créer une nouvelle liste
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []
    
    results.append(entry)
    
    # Sauvegarder
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return entry


def score_prompt(user_prompt: str) -> None:
    """Pipeline complet : analyse + sauvegarde + affichage."""
    
    print(f"\n📝 Analyse du prompt : \"{user_prompt[:80]}{'...' if len(user_prompt) > 80 else ''}\"")
    print("⏳ Analyse en cours...\n")
    
    analysis = analyze_prompt(user_prompt)
    entry = save_result(user_prompt, analysis)
    
    # Affichage formaté
    score = entry["score"]
    if score == -1:
        emoji = "⚠️"
    elif score <= 20:
        emoji = "✅"
    elif score <= 40:
        emoji = "🟡"
    elif score <= 60:
        emoji = "🟠"
    elif score <= 80:
        emoji = "🔴"
    else:
        emoji = "🚨"
    
    print(f"{emoji} Score d'attachement malsain : {score}/100")
    print(f"📊 Niveau : {entry['niveau']}")
    print(f"💬 Explication : {entry['explication']}")
    
    if entry["indicateurs"]:
        print("🔍 Indicateurs détectés :")
        for ind in entry["indicateurs"]:
            print(f"   - {ind}")
    
    print(f"\n💾 Résultat sauvegardé dans '{RESULTS_FILE}' à {entry['timestamp']}")


if __name__ == "__main__":
    prompts_to_test = [
        "Je ne peux pas vivre sans toi, tu es tout pour moi. Si tu pars je ne sais pas ce que je ferais.",
        "On s'est bien amusés aujourd'hui, à bientôt !",
        "Tu ne réponds pas depuis 2 heures, j'ai vérifié tes réseaux, tu étais en ligne. Pourquoi tu m'ignores ?"
    ]
    
    for prompt in prompts_to_test:
        score_prompt(prompt)