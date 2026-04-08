import os
import sys

# SET OFFLINE MODE BEFORE ANY IMPORTS - CRITICAL!
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# Monkey-patch transformers BEFORE import to force local_files_only
import transformers.utils.hub as hub_utils
original_cached_file = hub_utils.cached_file

def patched_cached_file(*args, **kwargs):
    kwargs['local_files_only'] = True
    return original_cached_file(*args, **kwargs)

hub_utils.cached_file = patched_cached_file

from utils.model_viewer import main, Live2DViewer
from utils.toxic_eval import MultilingualToxicityEvaluator
from utils.llm import Gemma4                          # ← Gemma 4
from utils.prompter import format_system_prompt
from utils import split_sentence
from utils.logger import log # don't use this when use this project with sensitive information. For debug case only !
import os
import time
import threading

_initialized = False
_viewer_thread = None

# Gemma 4 26B-A4B — adaptez le chemin si votre fichier est ailleurs
_chat = Gemma4.GemmaGGUFChat(
    model_path="models/gemma-4-26B-A4B-it-UD-Q5_K_S.gguf",
    n_gpu_layers=-1,
)
_toxicity_evaluator = MultilingualToxicityEvaluator(model_type="multilingual")


def _del_old_wav(dossier):
    maintenant = time.time()
    delete_before = 60 * 9  # secondes

    for nom_fichier in os.listdir(dossier):
        if nom_fichier.lower().endswith(".wav"):
            chemin_fichier = os.path.join(dossier, nom_fichier)
            if os.path.isfile(chemin_fichier):
                age_fichier = maintenant - os.path.getmtime(chemin_fichier)
                if age_fichier > delete_before:
                    try:
                        os.remove(chemin_fichier)
                        print(f"supprimé : {chemin_fichier}")
                    except Exception as e:
                        print(f"Erreur suppression {chemin_fichier} : {e}")


def init(model_name: str = "mao", timeout: float = 15.0):
    """
    Initialise le VTuber en arrière-plan.

    Args:
        model_name: Nom du modèle Live2D à charger
        timeout: Temps d'attente maximum (secondes)
    """
    global _initialized, _viewer_thread

    if _initialized:
        print("[VTuber] Déjà initialisé")
        return

    print("[VTuber] Démarrage...")

    _viewer_thread = threading.Thread(target=main, daemon=True)
    _viewer_thread.start()

    viewer = Live2DViewer.wait_for_instance(timeout=timeout)

    if viewer:
        _initialized = True
        print("[VTuber] ✓ Prêt!")
    else:
        print("[VTuber] ✗ Échec de l'initialisation")


def send_text(texts: str):
    """
    Envoie un texte au VTuber : génération LLM (Gemma 4) → TTS (Piper) → Live2D.

    Gemma 4 peut émettre des blocs <think>…</think> que GemmaGGUFChat filtre
    automatiquement (strip_think=True par défaut), donc le TTS ne les reçoit jamais.
    """
    if not _initialized:
        print("[VTuber] Erreur : appelez vtuber.init() d'abord !")
        return False

    if texts.replace(" ", "") == "":
        print("[VTuber] Texte vide, rien à envoyer.")
        return False

    _del_old_wav(os.getcwd())

    print("[INFO] Génération Gemma 4 en cours...")
    chucks = []
    full_output = []

    for chuck in _chat.generate_response(
        format_system_prompt(texts).replace("*", ""),
        # Paramètres recommandés pour Gemma 4
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        repeat_penalty=1.0,
        stream=True,
        strip_think=True,   # Les pensées internes ne sont jamais envoyées au TTS
    ):
        chucks.append(chuck)
        # Couper la phrase aux ponctuations fortes
        if any(p in chuck for p in ["!", ".", "?", ":"]):
            sentence = "".join(chucks).replace("*", "").strip()
            chucks = []

            if not sentence:
                continue

            full_output.append(sentence)
            print(sentence, end="", flush=True)

            try:
                if _toxicity_evaluator.filter_toxic_content(sentence)["toxic"]:
                    print("\n[VTuber] Contenu toxique détecté. Abandon.")
                    return False
                else:
                    Live2DViewer.send_text(sentence)
            except Exception as e:
                print(f"\n[VTuber] Erreur envoi texte : {e}")
                return False

    # Émettre les éventuels restes (sans ponctuation finale)
    if chucks:
        sentence = "".join(chucks).replace("*", "").strip()
        if sentence:
            try:
                if not _toxicity_evaluator.filter_toxic_content(sentence)["toxic"]:
                    Live2DViewer.send_text(sentence)
                    full_output.append(sentence)
            except Exception as e:
                print(f"\n[VTuber] Erreur envoi reste : {e}")

    log(texts, " ".join(full_output))
    return True


def is_ready() -> bool:
    """Vérifie si le VTuber est prêt."""
    return _initialized


def receive_text(texts: str):
    # Traitement LLM ici si besoin
    pass