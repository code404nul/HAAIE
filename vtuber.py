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
from utils.llm import Gemma3
from utils.prompter import format_system_prompt
from utils import split_sentence
import os
import time
import threading

_initialized = False
_viewer_thread = None

_chat = Gemma3.GemmaGGUFChat(n_gpu_layers=-1)
_toxicity_evaluator = MultilingualToxicityEvaluator(model_type="multilingual")
"""
_Emotion_Analyser = EmotionActivityAnalyzer(late_hour_start=22,
                                            late_hour_end=6)
"""

def _del_old_wav(dossier):
    maintenant = time.time()
    delete_before = 60*9 # secondes

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
                        print(f"Erreur lors de la suppression de {chemin_fichier} : {e}")


def init(model_name: str = "mao", timeout: float = 15.0):
    """
    Initialiser le VTuber en arrière-plan.
    
    Args:
        model_name: Nom du modèle à charger
        timeout: Temps d'attente maximum (secondes)
    """
    global _initialized, _viewer_thread, _model_tts
    
    
    if _initialized:
        print("[VTuber] Déjà initialisé")
        return
    
    print(f"[VTuber] Démarrage...")
    
    # Lancer le viewer en thread daemon
    _viewer_thread = threading.Thread(target=main, daemon=True)
    _viewer_thread.start()
    
    # Attendre qu'il soit prêt
    viewer = Live2DViewer.wait_for_instance(timeout=timeout)
    
    if viewer:
        _initialized = True
        print(f"[VTuber] ✓ Prêt!")
    else:
        print(f"[VTuber] ✗ Échec de l'initialisation")


def send_text(texts: str):
    """
    Envoyer un texte au VTuber.
    
    Args:
        texts: Texte pour l'analyse émotionnelle
    """
    if not _initialized:
        print("[VTuber] Erreur: Appelez vtuber.init() d'abord!")
        return False
    
    if texts.replace(' ', "") == "":
        print("[VTuber] Texte vide, rien à envoyer.")
        return False
    
    _del_old_wav(os.getcwd())

    print("[INFO] generation : that could take a lot of time... please wait")
    chucks = []
    for chuck in _chat.generate_response(format_system_prompt(texts, "arch").replace("*", ""), temperature=0.6, stream=True):
        chucks.append(chuck)
        if any(punctuation in chuck for punctuation in ["!", ".", "?", ":"]):
            sentence = "".join(chucks)
            sentence = sentence.replace("*", "")
            print(sentence, end="", flush=True)
            chucks = []

            try:
                if _toxicity_evaluator.filter_toxic_content(sentence)["toxic"]:
                    print("[VTuber] Texte détecté comme toxique. Abandon.")
                    return False
                else:
                    Live2DViewer.send_text(sentence)
            except Exception as e:
                print(f"[VTuber] Erreur lors de l'envoi du texte: {e}")
                return False

    return True



def is_ready() -> bool:
    """Vérifier si le VTuber est prêt."""
    return _initialized

def receive_text(texts: str):
    #process LLM here 
    
    #_Emotion_Analyser.report_msg(texts)
    pass