from utils.speech.STT import transcription_loop
from utils.config_manager import language

import vtuber
import time
import threading

WARNING = {
    "en": (
        "Hello! Please be aware that I don't replace a professional therapist. "
        "If you are in crisis or need urgent help, please reach out to a qualified "
        "mental health professional or contact emergency services immediately. "
        "I'm just an AI, take care of yourself!"
    ),
    "fr": (
        "Bonjour ! Veuillez noter que je ne remplace pas un thérapeute professionnel. "
        "Si vous êtes en crise ou avez besoin d'une aide urgente, veuillez contacter "
        "un professionnel de la santé mentale qualifié ou les services d'urgence "
        "immédiatement. Je suis juste une IA, prenez soin de vous !"
    ),
}

# ── Init VTuber ──────────────────────────────────────────────────────────────
vtuber.init()
print("[MAIN] Vtuber lancé.")
time.sleep(1)


# ── Callbacks STT ─────────────────────────────────────────────────────────────

def handle_transcription(text: str) -> None:
    """
    Callback final : appelé avec l'énoncé complet reconnu par Whisper.
    Envoie le texte au modèle LLM → TTS → Live2D.
    """
    text = text.strip()
    if not text:
        return

    print(f"\n[MAIN] Énoncé reçu : {text}")
    is_success = vtuber.send_text(text)

    if not is_success:
        print("[MAIN] Échec envoi, nouvelle tentative…")
        handle_transcription(text)


def handle_partial(text: str) -> None:
    """
    Callback partiel (optionnel) : segments Whisper intermédiaires.
    Ici on les affiche uniquement — pas d'envoi au LLM pour éviter
    les doublons avec le callback final.
    """
    print(f"  ✏️  {text}", end="\r", flush=True)


# ── Lancement de la boucle STT dans un thread dédié ─────────────────────────

thread = threading.Thread(
    target=transcription_loop,
    kwargs=dict(
        callback         = handle_transcription,
        partial_callback = handle_partial,
        language         = language(),
        # Ajuster selon votre micro / environnement sonore :
        #   ↓ si le micro est sensible ou la pièce silencieuse
        #   ↑ si beaucoup de bruit de fond
        energy_threshold = 0.05,
        # Durée de silence (secondes) avant de valider l'énoncé :
        silence_duration = 1.2,
    ),
    daemon=True,
    name="STT-Loop",
)

thread.start()
print("[MAIN] Boucle STT démarrée — parlez !")

# ── Boucle principale ────────────────────────────────────────────────────────

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nArrêt du VTuber…")