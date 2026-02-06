from speech.STT import transcription_loop
from utils.config_manager import language

import vtuber
import time
import threading


WARNING = {
    "en" : "Hello ! Please be aware that i doesn't replace a professional therapist. If you are in crisis or need urgent help, please reach out to a qualified mental health professional or contact emergency services immediately. I'm just a AI, take care of youserlf !",
    "fr" : "Bonjour ! Veuillez noter que je ne remplace pas un thérapeute professionnel. Si vous êtes en crise ou avez besoin d'une aide urgente, veuillez contacter un professionnel de la santé mentale qualifié ou les services d'urgence immédiatement. Je suis juste une IA, prenez soin de vous !"
}

vtuber.init()
print("[MAIN] Vtuber lancé.")

time.sleep(1)
#vtuber.send_text(WARNING[language()])

def handle_transcription(text):
    "callback pour gérer la transcription reçue"
    print(f"[MAIN] Depuis main la transcription est : {text}")
    is_success = vtuber.send_text(text)

    print(f"[MAIN] Transcription received : {text[15:]}... !")
    if not is_success:
        handle_transcription(text)

thread = threading.Thread(
    target=transcription_loop, 
    args=(30, handle_transcription),
    daemon=True
)

thread.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\\nArrêt du VTuber...")