import os
import wave
from huggingface_hub import snapshot_download
from piper import PiperVoice
from pydub import AudioSegment
from pydub.playback import play

def download_full_directory():
    """Télécharge tous les fichiers du dossier fr/fr_FR/upmc/medium"""
    local_dir = snapshot_download(
        repo_id="rhasspy/piper-voices",
        allow_patterns="fr/fr_FR/upmc/medium/*",
        local_dir="./models",
        local_dir_use_symlinks=False
    )
    print(f"Fichiers téléchargés dans : {local_dir}")
    return local_dir

def synthesize_audio(voice, text, file_path = "output.wav", play_audio = False):
    """
    Synthétise du texte en audio avec Piper et sauvegarde avec pydub
    
    Args:
        voice: Instance PiperVoice
        text: Texte à synthétiser
        output_path: Chemin du fichier de sortie
        play_audio: Si True, joue l'audio après la synthèse
    
    Returns:
        AudioSegment: L'objet audio créé
    """
    print(f"Synthèse en cours : '{text}'")
    
    with wave.open(file_path, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    
    audio = AudioSegment.from_wav(file_path)
    
    duration = len(audio) / 1000.0 
    print(f"  Sample rate : {audio.frame_rate} Hz")
    print(f"  Durée : {duration:.2f} secondes")
    
    return (audio, duration)

def play_audio(audio): play(audio)

def init_model_TTS(): return PiperVoice.load("models/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx")

if __name__ == "__main__":
    # Télécharger le modèle si nécessaire
    if not os.path.isfile("models/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx"):
        print("Téléchargement du modèle...")
        download_full_directory()
    
    # Charger la voix
    print("Chargement du modèle...")
    voice = PiperVoice.load("models/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx")
    
    # Synthétiser du texte
    text = "Ceci est un test de synthèse vocale avec Piper et pydub."
    
    # Créer le fichier audio
    audio = synthesize_audio(voice, text)