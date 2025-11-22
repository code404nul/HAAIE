import whisper
import sounddevice as sd
import numpy as np
from time import time
import threading
from queue import Queue, Empty
from utils.config_manager import size_stt, device
import torch

# Configuration globale
model = None
model_lock = threading.Lock()
SAMPLE_RATE = 16000

def load_model():
    """Charge le modèle Whisper une seule fois (thread-safe) avec support GPU"""
    global model
    with model_lock:
        if model is None:
            device_name = device()  # Appeler device() ici pour obtenir la chaîne
            print(f"🔄 Chargement du modèle Whisper sur {device_name}...")
            
            if device_name == "gpu":
                model = whisper.load_model(size_stt(), device="cuda")
                print(f"✓ Modèle chargé sur GPU (CUDA) - {torch.cuda.get_device_name(0)}")
            else:
                model = whisper.load_model(size_stt(), device="cpu")
                print("✓ Modèle chargé sur CPU")
    
    return model

def detect_voice_activity(audio, threshold=0.01, min_speech_duration=0.5):
    """
    Détecte si l'audio contient de la parole (VAD simple)
    Args:
        audio: signal audio
        threshold: seuil d'énergie pour détecter la parole
        min_speech_duration: durée minimale de parole (en secondes)
    Returns:
        bool: True si de la parole est détectée
    """
    # Calculer l'énergie RMS par fenêtre
    window_size = int(SAMPLE_RATE * 0.1)  # Fenêtres de 100ms
    energy = np.array([
        np.sqrt(np.mean(audio[i:i+window_size]**2))
        for i in range(0, len(audio) - window_size, window_size)
    ])
    
    # Compter les fenêtres avec énergie significative
    speech_windows = np.sum(energy > threshold)
    speech_duration = speech_windows * 0.1
    
    return speech_duration >= min_speech_duration

def record_audio(duration=30, sample_rate=SAMPLE_RATE):
    """
    Enregistre l'audio depuis le microphone
    Args:
        duration: durée d'enregistrement en secondes
        sample_rate: fréquence d'échantillonnage (16kHz recommandé pour Whisper)
    """
    print(f"🎤 Enregistrement en cours ({duration}s)...")
    
    # Pré-allocation du buffer pour éviter les réallocations
    audio = sd.rec(
        int(duration * sample_rate), 
        samplerate=sample_rate, 
        channels=1, 
        dtype='float32',
        blocking=True
    )
    
    print("✓ Enregistrement terminé")
    return audio.flatten()

def recording_worker(audio_queue, duration, stop_event):
    """
    Thread worker pour l'enregistrement continu
    Args:
        audio_queue: file d'attente pour stocker les audios enregistrés
        duration: durée de chaque enregistrement
        stop_event: événement pour arrêter le thread proprement
    """
    while not stop_event.is_set():
        try:
            audio = record_audio(duration=duration)
            
            # Vérifier si l'audio contient de la parole
            if detect_voice_activity(audio):
                print("✓ Parole détectée, ajout à la queue de transcription")
                audio_queue.put(audio)
            else:
                print("⊘ Silence détecté, transcription ignorée")
                
        except Exception as e:
            print(f"❌ Erreur lors de l'enregistrement : {e}")
            if not stop_event.is_set():
                continue

def transcription_worker(audio_queue, stop_event, callback=None):
    """
    Thread worker pour la transcription avec optimisations GPU
    Args:
        audio_queue: file d'attente contenant les audios à transcrire
        stop_event: événement pour arrêter le thread proprement
        callback: fonction appelée avec le texte transcrit (optionnel)
    """
    mdl = load_model()
    device_name = device()  # Obtenir le nom du device
    
    # Options de transcription optimisées
    transcribe_options = {
        "fp16": device_name == "gpu",
        "language": "fr",  # Spécifier la langue pour accélérer
        "beam_size": 5,  # Réduire pour plus de vitesse (défaut: 5)
        "best_of": 5,  # Réduire pour plus de vitesse (défaut: 5)
    }
    
    print(f"🔧 Options de transcription: fp16={transcribe_options['fp16']}")
    
    while not stop_event.is_set() or not audio_queue.empty():
        try:
            # Attendre un audio avec timeout
            audio = audio_queue.get(timeout=1)
            
            print("🔄 Transcription en cours...")
            start = time()
            
            # Transcription avec options optimisées
            result = mdl.transcribe(audio, **transcribe_options)
            
            end = time()
            
            print(f"⏱️  Temps de transcription : {end - start:.2f} secondes")
            print(f"📝 Transcription : {result['text']}\n")
            
            # Appeler le callback si fourni
            if callback:
                callback(result['text'])
            
            # Nettoyer la mémoire GPU si utilisée
            if device_name == "gpu":
                torch.cuda.empty_cache()
            
            audio_queue.task_done()
            
        except Empty:
            continue
        """
        except Exception as e:
            print(f"❌ Erreur pendant la transcription : {e}")
            if not audio_queue.empty():
                audio_queue.task_done()
        """

def transcription_loop(interval=30, callback=None):
    """
    Boucle de transcription continue avec enregistrement et analyse en parallèle
    Args:
        interval: durée d'enregistrement (en secondes)
        callback: fonction appelée avec le texte transcrit (optionnel)
    """
    # Queue avec taille limitée pour éviter l'accumulation
    audio_queue = Queue(maxsize=2)
    stop_event = threading.Event()
    
    # Créer les threads
    recorder_thread = threading.Thread(
        target=recording_worker,
        args=(audio_queue, interval, stop_event),
        daemon=True,
        name="AudioRecorder"
    )
    transcriber_thread = threading.Thread(
        target=transcription_worker,
        args=(audio_queue, stop_event, callback),
        daemon=True,
        name="AudioTranscriber"
    )
    
    print("🚀 Démarrage de la transcription continue...")
    print(f"   Device: {device()}")
    print("   (Appuyez sur Ctrl+C pour arrêter)\n")
    
    # Démarrer les threads
    recorder_thread.start()
    transcriber_thread.start()
    
    try:
        # Attendre indéfiniment
        while recorder_thread.is_alive() or transcriber_thread.is_alive():
            recorder_thread.join(timeout=1)
            transcriber_thread.join(timeout=1)
            
    except KeyboardInterrupt:
        print("\n🛑 Arrêt de la transcription...")
        stop_event.set()
        
        # Attendre que les threads se terminent
        recorder_thread.join(timeout=5)
        transcriber_thread.join(timeout=10)
        
        # Nettoyer la mémoire GPU
        if device() == "gpu":
            torch.cuda.empty_cache()
        
        print("✓ Arrêt terminé")

if __name__ == "__main__":
    # Mode continu avec enregistrement et transcription en parallèle
    transcription_loop(interval=30)