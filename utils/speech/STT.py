import time
import threading
import whisper
import sounddevice as sd
import numpy as np
import torch
from queue import Queue, Empty
from collections import deque
from typing import Optional

from utils.config_manager import size_stt, device

# ── Constantes (alignées sur fireredvad/core/constants.py) ──────────────────
SAMPLE_RATE     = 16_000
CHANNELS        = 1
DTYPE           = "int16"    # int16 requis par FireRedVAD
FRAME_SHIFT     = 160        # FRAME_SHIFT_SAMPLE  (10 ms)
FRAME_LENGTH    = 400        # FRAME_LENGTH_SAMPLE (25 ms) — taille exacte de detect_frame()
MAX_SILENCE_SEC = 5.0
PRE_ROLL_FRAMES = 10         # ~100 ms de pre-roll

_whisper_model = None
_whisper_lock  = threading.Lock()

_vad_model = None
_vad_lock  = threading.Lock()


# ── Chargement des modèles ───────────────────────────────────────────────────

def load_model():
    """Charge le modèle Whisper une seule fois (thread-safe)."""
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            device_name = device()
            print(f"🔄 Chargement du modèle Whisper sur {device_name}…")
            torch_device = "cuda" if device_name == "gpu" else "cpu"
            _whisper_model = whisper.load_model(size_stt(), device=torch_device)
            label = torch.cuda.get_device_name(0) if torch_device == "cuda" else "CPU"
            print(f"✓ Modèle Whisper chargé sur {label}")
    return _whisper_model


def load_stream_vad(model_dir: str = "models/stream_VAD", use_gpu: bool = False):
    """
    Charge FireRedStreamVad une seule fois (thread-safe).

    model_dir : dossier contenant model.pth.tar et cmvn.ark
                (ex: "pretrained_models/FireRedVAD-VAD-stream-251104")
    """
    global _vad_model
    with _vad_lock:
        if _vad_model is None:
            # fireredvad est installé comme package pip — import direct
            from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig

            cfg = FireRedStreamVadConfig(
                use_gpu            = use_gpu,
                smooth_window_size = 5,
                speech_threshold   = 0.5,
                pad_start_frame    = 5,
                min_speech_frame   = 8,
                min_silence_frame  = 10,
            )
            _vad_model = FireRedStreamVad.from_pretrained(model_dir, cfg)
            print(f"[VAD] Modèle chargé depuis : {model_dir}")
    return _vad_model


# ── Enregistrement avec VAD ──────────────────────────────────────────────────

def record_on_speech(
    model_dir: str         = "models/stream_VAD",
    use_gpu: bool          = False,
    max_silence_sec: float = MAX_SILENCE_SEC,
    mic_device: Optional[int] = None,
) -> np.ndarray:
    """
    Bloque jusqu'à la fin d'une session de parole et retourne
    un numpy array float32 16 kHz (prêt pour Whisper).

    Une session commence au premier frame de parole détecté et se termine
    après `max_silence_sec` secondes de silence consécutif.
    """
    vad = load_stream_vad(model_dir, use_gpu)
    vad.reset()   # réinitialise l'état interne entre chaque session

    audio_q: Queue[np.ndarray] = Queue()
    pre_roll: deque             = deque(maxlen=PRE_ROLL_FRAMES)

    # ── callback micro : reçoit des blocs int16 mono ──
    def _callback(indata, frames, time_info, status):
        if status:
            print(f"[MIC] {status}")
        audio_q.put(indata[:, 0].copy())   # int16 [N]

    # ── générateur de frames exactement FRAME_LENGTH samples (400) ──
    # avec un pas de FRAME_SHIFT (160) pour ne pas perdre de signal
    remainder = np.array([], dtype=np.int16)

    def iter_frames():
        nonlocal remainder
        while True:
            chunk    = audio_q.get()
            combined = np.concatenate([remainder, chunk])

            n_frames = max(0, (len(combined) - FRAME_LENGTH) // FRAME_SHIFT + 1)
            for i in range(n_frames):
                start = i * FRAME_SHIFT
                yield combined[start : start + FRAME_LENGTH]

            # garder les samples non encore consommés
            consumed  = n_frames * FRAME_SHIFT
            remainder = combined[consumed:]

    print("[VAD] En écoute…")

    speech_buffer: list[np.ndarray] = []
    in_session    = False
    silence_start: Optional[float]  = None

    with sd.InputStream(
        samplerate = SAMPLE_RATE,
        channels   = CHANNELS,
        dtype      = DTYPE,
        blocksize  = FRAME_SHIFT * 4,   # livraison fréquente → faible latence
        device     = mic_device,
        callback   = _callback,
    ):
        for frame in iter_frames():
            # detect_frame() attend exactement FRAME_LENGTH_SAMPLE (400) samples int16
            result    = vad.detect_frame(frame)
            is_speech = bool(result.is_speech)

            pre_roll.append(frame.copy())

            if is_speech:
                if not in_session:
                    in_session    = True
                    silence_start = None
                    # pre-roll sans le frame courant (déjà ajouté ci-dessous)
                    speech_buffer.extend(list(pre_roll)[:-1])
                    print("[VAD] ▶ Parole détectée – enregistrement…")

                silence_start = None
                speech_buffer.append(frame)

            else:
                if in_session:
                    speech_buffer.append(frame)
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif time.monotonic() - silence_start >= max_silence_sec:
                        print(f"[VAD] ⏹  Silence > {max_silence_sec}s – session terminée.")
                        break

    if not speech_buffer:
        return np.array([], dtype=np.float32)

    # int16 → float32 normalisé [-1, 1] pour Whisper
    audio_int16 = np.concatenate(speech_buffer)
    audio_fp32  = audio_int16.astype(np.float32) / 32768.0

    duration = len(audio_fp32) / SAMPLE_RATE
    print(f"[VAD] Capturé : {duration:.2f}s | {len(audio_fp32)} samples | float32")
    return audio_fp32


# ── Workers ──────────────────────────────────────────────────────────────────

def recording_worker(
    audio_queue: Queue,
    stop_event: threading.Event,
    model_dir: str         = "models/stream_VAD",
    use_gpu: bool          = False,
    max_silence_sec: float = MAX_SILENCE_SEC,
):
    """Thread dédié à l'enregistrement. Pousse chaque session dans audio_queue."""
    while not stop_event.is_set():
        try:
            audio = record_on_speech(
                model_dir       = model_dir,
                use_gpu         = use_gpu,
                max_silence_sec = max_silence_sec,
            )
            if len(audio) == 0:
                continue

            while not stop_event.is_set():
                try:
                    audio_queue.put(audio, timeout=1)
                    break
                except Exception:
                    continue   # queue pleine, on réessaie

        except Exception as e:
            print(f"❌ Erreur enregistrement : {e}")


def transcription_worker(
    audio_queue: Queue,
    stop_event: threading.Event,
    callback=None,
):
    """Thread dédié à la transcription Whisper. Consomme audio_queue."""
    mdl         = load_model()
    device_name = device()

    transcribe_options = {
        "fp16"     : device_name == "gpu",
        "language" : "fr",
        "beam_size": 5,
        "best_of"  : 5,
    }
    print(f"🔧 Transcription : fp16={transcribe_options['fp16']}")

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            audio = audio_queue.get(timeout=1)
        except Empty:
            continue

        try:
            print("🔄 Transcription en cours…")
            t0     = time.time()
            result = mdl.transcribe(audio, **transcribe_options)
            print(f"⏱️  {time.time() - t0:.2f}s  |  📝 {result['text']}\n")

            if callback:
                callback(result["text"])

            if device_name == "gpu":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Erreur transcription : {e}")
        finally:
            audio_queue.task_done()


# ── Point d'entrée public ────────────────────────────────────────────────────

def transcription_loop(
    callback=None,
    model_dir: str         = "models/stream_VAD",
    use_gpu: bool          = True,
    max_silence_sec: float = MAX_SILENCE_SEC,
):
    """
    Lance enregistrement + transcription en parallèle (deux threads daemon).
    Appelle callback(texte) à chaque session transcrite.
    Arrêt propre sur Ctrl+C.
    """
    # Pré-charger les modèles dans le thread principal (évite les races)
    load_stream_vad(model_dir, use_gpu)
    load_model()

    audio_queue = Queue(maxsize=3)
    stop_event  = threading.Event()

    recorder_thread = threading.Thread(
        target = recording_worker,
        kwargs = dict(
            audio_queue     = audio_queue,
            stop_event      = stop_event,
            model_dir       = model_dir,
            use_gpu         = use_gpu,
            max_silence_sec = max_silence_sec,
        ),
        daemon = True,
        name   = "AudioRecorder",
    )
    transcriber_thread = threading.Thread(
        target = transcription_worker,
        kwargs = dict(
            audio_queue = audio_queue,
            stop_event  = stop_event,
            callback    = callback,
        ),
        daemon = True,
        name   = "AudioTranscriber",
    )

    print("🚀 Transcription continue démarrée")
    print(f"   Device      : {device()}")
    print(f"   Silence max : {max_silence_sec}s")
    print("   (Ctrl+C pour arrêter)\n")

    recorder_thread.start()
    transcriber_thread.start()

    try:
        while recorder_thread.is_alive() or transcriber_thread.is_alive():
            recorder_thread.join(timeout=1)
            transcriber_thread.join(timeout=1)

    except KeyboardInterrupt:
        print("\n🛑 Arrêt…")
        stop_event.set()
        recorder_thread.join(timeout=5)
        transcriber_thread.join(timeout=10)
        if device() == "gpu":
            torch.cuda.empty_cache()
        print("✓ Arrêt terminé")


if __name__ == "__main__":
    transcription_loop()