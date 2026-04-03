"""
STT avec détection de voix en temps réel (VAD) et transcription streaming.

Architecture :
  ┌─────────────────────────────────────────────────────────────────┐
  │  sounddevice callback (thread audio)                            │
  │    └─► RingBuffer (audio brut, 100ms chunks)                    │
  │          └─► VADProcessor (machine à états)                     │
  │                ├─ SILENCE  : discard                            │
  │                ├─ SPEAKING : accumule les chunks                 │
  │                └─ END      : enqueue utterance ──► WhisperQueue │
  │                                                   └─► callback  │
  └─────────────────────────────────────────────────────────────────┘

Nouveautés vs ancienne version :
- Capture en chunks 100ms (au lieu d'enregistrements 30s bloquants)
- VAD par énergie RMS + hystérésis (évite les faux déclenchements)
- Transcription Whisper déclenchée automatiquement à la fin d'un énoncé
- Callback partiel optionnel (affichage en direct du texte reconnu)
- Timeout de sécurité : si quelqu'un parle plus de MAX_SPEECH_DURATION,
  on force une transcription intermédiaire
"""

import threading
import time
import numpy as np
import sounddevice as sd
import whisper
import torch
from collections import deque
from queue import Queue, Empty
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from utils.config_manager import size_stt, device as cfg_device

# ─────────────────────────────────────────────
# Constantes audio
# ─────────────────────────────────────────────

SAMPLE_RATE        = 16_000      # Hz (requis par Whisper)
CHUNK_MS           = 100         # Durée d'un chunk en ms
CHUNK_SAMPLES      = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 1600 samples

# ─────────────────────────────────────────────
# Paramètres VAD
# ─────────────────────────────────────────────

VAD_ENERGY_THRESHOLD   = 0.015   # Seuil RMS pour « est-ce de la parole ? »
VAD_SPEECH_PAD_CHUNKS  = 3       # Chunks de marge avant/après la parole (300ms)
VAD_SILENCE_CHUNKS     = 12      # Chunks de silence pour clore un énoncé (1.2s)
VAD_MIN_SPEECH_CHUNKS  = 4       # Durée minimale d'une vraie prise de parole (400ms)
MAX_SPEECH_DURATION    = 30.0    # Forcer la coupure après N secondes de parole

# ─────────────────────────────────────────────
# Modèle Whisper (singleton thread-safe)
# ─────────────────────────────────────────────

_model: Optional[whisper.Whisper] = None
_model_lock = threading.Lock()


def _load_model() -> whisper.Whisper:
    global _model
    with _model_lock:
        if _model is None:
            dev = cfg_device()
            print(f"⏳ Chargement Whisper ({size_stt()}) sur {dev}…")
            torch_device = "cuda" if dev == "gpu" else "cpu"
            _model = whisper.load_model(size_stt(), device=torch_device)
            print("✓ Whisper prêt")
    return _model


# ─────────────────────────────────────────────
# Machine à états VAD
# ─────────────────────────────────────────────

class _VadState(Enum):
    SILENCE  = auto()
    SPEAKING = auto()


@dataclass
class VADProcessor:
    """
    Consomme des chunks audio (numpy float32) et produit des
    utterances complètes dans `utterance_queue`.

    Paramètres ajustables :
        energy_threshold  : RMS minimum pour considérer un chunk comme « voix »
        silence_chunks    : nb de chunks silencieux consécutifs pour clore un énoncé
        min_speech_chunks : durée minimale pour qu'un segment soit considéré valide
        max_speech_sec    : durée max avant coupure forcée
    """

    utterance_queue: Queue
    energy_threshold:  float = VAD_ENERGY_THRESHOLD
    silence_chunks:    int   = VAD_SILENCE_CHUNKS
    min_speech_chunks: int   = VAD_MIN_SPEECH_CHUNKS
    max_speech_sec:    float = MAX_SPEECH_DURATION

    # État interne
    _state:          _VadState  = field(default=_VadState.SILENCE, init=False)
    _speech_buffer:  list       = field(default_factory=list, init=False)
    _silence_count:  int        = field(default=0, init=False)
    _speech_start:   float      = field(default=0.0, init=False)

    def process(self, chunk: np.ndarray) -> None:
        """Traite un chunk et met une utterance dans la queue si elle est terminée."""
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        is_voice = rms > self.energy_threshold

        if self._state == _VadState.SILENCE:
            if is_voice:
                self._state       = _VadState.SPEAKING
                self._speech_buffer = [chunk.copy()]
                self._silence_count = 0
                self._speech_start  = time.time()
                print("\n🎙️  Parole détectée…", end="", flush=True)

        elif self._state == _VadState.SPEAKING:
            self._speech_buffer.append(chunk.copy())
            elapsed = time.time() - self._speech_start

            if is_voice:
                self._silence_count = 0
                print(".", end="", flush=True)  # indicateur visuel
            else:
                self._silence_count += 1

            # ── Fin d'énoncé : silence prolongé ──
            if self._silence_count >= self.silence_chunks:
                self._finalize()

            # ── Coupure forcée : trop long ──
            elif elapsed >= self.max_speech_sec:
                print(f"\n⚠️  Coupure après {self.max_speech_sec:.0f}s de parole continue")
                self._finalize()

    def _finalize(self) -> None:
        """Valide et enqueue l'utterance courante, puis reset l'état."""
        if len(self._speech_buffer) >= self.min_speech_chunks:
            audio = np.concatenate(self._speech_buffer, axis=0)
            self.utterance_queue.put(audio)
            duration = len(audio) / SAMPLE_RATE
            print(f"\n✅ Énoncé capturé ({duration:.1f}s), transcription en cours…")
        else:
            print("\n⊘ Segment trop court, ignoré")

        self._state         = _VadState.SILENCE
        self._speech_buffer = []
        self._silence_count = 0


# ─────────────────────────────────────────────
# Worker de transcription
# ─────────────────────────────────────────────

def _transcription_worker(
    utterance_queue:  Queue,
    stop_event:       threading.Event,
    on_final:         Callable[[str], None],
    on_partial:       Optional[Callable[[str], None]] = None,
    language:         str = "fr",
) -> None:
    """
    Consomme les utterances et appelle `on_final(text)` pour chaque résultat.
    `on_partial` est appelé avec les segments intermédiaires si disponible.
    """
    model = _load_model()
    dev   = cfg_device()

    transcribe_opts = {
        "language": language,
        "fp16":     dev == "gpu",
        "beam_size": 5,
        "best_of":   5,
        "task":      "transcribe",
        "verbose":   False,
        # Retourner les segments permet un retour partiel
        "word_timestamps": False,
    }

    print(f"[STT Worker] Prêt (device={dev}, lang={language})")

    while not stop_event.is_set() or not utterance_queue.empty():
        try:
            audio = utterance_queue.get(timeout=0.5)
        except Empty:
            continue

        t0 = time.time()
        try:
            result = model.transcribe(audio, **transcribe_opts)
        except Exception as exc:
            print(f"[STT] Erreur transcription : {exc}")
            utterance_queue.task_done()
            continue

        elapsed = time.time() - t0
        full_text = result["text"].strip()

        # Retour partiel segment par segment
        if on_partial and result.get("segments"):
            for seg in result["segments"]:
                seg_text = seg["text"].strip()
                if seg_text:
                    on_partial(seg_text)

        if full_text:
            print(f"\n📝 [{elapsed:.2f}s] {full_text}")
            on_final(full_text)
        else:
            print("\n⊘ Résultat vide, ignoré")

        # Libération mémoire GPU
        if dev == "gpu":
            torch.cuda.empty_cache()

        utterance_queue.task_done()

    print("[STT Worker] Arrêt")


# ─────────────────────────────────────────────
# Entrée publique : transcription_loop
# ─────────────────────────────────────────────

def transcription_loop(
    interval:          int                         = 30,   # ignoré (kept for API compat)
    callback:          Optional[Callable[[str], None]] = None,
    partial_callback:  Optional[Callable[[str], None]] = None,
    language:          str                         = "fr",
    energy_threshold:  float                       = VAD_ENERGY_THRESHOLD,
    silence_duration:  float                       = VAD_SILENCE_CHUNKS * CHUNK_MS / 1000,
) -> None:
    """
    Boucle de transcription continue avec VAD temps réel.

    Args:
        interval:         Ignoré (compat. avec l'ancienne API) — la détection est
                          maintenant automatique.
        callback:         Appelé avec le texte final de chaque énoncé.
        partial_callback: Appelé avec les segments intermédiaires (optionnel).
        language:         Langue pour Whisper (ex: "fr", "en").
        energy_threshold: Seuil RMS pour détecter la parole (0.0–1.0).
        silence_duration: Durée de silence (secondes) avant de clore un énoncé.
    """

    utterance_queue = Queue(maxsize=8)
    stop_event      = threading.Event()

    silence_chunks = max(1, int(silence_duration * 1000 / CHUNK_MS))
    vad = VADProcessor(
        utterance_queue   = utterance_queue,
        energy_threshold  = energy_threshold,
        silence_chunks    = silence_chunks,
    )

    # ── Thread de transcription ──
    transcriber = threading.Thread(
        target     = _transcription_worker,
        args       = (utterance_queue, stop_event, callback, partial_callback, language),
        daemon     = True,
        name       = "STT-Transcriber",
    )
    transcriber.start()

    # ── Buffer inter-thread pour sounddevice ──
    raw_queue: Queue = Queue(maxsize=200)  # ~20s de tampon max

    def _audio_callback(indata: np.ndarray, frames: int, t, status) -> None:
        """Callback sounddevice — s'exécute dans le thread audio OS."""
        if status:
            print(f"[Audio] {status}", flush=True)
        chunk = indata[:, 0].copy()   # mono
        try:
            raw_queue.put_nowait(chunk)
        except Exception:
            pass  # buffer plein → on lâche le chunk

    print("🚀 Démarrage de la transcription en temps réel…")
    print(f"   Seuil énergie : {energy_threshold}  |  Silence : {silence_duration:.1f}s")
    print("   (Ctrl+C pour arrêter)\n")

    # ── Thread VAD (consomme raw_queue → vad.process) ──
    def _vad_loop() -> None:
        while not stop_event.is_set():
            try:
                chunk = raw_queue.get(timeout=0.3)
                vad.process(chunk)
            except Empty:
                continue
        print("[VAD] Arrêt")

    vad_thread = threading.Thread(target=_vad_loop, daemon=True, name="STT-VAD")
    vad_thread.start()

    # ── Flux audio principal (bloquant) ──
    try:
        with sd.InputStream(
            samplerate = SAMPLE_RATE,
            channels   = 1,
            dtype      = "float32",
            blocksize  = CHUNK_SAMPLES,
            callback   = _audio_callback,
        ):
            print("🎤 Micro ouvert — parlez !")
            while not stop_event.is_set():
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Arrêt…")
    finally:
        stop_event.set()
        vad_thread.join(timeout=2.0)
        # Attendre que la queue soit vidée
        utterance_queue.join()
        transcriber.join(timeout=5.0)
        if cfg_device() == "gpu":
            torch.cuda.empty_cache()
        print("✓ Transcription arrêtée")


# ─────────────────────────────────────────────
# Test standalone
# ─────────────────────────────────────────────

if __name__ == "__main__":
    def on_text(text: str) -> None:
        print(f"\n>>> CALLBACK: {text}\n")

    def on_partial(text: str) -> None:
        print(f"  … {text}", end="\r", flush=True)

    transcription_loop(
        callback         = on_text,
        partial_callback = on_partial,
        language         = "fr",
    )