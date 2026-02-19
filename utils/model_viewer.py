"""Live2D model viewer avec gestion synchronisée TTS + Expression + Multi-modèles."""

import math
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, ClassVar, List
import os

import pygame
from pygame.locals import DOUBLEBUF, OPENGL

import live2d.v3 as live2d
from live2d.v3 import StandardParams
from live2d.utils.lipsync import WavHandler

from utils.manage_model import ModelManager
from utils.emotion.get_emotion import corresp_emotion
from utils import lenght_to_duration

from speech.TTS import init_model_TTS, synthesize_audio


@dataclass
class ViewConfig:
    """Configuration for the Live2D viewer."""
    width: int = 800
    height: int = 1000
    title: str = "Live2D Viewer"
    frame_delay: int = 10
    background_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class TransformState:
    """State for model transformations."""
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    rotation_speed: float = math.pi * 10 / 1000 * 0.5
    rotation_amplitude: float = 1.0


@dataclass
class SceneModelConfig:
    """Configuration d'un modèle sur la scène."""
    model_id: str                        # ID du modèle (ex: "ahiru")
    offset_x: float = 0.0               # Décalage X normalisé (-1.0 à 1.0)
    offset_y: float = 0.0               # Décalage Y normalisé
    scale: float = 1.0                   # Échelle indépendante
    z_order: int = 0                     # Ordre de rendu (plus grand = premier plan)
    apply_head_direction: bool = True    # Appliquer le suivi du regard/souris
    apply_lip_sync: bool = False         # Appliquer le lip sync (principal uniquement)
    auto_blink: bool = True
    auto_breath: bool = False


@dataclass
class SceneModel:
    """Conteneur pour un modèle Live2D sur la scène."""
    config: SceneModelConfig
    model: Optional[live2d.LAppModel] = None
    expressions: List[str] = field(default_factory=list)
    part_ids: List[str] = field(default_factory=list)
    transform: TransformState = field(default_factory=TransformState)

    def load(self, model_manager: ModelManager, width: int, height: int) -> None:
        """Charge le modèle Live2D."""
        self.model = live2d.LAppModel()
        self.model.LoadModelJson(str(model_manager.path))

        self.expressions = self.model.GetExpressionIds()
        print(f"[SceneModel:{self.config.model_id}] Expressions: {self.expressions}")

        if self.expressions:
            self.model.AddExpression(self.expressions[0])

        self.model.Resize(width, height)
        self.model.SetAutoBlinkEnable(self.config.auto_blink)
        self.model.SetAutoBreathEnable(self.config.auto_breath)
        self.model.SetOffset(self.config.offset_x, self.config.offset_y)
        self.model.SetScale(self.config.scale)

        self.part_ids = self.model.GetPartIds()

    def update(self, rotation: float, mouse_pos: Optional[tuple] = None) -> None:
        """Met à jour le modèle (transformations + suivi souris)."""
        if self.model is None:
            return

        rotation_deg = math.sin(rotation) * self.transform.rotation_amplitude
        self.model.Rotate(rotation_deg)
        self.model.SetOffset(self.config.offset_x + self.transform.dx,
                             self.config.offset_y + self.transform.dy)
        self.model.SetScale(self.config.scale * self.transform.scale)

        # Suivi de la tête vers la souris
        if self.config.apply_head_direction and mouse_pos is not None:
            self.model.Drag(*mouse_pos)

        self.model.Update()

    def draw(self) -> None:
        """Dessine le modèle."""
        if self.model is not None:
            self.model.Draw()

    def set_expression(self, emotion_id: str) -> None:
        if self.model and emotion_id in self.expressions:
            self.model.ResetExpressions()
            self.model.AddExpression(emotion_id)

    def reset_expression(self) -> None:
        if self.model:
            self.model.ResetExpressions()

    def set_mouth(self, value: float) -> None:
        if self.model:
            self.model.SetParameterValue(StandardParams.ParamMouthOpenY, value)


@dataclass
class TTSRequest:
    """Requête TTS avec texte et émotion."""
    text: str
    emotion_id: Optional[str] = None
    priority: bool = False
    timestamp: float = 0.0


class TTSProcessor:
    """Processeur TTS thread-safe."""

    def __init__(self, tts_model):
        self.tts_model = tts_model
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.running = False

    def start(self):
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._process_worker, daemon=True, name="TTSProcessorThread"
        )
        self.worker_thread.start()
        print("[TTSProcessor] Thread démarré")

    def stop(self):
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            try:
                self.request_queue.put(None, timeout=0.1)
            except queue.Full:
                pass
            self.worker_thread.join(timeout=2.0)
        print("[TTSProcessor] Thread arrêté")

    def _process_worker(self):
        print("[TTSProcessor] Worker en cours d'exécution")
        while self.running:
            try:
                request = self.request_queue.get(timeout=0.1)
                if request is None:
                    break
                print(f"[TTSProcessor] Traitement: '{request.text}'")
                try:
                    audio_path = self._text_to_file_path(request.text)
                    audio, duration = synthesize_audio(self.tts_model, request.text, audio_path)
                    emotion_id = request.emotion_id if request.emotion_id else corresp_emotion(request.text)
                    self.result_queue.put({
                        'success': True,
                        'text': request.text,
                        'audio_path': audio_path,
                        'duration': duration,
                        'emotion_id': emotion_id,
                        'timestamp': time.time()
                    })
                    print(f"[TTSProcessor] Terminé: audio={audio_path}, émotion={emotion_id}, durée={duration:.2f}s")
                except Exception as e:
                    print(f"[TTSProcessor] Erreur: {e}")
                    self.result_queue.put({
                        'success': False,
                        'text': request.text,
                        'error': str(e),
                        'timestamp': time.time()
                    })
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTSProcessor] Erreur dans le worker: {e}")

    def _text_to_file_path(self, text: str) -> str:
        FORBIDDEN_CHARS = [" ", ".", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        safe_text = text[:50]
        for char in FORBIDDEN_CHARS:
            safe_text = safe_text.replace(char, "_")
        return f"tts_{hash(text) & 0xFFFFFFFF}.wav"

    def submit_request(self, text: str, emotion_id: Optional[str] = None, priority: bool = False) -> bool:
        try:
            request = TTSRequest(text=text, emotion_id=emotion_id, priority=priority, timestamp=time.time())
            self.request_queue.put_nowait(request)
            print(f"[TTSProcessor] Requête ajoutée: '{text}'")
            return True
        except queue.Full:
            print("[TTSProcessor] Queue pleine, requête ignorée")
            return False

    def get_result(self) -> Optional[dict]:
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def has_pending_requests(self) -> bool:
        return not self.request_queue.empty()


class Live2DViewer:
    """Interactive Live2D model viewer avec synchronisation TTS + Expression + Multi-modèles."""

    _instance: ClassVar[Optional['Live2DViewer']] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _initialized: ClassVar[threading.Event] = threading.Event()
    _external_queue: ClassVar[queue.Queue] = queue.Queue(maxsize=50)

    def __init__(self, config: ViewConfig = ViewConfig()):
        self.config = config

        # --- Définition des modèles sur la scène ---
        # z_order plus élevé = dessiné en dernier = premier plan.
        # keserannpasarann : arrière-plan, légèrement à gauche, lip sync actif.
        # ahiru             : premier plan, côté droit, suivi de tête actif.
        self.scene_models_configs = [
            SceneModelConfig(
                model_id="keserannpasarann",
                offset_x=0,
                offset_y=0.0,
                scale=1.0,
                z_order=0,              # arrière-plan
                apply_head_direction=True,
                apply_lip_sync=True,
                auto_blink=True,
                auto_breath=False,
            ),
            SceneModelConfig(
                model_id="ahiru",
                offset_x=0.5,           # côté droit
                offset_y=-0.5,
                scale=0.6,
                z_order=1,              # premier plan (dessiné en dernier, par-dessus)
                apply_head_direction=True,
                apply_lip_sync=False,
                auto_blink=True,
                auto_breath=False,
            ),
        ]

        self.scene_models: dict[str, SceneModel] = {}
        self.main_model_id = "keserannpasarann"
        self.model_managers: dict[str, ModelManager] = {}

        self.transform = TransformState()   # rotation globale partagée
        self.running = False
        self.current_expression_idx = 0

        # TTS + Audio
        self.tts_model = init_model_TTS()
        self.tts_processor = TTSProcessor(self.tts_model)
        self.wavHandler = None
        self.lipSyncN = 3

        # État de lecture
        self.current_audio_path: Optional[str] = None
        self.current_emotion_id: Optional[str] = None
        self.audio_start_time: Optional[float] = None
        self.audio_duration: Optional[float] = None
        self.is_playing: bool = False

        # UI
        self.overlay_surface = None
        self.font = None
        self.ai_text_surface = None
        self.mouse_pos: tuple = (self.config.width // 2, self.config.height // 2)

    # ------------------------------------------------------------------
    # Singleton / API externe
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> Optional['Live2DViewer']:
        return cls._instance

    @classmethod
    def wait_for_instance(cls, timeout: float = 10.0) -> Optional['Live2DViewer']:
        if cls._initialized.wait(timeout):
            return cls._instance
        return None

    @classmethod
    def send_text(cls, text: str, priority: bool = False) -> bool:
        try:
            cls._external_queue.put_nowait({'text': text, 'priority': priority})
            print(f"[External] Texte ajouté: '{text}'")
            return True
        except queue.Full:
            print(f"[External] Queue pleine, texte ignoré")
            return False

    @classmethod
    def send_emotion_direct(cls, text: str, emotion_id: str, priority: bool = False) -> bool:
        try:
            cls._external_queue.put_nowait({'text': text, 'emotion_id': emotion_id, 'priority': priority})
            print(f"[External] Texte + émotion ajoutés: '{text}' -> {emotion_id}")
            return True
        except queue.Full:
            print(f"[External] Queue pleine, requête ignorée")
            return False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        with self._lock:
            if Live2DViewer._instance is not None:
                raise RuntimeError("Une instance de Live2DViewer existe déjà!")
            Live2DViewer._instance = self

        pygame.init()
        pygame.mixer.init()
        live2d.init()
        live2d.setLogEnable(True)

        pygame.display.set_mode((self.config.width, self.config.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption(self.config.title)
        self.overlay_surface = pygame.Surface((self.config.width, self.config.height), pygame.SRCALPHA)

        if live2d.LIVE2D_VERSION == 3:
            live2d.glewInit()

        self._load_all_models()

        self.wavHandler = WavHandler()
        self.font = pygame.font.Font(None, 48)
        self.ai_text_surface = self.font.render("AI", True, (255, 255, 255))

        self.tts_processor.start()
        Live2DViewer._initialized.set()

        print("=== Live2D Viewer (Multi-modèles) ===")
        print("Modèles sur la scène:")
        for sc in sorted(self.scene_models_configs, key=lambda c: c.z_order):
            plan = "premier plan" if sc.z_order > 0 else "arrière-plan"
            print(f"  [{plan}] {sc.model_id} | offset=({sc.offset_x}, {sc.offset_y}) | scale={sc.scale}")
        print("\nPour ajouter un modèle : éditez scene_models_configs dans __init__")
        print("=====================================")

    def _load_all_models(self) -> None:
        """Charge tous les modèles, triés par z_order."""
        sorted_configs = sorted(self.scene_models_configs, key=lambda c: c.z_order)
        for cfg in sorted_configs:
            manager = ModelManager(cfg.model_id)
            self.model_managers[cfg.model_id] = manager
            scene_model = SceneModel(config=cfg)
            scene_model.load(manager, self.config.width, self.config.height)
            self.scene_models[cfg.model_id] = scene_model
            print(f"[Viewer] Modèle chargé: {cfg.model_id}")

    # ------------------------------------------------------------------
    # Propriétés de commodité
    # ------------------------------------------------------------------

    @property
    def main_scene_model(self) -> Optional[SceneModel]:
        return self.scene_models.get(self.main_model_id)

    @property
    def expressions(self) -> List[str]:
        sm = self.main_scene_model
        return sm.expressions if sm else []

    # ------------------------------------------------------------------
    # Gestion des inputs / TTS
    # ------------------------------------------------------------------

    def _check_inputs(self) -> None:
        if self.is_playing:
            return
        try:
            data = self._external_queue.get_nowait()
            text = data.get('text')
            emotion_id = data.get('emotion_id')
            priority = data.get('priority', False)
            if text:
                print(f"[Main] Nouvelle requête: '{text}'")
                self.tts_processor.submit_request(text, emotion_id, priority)
        except queue.Empty:
            pass

    def _check_tts_results(self) -> None:
        if self.is_playing:
            return
        result = self.tts_processor.get_result()
        if result and result['success']:
            self._start_playback(result)

    def _start_playback(self, result: dict) -> None:
        audio_path = result['audio_path']
        emotion_id = result['emotion_id']
        duration = result['duration']
        print(f"[Main] Démarrage: audio={audio_path}, émotion={emotion_id}, durée={duration:.2f}s")
        try:
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            self.wavHandler.Start(audio_path)

            main = self.main_scene_model
            if main and emotion_id and emotion_id in main.expressions:
                main.set_expression(emotion_id)
                print(f"[Main] Expression appliquée sur {self.main_model_id}: {emotion_id}")

            self.current_audio_path = audio_path
            self.current_emotion_id = emotion_id
            self.audio_start_time = time.time()
            self.audio_duration = duration
            self.is_playing = True
        except Exception as e:
            print(f"[Main] Erreur lors du démarrage: {e}")

    def _update_playback(self) -> None:
        if not self.is_playing:
            return
        audio_playing = pygame.mixer.music.get_busy()
        elapsed = time.time() - self.audio_start_time
        duration_exceeded = elapsed > self.audio_duration + 0.5

        if not audio_playing or duration_exceeded:
            print(f"[Main] Lecture terminée (elapsed={elapsed:.2f}s)")
            main = self.main_scene_model
            if main:
                main.reset_expression()
            self.current_audio_path = None
            self.current_emotion_id = None
            self.audio_start_time = None
            self.audio_duration = None
            self.is_playing = False

    def update_wav_handler(self) -> None:
        """Met à jour le lip sync sur les modèles configurés."""
        if self.wavHandler.Update():
            rms_value = self.wavHandler.GetRms()
            mouth_value = rms_value * self.lipSyncN
        else:
            mouth_value = 0.0 if not pygame.mixer.music.get_busy() else None

        if mouth_value is not None:
            for sm in self.scene_models.values():
                if sm.config.apply_lip_sync:
                    sm.set_mouth(mouth_value)

    # ------------------------------------------------------------------
    # Clavier / Souris
    # ------------------------------------------------------------------

    def _handle_keyboard(self, key: int) -> None:
        transform_map = {
            pygame.K_LEFT: ('dx', -0.1),
            pygame.K_RIGHT: ('dx', 0.1),
            pygame.K_o: ('dy', 0.1),
            pygame.K_l: ('dy', -0.1),
            pygame.K_i: ('scale', 0.1),
            pygame.K_u: ('scale', -0.1),
        }
        if key in transform_map:
            attr, delta = transform_map[key]
            setattr(self.transform, attr, getattr(self.transform, attr) + delta)
        elif key == pygame.K_SPACE:
            self.send_text("Bonjour le monde!")
        elif key == pygame.K_r:
            self._reset_all_models()
            print("Modèles réinitialisés")
        elif key == pygame.K_e:
            self._cycle_expression()

    def _reset_all_models(self) -> None:
        for sm in self.scene_models.values():
            if sm.model:
                sm.model.StopAllMotions()
                sm.model.ResetPose()
                sm.model.ResetExpression()

    def _cycle_expression(self) -> None:
        main = self.main_scene_model
        if not main or not main.expressions:
            return
        self.current_expression_idx = (self.current_expression_idx + 1) % len(main.expressions)
        expr = main.expressions[self.current_expression_idx]
        main.set_expression(expr)
        print(f"Expression: {expr}")

    def _handle_mouse_motion(self, pos: tuple[int, int]) -> None:
        self.mouse_pos = pos

    def _process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            elif event.type == pygame.KEYDOWN:
                self._handle_keyboard(event.key)
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(pygame.mouse.get_pos())

    # ------------------------------------------------------------------
    # Rendu
    # ------------------------------------------------------------------

    def _render_gradient(self) -> None:
        from OpenGL.GL import (
            glMatrixMode, glLoadIdentity, glOrtho, GL_PROJECTION, GL_MODELVIEW,
            glBegin, glEnd, glVertex2f, glColor4f, GL_QUADS,
            glEnable, glDisable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
        )
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.config.width, self.config.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        w, h = self.config.width, self.config.height
        glBegin(GL_QUADS)
        glColor4f(0.48, 0.18, 0.74, 1.0)
        glVertex2f(0, 0)
        glColor4f(0.29, 0.31, 0.79, 1.0)
        glVertex2f(w, 0)
        glColor4f(0.10, 0.44, 0.83, 1.0)
        glVertex2f(w, h)
        glColor4f(0.29, 0.31, 0.79, 1.0)
        glVertex2f(0, h)
        glEnd()
        glDisable(GL_BLEND)

    def _render_ai_label(self) -> None:
        from OpenGL.GL import (
            glMatrixMode, glLoadIdentity, glOrtho, GL_PROJECTION, GL_MODELVIEW,
            glEnable, glDisable, glBlendFunc, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
            glColor4f, glBegin, glEnd, glVertex2f, GL_QUADS,
            glRasterPos2f, glDrawPixels, GL_RGBA, GL_UNSIGNED_BYTE
        )
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.config.width, self.config.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        padding = 10
        box_width = 60
        box_height = 40
        x = self.config.width - box_width - padding
        y = self.config.height - box_height - padding

        glColor4f(0.0, 0.0, 0.0, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + box_width, y)
        glVertex2f(x + box_width, y + box_height)
        glVertex2f(x, y + box_height)
        glEnd()

        self.overlay_surface.fill((0, 0, 0, 0))
        text_surf = self.font.render("AI", True, (255, 255, 255))
        text_data = pygame.image.tostring(text_surf, "RGBA", True)
        text_x = x + (box_width - text_surf.get_width()) // 2
        text_y = y + (box_height - text_surf.get_height()) // 2

        glRasterPos2f(text_x, text_y + text_surf.get_height())
        glDrawPixels(text_surf.get_width(), text_surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        glDisable(GL_BLEND)

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.running = True
        print("\n[Main] Boucle principale démarrée")

        # Tri stable par z_order pour le rendu
        sorted_scene_models = sorted(self.scene_models.values(), key=lambda sm: sm.config.z_order)

        while self.running:
            self._check_inputs()
            self._check_tts_results()
            self._update_playback()
            self._process_events()

            if not self.running:
                break

            # Avancer la rotation globale
            self.transform.rotation += self.transform.rotation_speed

            # Lip sync
            self.update_wav_handler()

            # Mise à jour de tous les modèles
            for sm in sorted_scene_models:
                mouse = self.mouse_pos if sm.config.apply_head_direction else None
                sm.update(self.transform.rotation, mouse)

            # Rendu : arrière-plan → premier plan
            live2d.clearBuffer(*self.config.background_color)
            self._render_gradient()

            for sm in sorted_scene_models:   # z_order 0 d'abord, puis 1 (ahiru)
                sm.draw()

            self._render_ai_label()
            pygame.display.flip()
            pygame.time.wait(self.config.frame_delay)

        print("[Main] Boucle principale terminée")

    def cleanup(self) -> None:
        print("[Main] Nettoyage en cours...")
        self.tts_processor.stop()

        with self._lock:
            Live2DViewer._instance = None
            Live2DViewer._initialized.clear()

        time.sleep(0.2)

        try:
            live2d.dispose()
        except Exception as e:
            print(f"[Main] Erreur dispose: {e}")

        try:
            pygame.quit()
        except Exception as e:
            print(f"[Main] Erreur quit: {e}")

        print("[Main] Nettoyage terminé")


def main():
    """Entry point for the Live2D viewer."""
    config = ViewConfig(width=800, height=1000)
    viewer = Live2DViewer(config=config)

    try:
        viewer.initialize()
        viewer.run()
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C détecté")
    except Exception as e:
        print(f"[Main] Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        viewer.cleanup()


if __name__ == "__main__":
    main()