"""Live2D model viewer avec gestion synchronisée TTS + Expression."""

import math
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional, ClassVar
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
    background_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # Transparent pour le dégradé


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
class TTSRequest:
    """Requête TTS avec texte et émotion."""
    text: str
    emotion_id: Optional[str] = None
    priority: bool = False
    timestamp: float = 0.0


class TTSProcessor:
    """
    Processeur TTS thread-safe qui :
    - Traite les requêtes une par une
    - Génère l'audio
    - Détecte l'émotion
    - Retourne le tout ensemble
    """
    
    def __init__(self, tts_model):
        self.tts_model = tts_model
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        
    def start(self):
        """Démarre le thread de traitement."""
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._process_worker,
            daemon=True,
            name="TTSProcessorThread"
        )
        self.worker_thread.start()
        print("[TTSProcessor] Thread démarré")
    
    def stop(self):
        """Arrête le thread de traitement."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            try:
                self.request_queue.put(None, timeout=0.1)
            except queue.Full:
                pass
            self.worker_thread.join(timeout=2.0)
        print("[TTSProcessor] Thread arrêté")
    
    def _process_worker(self):
        """Worker thread qui traite les requêtes TTS."""
        print("[TTSProcessor] Worker en cours d'exécution")
        
        while self.running:
            try:
                request = self.request_queue.get(timeout=0.1)
                
                if request is None:
                    break
                
                print(f"[TTSProcessor] Traitement: '{request.text}'")
                
                try:
                    # Génération de l'audio
                    audio_path = self._text_to_file_path(request.text)
                    audio, duration = synthesize_audio(
                        self.tts_model, 
                        request.text, 
                        audio_path
                    )
                    
                    # Détection de l'émotion si nécessaire
                    if request.emotion_id is None:
                        emotion_id = corresp_emotion(request.text)
                    else:
                        emotion_id = request.emotion_id
                    
                    # Résultat complet
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
        """Génère un nom de fichier sûr à partir du texte."""
        FORBIDDEN_CHARS = [" ", ".", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        safe_text = text[:50]  # Limiter la longueur
        for char in FORBIDDEN_CHARS:
            safe_text = safe_text.replace(char, "_")
        return f"tts_{hash(text) & 0xFFFFFFFF}.wav"
    
    def submit_request(self, text: str, emotion_id: Optional[str] = None, priority: bool = False) -> bool:
        """Soumet une requête TTS."""
        try:
            request = TTSRequest(
                text=text,
                emotion_id=emotion_id,
                priority=priority,
                timestamp=time.time()
            )
            self.request_queue.put_nowait(request)
            print(f"[TTSProcessor] Requête ajoutée: '{text}'")
            return True
        except queue.Full:
            print("[TTSProcessor] Queue pleine, requête ignorée")
            return False
    
    def get_result(self) -> Optional[dict]:
        """Récupère un résultat si disponible."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def has_pending_requests(self) -> bool:
        """Vérifie s'il y a des requêtes en attente."""
        return not self.request_queue.empty()


class Live2DViewer:
    """Interactive Live2D model viewer avec synchronisation TTS + Expression."""
    
    # Singleton pattern
    _instance: ClassVar[Optional['Live2DViewer']] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _initialized: ClassVar[threading.Event] = threading.Event()
    
    # Queue d'entrée externe
    _external_queue: ClassVar[queue.Queue] = queue.Queue(maxsize=50)

    def __init__(self, model_manager: ModelManager, config: ViewConfig = ViewConfig()):
        self.config = config
        self.model_manager = model_manager
        self.model: Optional[live2d.LAppModel] = None
        self.transform = TransformState()
        self.running = False
        self.current_expression_idx = 0
        self.expressions = []
        self.part_ids = []
        
        # TTS + Audio
        self.tts_model = init_model_TTS()
        self.tts_processor = TTSProcessor(self.tts_model)
        self.wavHandler = None
        self.lipSyncN = 3
        
        # État de lecture actuel
        self.current_audio_path: Optional[str] = None
        self.current_emotion_id: Optional[str] = None
        self.audio_start_time: Optional[float] = None
        self.audio_duration: Optional[float] = None
        self.is_playing: bool = False
        
        # UI Elements
        self.font = None
        self.ai_text_surface = None

    @classmethod
    def get_instance(cls) -> Optional['Live2DViewer']:
        """Récupère l'instance singleton."""
        return cls._instance
    
    @classmethod
    def wait_for_instance(cls, timeout: float = 10.0) -> Optional['Live2DViewer']:
        """Attend que l'instance soit initialisée."""
        if cls._initialized.wait(timeout):
            return cls._instance
        return None
    
    @classmethod
    def send_text(cls, text: str, priority: bool = False) -> bool:
        """Envoie du texte depuis n'importe où."""
        try:
            cls._external_queue.put_nowait({
                'text': text,
                'priority': priority
            })
            print(f"[External] Texte ajouté: '{text}'")
            return True
        except queue.Full:
            print(f"[External] Queue pleine, texte ignoré")
            return False
    
    @classmethod
    def send_emotion_direct(cls, text: str, emotion_id: str, priority: bool = False) -> bool:
        """Envoie du texte avec une émotion pré-définie."""
        try:
            cls._external_queue.put_nowait({
                'text': text,
                'emotion_id': emotion_id,
                'priority': priority
            })
            print(f"[External] Texte + émotion ajoutés: '{text}' -> {emotion_id}")
            return True
        except queue.Full:
            print(f"[External] Queue pleine, requête ignorée")
            return False

    def initialize(self) -> None:
        """Initialize pygame, Live2D, and load the model."""
        with self._lock:
            if Live2DViewer._instance is not None:
                raise RuntimeError("Une instance de Live2DViewer existe déjà!")
            Live2DViewer._instance = self
        
        pygame.init()
        pygame.mixer.init()
        live2d.init()
        live2d.setLogEnable(True)

        pygame.display.set_mode(
            (self.config.width, self.config.height),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption(self.config.title)
        
        # Créer une surface 2D pour l'overlay
        self.overlay_surface = pygame.Surface((self.config.width, self.config.height), pygame.SRCALPHA)

        if live2d.LIVE2D_VERSION == 3:
            live2d.glewInit()

        self._load_model()
        
        # Initialiser le WavHandler
        self.wavHandler = WavHandler()
        
        # Initialiser la police pour le texte "AI"
        self.font = pygame.font.Font(None, 48)
        self.ai_text_surface = self.font.render("AI", True, (255, 255, 255))
        
        # Démarrer le processeur TTS
        self.tts_processor.start()
        
        # Signaler que l'instance est prête
        Live2DViewer._initialized.set()
        
        print("=== Live2D Viewer (Queue Refactorisée) ===")
        print("Contrôles:")
        print("- SPACE: Dire 'Bonjour le monde!'")
        print("- E: Changer d'expression")
        print("- R: Réinitialiser")
        print("\nAPI externe:")
        print("  Live2DViewer.send_text('texte')")
        print("  Live2DViewer.send_emotion_direct('texte', 'f01')")
        print(f"\nExpressions: {self.expressions}")
        print("==========================================")

    def _load_model(self) -> None:
        """Load and configure the Live2D model."""
        self.model = live2d.LAppModel()
        self.model.LoadModelJson(str(self.model_manager.path))
        
        self.expressions = self.model.GetExpressionIds()
        print(f"Expressions disponibles: {self.expressions}")
        
        if self.expressions:
            self.model.AddExpression(self.expressions[0])

        self.model.Resize(self.config.width, self.config.height)
        self.model.SetAutoBlinkEnable(True)
        self.model.SetAutoBreathEnable(False)
        
        self.part_ids = self.model.GetPartIds()

    def _check_inputs(self) -> None:
        """Vérifie les inputs de la queue externe."""
        # Ne traiter de nouvelles requêtes que si rien n'est en cours de lecture
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
        """Vérifie les résultats du processeur TTS."""
        # Ne récupérer un résultat que si rien n'est en cours
        if self.is_playing:
            return
        
        result = self.tts_processor.get_result()
        if result and result['success']:
            self._start_playback(result)

    def _start_playback(self, result: dict) -> None:
        """Démarre la lecture audio + expression."""
        audio_path = result['audio_path']
        emotion_id = result['emotion_id']
        duration = result['duration']
        
        print(f"[Main] Démarrage: audio={audio_path}, émotion={emotion_id}, durée={duration:.2f}s")
        
        try:
            # Charger et jouer l'audio
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Démarrer le lip sync
            self.wavHandler.Start(audio_path)
            
            # Appliquer l'expression
            if emotion_id and emotion_id in self.expressions:
                self.model.ResetExpressions()
                self.model.AddExpression(emotion_id)
                print(f"[Main] Expression appliquée: {emotion_id}")
            
            # Enregistrer l'état
            self.current_audio_path = audio_path
            self.current_emotion_id = emotion_id
            self.audio_start_time = time.time()
            self.audio_duration = duration
            self.is_playing = True
            
        except Exception as e:
            print(f"[Main] Erreur lors du démarrage: {e}")

    def _update_playback(self) -> None:
        """Met à jour l'état de lecture et reset l'expression quand terminé."""
        if not self.is_playing:
            return
        
        # Vérifier si l'audio est toujours en cours
        audio_playing = pygame.mixer.music.get_busy()
        
        # Vérifier si la durée est dépassée
        elapsed = time.time() - self.audio_start_time
        duration_exceeded = elapsed > self.audio_duration + 0.5  # Marge de 0.5s
        
        if not audio_playing or duration_exceeded:
            print(f"[Main] Lecture terminée (elapsed={elapsed:.2f}s)")
            
            # Reset l'expression
            self.model.ResetExpressions()
            print(f"[Main] Expression '{self.current_emotion_id}' retirée")
            
            # Reset l'état
            self.current_audio_path = None
            self.current_emotion_id = None
            self.audio_start_time = None
            self.audio_duration = None
            self.is_playing = False

    def update_wav_handler(self) -> None:
        """Met à jour le lip sync."""
        if self.wavHandler.Update():
            rms_value = self.wavHandler.GetRms()
            mouth_value = rms_value * self.lipSyncN
            self.model.SetParameterValue(StandardParams.ParamMouthOpenY, mouth_value)
        else:
            if not pygame.mixer.music.get_busy():
                self.model.SetParameterValue(StandardParams.ParamMouthOpenY, 0.0)

    def _handle_keyboard(self, key: int) -> None:
        """Handle keyboard input."""
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
            self._reset_model()
            print("Modèle réinitialisé")
        
        elif key == pygame.K_e:
            self._cycle_expression()

    def _reset_model(self) -> None:
        """Reset model to default state."""
        self.model.StopAllMotions()
        self.model.ResetPose()
        self.model.ResetExpression()

    def _cycle_expression(self) -> None:
        """Cycle to the next expression."""
        if not self.expressions:
            return
        
        self.current_expression_idx = (self.current_expression_idx + 1) % len(self.expressions)
        expr = self.expressions[self.current_expression_idx]
        self.model.ResetExpressions()
        self.model.AddExpression(expr)
        print(f"Expression: {expr}")

    def _handle_mouse_motion(self, pos: tuple[int, int]) -> None:
        """Handle mouse motion."""
        self.model.Drag(*pos)

    def _apply_transformations(self) -> None:
        """Apply transformations."""
        self.transform.rotation += self.transform.rotation_speed
        rotation_deg = math.sin(self.transform.rotation) * self.transform.rotation_amplitude
        
        self.model.Rotate(rotation_deg)
        self.model.SetOffset(self.transform.dx, self.transform.dy)
        self.model.SetScale(self.transform.scale)

    def _render_gradient(self) -> None:
        """Dessine un dégradé violet (haut-gauche) → bleu (bas-droite) en fond."""
        from OpenGL.GL import (
            glMatrixMode, glLoadIdentity, glOrtho, GL_PROJECTION, GL_MODELVIEW,
            glBegin, glEnd, glVertex2f, glColor4f, GL_QUADS,
            glEnable, glDisable, GL_BLEND, glBlendFunc,
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
        )

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.config.width, self.config.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        w, h = self.config.width, self.config.height

        # Violet (haut-gauche)  #7B2FBE → (0.48, 0.18, 0.74)
        # Bleu   (bas-droite)   #1A6FD4 → (0.10, 0.44, 0.83)
        # Intermédiaire (coins restants) pour un dégradé diagonal naturel

        glBegin(GL_QUADS)
        glColor4f(0.48, 0.18, 0.74, 1.0)  # haut-gauche  — violet
        glVertex2f(0, 0)

        glColor4f(0.29, 0.31, 0.79, 1.0)  # haut-droite  — intermédiaire
        glVertex2f(w, 0)

        glColor4f(0.10, 0.44, 0.83, 1.0)  # bas-droite   — bleu
        glVertex2f(w, h)

        glColor4f(0.29, 0.31, 0.79, 1.0)  # bas-gauche   — intermédiaire
        glVertex2f(0, h)
        glEnd()

        glDisable(GL_BLEND)

    def _render_ai_label(self) -> None:
        """Affiche le label 'AI' en bas à droite avec OpenGL."""
        from OpenGL.GL import glMatrixMode, glLoadIdentity, glOrtho, GL_PROJECTION, GL_MODELVIEW
        from OpenGL.GL import glEnable, glDisable, glBlendFunc, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
        from OpenGL.GL import glColor4f, glBegin, glEnd, glVertex2f, GL_QUADS
        
        # Passer en mode 2D
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.config.width, self.config.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Activer le blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Position en bas à droite
        padding = 10
        box_width = 60
        box_height = 40
        x = self.config.width - box_width - padding
        y = self.config.height - box_height - padding
        
        # Dessiner un rectangle semi-transparent noir
        glColor4f(0.0, 0.0, 0.0, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + box_width, y)
        glVertex2f(x + box_width, y + box_height)
        glVertex2f(x, y + box_height)
        glEnd()
        
        # Dessiner le texte "AI" sur la surface pygame
        self.overlay_surface.fill((0, 0, 0, 0))
        text_x = x + (box_width - self.ai_text_surface.get_width()) // 2
        text_y = y + (box_height - self.ai_text_surface.get_height()) // 2
        
        # Utiliser pygame pour le texte
        import pygame
        from OpenGL.GL import glRasterPos2f, glDrawPixels, GL_RGBA, GL_UNSIGNED_BYTE
        
        # Créer une petite surface pour le texte
        text_surf = self.font.render("AI", True, (255, 255, 255))
        text_data = pygame.image.tostring(text_surf, "RGBA", True)
        
        glRasterPos2f(text_x, text_y + text_surf.get_height())
        glDrawPixels(text_surf.get_width(), text_surf.get_height(), 
                     GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        glDisable(GL_BLEND)

    def _process_events(self) -> None:
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            
            elif event.type == pygame.KEYDOWN:
                self._handle_keyboard(event.key)
            
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(pygame.mouse.get_pos())

    def run(self) -> None:
        """Main rendering loop."""
        self.running = True
        
        print("\n[Main] Boucle principale démarrée")
        
        while self.running:
            # Vérifier les inputs externes
            self._check_inputs()
            
            # Vérifier les résultats TTS
            self._check_tts_results()
            
            # Mettre à jour l'état de lecture
            self._update_playback()
            
            # Traiter les événements pygame
            self._process_events()
            
            if not self.running:
                break
            
            # Appliquer les transformations
            self._apply_transformations()
            
            # Mettre à jour le lip sync
            self.update_wav_handler()
            
            # Mise à jour du modèle
            self.model.Update()
            
            # Rendu
            live2d.clearBuffer(*self.config.background_color)
            self._render_gradient()  # Dégradé violet → bleu
            self.model.Draw()
            
            # Afficher le label "AI"
            self._render_ai_label()
            
            pygame.display.flip()
            pygame.time.wait(self.config.frame_delay)
        
        print("[Main] Boucle principale terminée")

    def cleanup(self) -> None:
        """Cleanup resources."""
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
    model_manager = ModelManager("keserannpasarann")
    viewer = Live2DViewer(model_manager)

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