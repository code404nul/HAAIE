"""
Gestionnaire d'animation de la direction du regard basé sur l'émotion.

Chaque émotion/expression Live2D possède un comportement de regard distinct :
  - amplitude des mouvements
  - vitesse
  - pattern (aléatoire, fixe, nerveux, etc.)
  - zones cibles préférentielles

Ce module est indépendant de Live2DViewer et s'intègre via un appel à
GazeAnimator.update() dans la boucle principale, qui retourne (x, y)
à passer à model.Drag().
"""

import math
import time
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration par expression
# ---------------------------------------------------------------------------

@dataclass
class GazeBehavior:
    """Paramètres de comportement du regard pour une expression donnée."""

    # Amplitude maximale du regard (0.0 = immobile, 1.0 = plein écran)
    amplitude_x: float = 0.3
    amplitude_y: float = 0.2

    # Vitesse de déplacement vers la cible (lerp factor par seconde, 0-1)
    speed: float = 2.5

    # Durée entre deux changements de cible (secondes)
    target_change_interval_min: float = 1.5
    target_change_interval_max: float = 3.5

    # Décalage de base (ex: regard vers le bas pour la honte)
    base_offset_x: float = 0.0
    base_offset_y: float = 0.0

    # Probabilité de regarder vers un coin plutôt que le centre
    corner_bias: float = 0.0   # 0 = centre, 1 = toujours dans un coin

    # Ajout de micro-tremblements (nervosité, peur)
    jitter: float = 0.0        # amplitude du tremblement

    # Probabilité de rester fixe sur la position courante
    fixation_chance: float = 0.0

    # Le regard suit-il la souris ? (0 = non, 1 = oui à 100%)
    mouse_follow_weight: float = 0.0

    # Description lisible
    label: str = ""


# ---------------------------------------------------------------------------
# Table des comportements par expression Live2D
# ---------------------------------------------------------------------------

GAZE_BEHAVIORS: dict[str, GazeBehavior] = {

    # --- idle : regard naturel, détendu, suit légèrement la souris ----------
    "idle": GazeBehavior(
        amplitude_x=0.35,
        amplitude_y=0.25,
        speed=1.8,
        target_change_interval_min=2.0,
        target_change_interval_max=4.5,
        corner_bias=0.1,
        mouse_follow_weight=0.5,
        label="idle — détendu",
    ),

    # --- impressed/dreamer : regard vers le haut, rêveur ---------------------
    "impressed/dreamer": GazeBehavior(
        amplitude_x=0.25,
        amplitude_y=0.15,
        base_offset_y=-0.3,     # légèrement vers le haut
        speed=1.2,
        target_change_interval_min=3.0,
        target_change_interval_max=6.0,
        corner_bias=0.2,
        fixation_chance=0.4,    # s'arrête souvent, comme perdu dans ses pensées
        mouse_follow_weight=0.1,
        label="impressed/dreamer — rêveur",
    ),

    # --- realizing : regard qui cherche, mobile, curieux --------------------
    "realizing": GazeBehavior(
        amplitude_x=0.55,
        amplitude_y=0.35,
        speed=3.5,
        target_change_interval_min=0.6,
        target_change_interval_max=1.8,
        corner_bias=0.3,
        mouse_follow_weight=0.3,
        label="realizing — cherche/curieux",
    ),

    # --- angry : regard direct et fixe, légèrement baissé ------------------
    "angry": GazeBehavior(
        amplitude_x=0.15,
        amplitude_y=0.10,
        base_offset_y=0.15,     # légèrement vers le bas (froncement)
        speed=4.0,
        target_change_interval_min=2.5,
        target_change_interval_max=5.0,
        fixation_chance=0.6,    # fixe la cible longtemps
        mouse_follow_weight=0.8,   # suit agressivement la souris
        label="angry — fixe/direct",
    ),

    # --- emu (émouvant / tristesse / amour / soulagement) : regard doux ----
    "emu": GazeBehavior(
        amplitude_x=0.20,
        amplitude_y=0.15,
        base_offset_y=0.10,     # regard légèrement baissé
        speed=1.0,
        target_change_interval_min=3.0,
        target_change_interval_max=6.0,
        fixation_chance=0.3,
        mouse_follow_weight=0.25,
        label="emu — doux/triste",
    ),

    # --- embarrassed : regard qui fuit, vers le bas -------------------------
    "embarrassed": GazeBehavior(
        amplitude_x=0.40,
        amplitude_y=0.10,
        base_offset_x=0.0,
        base_offset_y=0.35,     # nettement vers le bas
        speed=2.0,
        target_change_interval_min=0.8,
        target_change_interval_max=2.0,
        corner_bias=0.5,        # fuit sur les côtés
        mouse_follow_weight=0.0,   # évite activement le regard
        label="embarrassed — fuit",
    ),

    # --- shy/neutral : timide, petit regard fuyant --------------------------
    "shy/neutral": GazeBehavior(
        amplitude_x=0.45,
        amplitude_y=0.20,
        base_offset_y=0.20,
        speed=2.8,
        target_change_interval_min=0.5,
        target_change_interval_max=1.5,
        jitter=0.015,           # léger tremblement de nervosité
        corner_bias=0.4,
        mouse_follow_weight=0.0,
        fixation_chance=0.1,
        label="shy/neutral — nerveux/timide",
    ),
}

# Comportement par défaut si l'expression est inconnue
_DEFAULT_BEHAVIOR = GAZE_BEHAVIORS["idle"]


# ---------------------------------------------------------------------------
# Gestionnaire principal
# ---------------------------------------------------------------------------

class GazeAnimator:
    """
    Gère l'animation de la direction du regard d'un modèle Live2D.

    Utilisation dans la boucle principale :
    ----------------------------------------
        animator = GazeAnimator(screen_width=800, screen_height=1000)
        animator.set_expression("shy/neutral")

        # dans la boucle :
        mouse_pos = pygame.mouse.get_pos()
        gaze_x, gaze_y = animator.update(mouse_pos)
        model.Drag(gaze_x, gaze_y)
    """

    def __init__(self, screen_width: int = 800, screen_height: int = 1000):
        self.screen_width = screen_width
        self.screen_height = screen_height

        self._behavior: GazeBehavior = _DEFAULT_BEHAVIOR

        # Position actuelle du regard (coordonnées écran)
        self._current_x: float = screen_width / 2
        self._current_y: float = screen_height / 2

        # Cible courante
        self._target_x: float = screen_width / 2
        self._target_y: float = screen_height / 2

        # Timer pour changer de cible
        self._last_target_change: float = time.time()
        self._next_interval: float = self._new_interval()

        # État de fixation
        self._is_fixating: bool = False

        self._last_update: float = time.time()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def set_expression(self, expression_id: str) -> None:
        """
        Change le comportement du regard selon l'expression Live2D.
        Accepte directement les valeurs retournées par corresp_emotion().
        """
        behavior = GAZE_BEHAVIORS.get(expression_id, _DEFAULT_BEHAVIOR)
        if behavior is not self._behavior:
            self._behavior = behavior
            # Réinitialise le timer pour éviter un changement immédiat brutal
            self._next_interval = self._new_interval()
            self._last_target_change = time.time()
            print(f"[GazeAnimator] Comportement → '{behavior.label}'")

    def update(self, mouse_pos: Optional[tuple[int, int]] = None) -> tuple[float, float]:
        """
        Met à jour la position du regard et retourne (x, y) pour model.Drag().

        Args:
            mouse_pos: position souris en pixels (peut être None).

        Returns:
            (x, y) en coordonnées écran à passer directement à model.Drag().
        """
        now = time.time()
        dt = min(now - self._last_update, 0.1)   # cap à 100 ms
        self._last_update = now

        b = self._behavior
        cx = self.screen_width / 2
        cy = self.screen_height / 2

        # --- Changement de cible ---
        if now - self._last_target_change >= self._next_interval:
            self._pick_new_target()
            self._last_target_change = now
            self._next_interval = self._new_interval()

        # --- Cible effective : mélange cible aléatoire + souris ---
        eff_target_x = self._target_x
        eff_target_y = self._target_y

        if mouse_pos is not None and b.mouse_follow_weight > 0.0:
            w = b.mouse_follow_weight
            eff_target_x = eff_target_x * (1 - w) + mouse_pos[0] * w
            eff_target_y = eff_target_y * (1 - w) + mouse_pos[1] * w

        # --- Lerp vers la cible ---
        lerp_factor = 1.0 - math.exp(-b.speed * dt)
        self._current_x += (eff_target_x - self._current_x) * lerp_factor
        self._current_y += (eff_target_y - self._current_y) * lerp_factor

        # --- Jitter (nervosité) ---
        jx, jy = 0.0, 0.0
        if b.jitter > 0.0:
            jx = random.gauss(0, b.jitter * self.screen_width)
            jy = random.gauss(0, b.jitter * self.screen_height)

        final_x = self._current_x + jx
        final_y = self._current_y + jy

        return (final_x, final_y)

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _pick_new_target(self) -> None:
        """Choisit une nouvelle cible de regard selon le comportement actif."""
        b = self._behavior
        cx = self.screen_width / 2
        cy = self.screen_height / 2

        # Fixation : rester sur la position actuelle
        if b.fixation_chance > 0.0 and random.random() < b.fixation_chance:
            self._is_fixating = True
            return  # on garde la cible précédente

        self._is_fixating = False

        # Offset de base (ex: regard vers le bas pour emu/angry)
        base_x = cx + b.base_offset_x * self.screen_width * 0.5
        base_y = cy + b.base_offset_y * self.screen_height * 0.5

        # Amplitude en pixels
        amp_x = b.amplitude_x * self.screen_width * 0.5
        amp_y = b.amplitude_y * self.screen_height * 0.5

        if b.corner_bias > 0.0 and random.random() < b.corner_bias:
            # Choisir un coin/bord
            corner_x = random.choice([-1, 1])
            corner_y = random.choice([-1, 1])
            self._target_x = base_x + corner_x * amp_x
            self._target_y = base_y + corner_y * amp_y
        else:
            # Position aléatoire dans l'ellipse d'amplitude
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, 1) ** 0.5   # distribution uniforme dans l'ellipse
            self._target_x = base_x + math.cos(angle) * amp_x * r
            self._target_y = base_y + math.sin(angle) * amp_y * r

    def _new_interval(self) -> float:
        """Tire un nouvel intervalle aléatoire entre deux changements de cible."""
        b = self._behavior
        return random.uniform(b.target_change_interval_min, b.target_change_interval_max)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def debug_info(self) -> str:
        return (
            f"[GazeAnimator] behavior='{self._behavior.label}' | "
            f"target=({self._target_x:.0f}, {self._target_y:.0f}) | "
            f"current=({self._current_x:.0f}, {self._current_y:.0f}) | "
            f"fixating={self._is_fixating}"
        )