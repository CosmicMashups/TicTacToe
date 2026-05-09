"""Rendering helpers and a small animation engine for Pygame UI."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Optional

import pygame


Color = pygame.Color


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(a: Color, b: Color, t: float) -> Color:
    t = clamp01(t)
    return Color(
        int(lerp(a.r, b.r, t)),
        int(lerp(a.g, b.g, t)),
        int(lerp(a.b, b.b, t)),
        int(lerp(a.a, b.a, t)),
    )


def ease_in_out(t: float) -> float:
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_in_cubic(t: float) -> float:
    t = clamp01(t)
    return t * t * t


def ease_out_back(t: float) -> float:
    t = clamp01(t)
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


@dataclass
class Tween:
    duration_s: float
    easing: Callable[[float], float] = ease_in_out
    elapsed_s: float = 0.0
    done: bool = False

    def reset(self) -> None:
        self.elapsed_s = 0.0
        self.done = False

    def step(self, dt_s: float) -> float:
        if self.done:
            return 1.0
        self.elapsed_s += max(0.0, dt_s)
        t = 1.0 if self.duration_s <= 0 else self.elapsed_s / self.duration_s
        if t >= 1.0:
            self.done = True
            t = 1.0
        return self.easing(t)


@dataclass
class Particle:
    pos: pygame.Vector2
    vel: pygame.Vector2
    radius: float
    color: Color
    life_s: float
    age_s: float = 0.0

    def step(self, dt_s: float) -> bool:
        self.age_s += dt_s
        self.pos += self.vel * dt_s
        self.vel.y += 420.0 * dt_s
        return self.age_s < self.life_s

    def draw(self, surf: pygame.Surface) -> None:
        t = clamp01(self.age_s / max(1e-6, self.life_s))
        alpha = int(255 * (1.0 - t))
        c = Color(self.color.r, self.color.g, self.color.b, alpha)
        pygame.draw.circle(surf, c, (int(self.pos.x), int(self.pos.y)), int(max(1, self.radius)))


@dataclass
class ParticleSystem:
    particles: list[Particle] = field(default_factory=list)

    def burst(self, center: tuple[int, int], palette: list[Color], count: int = 30) -> None:
        cx, cy = center
        for _ in range(count):
            angle = random.random() * math.tau
            speed = random.uniform(80, 260)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            p = Particle(
                pos=pygame.Vector2(cx, cy),
                vel=vel,
                radius=random.uniform(2.0, 4.5),
                color=random.choice(palette),
                life_s=random.uniform(0.6, 1.2),
            )
            self.particles.append(p)

    def step(self, dt_s: float) -> None:
        alive: list[Particle] = []
        for p in self.particles:
            if p.step(dt_s):
                alive.append(p)
        self.particles = alive

    def draw(self, surf: pygame.Surface) -> None:
        for p in self.particles:
            p.draw(surf)


@dataclass
class Theme:
    name: str
    accent: Color
    accent_soft: Color
    bg_top: Color
    bg_bottom: Color
    board_glow: Color


HAPPY_THEME = Theme(
    name="Happy",
    accent=Color(60, 220, 120),
    accent_soft=Color(120, 255, 180),
    bg_top=Color(14, 28, 20),
    bg_bottom=Color(10, 18, 14),
    board_glow=Color(70, 255, 150),
)

NEUTRAL_THEME = Theme(
    name="Neutral",
    accent=Color(80, 160, 255),
    accent_soft=Color(150, 210, 255),
    bg_top=Color(10, 16, 28),
    bg_bottom=Color(8, 12, 20),
    board_glow=Color(120, 190, 255),
)


EMOTION_THEMES: dict[str, Theme] = {
    "calm_blue": NEUTRAL_THEME,
    "soft_blue": Theme(
        name="Soft Blue",
        accent=Color(110, 180, 255),
        accent_soft=Color(180, 225, 255),
        bg_top=Color(10, 18, 34),
        bg_bottom=Color(8, 12, 22),
        board_glow=Color(160, 220, 255),
    ),
    "warm_pink": Theme(
        name="Warm Pink",
        accent=Color(255, 110, 180),
        accent_soft=Color(255, 170, 215),
        bg_top=Color(30, 14, 24),
        bg_bottom=Color(18, 8, 14),
        board_glow=Color(255, 150, 210),
    ),
    "bright_glow": Theme(
        name="Bright Glow",
        accent=Color(255, 215, 90),
        accent_soft=Color(255, 235, 150),
        bg_top=Color(22, 20, 10),
        bg_bottom=Color(14, 12, 8),
        board_glow=Color(255, 235, 140),
    ),
    "party": Theme(
        name="Party",
        accent=Color(170, 120, 255),
        accent_soft=Color(210, 180, 255),
        bg_top=Color(14, 10, 22),
        bg_bottom=Color(8, 6, 14),
        board_glow=Color(210, 170, 255),
    ),
    "dynamic": Theme(
        name="Dynamic",
        accent=Color(80, 240, 220),
        accent_soft=Color(150, 255, 240),
        bg_top=Color(8, 18, 18),
        bg_bottom=Color(6, 12, 12),
        board_glow=Color(130, 255, 240),
    ),
    "supportive": Theme(
        name="Supportive",
        accent=Color(120, 210, 190),
        accent_soft=Color(170, 240, 225),
        bg_top=Color(8, 18, 22),
        bg_bottom=Color(6, 12, 16),
        board_glow=Color(150, 245, 230),
    ),
    "flash": Theme(
        name="Flash",
        accent=Color(255, 255, 255),
        accent_soft=Color(220, 235, 255),
        bg_top=Color(10, 16, 28),
        bg_bottom=Color(8, 12, 20),
        board_glow=Color(200, 220, 255),
    ),
    "subtle_pulse": Theme(
        name="Subtle Pulse",
        accent=Color(150, 200, 255),
        accent_soft=Color(190, 230, 255),
        bg_top=Color(10, 16, 28),
        bg_bottom=Color(8, 12, 20),
        board_glow=Color(140, 210, 255),
    ),
}


def draw_vertical_gradient(surf: pygame.Surface, rect: pygame.Rect, top: Color, bottom: Color) -> None:
    h = max(1, rect.height)
    for i in range(h):
        t = i / float(h - 1) if h > 1 else 1.0
        c = lerp_color(top, bottom, t)
        pygame.draw.line(surf, c, (rect.left, rect.top + i), (rect.right - 1, rect.top + i))


def try_load_sound(path: str) -> Optional[pygame.mixer.Sound]:
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        return pygame.mixer.Sound(path)
    except Exception:
        return None

