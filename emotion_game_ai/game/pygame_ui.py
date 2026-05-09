"""Modern Pygame UI for the Emotion-Aware Tic-Tac-Toe game."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random
import time
from typing import Optional

import numpy as np
import pygame

from emotion_game_ai.game.ai_player import decide_move
from emotion_game_ai.game.board import Board
from emotion_game_ai.game.renderer import (
    HAPPY_THEME,
    EMOTION_THEMES,
    NEUTRAL_THEME,
    ParticleSystem,
    Theme,
    Tween,
    draw_vertical_gradient,
    ease_in_cubic,
    ease_in_out,
    ease_out_back,
    lerp,
    lerp_color,
    try_load_sound,
)
from emotion_game_ai.nlp.sentiment_model import SentimentModel
from emotion_game_ai.nlp.sentiment_model import sentiment_from_emotion
from emotion_game_ai.utils.threading_utils import SharedState


WINDOW_W, WINDOW_H = 1280, 720
HUD_H = 80
SIDE_W = 300
BOARD_SIZE = 500


@dataclass
class CellAnim:
    kind: str  # "X" | "O"
    started_s: float
    duration_s: float = 0.30

    def t(self, now_s: float) -> float:
        if self.duration_s <= 0:
            return 1.0
        return max(0.0, min(1.0, (now_s - self.started_s) / self.duration_s))


class PygameApp:
    def __init__(self, shared: SharedState, sentiment: SentimentModel) -> None:
        self.shared = shared
        self.sentiment = sentiment

        pygame.init()
        pygame.display.set_caption("Emotion-Aware Tic-Tac-Toe")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("segoe ui", 28, bold=True)
        self.font_ui = pygame.font.SysFont("segoe ui", 18)
        self.font_small = pygame.font.SysFont("segoe ui", 14)

        self.rng = random.Random(7)
        self.particles = ParticleSystem()

        self.board = Board()
        self.turn = "X"
        self.winner: Optional[str] = None
        self.draw = False
        self.win_line: Optional[list[tuple[int, int]]] = None

        self.cell_anims: dict[tuple[int, int], CellAnim] = {}
        self.cell_pulse: dict[tuple[int, int], Tween] = {}
        self.board_shake = Tween(duration_s=0.45, easing=ease_in_out)
        self.board_shake.done = True

        self.neutral_player_turns = 0
        self.hint_active = False
        self.hint_cell: Optional[tuple[int, int]] = None
        self.hint_slide = Tween(duration_s=0.35, easing=ease_out_back)
        self.hint_slide.done = True

        self.theme = NEUTRAL_THEME
        self.theme_target = NEUTRAL_THEME
        self.theme_tween = Tween(duration_s=0.40, easing=ease_in_out)
        self.theme_tween.done = True

        self.ai_thinking_until_s = 0.0
        self.ai_pending_move: Optional[tuple[int, int]] = None
        self.ai_typing = ""
        self.ai_typing_until_s = 0.0

        self.sfx_move = try_load_sound(os.path.join("assets", "sounds", "move.mp3"))
        self.sfx_win = try_load_sound(os.path.join("assets", "sounds", "win.mp3"))
        self.sfx_click = try_load_sound(os.path.join("assets", "sounds", "click.mp3"))

        self.scene = "play"  # "play" | "postgame"
        self.feedback_text = ""
        self.last_sentiment = "neutral"
        self.postgame_message = ""

    def run(self) -> None:
        while True:
            dt_s = self.clock.tick(60) / 1000.0
            if not self._handle_events():
                return

            self._step(dt_s)
            self._draw()
            pygame.display.flip()

    def _layout(self) -> dict[str, pygame.Rect]:
        hud = pygame.Rect(0, 0, WINDOW_W, HUD_H)
        left = pygame.Rect(0, HUD_H, SIDE_W, WINDOW_H - HUD_H)
        right = pygame.Rect(WINDOW_W - SIDE_W, HUD_H, SIDE_W, WINDOW_H - HUD_H)
        center = pygame.Rect(SIDE_W, HUD_H, WINDOW_W - 2 * SIDE_W, WINDOW_H - HUD_H)

        board = pygame.Rect(0, 0, BOARD_SIZE, BOARD_SIZE)
        board.center = center.center
        return {"hud": hud, "left": left, "right": right, "center": center, "board": board}

    def _handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_c:
                idx = self.shared.request_next_camera()
                self.shared.set_dialogue(f"Switching camera to index {idx} (press C to cycle).", ttl_s=2.5)

            if self.scene == "play":
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self._on_click(pygame.mouse.get_pos())
            else:
                self._handle_postgame_input(ev)
        return True

    def _handle_postgame_input(self, ev: pygame.event.Event) -> None:
        if ev.type != pygame.KEYDOWN:
            return
        if ev.key == pygame.K_RETURN:
            emotion = self.sentiment.predict_emotion(self.feedback_text)
            self.shared.set_feedback_emotion(emotion)
            sentiment = sentiment_from_emotion(emotion)
            self.last_sentiment = sentiment

            # Keep legacy sentiment tuning alongside full behavior profile.
            _, _, _, tuning, _ = self.shared.get_snapshot()
            tuning.adjust_for_sentiment(sentiment)

            # UI/system feedback
            behavior = self.shared.last_behavior
            self.shared.set_dialogue(behavior.system_message, ttl_s=4.0)
            self._emit_ui_effect(behavior.ui_effect)
            self.feedback_text = ""
            self._start_new_match()
            return
        if ev.key == pygame.K_BACKSPACE:
            self.feedback_text = self.feedback_text[:-1]
            return
        if ev.unicode and ev.unicode.isprintable():
            if len(self.feedback_text) < 140:
                self.feedback_text += ev.unicode

    def _on_click(self, pos: tuple[int, int]) -> None:
        if self.board.game_over() or self.turn != "X":
            return
        rects = self._layout()
        board_rect = rects["board"]
        if not board_rect.collidepoint(pos):
            return
        cell = self._cell_from_pos(pos, board_rect)
        if cell is None:
            return
        r, c = cell
        if not self.board.place(r, c, "X"):
            return

        now_s = time.perf_counter()
        self.cell_anims[(r, c)] = CellAnim(kind="X", started_s=now_s)
        self.cell_pulse[(r, c)] = Tween(duration_s=0.20, easing=ease_out_back)
        if self.sfx_move:
            self.sfx_move.play()

        # Per-turn analytics emotion sampling
        emotion, _, stats, _, _ = self.shared.get_snapshot()
        stats.record_emotion(emotion)

        # Good move highlight heuristic: center move
        if (r, c) == (1, 1):
            self.particles.burst(board_rect.center, self._confetti_palette(emotion), count=18)

        self._post_move_updates(player="X")
        self.turn = "O"
        self._queue_ai_move()

    def _queue_ai_move(self) -> None:
        emotion, _, _, tuning, _ = self.shared.get_snapshot()
        decision = decide_move(self.board, emotion, tuning, self.rng)
        self.shared.ai_mode = decision.mode
        self.shared.set_dialogue(decision.message, ttl_s=3.0)

        self.ai_pending_move = decision.move
        self.ai_thinking_until_s = time.perf_counter() + decision.thinking_delay_s
        self.ai_typing = ""
        self.ai_typing_until_s = time.perf_counter() + 1.2

    def _post_move_updates(self, player: str) -> None:
        self.winner = self.board.winner()
        self.draw = self.board.is_draw()
        self.win_line = Board.winning_line_coords(self.board.grid)
        if self.winner or self.draw:
            self._finish_match()

    def _finish_match(self) -> None:
        emotion, _, stats, _, _ = self.shared.get_snapshot()
        stats.games_played += 1
        if self.winner == "X":
            stats.wins += 1
            if emotion == "Happy":
                self.postgame_message = "Great victory! You're improving fast."
            else:
                self.postgame_message = "Nice win. Keep building momentum."
        elif self.winner == "O":
            stats.losses += 1
            if emotion == "Neutral":
                self.postgame_message = "That was a tough round. Want a tip for next time?"
            else:
                self.postgame_message = "Close one. Ready for another?"
        else:
            stats.draws += 1
            self.postgame_message = "Draw! Well played."

        if self.winner and self.sfx_win:
            self.sfx_win.play()

        rects = self._layout()
        if self.winner:
            self.particles.burst(rects["board"].center, self._confetti_palette(emotion), count=60)
        else:
            self.board_shake.reset()

        self.scene = "postgame"
        self.turn = "X"
        self.ai_pending_move = None
        self.ai_thinking_until_s = 0.0

    def _start_new_match(self) -> None:
        self.board = Board()
        self.turn = "X"
        self.winner = None
        self.draw = False
        self.win_line = None
        self.cell_anims.clear()
        self.cell_pulse.clear()
        self.neutral_player_turns = 0
        self.hint_active = False
        self.hint_cell = None
        self.scene = "play"
        # Apply post-game NLP theme preference (if any).
        self._set_theme_target(self.shared.current_emotion)

    def _step(self, dt_s: float) -> None:
        emotion, vision, _, _, ai_mode = self.shared.get_snapshot()
        self._set_theme_target(emotion)
        self._step_theme(dt_s)
        self.particles.step(dt_s)

        if self.scene == "play":
            if self.turn == "O" and not self.board.game_over():
                now = time.perf_counter()
                if now >= self.ai_thinking_until_s and self.ai_pending_move is not None:
                    r, c = self.ai_pending_move
                    if self.board.place(r, c, "O"):
                        self.cell_anims[(r, c)] = CellAnim(kind="O", started_s=now)
                        if self.sfx_move:
                            self.sfx_move.play()
                        self.shared.set_dialogue(self._random_ai_dialogue(emotion), ttl_s=2.5)
                        self._post_move_updates(player="O")
                    self.ai_pending_move = None
                    self.turn = "X"

            self._update_hints(emotion)
            self._typing_effect()

    def _typing_effect(self) -> None:
        if self.turn != "O" or self.scene != "play" or self.board.game_over():
            return
        now = time.perf_counter()
        if now > self.ai_typing_until_s:
            return
        target = "Analyzing best move..."
        if len(self.ai_typing) < len(target):
            self.ai_typing += target[len(self.ai_typing)]

    def _update_hints(self, emotion: str) -> None:
        if self.scene != "play" or self.turn != "X" or self.board.game_over():
            return
        if emotion == "Neutral":
            # Count player turns spent neutral.
            self.neutral_player_turns = min(99, self.neutral_player_turns + 1)
        else:
            self.neutral_player_turns = 0
            self.hint_active = False
            self.hint_cell = None

        if self.neutral_player_turns >= 3 and not self.hint_active:
            self.hint_active = True
            self.hint_cell = self._suggest_hint_cell()
            self.hint_slide.reset()
            self.shared.set_dialogue("Hint: The center square is often the strongest opening move.", ttl_s=4.0)

    def _suggest_hint_cell(self) -> Optional[tuple[int, int]]:
        if self.board.grid[1][1] == "":
            return (1, 1)
        for cell in [(0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)]:
            r, c = cell
            if self.board.grid[r][c] == "":
                return cell
        return None

    def _set_theme_target(self, emotion: str) -> None:
        # Primary theme from vision emotion (Happy/Neutral).
        target = HAPPY_THEME if emotion == "Happy" else NEUTRAL_THEME
        # Optional override from post-game NLP behavior mapping.
        try:
            ui_theme = self.shared.last_behavior.ui_theme
        except Exception:
            ui_theme = ""
        if ui_theme:
            target = EMOTION_THEMES.get(ui_theme, target)
        if target.name != self.theme_target.name:
            self.theme_target = target
            self.theme_tween.reset()

    def _emit_ui_effect(self, effect: str) -> None:
        rects = self._layout()
        center = rects["board"].center
        eff = (effect or "").strip().lower()
        if eff == "hearts":
            palette = [pygame.Color(255, 110, 180), pygame.Color(255, 170, 215), pygame.Color(255, 70, 140)]
            self.particles.burst(center, palette, count=42)
        elif eff in {"particles", "dynamic_board"}:
            palette = [self.theme.accent, self.theme.accent_soft, pygame.Color(255, 235, 150)]
            self.particles.burst(center, palette, count=60)
        elif eff == "flash":
            palette = [pygame.Color(255, 255, 255), pygame.Color(200, 220, 255)]
            self.particles.burst(center, palette, count=24)
        elif eff in {"calm_bg", "calming_overlay", "slow_fade", "subtle_pulse", "supportive_theme", "dynamic_lighting", "board_glow"}:
            # These are represented mostly via theme + a small accent burst.
            palette = [self.theme.accent_soft, self.theme.accent]
            self.particles.burst(center, palette, count=18)

    def _step_theme(self, dt_s: float) -> None:
        if self.theme_tween.done:
            self.theme = self.theme_target
            return
        t = self.theme_tween.step(dt_s)
        self.theme = Theme(
            name=self.theme_target.name,
            accent=lerp_color(self.theme.accent, self.theme_target.accent, t),
            accent_soft=lerp_color(self.theme.accent_soft, self.theme_target.accent_soft, t),
            bg_top=lerp_color(self.theme.bg_top, self.theme_target.bg_top, t),
            bg_bottom=lerp_color(self.theme.bg_bottom, self.theme_target.bg_bottom, t),
            board_glow=lerp_color(self.theme.board_glow, self.theme_target.board_glow, t),
        )

    def _draw(self) -> None:
        rects = self._layout()
        draw_vertical_gradient(self.screen, self.screen.get_rect(), self.theme.bg_top, self.theme.bg_bottom)

        self._draw_hud(rects["hud"])
        self._draw_webcam_panel(rects["left"])
        self._draw_status_panel(rects["right"])
        self._draw_board(rects["board"])
        self.particles.draw(self.screen)

        if self.scene == "postgame":
            self._draw_postgame_overlay()

    def _draw_hud(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, pygame.Color(0, 0, 0, 80), rect)
        emotion, _, stats, _, ai_mode = self.shared.get_snapshot()

        title = self.font_title.render("Emotion-Aware Tic-Tac-Toe", True, pygame.Color(240, 245, 255))
        self.screen.blit(title, (16, 18))

        feedback_emotion = (self.shared.last_feedback_emotion or "none").lower()
        hud_text = (
            f"Face: {emotion}    "
            f"Feedback: {feedback_emotion} ({self.last_sentiment})    "
            f"AI Mode: {ai_mode}    "
            f"Games Played: {stats.games_played}    Wins: {stats.wins}"
        )
        t = self.font_ui.render(hud_text, True, pygame.Color(200, 210, 230))
        self.screen.blit(t, (420, 28))

        msg = self.shared.get_dialogue()
        if msg:
            m = self.font_small.render(msg, True, self.theme.accent_soft)
            self.screen.blit(m, (16, 52))

    def _draw_webcam_panel(self, rect: pygame.Rect) -> None:
        emotion, vision, _, _, _ = self.shared.get_snapshot()
        border = self.theme.accent if emotion == "Happy" else self.theme.accent_soft
        pygame.draw.rect(self.screen, pygame.Color(18, 22, 30), rect, border_radius=12)
        pygame.draw.rect(self.screen, border, rect, width=2, border_radius=12)

        pad = 12
        inner = rect.inflate(-2 * pad, -2 * pad)

        header = self.font_ui.render("WEBCAM FEED", True, pygame.Color(235, 240, 255))
        self.screen.blit(header, (inner.left, inner.top))

        # Reserve the lower portion of the panel for camera diagnostics text.
        preview_height = int(inner.height * 0.55)
        preview_rect = pygame.Rect(inner.left, inner.top + 26, inner.width, max(40, preview_height))
        if isinstance(vision.preview_rgb, np.ndarray):
            frame = vision.preview_rgb
            surf = self._rgb_to_surface(frame)
            surf = pygame.transform.smoothscale(surf, (preview_rect.width, preview_rect.height))
            self.screen.blit(surf, preview_rect.topleft)
        else:
            txt = self.font_small.render("Camera unavailable (press C to switch)", True, pygame.Color(170, 180, 200))
            self.screen.blit(txt, (preview_rect.left, preview_rect.top + 10))

        info_y = preview_rect.bottom + 6
        cam_line = self.font_small.render(
            f"Camera: {vision.camera_index}  ({'OK' if vision.camera_ok else 'NOT FOUND'})",
            True,
            pygame.Color(170, 180, 200),
        )
        info1 = self.font_small.render(f"Landmarks: {vision.landmarks_detected}", True, pygame.Color(170, 180, 200))
        info2 = self.font_small.render(f"Mouth width: {vision.mouth_width:.2f}", True, pygame.Color(170, 180, 200))
        info3 = self.font_small.render(f"Emotion: {vision.emotion_smoothed}  ({vision.fps:.1f} FPS)", True, border)
        self.screen.blit(cam_line, (inner.left, info_y))
        self.screen.blit(info1, (inner.left, info_y + 18))
        self.screen.blit(info2, (inner.left, info_y + 36))
        self.screen.blit(info3, (inner.left, info_y + 54))

    def _draw_status_panel(self, rect: pygame.Rect) -> None:
        emotion, _, stats, tuning, ai_mode = self.shared.get_snapshot()
        pygame.draw.rect(self.screen, pygame.Color(18, 22, 30), rect, border_radius=12)
        pygame.draw.rect(self.screen, self.theme.accent_soft, rect, width=2, border_radius=12)

        pad = 12
        x, y = rect.left + pad, rect.top + pad
        title = self.font_ui.render("AI STATUS", True, pygame.Color(235, 240, 255))
        self.screen.blit(title, (x, y))
        y += 30

        lines = [
            f"AI Status: {'Thinking...' if self.turn == 'O' and self.scene=='play' and not self.board.game_over() else 'Idle'}",
            f"Face emotion: {emotion}",
            f"Feedback emotion: {(self.shared.last_feedback_emotion or 'none').lower()}",
            f"Last sentiment: {self.last_sentiment}",
            f"Difficulty: {ai_mode} Mode",
            f"Assistive mistake prob: {min(0.30, max(0.20, tuning.assistive_mistake_prob)):.2f}",
            f"Cameras: {self.shared.camera_status_summary()}",
        ]
        for line in lines:
            t = self.font_small.render(line, True, pygame.Color(170, 180, 200))
            self.screen.blit(t, (x, y))
            y += 18

        y += 10
        if self.ai_typing:
            typing = self.font_small.render(self.ai_typing, True, self.theme.accent_soft)
            self.screen.blit(typing, (x, y))
            y += 24

        if self.hint_active and self.hint_cell is not None:
            slide = self.hint_slide.step(1.0 / 60.0) if not self.hint_slide.done else 1.0
            hint_x = int(lerp(rect.right + 180, x, slide))
            hint = self.font_small.render("Tip: Controlling the center helps.", True, self.theme.accent_soft)
            self.screen.blit(hint, (hint_x, rect.bottom - 48))

        y = rect.bottom - 72
        dist = stats.emotion_counts
        total = max(1, dist.get("Happy", 0) + dist.get("Neutral", 0))
        h_pct = int(100 * dist.get("Happy", 0) / total)
        n_pct = 100 - h_pct
        d1 = self.font_small.render(f"Emotion dist: Happy {h_pct}% / Neutral {n_pct}%", True, pygame.Color(170, 180, 200))
        self.screen.blit(d1, (x, y))

    def _draw_board(self, rect: pygame.Rect) -> None:
        emotion, _, _, _, _ = self.shared.get_snapshot()

        shake_off = pygame.Vector2(0, 0)
        if not self.board_shake.done:
            t = self.board_shake.step(1.0 / 60.0)
            amp = (1.0 - t) * 10.0
            shake_off.x = math.sin(time.perf_counter() * 55.0) * amp
            shake_off.y = math.cos(time.perf_counter() * 42.0) * amp

        rect2 = rect.move(int(shake_off.x), int(shake_off.y))

        glow = pygame.Surface((rect2.width + 16, rect2.height + 16), pygame.SRCALPHA)
        pygame.draw.rect(glow, pygame.Color(self.theme.board_glow.r, self.theme.board_glow.g, self.theme.board_glow.b, 60), glow.get_rect(), border_radius=24)
        self.screen.blit(glow, (rect2.left - 8, rect2.top - 8))

        pygame.draw.rect(self.screen, pygame.Color(22, 26, 36), rect2, border_radius=18)
        pygame.draw.rect(self.screen, self.theme.accent_soft, rect2, width=2, border_radius=18)

        # Grid
        for i in range(1, 3):
            x = rect2.left + i * rect2.width // 3
            pygame.draw.line(self.screen, pygame.Color(60, 70, 90), (x, rect2.top + 18), (x, rect2.bottom - 18), 4)
            y = rect2.top + i * rect2.height // 3
            pygame.draw.line(self.screen, pygame.Color(60, 70, 90), (rect2.left + 18, y), (rect2.right - 18, y), 4)

        # Hover highlight
        if self.scene == "play" and self.turn == "X" and not self.board.game_over():
            mx, my = pygame.mouse.get_pos()
            if rect2.collidepoint((mx, my)):
                cell = self._cell_from_pos((mx, my), rect2)
                if cell and self.board.grid[cell[0]][cell[1]] == "":
                    cell_rect = self._cell_rect(cell, rect2)
                    col = self.theme.accent if emotion == "Happy" else self.theme.accent_soft
                    self._draw_cell_glow(cell_rect, col)

        # Hint highlight
        if self.hint_active and self.hint_cell is not None:
            cell_rect = self._cell_rect(self.hint_cell, rect2)
            self._draw_cell_glow(cell_rect, self.theme.accent_soft)

        # Pieces
        now_s = time.perf_counter()
        for r in range(3):
            for c in range(3):
                mark = self.board.grid[r][c]
                if not mark:
                    continue
                cell_rect = self._cell_rect((r, c), rect2)
                anim = self.cell_anims.get((r, c))
                t = anim.t(now_s) if anim else 1.0
                if mark == "X":
                    self._draw_x(cell_rect, t, emotion)
                else:
                    self._draw_o(cell_rect, t, emotion)

                pulse = self.cell_pulse.get((r, c))
                if pulse and not pulse.done:
                    p = pulse.step(1.0 / 60.0)
                    s = 1.0 + 0.10 * p
                    pr = cell_rect.copy()
                    pr.width = int(pr.width * s)
                    pr.height = int(pr.height * s)
                    pr.center = cell_rect.center
                    pygame.draw.rect(self.screen, pygame.Color(255, 255, 255, 0), pr, border_radius=12)

        # Win line highlight
        if self.win_line:
            pts = [self._cell_rect(cell, rect2).center for cell in self.win_line]
            pygame.draw.line(self.screen, self.theme.accent, pts[0], pts[-1], 10)

    def _draw_postgame_overlay(self) -> None:
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        box = pygame.Rect(0, 0, 760, 320)
        box.center = (WINDOW_W // 2, WINDOW_H // 2)
        pygame.draw.rect(self.screen, pygame.Color(20, 24, 34), box, border_radius=18)
        pygame.draw.rect(self.screen, self.theme.accent_soft, box, width=2, border_radius=18)

        emotion, _, stats, _, _ = self.shared.get_snapshot()
        header = "Game Over"
        if self.winner == "X":
            header = "You Win!"
        elif self.winner == "O":
            header = "AI Wins"
        elif self.draw:
            header = "Draw"

        t1 = self.font_title.render(header, True, pygame.Color(240, 245, 255))
        self.screen.blit(t1, (box.left + 24, box.top + 18))

        msg = self.font_ui.render(self.postgame_message, True, self.theme.accent_soft)
        self.screen.blit(msg, (box.left + 24, box.top + 60))

        q = self.font_ui.render("How did you feel about this game?", True, pygame.Color(200, 210, 230))
        self.screen.blit(q, (box.left + 24, box.top + 110))

        input_rect = pygame.Rect(box.left + 24, box.top + 148, box.width - 48, 44)
        pygame.draw.rect(self.screen, pygame.Color(12, 14, 20), input_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.theme.accent, input_rect, width=2, border_radius=10)

        typed = self.feedback_text or ""
        t_in = self.font_ui.render(typed + ("|" if int(time.time() * 2) % 2 == 0 else ""), True, pygame.Color(230, 235, 245))
        self.screen.blit(t_in, (input_rect.left + 12, input_rect.top + 10))

        hint = self.font_small.render("Press Enter to submit and start next match.", True, pygame.Color(170, 180, 200))
        self.screen.blit(hint, (box.left + 24, box.bottom - 34))

        # Stats summary
        dist = stats.emotion_counts
        total = max(1, dist.get("Happy", 0) + dist.get("Neutral", 0))
        h_pct = int(100 * dist.get("Happy", 0) / total)
        n_pct = 100 - h_pct
        stats_line = f"Games Played: {stats.games_played}   Wins: {stats.wins}   Losses: {stats.losses}   Draws: {stats.draws}"
        stats2 = f"Emotion Distribution: Happy {h_pct}% / Neutral {n_pct}%"
        s1 = self.font_small.render(stats_line, True, pygame.Color(170, 180, 200))
        s2 = self.font_small.render(stats2, True, pygame.Color(170, 180, 200))
        self.screen.blit(s1, (box.left + 24, box.top + 210))
        self.screen.blit(s2, (box.left + 24, box.top + 232))

    def _cell_from_pos(self, pos: tuple[int, int], board_rect: pygame.Rect) -> Optional[tuple[int, int]]:
        x, y = pos
        if not board_rect.collidepoint(pos):
            return None
        rel_x = x - board_rect.left
        rel_y = y - board_rect.top
        c = int(rel_x / (board_rect.width / 3))
        r = int(rel_y / (board_rect.height / 3))
        if 0 <= r < 3 and 0 <= c < 3:
            return (r, c)
        return None

    def _cell_rect(self, cell: tuple[int, int], board_rect: pygame.Rect) -> pygame.Rect:
        r, c = cell
        cw = board_rect.width // 3
        ch = board_rect.height // 3
        x = board_rect.left + c * cw
        y = board_rect.top + r * ch
        return pygame.Rect(x + 8, y + 8, cw - 16, ch - 16)

    def _draw_cell_glow(self, rect: pygame.Rect, color: pygame.Color) -> None:
        glow = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(glow, pygame.Color(color.r, color.g, color.b, 70), glow.get_rect(), border_radius=14)
        self.screen.blit(glow, rect.topleft)
        pygame.draw.rect(self.screen, pygame.Color(color.r, color.g, color.b, 180), rect, width=2, border_radius=14)

    def _draw_x(self, rect: pygame.Rect, t: float, emotion: str) -> None:
        col = self.theme.accent if emotion == "Happy" else self.theme.accent_soft
        pad = 18
        a = (rect.left + pad, rect.top + pad)
        b = (rect.right - pad, rect.bottom - pad)
        c = (rect.left + pad, rect.bottom - pad)
        d = (rect.right - pad, rect.top + pad)

        # Two-stroke animation
        if t < 0.5:
            tt = ease_in_cubic(t / 0.5)
            p = (int(a[0] + (b[0] - a[0]) * tt), int(a[1] + (b[1] - a[1]) * tt))
            pygame.draw.line(self.screen, col, a, p, 10)
        else:
            pygame.draw.line(self.screen, col, a, b, 10)
            tt = ease_in_cubic((t - 0.5) / 0.5)
            p = (int(c[0] + (d[0] - c[0]) * tt), int(c[1] + (d[1] - c[1]) * tt))
            pygame.draw.line(self.screen, col, c, p, 10)

    def _draw_o(self, rect: pygame.Rect, t: float, emotion: str) -> None:
        col = self.theme.accent if emotion == "Happy" else self.theme.accent_soft
        center = rect.center
        radius = min(rect.width, rect.height) // 2 - 14
        start_angle = -np.pi / 2
        end_angle = start_angle + (np.pi * 2) * ease_in_out(t)
        pygame.draw.arc(
            self.screen,
            col,
            pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2),
            start_angle,
            end_angle,
            10,
        )

    def _confetti_palette(self, emotion: str) -> list[pygame.Color]:
        if emotion == "Happy":
            return [pygame.Color(70, 255, 150), pygame.Color(240, 230, 90), pygame.Color(80, 200, 120)]
        return [pygame.Color(120, 190, 255), pygame.Color(160, 220, 255), pygame.Color(80, 160, 255)]

    def _random_ai_dialogue(self, emotion: str) -> str:
        if emotion == "Happy":
            return self.rng.choice(["Nice move!", "You’re playing well!", "Keep it up!"])
        return self.rng.choice(["You're doing great. Keep going.", "Remember to block your opponent.", "Take your time."])

    @staticmethod
    def _rgb_to_surface(rgb: np.ndarray) -> pygame.Surface:
        # pygame.surfarray expects (w, h, 3)
        arr = np.transpose(rgb, (1, 0, 2))
        return pygame.surfarray.make_surface(arr)

