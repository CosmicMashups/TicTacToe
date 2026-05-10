"""Modern Pygame UI for the Emotion-Aware Tic-Tac-Toe game with Zoomable/Pannable Tree View."""

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
from emotion_game_ai.game.game_tree import GameTreeNode
from emotion_game_ai.game.game_tree_evaluation import build_evaluated_gameplay_tree_by_levels
from emotion_game_ai.game.search_stats import SearchStats
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

HUD_H = 68
LEFT_PANEL_RATIO = 0.16
RIGHT_PANEL_RATIO = 0.40
MIN_PANEL_W = 210
MAX_PANEL_W = 800
BOARD_MIN_SIZE = 420
BOARD_MAX_SIZE = 700

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
        display_info = pygame.display.Info()
        self.screen_w = display_info.current_w
        self.screen_h = display_info.current_h
        self.screen = pygame.display.set_mode((self.screen_w - 100, self.screen_h - 100), pygame.SCALED | pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("segoe ui", 28, bold=True)
        self.font_ui = pygame.font.SysFont("segoe ui", 18)
        self.font_small = pygame.font.SysFont("segoe ui", 14)
        self.font_tree = pygame.font.SysFont("segoe ui", 12)

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
        self.show_ai_diagnostics = False
        self.last_ai_search_stats: Optional[SearchStats] = None
        self.tree_graph_root: Optional[GameTreeNode] = None
        self.tree_graph_levels = 4
        self.tree_graph_min_levels = 2
        self.tree_graph_max_levels = 4
        self.live_tree_graph_enabled = True
        self.tree_graph_message = "Live tree simulation is enabled."

        self.tree_zoom = 1.0
        self.tree_pan = pygame.Vector2(0, 0)
        self.is_panning_tree = False

        self.sfx_move = try_load_sound(os.path.join("assets", "sounds", "move.mp3"))
        self.sfx_win = try_load_sound(os.path.join("assets", "sounds", "win.mp3"))
        self.sfx_click = try_load_sound(os.path.join("assets", "sounds", "click.mp3"))

        self.scene = "play"  # "play" | "postgame" | "tree_view"
        self.feedback_text = ""
        self.last_sentiment = "neutral"
        self.postgame_message = ""
        self._refresh_tree_cache(notify=False)

    def run(self) -> None:
        while True:
            dt_s = self.clock.tick(60) / 1000.0
            if not self._handle_events():
                return

            self._step(dt_s)
            self._draw()
            pygame.display.flip()

    def _layout(self) -> dict[str, pygame.Rect]:
        window_w, window_h = self.screen.get_size()
        hud = pygame.Rect(0, 0, window_w, HUD_H)
        main_h = window_h - HUD_H
        left_w = max(MIN_PANEL_W, min(MAX_PANEL_W, int(window_w * LEFT_PANEL_RATIO)))
        right_w = max(MIN_PANEL_W, min(MAX_PANEL_W, int(window_w * RIGHT_PANEL_RATIO)))
        center_w = max(420, window_w - left_w - right_w)
        left = pygame.Rect(0, HUD_H, left_w, main_h)
        center = pygame.Rect(left_w, HUD_H, center_w, main_h)
        right = pygame.Rect(left_w + center_w, HUD_H, right_w, main_h)

        board_size = min(center.width - 24, center.height - 34)
        board_size = max(BOARD_MIN_SIZE, min(BOARD_MAX_SIZE, board_size))
        board = pygame.Rect(0, 0, board_size, board_size)
        board.center = center.center

        full_tree = pygame.Rect(40, HUD_H + 40, window_w - 80, window_h - HUD_H - 80)
        
        return {
            "hud": hud, "left": left, "right": right, 
            "center": center, "board": board, "full_tree": full_tree
        }

    def _handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return False
                if ev.key == pygame.K_v:
                    if self.scene == "play":
                        self.scene = "tree_view"
                        self.shared.set_dialogue("Switched to Full Tree View.", ttl_s=2.0)
                    elif self.scene == "tree_view":
                        self.scene = "play"
                        self.shared.set_dialogue("Returned to Game.", ttl_s=2.0)
                    continue
                
                if self.scene == "tree_view" and ev.key == pygame.K_r:
                    self.tree_zoom = 1.0
                    self.tree_pan = pygame.Vector2(0, 0)
                    continue

                if ev.key == pygame.K_c:
                    idx = self.shared.request_next_camera()
                    self.shared.set_dialogue(f"Switching camera to index {idx} (press C to cycle).", ttl_s=2.5)
                if ev.key == pygame.K_TAB:
                    self.show_ai_diagnostics = not self.show_ai_diagnostics
                    state = "shown" if self.show_ai_diagnostics else "hidden"
                    self.shared.set_dialogue(f"AI diagnostics {state}.", ttl_s=2.0)
                
                if ev.key == pygame.K_g:
                    self.live_tree_graph_enabled = not self.live_tree_graph_enabled
                    if self.live_tree_graph_enabled:
                        self._refresh_tree_cache()
                    else:
                        self.shared.set_dialogue("Live tree simulation hidden.", ttl_s=2.0)
                    continue
                if ev.key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
                    self._adjust_tree_graph_levels(1)
                    continue
                if ev.key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
                    self._adjust_tree_graph_levels(-1)
                    continue

            if self.scene == "play":
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self._on_click(pygame.mouse.get_pos())
            
            elif self.scene == "tree_view":
                if ev.type == pygame.MOUSEWHEEL:
                    zoom_speed = 0.1
                    self.tree_zoom = max(0.2, min(8.0, self.tree_zoom + ev.y * zoom_speed))
                
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self.is_panning_tree = True
                if ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    self.is_panning_tree = False
                if ev.type == pygame.MOUSEMOTION and self.is_panning_tree:
                    self.tree_pan += pygame.Vector2(ev.rel)

            elif self.scene == "postgame":
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

            _, _, _, tuning, _ = self.shared.get_snapshot()
            tuning.adjust_for_sentiment(sentiment)

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

        emotion, _, stats, _, _ = self.shared.get_snapshot()
        stats.record_emotion(emotion)

        if (r, c) == (1, 1):
            self.particles.burst(board_rect.center, self._confetti_palette(emotion), count=18)

        self._post_move_updates(player="X")
        self.turn = "O"
        self._refresh_tree_graph_if_live()
        self._queue_ai_move()

    def _queue_ai_move(self) -> None:
        emotion, _, _, tuning, _ = self.shared.get_snapshot()
        decision = decide_move(self.board, emotion, tuning, self.rng)
        self.shared.ai_mode = decision.mode
        self.shared.set_dialogue(decision.message, ttl_s=3.0)
        self.last_ai_search_stats = decision.search_stats

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
        self._set_theme_target(self.shared.current_emotion)
        self._refresh_tree_graph_if_live()

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
                    self._refresh_tree_graph_if_live()

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
        target = HAPPY_THEME if emotion == "Happy" else NEUTRAL_THEME
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

        if self.scene == "play" or self.scene == "postgame":
            self._draw_webcam_panel(rects["left"])
            self._draw_status_panel(rects["right"])
            self._draw_board(rects["board"])
            if self.scene == "postgame":
                self._draw_postgame_overlay()
        elif self.scene == "tree_view":
            self._draw_full_tree_view(rects["full_tree"])

        self.particles.draw(self.screen)

    def _draw_full_tree_view(self, rect: pygame.Rect) -> None:
        """Draw a large version of the game tree with zoom and pan functionality."""
        pygame.draw.rect(self.screen, pygame.Color(18, 22, 30, 230), rect, border_radius=18)
        pygame.draw.rect(self.screen, self.theme.accent_soft, rect, width=2, border_radius=18)
        
        viewport = rect.inflate(-40, -40)
        
        old_clip = self.screen.get_clip()
        self.screen.set_clip(viewport)
        
        if self.tree_graph_root and self.live_tree_graph_enabled:
            v_w = viewport.width * self.tree_zoom
            v_h = viewport.height * self.tree_zoom
            
            v_rect = pygame.Rect(0, 0, int(v_w), int(v_h))
            v_rect.center = viewport.center + self.tree_pan
            
            self._draw_native_tree_graph(v_rect, self.tree_graph_root, is_expanded=True)
        else:
            msg = "Tree data unavailable. Press G to toggle live view."
            txt = self.font_ui.render(msg, True, pygame.Color(170, 180, 200))
            self.screen.blit(txt, txt.get_rect(center=rect.center))

        self.screen.set_clip(old_clip)

        nav_help = "Left Click & Drag: Pan  |  Scroll: Zoom  |  'R': Reset View  |  'V': Return to Game"
        hint = self.font_small.render(nav_help, True, self.theme.accent_soft)
        self.screen.blit(hint, (rect.left + 20, rect.bottom - 30))

    def _refresh_tree_cache(self, notify: bool = True) -> None:
        try:
            is_maximizing = self.turn == "O"
            self.tree_graph_root = build_evaluated_gameplay_tree_by_levels(
                self.board,
                is_maximizing=is_maximizing,
                levels=self.tree_graph_levels,
            )
            self.tree_graph_message = (
                f"Live tree: {self.tree_graph_levels} levels from current board."
            )
            if notify:
                self.shared.set_dialogue(
                    f"Live tree updated ({self.tree_graph_levels} levels).",
                    ttl_s=2.5,
                )
        except Exception as exc:
            self.tree_graph_root = None
            self.tree_graph_message = f"Tree graph unavailable: {exc}"
            if notify:
                self.shared.set_dialogue(self.tree_graph_message, ttl_s=4.0)

    def _refresh_tree_graph_if_live(self) -> None:
        if self.live_tree_graph_enabled:
            self._refresh_tree_cache(notify=False)

    def _adjust_tree_graph_levels(self, delta: int) -> None:
        old_levels = self.tree_graph_levels
        self.tree_graph_levels = max(
            self.tree_graph_min_levels,
            min(self.tree_graph_max_levels, self.tree_graph_levels + delta),
        )
        if self.tree_graph_levels == old_levels:
            self.shared.set_dialogue(f"Tree graph already at {self.tree_graph_levels} levels.", ttl_s=2.0)
            return
        self.shared.set_dialogue(f"Tree graph levels set to {self.tree_graph_levels}.", ttl_s=2.0)
        self._refresh_tree_graph_if_live()

    def _draw_hud(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, pygame.Color(0, 0, 0, 80), rect)
        emotion, _, stats, _, ai_mode = self.shared.get_snapshot()

        title = self.font_title.render("Emotion-Aware Tic-Tac-Toe", True, pygame.Color(240, 245, 255))
        self.screen.blit(title, (16, 10))

        feedback_emotion = (self.shared.last_feedback_emotion or "none").lower()
        hud_text = (
            f"Face: {emotion}    "
            f"Feedback: {feedback_emotion} ({self.last_sentiment})    "
            f"AI Mode: {ai_mode}    "
            f"Games Played: {stats.games_played}    Wins: {stats.wins}"
        )
        t = self.font_small.render(hud_text, True, pygame.Color(200, 210, 230))
        self.screen.blit(t, (370, 22))

        msg = self.shared.get_dialogue()
        if msg:
            m = self.font_small.render(msg, True, self.theme.accent_soft)
            self.screen.blit(m, (16, 46))

    def _draw_webcam_panel(self, rect: pygame.Rect) -> None:
        emotion, vision, stats, _, ai_mode = self.shared.get_snapshot()
        border = self.theme.accent if emotion == "Happy" else self.theme.accent_soft
        pygame.draw.rect(self.screen, pygame.Color(18, 22, 30), rect, border_radius=12)
        pygame.draw.rect(self.screen, border, rect, width=2, border_radius=12)

        pad = 10
        inner = rect.inflate(-2 * pad, -2 * pad)

        header = self.font_ui.render("WEBCAM FEED", True, pygame.Color(235, 240, 255))
        self.screen.blit(header, (inner.left, inner.top))

        preview_height = int(inner.height * 0.42)
        preview_rect = pygame.Rect(inner.left, inner.top + 24, inner.width, max(40, preview_height))
        if isinstance(vision.preview_rgb, np.ndarray):
            frame = vision.preview_rgb
            surf = self._rgb_to_surface(frame)
            surf = pygame.transform.smoothscale(surf, (preview_rect.width, preview_rect.height))
            self.screen.blit(surf, preview_rect.topleft)
        else:
            self._draw_wrapped_text(
                "Camera unavailable (press C to switch)",
                self.font_small,
                pygame.Color(170, 180, 200),
                pygame.Rect(preview_rect.left, preview_rect.top + 10, preview_rect.width, preview_rect.height - 10),
                line_spacing=3,
            )

        info_y = preview_rect.bottom + 3
        info_area = pygame.Rect(inner.left, info_y, inner.width, 60)
        line_h = self.font_small.get_linesize()
        self._draw_wrapped_text(
            f"Camera (C): {vision.camera_index}  ({'OK' if vision.camera_ok else 'NOT FOUND'})",
            self.font_small,
            pygame.Color(170, 180, 200),
            pygame.Rect(info_area.left, info_area.top, info_area.width, line_h + 2),
            line_spacing=2,
        )
        self._draw_wrapped_text(
            f"Landmarks: {vision.landmarks_detected}",
            self.font_small,
            pygame.Color(170, 180, 200),
            pygame.Rect(info_area.left, info_area.top + 15, info_area.width, line_h + 2),
            line_spacing=2,
        )
        self._draw_wrapped_text(
            f"Mouth width: {vision.mouth_width:.2f}",
            self.font_small,
            pygame.Color(170, 180, 200),
            pygame.Rect(info_area.left, info_area.top + 30, info_area.width, line_h + 2),
            line_spacing=2,
        )
        self._draw_wrapped_text(
            f"Emotion: {vision.emotion_smoothed}  ({vision.fps:.1f} FPS)",
            self.font_small,
            border,
            pygame.Rect(info_area.left, info_area.top + 45, info_area.width, line_h + 2),
            line_spacing=2,
        )

        status_y = info_y + 67
        dist = stats.emotion_counts
        total = max(1, dist.get("Happy", 0) + dist.get("Neutral", 0))
        h_pct = int(100 * dist.get("Happy", 0) / total)
        n_pct = 100 - h_pct
        status_lines = [
            "AI STATUS",
            f"AI Status: {'Thinking...' if self.turn == 'O' and self.scene == 'play' and not self.board.game_over() else 'Idle'}",
            f"Face emotion: {emotion}",
            f"Mode: {ai_mode}",
            f"Diagnostics: {'ON' if self.show_ai_diagnostics else 'OFF'} (TAB)",
            f"Live tree: {'ON' if self.live_tree_graph_enabled else 'OFF'}  Lvl: {self.tree_graph_levels}",
            f"Emotion dist: Happy {h_pct}% / Neutral {n_pct}%",
        ]
        line_step = 18
        max_status_y = inner.bottom - 14
        for idx, line in enumerate(status_lines):
            if status_y > max_status_y:
                break
            color = pygame.Color(235, 240, 255) if idx == 0 else pygame.Color(170, 180, 200)
            font = self.font_ui if idx == 0 else self.font_small
            self._draw_wrapped_text(
                line,
                font,
                color,
                pygame.Rect(inner.left, status_y, inner.width, line_step),
                line_spacing=2,
            )
            status_y += line_step

    def _draw_status_panel(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, pygame.Color(18, 22, 30), rect, border_radius=12)
        pygame.draw.rect(self.screen, self.theme.accent_soft, rect, width=2, border_radius=12)

        pad = 10
        x, y = rect.left + pad, rect.top + pad
        title = self.font_ui.render("TREE VIEW", True, pygame.Color(235, 240, 255))
        self.screen.blit(title, (x, y))
        y += 20
        if self.ai_typing:
            typing = self.font_small.render(self.ai_typing, True, self.theme.accent_soft)
            self.screen.blit(typing, (x, y))
            y += 20

        if self.show_ai_diagnostics and self.last_ai_search_stats is not None:
            diag = self.last_ai_search_stats
            diag_lines = [
                f"Search: {diag.algorithm or 'alpha-beta'}",
                f"Move/value: {diag.best_move} / {diag.best_value}",
                f"Visited: {diag.nodes_visited}  Leaves: {diag.leaf_nodes}",
                f"Pruned: {diag.pruned_nodes}  Events: {diag.pruning_events}",
                f"Depth: {diag.max_depth}  Time: {diag.execution_time_ms:.2f} ms",
            ]
            for line in diag_lines:
                t = self.font_small.render(line, True, self.theme.accent_soft)
                self.screen.blit(t, (x, y))
                y += 14

        y = self._draw_tree_graph_in_status_panel(rect, x, y)

        if self.hint_active and self.hint_cell is not None:
            slide = self.hint_slide.step(1.0 / 60.0) if not self.hint_slide.done else 1.0
            hint_x = int(lerp(rect.right + 180, x, slide))
            hint = self.font_small.render("Tip: Controlling the center helps.", True, self.theme.accent_soft)
            self.screen.blit(hint, (hint_x, rect.bottom - 44))

    def _draw_tree_graph_in_status_panel(self, rect: pygame.Rect, x: int, y: int) -> int:
        y += 6
        header = f"Live Game Tree ({self.tree_graph_levels} levels)"
        header_color = self.theme.accent_soft if self.live_tree_graph_enabled else pygame.Color(130, 140, 160)
        title = self.font_small.render(header, True, header_color)
        self.screen.blit(title, (x, y))
        y += 16
        root_value = "N/A"
        if self.tree_graph_root is not None and self.tree_graph_root.value is not None:
            root_value = str(self.tree_graph_root.value)
        legend_rect = pygame.Rect(x, y, rect.width - 20, 28)
        self._draw_wrapped_text(
            f"Root value: {root_value}    P = pruned branch",
            self.font_tree,
            pygame.Color(160, 170, 190),
            legend_rect,
            line_spacing=2,
        )
        y += 20

        available_h = max(180, rect.bottom - y - 36)
        graph_h = min(max(320, int(rect.height * 0.64)), available_h)
        graph_margin_x = 18
        graph_rect = pygame.Rect(x + graph_margin_x, y, rect.width - 20 - (2 * graph_margin_x), graph_h)
        pygame.draw.rect(self.screen, pygame.Color(12, 14, 20), graph_rect, border_radius=10)
        pygame.draw.rect(self.screen, pygame.Color(50, 42, 60), graph_rect, width=1, border_radius=10)

        if self.live_tree_graph_enabled and self.tree_graph_root is not None:
            self._draw_native_tree_graph(graph_rect, self.tree_graph_root)
        else:
            msg = "Press G to show live tree."
            if self.live_tree_graph_enabled:
                msg = self.tree_graph_message
            text = self.font_small.render(msg[:58], True, pygame.Color(170, 180, 200))
            self.screen.blit(text, (graph_rect.left + 8, graph_rect.top + 12))

        controls_rect = pygame.Rect(x, graph_rect.bottom + 5, rect.width - 20, 18)
        self._draw_wrapped_text(
            "G toggle   +/- levels   V expand",
            self.font_small,
            pygame.Color(150, 160, 180),
            controls_rect,
            line_spacing=2,
        )
        return graph_rect.bottom + 22

    def _draw_native_tree_graph(self, rect: pygame.Rect, root: GameTreeNode, is_expanded: bool = False) -> None:
        inner = rect.inflate(-20, -20)
        positions = self._tree_node_positions(root, inner)
        nodes = self._walk_tree_nodes(root)
        
        node_scale = (rect.width / 800.0) if is_expanded else (rect.width / 400.0)
        node_scale = max(0.4, min(4.0, node_scale))
        
        edge_width = max(1, int(1.5 * node_scale))
        major_radius = max(6, int(8 * node_scale))
        minor_radius = max(3, int(5 * node_scale))

        for node in nodes:
            for child in node.children:
                start = positions.get(node.node_id)
                end = positions.get(child.node_id)
                if start is None or end is None:
                    continue
                color = pygame.Color(88, 78, 110) if child.pruned else pygame.Color(185, 96, 152)
                pygame.draw.line(self.screen, color, start, end, edge_width)

        for node in nodes:
            pos = positions.get(node.node_id)
            if pos is None:
                continue
            color = self._native_tree_node_color(node)
            radius = major_radius if node.depth <= 1 else minor_radius
            if node.pruned:
                radius = minor_radius
            pygame.draw.circle(self.screen, color, pos, radius)
            pygame.draw.circle(self.screen, pygame.Color(235, 210, 230), pos, radius, width=1)

            max_labeled_depth = 3 if is_expanded and self.tree_zoom > 1.2 else (2 if inner.width >= 360 else 1)
            if node.depth <= max_labeled_depth:
                self._draw_tree_node_label(node, pos, is_expanded)
            elif node.pruned:
                mark = self.font_tree.render("P", True, pygame.Color(190, 170, 185))
                self.screen.blit(mark, (pos[0] + 4, pos[1] - 8))

    def _tree_node_positions(self, root: GameTreeNode, rect: pygame.Rect) -> dict[int, tuple[int, int]]:
        levels: dict[int, list[GameTreeNode]] = {}
        for node in self._walk_tree_nodes(root):
            levels.setdefault(node.depth, []).append(node)

        max_depth = max(levels) if levels else 1
        positions: dict[int, tuple[int, int]] = {}
        for depth, nodes in levels.items():
            y = rect.top + rect.height // 2 if max_depth == 0 else rect.top + int(depth * rect.height / max_depth)
            count = len(nodes)
            for idx, node in enumerate(nodes):
                if count == 1:
                    x = rect.centerx
                else:
                    margin = rect.width * 0.05
                    usable_w = rect.width - (2 * margin)                    
                    x = rect.left + margin + int(idx * usable_w / (count - 1))
                positions[node.node_id] = (x, y)
        return positions

    def _walk_tree_nodes(self, root: GameTreeNode) -> list[GameTreeNode]:
        nodes = [root]
        for child in root.children:
            nodes.extend(self._walk_tree_nodes(child))
        return nodes

    def _native_tree_node_color(self, node: GameTreeNode) -> pygame.Color:
        if node.pruned:
            return pygame.Color(82, 74, 92)
        if node.depth == 0:
            return pygame.Color(240, 120, 170)
        if node.is_maximizing:
            return pygame.Color(220, 95, 150)
        return pygame.Color(130, 95, 185)

    def _draw_tree_node_label(self, node: GameTreeNode, pos: tuple[int, int], is_expanded: bool = False) -> None:
        role = "MAX" if node.is_maximizing else "MIN"
        if node.depth == 0:
            text = f"{role} v{node.value}"
        else:
            move_txt = node.move if node.move is not None else "-"
            text = f"{move_txt} v{node.value}"
        
        font = self.font_ui if is_expanded else self.font_tree
        label = font.render(text, True, pygame.Color(245, 235, 250))
        box = label.get_rect()
        box.center = (pos[0], pos[1] - (24 if is_expanded else 16))
        box.inflate_ip(8, 4)
        
        pygame.draw.rect(self.screen, pygame.Color(34, 24, 42), box, border_radius=5)
        pygame.draw.rect(self.screen, pygame.Color(100, 70, 120), box, width=1, border_radius=5)
        self.screen.blit(label, label.get_rect(center=box.center))

    def _draw_board(self, rect: pygame.Rect) -> None:
        emotion, _, _, _, _ = self.shared.get_snapshot()

        shake_off = pygame.Vector2(0, 0)
        if not self.board_shake.done:
            t = self.board_shake.step(1.0 / 60.0)
            amp = (1.0 - t) * 10.0
            shake_off.x = math.sin(time.perf_counter() * 55.0) * amp
            shake_off.y = math.cos(time.perf_counter() * 42.0) * amp

        rect2 = rect.move(int(shake_off.x), int(shake_off.y))
        board_scale = max(0.85, min(1.35, rect2.width / 500.0))
        grid_inset = int(18 * board_scale)
        grid_width = max(3, int(4 * board_scale))
        mark_width = max(7, int(10 * board_scale))
        win_width = max(8, int(10 * board_scale))

        glow = pygame.Surface((rect2.width + 16, rect2.height + 16), pygame.SRCALPHA)
        pygame.draw.rect(glow, pygame.Color(self.theme.board_glow.r, self.theme.board_glow.g, self.theme.board_glow.b, 60), glow.get_rect(), border_radius=24)
        self.screen.blit(glow, (rect2.left - 8, rect2.top - 8))

        pygame.draw.rect(self.screen, pygame.Color(22, 26, 36), rect2, border_radius=18)
        pygame.draw.rect(self.screen, self.theme.accent_soft, rect2, width=2, border_radius=18)

        for i in range(1, 3):
            x = rect2.left + i * rect2.width // 3
            pygame.draw.line(self.screen, pygame.Color(60, 70, 90), (x, rect2.top + grid_inset), (x, rect2.bottom - grid_inset), grid_width)
            y = rect2.top + i * rect2.height // 3
            pygame.draw.line(self.screen, pygame.Color(60, 70, 90), (rect2.left + grid_inset, y), (rect2.right - grid_inset, y), grid_width)

        if self.scene == "play" and self.turn == "X" and not self.board.game_over():
            mx, my = pygame.mouse.get_pos()
            if rect2.collidepoint((mx, my)):
                cell = self._cell_from_pos((mx, my), rect2)
                if cell and self.board.grid[cell[0]][cell[1]] == "":
                    cell_rect = self._cell_rect(cell, rect2)
                    col = self.theme.accent if emotion == "Happy" else self.theme.accent_soft
                    self._draw_cell_glow(cell_rect, col)

        if self.hint_active and self.hint_cell is not None:
            cell_rect = self._cell_rect(self.hint_cell, rect2)
            self._draw_cell_glow(cell_rect, self.theme.accent_soft)

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
                    self._draw_x(cell_rect, t, emotion, mark_width)
                else:
                    self._draw_o(cell_rect, t, emotion, mark_width)

                pulse = self.cell_pulse.get((r, c))
                if pulse and not pulse.done:
                    p = pulse.step(1.0 / 60.0)
                    s = 1.0 + 0.10 * p
                    pr = cell_rect.copy()
                    pr.width = int(pr.width * s)
                    pr.height = int(pr.height * s)
                    pr.center = cell_rect.center
                    pygame.draw.rect(self.screen, pygame.Color(255, 255, 255, 0), pr, border_radius=12)

        if self.win_line:
            pts = [self._cell_rect(cell, rect2).center for cell in self.win_line]
            pygame.draw.line(self.screen, self.theme.accent, pts[0], pts[-1], win_width)

    def _draw_postgame_overlay(self) -> None:
        window_w, window_h = self.screen.get_size()
        overlay = pygame.Surface((window_w, window_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        box = pygame.Rect(0, 0, 760, 320)
        box.center = (window_w // 2, window_h // 2)
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
        pad = max(6, min(12, int(min(cw, ch) * 0.06)))
        return pygame.Rect(x + pad, y + pad, cw - (2 * pad), ch - (2 * pad))

    def _draw_cell_glow(self, rect: pygame.Rect, color: pygame.Color) -> None:
        glow = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(glow, pygame.Color(color.r, color.g, color.b, 70), glow.get_rect(), border_radius=14)
        self.screen.blit(glow, rect.topleft)
        pygame.draw.rect(self.screen, pygame.Color(color.r, color.g, color.b, 180), rect, width=2, border_radius=14)

    def _draw_x(self, rect: pygame.Rect, t: float, emotion: str, stroke: int) -> None:
        col = self.theme.accent if emotion == "Happy" else self.theme.accent_soft
        pad = max(12, min(22, int(min(rect.width, rect.height) * 0.18)))
        a = (rect.left + pad, rect.top + pad)
        b = (rect.right - pad, rect.bottom - pad)
        c = (rect.left + pad, rect.bottom - pad)
        d = (rect.right - pad, rect.top + pad)

        if t < 0.5:
            tt = ease_in_cubic(t / 0.5)
            p = (int(a[0] + (b[0] - a[0]) * tt), int(a[1] + (b[1] - a[1]) * tt))
            pygame.draw.line(self.screen, col, a, p, stroke)
        else:
            pygame.draw.line(self.screen, col, a, b, stroke)
            tt = ease_in_cubic((t - 0.5) / 0.5)
            p = (int(c[0] + (d[0] - c[0]) * tt), int(c[1] + (d[1] - c[1]) * tt))
            pygame.draw.line(self.screen, col, c, p, stroke)

    def _draw_o(self, rect: pygame.Rect, t: float, emotion: str, stroke: int) -> None:
        col = self.theme.accent if emotion == "Happy" else self.theme.accent_soft
        center = rect.center
        radius = min(rect.width, rect.height) // 2 - max(10, stroke + 2)
        start_angle = -np.pi / 2
        end_angle = start_angle + (np.pi * 2) * ease_in_out(t)
        pygame.draw.arc(
            self.screen,
            col,
            pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2),
            start_angle,
            end_angle,
            stroke,
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
        arr = np.transpose(rgb, (1, 0, 2))
        return pygame.surfarray.make_surface(arr)

    def _draw_wrapped_text(
        self,
        text: str,
        font: pygame.font.Font,
        color: pygame.Color,
        rect: pygame.Rect,
        line_spacing: int = 2,
    ) -> int:
        if rect.width <= 4 or rect.height <= 4:
            return rect.top
        words = text.split(" ")
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= rect.width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = word
            else:
                chunk = word
                while chunk and font.size(chunk)[0] > rect.width:
                    split_idx = max(1, len(chunk) - 1)
                    while split_idx > 1 and font.size(chunk[:split_idx])[0] > rect.width:
                        split_idx -= 1
                    lines.append(chunk[:split_idx])
                    chunk = chunk[split_idx:]
                current = chunk
        if current:
            lines.append(current)

        y = rect.top
        line_h = font.get_linesize()
        max_bottom = rect.bottom
        for line in lines:
            if y + line_h > max_bottom:
                break
            surf = font.render(line, True, color)
            self.screen.blit(surf, (rect.left, y))
            y += line_h + line_spacing
        return y