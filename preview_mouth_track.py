#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preview_mouth_track.py

mouth_track.npz を使って、口スプライト(open.png 等)を quad にワープ合成し、
OpenCV でプレビュー再生する簡易ツール。

例:
  python preview_mouth_track.py --video loop.mp4 --track mouth_track.npz --sprite mouth/open.png --draw-quad

キー:
  - q / ESC : 終了
  - Space   : 一時停止/再開
  - [ / ]   : (一時停止中) 前/次フレーム
  - r       : 先頭へ戻る
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def load_rgba(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read sprite: {path}")

    # 2D(グレースケール) → BGRA
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

    # BGR → BGRA
    if img.shape[2] == 3:
        a = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, a], axis=2)

    if img.shape[2] != 4:
        raise RuntimeError(f"Unexpected sprite channels: {img.shape}")
    return img


def warp_sprite_to_quad(sprite_rgba: np.ndarray, dst_quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    sh, sw = sprite_rgba.shape[:2]
    # OpenCV の射影変換は 0..w-1 / 0..h-1 の座標系に揃える方が端が安定しやすい
    src_quad = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], dtype=np.float32)
    dst_quad = np.asarray(dst_quad, dtype=np.float32).reshape(4, 2)
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(
        sprite_rgba,
        M,
        (int(out_w), int(out_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


def alpha_blend(dst_bgr: np.ndarray, src_rgba_full: np.ndarray, alpha_mul: float = 1.0) -> np.ndarray:
    """dst_bgr に src_rgba_full をアルファ合成（同サイズ前提）"""
    if dst_bgr.shape[:2] != src_rgba_full.shape[:2]:
        raise ValueError("size mismatch")

    dst = dst_bgr.astype(np.float32)
    src_rgb = src_rgba_full[:, :, :3].astype(np.float32)

    a = (src_rgba_full[:, :, 3].astype(np.float32) / 255.0) * float(alpha_mul)
    a = np.clip(a, 0.0, 1.0)
    a = a[:, :, None]  # (H,W,1)

    out = dst * (1.0 - a) + src_rgb * a
    return np.clip(out, 0, 255).astype(np.uint8)


def maybe_scale_quads(quads: np.ndarray, track_w: int, track_h: int, vid_w: int, vid_h: int) -> np.ndarray:
    if track_w <= 0 or track_h <= 0:
        return quads
    if track_w == vid_w and track_h == vid_h:
        return quads
    sx = vid_w / track_w
    sy = vid_h / track_h
    out = quads.copy().astype(np.float32)
    out[:, :, 0] *= sx
    out[:, :, 1] *= sy
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--track", required=True, help="mouth_track.npz")
    ap.add_argument("--sprite", required=True, help="口スプライト (open.png 等)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1, help="-1で最後まで")
    ap.add_argument("--alpha", type=float, default=1.0, help="合成アルファ倍率(0..1)")
    ap.add_argument("--preview-scale", type=float, default=0.5, help="プレビュー表示の縮小率 (0.5=半分)")
    ap.add_argument("--window", default="preview")
    ap.add_argument("--show-quad", "--draw-quad", dest="show_quad", action="store_true", help="quad を線で表示")
    ap.add_argument("--no-loop", action="store_true", help="動画末尾で終了（デフォルトはループ）")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        raise RuntimeError(f"Video not found: {args.video}")
    if not os.path.exists(args.track):
        raise RuntimeError(f"Track not found: {args.track}")
    if not os.path.exists(args.sprite):
        raise RuntimeError(f"Sprite not found: {args.sprite}")

    track = np.load(args.track, allow_pickle=False)
    quads = track["quad"].astype(np.float32)
    valid = track["valid"].astype(np.uint8) if "valid" in track else np.ones((len(quads),), np.uint8)
    track_w = int(track["w"]) if "w" in track else 0
    track_h = int(track["h"]) if "h" in track else 0

    sprite = load_rgba(args.sprite)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    # プレビュー用縮小サイズ（処理も縮小サイズで行う）
    scale = float(args.preview_scale)
    vid_w = max(2, int(orig_w * scale))
    vid_h = max(2, int(orig_h * scale))
    print(f"[info] preview size: {vid_w}x{vid_h} (scale={scale})")

    # quads をプレビューサイズにスケーリング
    quads = maybe_scale_quads(quads, track_w, track_h, vid_w, vid_h)

    start = max(0, int(args.start))
    end = len(quads) if args.end < 0 else min(len(quads), int(args.end))
    if start >= end:
        raise RuntimeError("invalid start/end")

    loop_enabled = not bool(args.no_loop)

    paused = False
    i = start
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    last_frame = None  # Cache last rendered frame for paused state
    need_refresh = True  # Flag to indicate frame needs re-rendering

    print("[info] keys: q/ESC=quit, space=pause/resume, [/]=step (paused), r=restart")

    def render_frame(frame_idx: int, is_paused: bool) -> np.ndarray | None:
        """Render a frame with overlays. Returns None on failure."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        if scale < 1.0:
            frame = cv2.resize(frame, (vid_w, vid_h), interpolation=cv2.INTER_AREA)

        if int(valid[frame_idx]) != 0:
            warped = warp_sprite_to_quad(sprite, quads[frame_idx], vid_w, vid_h)
            frame = alpha_blend(frame, warped, alpha_mul=args.alpha)

        if args.show_quad:
            q = quads[frame_idx].astype(np.int32).reshape(4, 2)
            cv2.polylines(frame, [q], isClosed=True, color=(0, 255, 0), thickness=2)

        status = "  (paused)" if is_paused else ""
        cv2.putText(
            frame,
            f"frame {frame_idx}/{end-1}  valid={int(valid[frame_idx])}{status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    while True:
        # --- render ---
        if not paused:
            if i >= end:
                if loop_enabled:
                    i = start
                else:
                    break

            last_frame = render_frame(i, is_paused=False)
            if last_frame is None:
                if loop_enabled:
                    i = start
                    continue
                break

            cv2.imshow(args.window, last_frame)
            i += 1
            need_refresh = False
        else:
            # Paused: refresh display if needed (e.g., after stepping or entering pause)
            if need_refresh:
                display_idx = max(start, min(end - 1, i - 1 if i > start else i))
                last_frame = render_frame(display_idx, is_paused=True)
                if last_frame is not None:
                    cv2.imshow(args.window, last_frame)
                need_refresh = False

        key = cv2.waitKey(int(1000 / max(1.0, fps))) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord(' '):
            paused = not paused
            need_refresh = True  # Refresh to update "(paused)" status
            continue
        if key == ord('r'):
            i = start
            need_refresh = True
            continue

        # Paused stepping
        # Note on index logic:
        #   - 'i' points to the "next frame to be played" (i.e., the frame that will be shown when resumed)
        #   - When paused, the currently displayed frame is i-1 (since we increment i after showing each frame)
        #   - '[' key: go to previous frame -> set i so that i-1 is one less than current display
        #   - ']' key: go to next frame -> set i = display_idx + 2, because we'll show i-1 = display_idx + 1
        if paused:
            stepped = False
            if key == ord('['):
                display_idx = max(start, min(end - 1, i - 1 if i > start else i))
                if display_idx > start:
                    i = display_idx
                    stepped = True
            elif key == ord(']'):
                display_idx = max(start, min(end - 1, i - 1 if i > start else i))
                if display_idx < end - 1:
                    i = display_idx + 2
                    stepped = True

            if stepped:
                need_refresh = True

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
