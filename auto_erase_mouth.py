#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_erase_mouth.py

erase_mouth_offline.py を「動画向けに安定するように」自動チューニングするラッパー。

やること:
- 参照フレーム(ref_frame)を "口閉じ優先" で自動選択（confidence最大に固定しない）
- coverage / valid-policy などを複数候補で実行
- 口元の残り/継ぎ目を簡易スコアリングして最良の出力を採用

注意:
- 完全な画質評価は難しいので、スコアは「口内の残り」と「周辺との輝度なじみ」の指標に寄せています。
- 速度より品質優先の安全設計。候補数は多すぎないデフォルトにしています。
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Track loading (compatible with face_track_anime_detector.py output)
# ----------------------------

@dataclass
class Track:
    quad: np.ndarray          # (N,4,2) float32
    valid: np.ndarray         # (N,) bool
    filled: np.ndarray        # (N,4,2) float32 (filled by policy)
    confidence: Optional[np.ndarray]
    total: int


def _fill_quads(quads: np.ndarray, valid: np.ndarray, policy: str) -> np.ndarray:
    filled = quads.copy()
    N = len(filled)
    if N == 0:
        return filled
    if policy == "hold":
        last = None
        for i in range(N):
            if valid[i]:
                last = filled[i]
            elif last is not None:
                filled[i] = last
        # fill before first valid
        idxs = np.where(valid)[0]
        if len(idxs):
            first = int(idxs[0])
            for i in range(first):
                filled[i] = filled[first]
    return filled


def load_track(path: str, target_w: int, target_h: int, policy: str) -> Track:
    npz = np.load(path, allow_pickle=False)
    quads = np.asarray(npz["quad"], dtype=np.float32)
    if quads.ndim != 3 or quads.shape[1:] != (4, 2):
        raise ValueError("quad must be (N,4,2)")
    valid = np.asarray(npz.get("valid", np.ones((quads.shape[0],), np.uint8)), dtype=np.uint8) > 0
    conf = None
    if "confidence" in npz:
        conf = np.asarray(npz["confidence"], dtype=np.float32)
        if conf.shape[0] != quads.shape[0]:
            conf = None

    # scale if track was saved for different size
    src_w = int(npz.get("w", target_w))
    src_h = int(npz.get("h", target_h))
    if src_w > 0 and src_h > 0 and (src_w != target_w or src_h != target_h):
        sx = float(target_w) / float(src_w)
        sy = float(target_h) / float(src_h)
        quads = quads.copy()
        quads[:, :, 0] *= sx
        quads[:, :, 1] *= sy

    filled = _fill_quads(quads, valid, policy=policy)
    return Track(quad=quads, valid=valid, filled=filled, confidence=conf, total=int(quads.shape[0]))


# ----------------------------
# Geometry helpers (mirrors erase_mouth_offline.py logic)
# ----------------------------

def quad_wh(quad: np.ndarray) -> Tuple[float, float]:
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    w = float(np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) * 0.5
    h = float(np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) * 0.5
    return w, h


def ensure_even_ge2(v: int) -> int:
    """Round to even integer >= 2.

    NOTE: Uses floor (n-1 for odd) to match erase_mouth_offline.py.
    """
    v = int(v)
    if v < 2:
        return 2
    return v if (v % 2 == 0) else (v - 1)


def warp_frame_to_norm(frame_bgr: np.ndarray, quad: np.ndarray, norm_w: int, norm_h: int) -> np.ndarray:
    src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [norm_w - 1, 0], [norm_w - 1, norm_h - 1], [0, norm_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(frame_bgr, M, dsize=(norm_w, norm_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return patch


def make_mouth_mask(w: int, h: int, rx: int, ry: int, center_y_offset_px: int = 0, top_clip_frac: float = 0.84) -> np.ndarray:
    """Mouth-eraser mask specialized for anime faces.

    NOTE: This must match erase_mouth_offline.py's make_mouth_mask() for QA scoring consistency.
    """
    mask = np.zeros((h, w), np.uint8)
    cx, cy0 = w // 2, h // 2
    cy = int(np.clip(cy0 + int(center_y_offset_px), 0, h - 1))
    rx = int(max(1, min(rx, w // 2 - 1)))
    ry = int(max(1, min(ry, h // 2 - 1)))
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 255, -1)
    # Clip the top portion to protect the nose area.
    top_clip_frac = float(np.clip(top_clip_frac, 0.6, 1.0))
    clip_y = int(round(cy - ry * top_clip_frac))
    clip_y = int(np.clip(clip_y, 0, h))
    if clip_y > 0:
        mask[:clip_y, :] = 0
    return mask


def inner_and_ring_masks(norm_w: int, norm_h: int, coverage: float,
                         mask_scale_x: float = 0.62, mask_scale_y: float = 0.62,
                         ring_px: int = 18, center_y_off: int = 0, top_clip_frac: float = 0.84) -> Tuple[np.ndarray, np.ndarray]:
    """Generate inner (erase) and ring (shading reference) masks.

    NOTE: Coverage mapping must match erase_mouth_offline.py for QA scoring consistency.
    """
    cov = float(coverage)
    msx, msy = float(mask_scale_x), float(mask_scale_y)
    rpx = int(ring_px)
    cyo = int(center_y_off)
    tcf = float(top_clip_frac)
    if cov >= 0.0:
        cov = float(np.clip(cov, 0.0, 1.0))
        # Match erase_mouth_offline.py exactly
        msx = float(0.50 + 0.18 * cov)   # 0.50..0.68
        msy = float(0.44 + 0.14 * cov)   # 0.44..0.58
        rpx = int(round(16 + 10 * cov))  # 16..26
        cyo = int(round(norm_h * (0.05 + 0.01 * cov)))
        tcf = float(0.84 - 0.06 * cov)   # 0.84..0.78

    rx = int((norm_w * msx) * 0.5)
    ry = int((norm_h * msy) * 0.5)

    inner_u8 = make_mouth_mask(norm_w, norm_h, rx=rx, ry=ry, center_y_offset_px=cyo, top_clip_frac=tcf)
    outer_u8 = make_mouth_mask(norm_w, norm_h, rx=rx + rpx, ry=ry + rpx, center_y_offset_px=cyo, top_clip_frac=tcf)
    ring_u8 = cv2.subtract(outer_u8, inner_u8)
    return inner_u8, ring_u8


# ----------------------------
# Ref-frame selection (smart)
# ----------------------------

def _candidate_indices(track: Track, n_out: int, top_k: int = 48) -> List[int]:
    valid_idxs = np.where(track.valid[:n_out])[0]
    if len(valid_idxs) == 0:
        return [0]
    if track.confidence is None:
        # evenly spaced
        if len(valid_idxs) <= top_k:
            return valid_idxs.tolist()
        step = max(1, len(valid_idxs) // top_k)
        return valid_idxs[::step].tolist()
    conf = track.confidence[:n_out].copy()
    conf[~track.valid[:n_out]] = -1.0
    # take top_k by confidence
    idxs = np.argsort(-conf)[: min(top_k, len(conf))]
    idxs = [int(i) for i in idxs if conf[i] >= 0.0]
    return idxs if idxs else [int(valid_idxs[0])]


def choose_ref_frame_smart(video_path: str, track: Track, n_out: int,
                           norm_w: int, norm_h: int, coverage_for_mask: float = 0.60) -> int:
    """Pick ref frame with closed-mouth preference among high-confidence candidates."""
    cand = _candidate_indices(track, n_out=n_out, top_k=48)

    inner_u8, _ring_u8 = inner_and_ring_masks(norm_w, norm_h, coverage=coverage_for_mask)
    inner = inner_u8 > 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return int(cand[0])

    best_idx = int(cand[0])
    best_score = float("inf")

    # Score: variance + edge energy inside inner mask (lower is "simpler"/closed)
    try:
        for idx in cand:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            patch = warp_frame_to_norm(bgr, track.filled[idx], norm_w, norm_h)
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
            v = float(np.var(gray[inner]))
            # edge energy
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            e = float(np.mean((gx[inner] ** 2 + gy[inner] ** 2)))
            score = v + 0.25 * e
            if score < best_score:
                best_score = score
                best_idx = int(idx)
    finally:
        cap.release()

    return best_idx


# ----------------------------
# Output QA scoring
# ----------------------------

def sample_indices(track: Track, n_out: int, n_samples: int = 12) -> List[int]:
    idxs = np.where(track.valid[:n_out])[0]
    if len(idxs) == 0:
        return [0]
    if len(idxs) <= n_samples:
        return [int(i) for i in idxs]
    # evenly spaced samples
    step = max(1, len(idxs) // n_samples)
    out = [int(i) for i in idxs[::step][:n_samples]]
    return out


# Invasion (over-erase) detection:
# We penalize large changes in areas that should stay stable (nose / cheeks) in the normalized patch.
# This helps avoid cases where fast motion / large coverage erases nose or cheek lines.
INVASION_WEIGHT = 2.0  # higher => more conservative (less invasion)
INVASION_FLOOR = 0.0   # ignore tiny diffs
INVASION_CAP = 50.0    # cap per-frame invasion score to prevent extreme outliers dominating

def invasion_masks(norm_w: int, norm_h: int, inner_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (nose_mask_u8, cheek_mask_u8) in normalized patch coords."""
    nose = np.zeros((norm_h, norm_w), np.uint8)
    cheek = np.zeros((norm_h, norm_w), np.uint8)

    # Nose-ish zone: upper center
    x0 = int(round(norm_w * 0.30))
    x1 = int(round(norm_w * 0.70))
    y0 = int(round(norm_h * 0.10))
    y1 = int(round(norm_h * 0.38))
    nose[y0:y1, x0:x1] = 255

    # Cheek zones: left/right sides (exclude very top to avoid hair/eye)
    y0c = int(round(norm_h * 0.38))
    y1c = int(round(norm_h * 0.90))
    # left
    cheek[y0c:y1c, int(round(norm_w * 0.05)):int(round(norm_w * 0.25))] = 255
    # right
    cheek[y0c:y1c, int(round(norm_w * 0.75)):int(round(norm_w * 0.95))] = 255

    # Never count mouth area as invasion
    inner = (inner_u8 > 0).astype(np.uint8) * 255
    nose = cv2.bitwise_and(nose, cv2.bitwise_not(inner))
    cheek = cv2.bitwise_and(cheek, cv2.bitwise_not(inner))
    return nose, cheek

def score_mouthless(video_in: str, video_out: str, track: Track, n_out: int, norm_w: int, norm_h: int,
                    coverage: float, n_samples: int = 12) -> float:
    """Lower is better."""
    inner_u8, ring_u8 = inner_and_ring_masks(norm_w, norm_h, coverage=coverage)
    inner = inner_u8 > 0
    ring = ring_u8 > 0
    nose_u8, cheek_u8 = invasion_masks(norm_w, norm_h, inner_u8)
    nose = nose_u8 > 0
    cheek = cheek_u8 > 0

    cap_in = cv2.VideoCapture(video_in)
    cap = cv2.VideoCapture(video_out)
    if (not cap.isOpened()) or (not cap_in.isOpened()):
        cap.release()
        cap_in.release()
        return float("inf")

    idxs = sample_indices(track, n_out=n_out, n_samples=n_samples)
    scores: List[float] = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        cap_in.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, bgr = cap.read()
        ok_in, bgr_in = cap_in.read()
        if not ok or bgr is None or not ok_in or bgr_in is None:
            continue
        patch = warp_frame_to_norm(bgr, track.filled[idx], norm_w, norm_h)
        patch_in = warp_frame_to_norm(bgr_in, track.filled[idx], norm_w, norm_h)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        # continuity: inner mean close to ring mean
        mu_i = float(np.mean(L[inner]))
        mu_r = float(np.mean(L[ring]))
        std_i = float(np.std(L[inner]))
        # texture inside inner should be low after erase
        # use gradient energy in inner
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
        edge_i = float(np.mean(np.sqrt(gx[inner] ** 2 + gy[inner] ** 2)))
        # invasion: compare in/out in LAB L channel (perceptual-ish brightness)
        L_in = cv2.cvtColor(patch_in, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        diff = np.abs(L - L_in)
        inv_n = float(np.mean(diff[nose])) if np.any(nose) else 0.0
        inv_c = float(np.mean(diff[cheek])) if np.any(cheek) else 0.0
        invasion = max(inv_n, inv_c)
        invasion = float(np.clip(invasion - INVASION_FLOOR, 0.0, INVASION_CAP))

        s = abs(mu_i - mu_r) + 0.35 * std_i + 0.20 * edge_i + (INVASION_WEIGHT * invasion)
        scores.append(float(s))

    cap.release()
    cap_in.release()
    if not scores:
        return float("inf")
    return float(np.mean(scores))


# ----------------------------
# Running erase_mouth_offline.py candidates
# ----------------------------

def run_erase(erase_py: str, video: str, track_path: str, out_path: str,
             valid_policy: str, ref_frame: int, coverage: float, keep_audio: bool,
             shading: str, debug_dir: str = "") -> int:
    cmd = [sys.executable, erase_py,
           "--video", video,
           "--track", track_path,
           "--out", out_path,
           "--valid-policy", valid_policy,
           "--ref-frame", str(int(ref_frame)),
           "--coverage", str(float(coverage)),
           "--shading", shading]
    if keep_audio:
        cmd.append("--keep-audio")
    if debug_dir:
        cmd += ["--debug", debug_dir]
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto tune mouth erase for videos.")
    ap.add_argument("--video", required=True, help="input video")
    ap.add_argument("--track", required=True, help="mouth_track(.npz)")
    ap.add_argument("--out", required=True, help="output mouthless video")
    ap.add_argument("--erase", default="erase_mouth_offline.py", help="path to erase_mouth_offline.py")
    ap.add_argument("--coverage", default="0.60,0.70,0.80", help="candidate coverages comma-separated")
    ap.add_argument("--valid-policy", default="hold", choices=["hold", "strict"], help="base policy")
    ap.add_argument("--try-strict", action="store_true", help="also try strict policy")
    ap.add_argument("--ref-mode", default="smart", choices=["smart", "confidence", "first"], help="ref frame selection")
    ap.add_argument("--ref-topk", type=int, default=48, help="ref selection candidates by confidence")
    ap.add_argument("--keep-audio", action="store_true")
    ap.add_argument("--shading", default="plane", choices=["plane", "none"])
    ap.add_argument("--oversample", type=float, default=1.2)
    ap.add_argument("--norm-w", type=int, default=0)
    ap.add_argument("--norm-h", type=int, default=0)
    ap.add_argument("--qa-samples", type=int, default=12)
    ap.add_argument("--debug", default="", help="optional debug dir (best candidate only)")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        print(f"[error] video not found: {args.video}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.track):
        print(f"[error] track not found: {args.track}", file=sys.stderr)
        return 2

    # resolve erase script path
    erase_py = args.erase
    if not os.path.isfile(erase_py):
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, erase_py)
        if os.path.isfile(cand):
            erase_py = cand
    if not os.path.isfile(erase_py):
        print(f"[error] erase script not found: {args.erase}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[error] failed to open video: {args.video}", file=sys.stderr)
        return 2
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if vid_w <= 0 or vid_h <= 0:
        print("[error] invalid video size", file=sys.stderr)
        return 2

    track = load_track(args.track, vid_w, vid_h, policy=args.valid_policy)
    n_out = min(total_frames if total_frames > 0 else track.total, track.total)
    if n_out <= 0:
        print("[error] empty track/video", file=sys.stderr)
        return 2

    # Decide normalized patch size (same as erase_mouth_offline.py)
    quads_for_size = track.filled[:n_out]
    ws = np.array([quad_wh(q)[0] for q in quads_for_size], dtype=np.float32)
    hs = np.array([quad_wh(q)[1] for q in quads_for_size], dtype=np.float32)
    ratio = float(np.median(ws / np.maximum(1e-6, hs)))
    p95w = float(np.percentile(ws, 95))

    if args.norm_w > 0:
        norm_w = int(args.norm_w)
    else:
        norm_w = int(round(p95w * float(args.oversample)))
    norm_w = ensure_even_ge2(max(96, norm_w))

    if args.norm_h > 0:
        norm_h = int(args.norm_h)
    else:
        norm_h = int(round(norm_w / max(0.25, min(4.0, ratio))))
    norm_h = ensure_even_ge2(max(64, norm_h))

    coverages = []
    for s in args.coverage.split(","):
        s = s.strip()
        if not s:
            continue
        coverages.append(float(s))
    if not coverages:
        coverages = [0.60, 0.70, 0.80]

    # choose ref frame
    if args.ref_mode == "first":
        ref_idx = int(np.where(track.valid[:n_out])[0][0]) if track.valid[:n_out].any() else 0
    elif args.ref_mode == "confidence":
        if track.confidence is None:
            ref_idx = int(np.where(track.valid[:n_out])[0][0]) if track.valid[:n_out].any() else 0
        else:
            m = track.confidence[:n_out].copy()
            m[~track.valid[:n_out]] = -1.0
            ref_idx = int(np.argmax(m))
    else:
        ref_idx = choose_ref_frame_smart(args.video, track, n_out=n_out, norm_w=norm_w, norm_h=norm_h, coverage_for_mask=coverages[0])
    print(f"[auto_erase] ref_frame={ref_idx} (mode={args.ref_mode})")

    policies = [args.valid_policy]
    if args.try_strict and "strict" not in policies:
        policies.append("strict")

    tmpdir = tempfile.mkdtemp(prefix="auto_erase_")
    best = None
    best_score = float("inf")
    best_meta = ""

    try:
        for pol in policies:
            for cov in coverages:
                out_tmp = os.path.join(tmpdir, f"out_{pol}_cov{cov:.2f}.mp4")
                print(f"[auto_erase] run policy={pol} coverage={cov:.2f}")
                rc = run_erase(erase_py, args.video, args.track, out_tmp,
                               valid_policy=pol, ref_frame=ref_idx, coverage=cov,
                               keep_audio=args.keep_audio, shading=args.shading,
                               debug_dir="")
                if rc != 0 or (not os.path.isfile(out_tmp)):
                    print(f"[auto_erase]   -> failed rc={rc}")
                    continue

                s = score_mouthless(args.video, out_tmp, track, n_out=n_out, norm_w=norm_w, norm_h=norm_h,
                                    coverage=cov, n_samples=int(args.qa_samples))
                print(f"[auto_erase]   qa_score={s:.4f} (lower is better)")
                if s < best_score:
                    best_score = s
                    best = out_tmp
                    best_meta = f"policy={pol}, coverage={cov:.2f}, ref={ref_idx}"
        if best is None:
            print("[auto_erase] no candidate succeeded", file=sys.stderr)
            return 3

        # If debug dir requested, re-run best candidate with debug enabled
        if args.debug:
            os.makedirs(args.debug, exist_ok=True)
            # parse best coverage/policy from best_meta
            pol = "hold" if "policy=hold" in best_meta else "strict"
            cov = float(best_meta.split("coverage=")[1].split(",")[0])
            out_dbg = os.path.join(tmpdir, "out_best_debug.mp4")
            run_erase(erase_py, args.video, args.track, out_dbg,
                     valid_policy=pol, ref_frame=ref_idx, coverage=cov,
                     keep_audio=args.keep_audio, shading=args.shading,
                     debug_dir=args.debug)
            if os.path.isfile(out_dbg):
                best = out_dbg

        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        shutil.copy2(best, args.out)
        print(f"[auto_erase] [OK] best: {best_meta} qa_score={best_score:.4f} -> {args.out}")
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
