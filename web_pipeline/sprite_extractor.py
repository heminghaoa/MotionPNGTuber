"""
嘴 PNG 提取模块 - 从视频中自动提取 5 种嘴型 PNG

输出文件：
- open.png: 口开
- closed.png: 口闭
- half.png: 半开
- e.png: 横长（え）
- u.png: 嘟嘴（う）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .mouth_detector import MouthTrack


@dataclass
class MouthFrameInfo:
    """单帧嘴部信息"""
    frame_idx: int
    quad: np.ndarray  # (4, 2)
    width: float
    height: float
    aspect_ratio: float  # width / height
    valid: bool
    openness: float = 0.0  # 嘴部开合程度 (0-1, 基于图像分析)


@dataclass
class SpriteSelection:
    """5 种嘴型的选择结果"""
    open_idx: int
    closed_idx: int
    half_idx: int
    e_idx: int
    u_idx: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "open": self.open_idx,
            "closed": self.closed_idx,
            "half": self.half_idx,
            "e": self.e_idx,
            "u": self.u_idx,
        }


class SpriteExtractor:
    """
    嘴 PNG 提取器

    用法:
        extractor = SpriteExtractor()
        extractor.extract(video_path, track, output_dir)
    """

    def __init__(
        self,
        feather_px: int = 15,
        mask_scale: float = 0.85,
        padding: float = 1.2,
    ):
        """
        Args:
            feather_px: 羽化像素
            mask_scale: 椭圆 mask 缩放
            padding: 输出尺寸 padding
        """
        self.feather_px = feather_px
        self.mask_scale = mask_scale
        self.padding = padding

    def _quad_size(self, quad: np.ndarray) -> Tuple[float, float]:
        """计算 quad 的宽高"""
        quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        w = float(np.linalg.norm(quad[1] - quad[0]))
        h = float(np.linalg.norm(quad[3] - quad[0]))
        return w, h

    def _analyze_openness(self, frame: np.ndarray, quad: np.ndarray) -> float:
        """
        分析嘴部开合程度 (基于图像分析)

        原理：张开的嘴内部较暗（口腔），闭合的嘴主要是皮肤和嘴唇色
        返回 0-1 的开合程度
        """
        quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        x1, y1 = int(quad[0][0]), int(quad[0][1])
        x2, y2 = int(quad[2][0]), int(quad[2][1])

        # 边界检查
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        # 裁剪嘴部区域
        roi = frame[y1:y2, x1:x2]

        # 转换到 HSV 分析暗色区域
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 暗色像素 (V < 80) 通常是口腔内部
        dark_mask = hsv[:, :, 2] < 80
        dark_ratio = np.sum(dark_mask) / dark_mask.size

        # 红色/深色像素 (嘴唇打开时露出的口腔)
        # H 在 0-10 或 170-180 范围，S > 50，V < 150
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        red_dark_mask = (
            ((h_channel < 15) | (h_channel > 165)) &
            (s_channel > 30) &
            (v_channel < 150)
        )
        red_dark_ratio = np.sum(red_dark_mask) / red_dark_mask.size

        # 综合评分
        openness = dark_ratio * 0.5 + red_dark_ratio * 0.5
        return min(1.0, openness * 3)  # 放大系数

    def _analyze_frames(
        self,
        track: MouthTrack,
        video_path: str | Path | None = None,
    ) -> List[MouthFrameInfo]:
        """分析所有帧的嘴部信息"""
        frames = []

        # 如果提供视频路径，分析图像内容
        cap = None
        if video_path:
            cap = cv2.VideoCapture(str(video_path))

        for i, f in enumerate(track.frames):
            quad = np.array(f["quad"], dtype=np.float32)
            valid = f.get("valid", True)

            if not valid:
                frames.append(MouthFrameInfo(
                    frame_idx=i,
                    quad=quad,
                    width=0,
                    height=0,
                    aspect_ratio=1,
                    valid=False,
                    openness=0,
                ))
                continue

            w, h = self._quad_size(quad)
            aspect = w / max(h, 1e-6)

            # 分析开合程度
            openness = 0.0
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    openness = self._analyze_openness(frame, quad)

            frames.append(MouthFrameInfo(
                frame_idx=i,
                quad=quad,
                width=w,
                height=h,
                aspect_ratio=aspect,
                valid=True,
                openness=openness,
            ))

        if cap is not None:
            cap.release()

        return frames

    def _select_5_types(
        self,
        frames: List[MouthFrameInfo],
    ) -> SpriteSelection:
        """
        选择 5 种嘴型

        选择标准（基于图像分析的开合程度）：
        - open: 开合程度最大
        - closed: 开合程度最小
        - half: 开合程度接近中位数
        - e: 宽高比最大且有一定开度（横长）
        - u: 开度较小且分散（嘟嘴）
        """
        valid_frames = [f for f in frames if f.valid]

        if len(valid_frames) < 5:
            raise ValueError(f"有效帧不足 5 个 ({len(valid_frames)})")

        openness = np.array([f.openness for f in valid_frames])
        heights = np.array([f.height for f in valid_frames])
        widths = np.array([f.width for f in valid_frames])
        aspects = np.array([f.aspect_ratio for f in valid_frames])

        # 打印调试信息
        print(f"[SpriteExtractor] 开合程度范围: {openness.min():.3f} - {openness.max():.3f}")

        used = set()

        def pick(scores: np.ndarray, maximize: bool = True) -> int:
            """选择最佳帧（排除已使用）"""
            order = np.argsort(scores)
            if maximize:
                order = order[::-1]

            for idx in order:
                frame_idx = valid_frames[idx].frame_idx
                if frame_idx not in used:
                    used.add(frame_idx)
                    return frame_idx

            return valid_frames[order[0]].frame_idx

        # 1. open: 开合程度最大
        open_idx = pick(openness, maximize=True)

        # 2. closed: 开合程度最小
        closed_idx = pick(openness, maximize=False)

        # 3. half: 开合程度接近中位数
        median_open = np.median(openness)
        half_scores = -np.abs(openness - median_open)
        half_idx = pick(half_scores, maximize=True)

        # 4. e: 宽高比大 + 有一定开度（横长张嘴）
        e_scores = aspects + openness * 0.5
        e_idx = pick(e_scores, maximize=True)

        # 5. u: 开度小 + 分布均匀（嘟嘴）
        u_scores = -openness - aspects * 0.3
        u_idx = pick(u_scores, maximize=True)

        return SpriteSelection(
            open_idx=open_idx,
            closed_idx=closed_idx,
            half_idx=half_idx,
            e_idx=e_idx,
            u_idx=u_idx,
        )

    def _make_ellipse_mask(self, w: int, h: int) -> np.ndarray:
        """创建椭圆 mask"""
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        rx = int((w * self.mask_scale) * 0.5)
        ry = int((h * self.mask_scale) * 0.5)
        rx = max(1, min(rx, w // 2 - 1))
        ry = max(1, min(ry, h // 2 - 1))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        return mask

    def _feather_mask(self, mask: np.ndarray) -> np.ndarray:
        """羽化 mask"""
        if self.feather_px <= 0:
            return (mask.astype(np.float32) / 255.0).clip(0, 1)

        k = 2 * self.feather_px + 1
        m = cv2.GaussianBlur(mask, (k, k), sigmaX=0)
        return (m.astype(np.float32) / 255.0).clip(0, 1)

    def _extract_sprite(
        self,
        frame: np.ndarray,
        quad: np.ndarray,
        out_w: int,
        out_h: int,
    ) -> np.ndarray:
        """
        从帧中提取嘴 sprite

        Returns:
            RGBA 图像 (H, W, 4)
        """
        # 透视变换
        src = quad.astype(np.float32).reshape(4, 2)
        dst = np.array([
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        patch = cv2.warpPerspective(
            frame, M, (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # 生成 mask
        mask_u8 = self._make_ellipse_mask(out_w, out_h)
        mask_f = self._feather_mask(mask_u8)

        # 合成 RGBA
        rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
        rgba[:, :, :3] = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        rgba[:, :, 3] = (mask_f * 255).astype(np.uint8)

        return rgba

    def _compute_unified_size(
        self,
        frames: List[MouthFrameInfo],
        selection: SpriteSelection,
    ) -> Tuple[int, int]:
        """计算统一的输出尺寸"""
        indices = list(selection.as_dict().values())
        idx_to_frame = {f.frame_idx: f for f in frames}

        max_w, max_h = 0.0, 0.0
        for idx in indices:
            if idx in idx_to_frame:
                f = idx_to_frame[idx]
                max_w = max(max_w, f.width)
                max_h = max(max_h, f.height)

        # 应用 padding 并确保偶数
        w = int(max_w * self.padding)
        h = int(max_h * self.padding)
        w = max(2, w + (w % 2))
        h = max(2, h + (h % 2))

        return w, h

    def extract(
        self,
        video_path: str | Path,
        track: MouthTrack,
        output_dir: str | Path,
        progress_callback: callable | None = None,
    ) -> Dict[str, Path]:
        """
        提取 5 种嘴型 PNG

        Args:
            video_path: 视频路径
            track: 嘴部轨迹
            output_dir: 输出目录
            progress_callback: 进度回调

        Returns:
            {"open": path, "closed": path, ...}
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 分析帧（包括图像内容分析）
        frames = self._analyze_frames(track, video_path)

        # 选择 5 种嘴型
        selection = self._select_5_types(frames)
        print(f"[SpriteExtractor] 选择的帧: {selection.as_dict()}")

        # 计算统一尺寸
        out_w, out_h = self._compute_unified_size(frames, selection)
        print(f"[SpriteExtractor] 输出尺寸: {out_w}x{out_h}")

        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        # 提取每种嘴型
        idx_to_frame = {f.frame_idx: f for f in frames}
        results = {}
        sprite_names = ["open", "closed", "half", "e", "u"]

        try:
            for i, (name, frame_idx) in enumerate(selection.as_dict().items()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    print(f"[SpriteExtractor] 警告: 无法读取帧 {frame_idx}")
                    continue

                quad = idx_to_frame[frame_idx].quad
                sprite = self._extract_sprite(frame, quad, out_w, out_h)

                # 保存 PNG
                out_path = output_dir / f"{name}.png"
                # OpenCV 需要 BGRA
                bgra = cv2.cvtColor(sprite, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(str(out_path), bgra)

                results[name] = out_path
                print(f"  {name}.png <- 帧 {frame_idx}")

                if progress_callback:
                    progress_callback(i + 1, 5)

        finally:
            cap.release()

        print(f"[SpriteExtractor] 完成，输出到: {output_dir}")
        return results


def extract_mouth_sprites(
    video_path: str | Path,
    track: MouthTrack,
    output_dir: str | Path,
) -> Dict[str, Path]:
    """
    便捷函数：提取嘴 PNG

    Args:
        video_path: 视频路径
        track: 嘴部轨迹
        output_dir: 输出目录

    Returns:
        {"open": path, "closed": path, ...}
    """
    extractor = SpriteExtractor()
    return extractor.extract(video_path, track, output_dir)
