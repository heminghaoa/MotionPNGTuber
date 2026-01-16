"""
口消し模块 - 使用 OpenCV inpaint 擦除嘴部

替代原项目依赖大模型 inpainting 的方案，使用传统 CV 方法
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import Config
from .mouth_detector import MouthTrack


@dataclass
class EraseConfig:
    """口消し配置"""

    pad: int = 10  # mask 膨胀像素
    inpaint_radius: int = 5  # inpaint 半径
    blend: bool = True  # 边缘融合
    feather_px: int = 8  # 羽化像素
    top_clip_frac: float = 0.85  # 顶部裁剪比例（保护鼻子）
    bottom_extend: float = 0.3  # 向下扩展比例（嘴张开时向下扩展）


class MouthEraser:
    """
    使用 OpenCV inpaint 擦除嘴部

    用法:
        eraser = MouthEraser()
        eraser.erase_video("video.mp4", track, "output.mp4")
    """

    def __init__(self, config: Config | None = None, erase_config: EraseConfig | None = None):
        self.config = config or Config()
        self.erase_config = erase_config or EraseConfig(
            pad=self.config.erase_pad,
            inpaint_radius=self.config.erase_inpaint_radius,
            blend=self.config.erase_blend,
        )

    def _create_ellipse_mask(
        self,
        width: int,
        height: int,
        cx: int,
        cy: int,
        rx: int,
        ry: int,
    ) -> np.ndarray:
        """
        创建椭圆 mask

        Args:
            width, height: mask 尺寸
            cx, cy: 椭圆中心
            rx, ry: 椭圆半径

        Returns:
            uint8 mask (0 或 255)
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        # 调整中心位置（稍微向下偏移，避免擦除鼻子）
        cy_offset = int(ry * 0.1)
        cy = min(cy + cy_offset, height - 1)

        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        # 裁剪顶部（保护鼻子区域）
        top_clip = self.erase_config.top_clip_frac
        clip_y = int(cy - ry * top_clip)
        clip_y = max(0, clip_y)
        if clip_y > 0:
            mask[:clip_y, :] = 0

        return mask

    def _feather_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        膨胀 + 羽化 mask

        Returns:
            float32 mask (0-1)
        """
        # 膨胀
        pad = self.erase_config.pad
        if pad > 0:
            kernel_size = 2 * pad + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.dilate(mask, kernel, iterations=1)

        # 羽化
        feather = self.erase_config.feather_px
        if feather > 0:
            kernel_size = 2 * feather + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigmaX=0)

        return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)

    def _quad_to_ellipse_params(
        self, quad: np.ndarray, frame_h: int, frame_w: int
    ) -> tuple[int, int, int, int]:
        """
        从 quad 计算椭圆参数

        Args:
            quad: (4, 2) 四边形顶点
            frame_h, frame_w: 帧尺寸

        Returns:
            (cx, cy, rx, ry)
        """
        quad = np.asarray(quad, dtype=np.float32)

        # 计算 bbox
        x_min = max(0, int(quad[:, 0].min()))
        y_min = max(0, int(quad[:, 1].min()))
        x_max = min(frame_w, int(quad[:, 0].max()))
        y_max = min(frame_h, int(quad[:, 1].max()))

        # 向下扩展（嘴张开时向下扩展）
        bbox_h = y_max - y_min
        bottom_ext = int(bbox_h * self.erase_config.bottom_extend)
        y_max = min(frame_h, y_max + bottom_ext)

        # 计算椭圆参数
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        rx = max(1, (x_max - x_min) // 2 + self.erase_config.pad)
        ry = max(1, (y_max - y_min) // 2 + self.erase_config.pad)

        return cx, cy, rx, ry

    def erase_frame(
        self,
        frame: np.ndarray,
        quad: np.ndarray,
        valid: bool = True,
    ) -> np.ndarray:
        """
        擦除单帧图像中的嘴部

        Args:
            frame: BGR 图像 (H, W, 3)
            quad: (4, 2) 嘴部四边形顶点
            valid: 是否有效

        Returns:
            擦除后的 BGR 图像
        """
        if not valid:
            return frame.copy()

        h, w = frame.shape[:2]
        cx, cy, rx, ry = self._quad_to_ellipse_params(quad, h, w)

        # 创建 mask
        mask_u8 = self._create_ellipse_mask(w, h, cx, cy, rx, ry)

        # 检查 mask 是否有效
        if mask_u8.max() == 0:
            return frame.copy()

        # OpenCV inpaint
        result = cv2.inpaint(
            frame,
            mask_u8,
            inpaintRadius=self.erase_config.inpaint_radius,
            flags=cv2.INPAINT_TELEA,
        )

        # 可选：边缘融合
        if self.erase_config.blend:
            mask_f = self._feather_mask(mask_u8)
            mask_f = mask_f[:, :, np.newaxis]  # (H, W, 1)
            result = (
                frame.astype(np.float32) * (1.0 - mask_f)
                + result.astype(np.float32) * mask_f
            )
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def erase_video(
        self,
        video_path: str | Path,
        track: MouthTrack,
        output_path: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """
        擦除视频中所有帧的嘴部

        Args:
            video_path: 输入视频路径
            track: 嘴部轨迹
            output_path: 输出视频路径
            progress_callback: 进度回调 fn(current, total)

        Returns:
            输出视频路径
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 打开输入视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建输出视频（尝试多种编码器）
        out = None
        codecs = [
            ("avc1", ".mp4"),  # H.264 for macOS
            ("mp4v", ".mp4"),  # MPEG-4
            ("XVID", ".avi"),  # Xvid
            ("MJPG", ".avi"),  # Motion JPEG
        ]

        output_path_final = output_path
        for codec, ext in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_path = output_path.with_suffix(ext)
                out = cv2.VideoWriter(str(test_path), fourcc, fps, (width, height))
                if out.isOpened():
                    output_path_final = test_path
                    break
                out.release()
                out = None
            except Exception:
                continue

        if out is None or not out.isOpened():
            cap.release()
            raise RuntimeError(f"无法创建输出视频: {output_path}")

        # 处理每一帧
        frame_idx = 0
        try:
            for frame_data in track.frames:
                ret, frame = cap.read()
                if not ret:
                    break

                quad = np.array(frame_data["quad"], dtype=np.float32)
                valid = frame_data.get("valid", True)

                # 擦除嘴部
                result = self.erase_frame(frame, quad, valid)
                out.write(result)

                if progress_callback:
                    progress_callback(frame_idx + 1, total_frames)

                frame_idx += 1

            # 处理剩余帧（如果有）
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_idx += 1

        finally:
            cap.release()
            out.release()

        print(f"[MouthEraser] 保存到: {output_path_final}")
        print(f"  处理帧数: {frame_idx}")

        return output_path_final


def erase_mouth_simple(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad: int = 10,
    inpaint_radius: int = 5,
) -> np.ndarray:
    """
    简化的单帧口消し函数

    Args:
        frame: BGR 图像 (H, W, 3)
        bbox: (x1, y1, x2, y2) 归一化坐标 0-1
        pad: mask 膨胀像素
        inpaint_radius: inpaint 半径

    Returns:
        擦除后的 BGR 图像
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1_px, y1_px = int(x1 * w), int(y1 * h)
    x2_px, y2_px = int(x2 * w), int(y2 * h)

    # 生成椭圆 mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = (x1_px + x2_px) // 2
    cy = (y1_px + y2_px) // 2
    rx = (x2_px - x1_px) // 2 + pad
    ry = (y2_px - y1_px) // 2 + pad

    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    # OpenCV inpaint
    result = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)

    return result
