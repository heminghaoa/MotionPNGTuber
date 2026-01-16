"""
嘴部检测模块 - 使用 Qwen-VL 检测嘴部位置

替代原项目的 anime-face-detector + MMPose 方案
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from .config import Config


@dataclass
class MouthBBox:
    """嘴部边界框"""

    x1: float  # 归一化坐标 0-1
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    valid: bool = True

    @property
    def center(self) -> tuple[float, float]:
        """中心点"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def size(self) -> tuple[float, float]:
        """宽高"""
        return (self.x2 - self.x1, self.y2 - self.y1)

    def to_quad(self, width: int, height: int) -> np.ndarray:
        """
        转换为四边形顶点 (4, 2)
        顺序: 左上, 右上, 右下, 左下
        """
        x1_px = self.x1 * width
        y1_px = self.y1 * height
        x2_px = self.x2 * width
        y2_px = self.y2 * height

        return np.array(
            [
                [x1_px, y1_px],  # 左上
                [x2_px, y1_px],  # 右上
                [x2_px, y2_px],  # 右下
                [x1_px, y2_px],  # 左下
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "confidence": self.confidence,
            "valid": self.valid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MouthBBox":
        """从字典创建"""
        bbox = data.get("bbox", [0, 0, 0, 0])
        return cls(
            x1=bbox[0],
            y1=bbox[1],
            x2=bbox[2],
            y2=bbox[3],
            confidence=data.get("confidence", 1.0),
            valid=data.get("valid", True),
        )

    @classmethod
    def invalid(cls) -> "MouthBBox":
        """创建无效的 bbox"""
        return cls(x1=0, y1=0, x2=0, y2=0, confidence=0, valid=False)


class MouthDetector:
    """
    使用 Qwen-VL 检测嘴部位置

    用法:
        detector = MouthDetector(config)
        track = detector.detect_video("video.mp4")
        track.save("mouth_track.json")
    """

    DETECT_PROMPT = """请精确检测图中动漫角色的嘴唇/嘴巴位置。

重要说明：
- 嘴唇位于脸部下方，鼻子下方，下巴上方
- 不要检测眼睛、鼻子、额头或其他部位
- 嘴唇通常是红色或粉色的

要求：
1. 输出嘴唇区域的边界框坐标 [x1, y1, x2, y2]
2. 坐标必须归一化到 0-1 范围（相对于图像宽高的比例）
3. x1,y1 是左上角，x2,y2 是右下角
4. 只输出 JSON，不要其他文字

输出格式：
{"bbox": [x1, y1, x2, y2], "confidence": 0.95}

示例（嘴唇通常在 y=0.3-0.45 范围）：
{"bbox": [0.4, 0.35, 0.6, 0.42], "confidence": 0.95}"""

    VIDEO_DETECT_PROMPT = """请检测视频中动漫角色的嘴部（嘴唇）位置。

重要要求：
1. 检测的是角色的嘴唇/嘴巴区域，不是其他部位
2. 坐标必须是归一化的 0-1 范围（相对于视频宽高的比例）
3. 格式：[x1, y1, x2, y2] 其中 (x1,y1) 是左上角，(x2,y2) 是右下角
4. 所有坐标值都应该在 0.0 到 1.0 之间

输出 JSON 数组，每帧一个对象：
[
  {"frame": 0, "bbox": [0.4, 0.35, 0.6, 0.45]},
  {"frame": 1, "bbox": [0.4, 0.35, 0.6, 0.45]},
  ...
]

注意：确保 bbox 坐标是 0-1 范围的归一化值！"""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.validate()

        if OpenAI is None:
            raise ImportError("请安装 openai: pip install openai")

        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

        # EMA 平滑状态
        self._last_bbox: MouthBBox | None = None

    def _encode_image_base64(self, image: np.ndarray) -> str:
        """将图像编码为 base64"""
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _parse_response(self, content: str, image_width: int = 0, image_height: int = 0) -> MouthBBox:
        """解析 VL 模型返回的 JSON"""
        # 尝试提取 JSON
        json_match = re.search(r"\{[^}]+\}", content)
        if not json_match:
            return MouthBBox.invalid()

        try:
            data = json.loads(json_match.group())
            bbox = data.get("bbox", [0, 0, 0, 0])

            if len(bbox) != 4:
                return MouthBBox.invalid()

            x1, y1, x2, y2 = bbox

            # 检查是否是像素坐标（任何值 > 1）
            if any(v > 1 for v in bbox):
                # 像素坐标，需要归一化
                if image_width > 0 and image_height > 0:
                    x1 = x1 / image_width
                    y1 = y1 / image_height
                    x2 = x2 / image_width
                    y2 = y2 / image_height
                else:
                    return MouthBBox.invalid()

            # 验证坐标范围
            if not all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
                return MouthBBox.invalid()

            if x2 <= x1 or y2 <= y1:
                return MouthBBox.invalid()

            return MouthBBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=data.get("confidence", 1.0),
                valid=data.get("valid", True),
            )
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return MouthBBox.invalid()

    def _smooth_bbox(self, bbox: MouthBBox) -> MouthBBox:
        """EMA 平滑"""
        if not bbox.valid:
            # 无效时使用上一帧
            if self._last_bbox and self._last_bbox.valid:
                return self._last_bbox
            return bbox

        if self._last_bbox is None or not self._last_bbox.valid:
            self._last_bbox = bbox
            return bbox

        beta = self.config.mouth_smoothing_beta
        smoothed = MouthBBox(
            x1=self._last_bbox.x1 * (1 - beta) + bbox.x1 * beta,
            y1=self._last_bbox.y1 * (1 - beta) + bbox.y1 * beta,
            x2=self._last_bbox.x2 * (1 - beta) + bbox.x2 * beta,
            y2=self._last_bbox.y2 * (1 - beta) + bbox.y2 * beta,
            confidence=bbox.confidence,
            valid=True,
        )
        self._last_bbox = smoothed
        return smoothed

    def detect_frame(self, image: np.ndarray, smooth: bool = True) -> MouthBBox:
        """
        检测单帧图像中的嘴部位置

        Args:
            image: BGR 图像 (H, W, 3)
            smooth: 是否应用 EMA 平滑

        Returns:
            MouthBBox
        """
        h, w = image.shape[:2]
        image_url = self._encode_image_base64(image)

        try:
            response = self.client.chat.completions.create(
                model=self.config.qwen_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": self.DETECT_PROMPT},
                        ],
                    }
                ],
                max_tokens=200,
                extra_headers={
                    # 关闭内容审查
                    "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}'
                },
            )

            content = response.choices[0].message.content or ""
            bbox = self._parse_response(content, image_width=w, image_height=h)

        except Exception as e:
            print(f"[MouthDetector] 检测失败: {e}")
            bbox = MouthBBox.invalid()

        if smooth:
            bbox = self._smooth_bbox(bbox)

        return bbox

    def _parse_video_response(
        self, content: str, video_width: int, video_height: int
    ) -> list[MouthBBox]:
        """解析视频批量检测的响应"""
        # 尝试提取 JSON 数组
        array_match = re.search(r"\[[\s\S]*\]", content)
        if not array_match:
            print(f"[MouthDetector] 无法解析视频响应: {content[:200]}...")
            return []

        try:
            data_list = json.loads(array_match.group())
            print(f"[MouthDetector] 解析到 {len(data_list)} 个数据项")
            if data_list:
                print(f"[MouthDetector] 第一个数据项: {data_list[0]}")
            results = []

            for item in data_list:
                bbox = item.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    results.append(MouthBBox.invalid())
                    continue

                x1, y1, x2, y2 = bbox

                # 检查是否是像素坐标
                if any(v > 1 for v in bbox):
                    if video_width > 0 and video_height > 0:
                        x1 = x1 / video_width
                        y1 = y1 / video_height
                        x2 = x2 / video_width
                        y2 = y2 / video_height
                    else:
                        results.append(MouthBBox.invalid())
                        continue

                # 验证坐标范围
                if not all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
                    results.append(MouthBBox.invalid())
                    continue

                if x2 <= x1 or y2 <= y1:
                    results.append(MouthBBox.invalid())
                    continue

                results.append(MouthBBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=item.get("confidence", 1.0),
                    valid=item.get("valid", True),
                ))

            return results

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[MouthDetector] JSON 解析失败: {e}")
            return []

    def _compress_video(
        self, video_path: Path, max_size: int, scale: float = 0.5
    ) -> Path | None:
        """压缩视频到指定大小以下"""
        try:
            # 创建临时文件
            temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
            temp_path = Path(temp_path)

            # 使用 ffmpeg 压缩
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"scale=iw*{scale}:ih*{scale}",
                "-c:v", "libx264", "-crf", "28",
                "-preset", "fast",
                "-an",  # 去除音频
                str(temp_path)
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )

            if result.returncode == 0 and temp_path.exists():
                return temp_path
            else:
                print(f"[MouthDetector] ffmpeg 错误: {result.stderr[:200]}")
                if temp_path.exists():
                    temp_path.unlink()
                return None

        except FileNotFoundError:
            print("[MouthDetector] ffmpeg 未安装，无法压缩视频")
            return None
        except subprocess.TimeoutExpired:
            print("[MouthDetector] ffmpeg 超时")
            return None
        except Exception as e:
            print(f"[MouthDetector] 压缩失败: {e}")
            return None

    def detect_video_batch(
        self,
        video_path: str | Path,
        fps: float = 2.0,
        progress_callback: callable | None = None,
    ) -> "MouthTrack":
        """
        使用视频输入模式批量检测嘴部位置（一次 API 调用）

        Args:
            video_path: 视频路径
            fps: 视频采样帧率 (0.1-10)，默认 2.0
            progress_callback: 进度回调 fn(current, total)

        Returns:
            MouthTrack
        """
        video_path = Path(video_path)
        original_video_path = video_path  # 保存原始路径用于回退

        # 获取视频信息
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 读取视频文件
        file_size = video_path.stat().st_size
        max_size = 10 * 1024 * 1024  # 10MB

        # 如果文件过大，尝试压缩
        temp_video_path = None
        compressed_scale = 1.0  # 压缩比例
        if file_size > max_size:
            print(f"[MouthDetector] 视频文件 {file_size/1024/1024:.1f}MB，尝试压缩...")
            compressed_scale = 0.5
            temp_video_path = self._compress_video(video_path, max_size, scale=compressed_scale)
            if temp_video_path:
                video_path = temp_video_path
                file_size = video_path.stat().st_size
                print(f"[MouthDetector] 压缩后 {file_size/1024/1024:.1f}MB")
            else:
                print(f"[MouthDetector] 压缩失败，回退到逐帧检测")
                return self.detect_video(video_path, progress_callback=progress_callback)

        # 仍然过大则回退
        if file_size > max_size:
            print(f"[MouthDetector] 压缩后仍过大，回退到逐帧检测")
            if temp_video_path and temp_video_path.exists():
                temp_video_path.unlink()
            return self.detect_video(video_path, progress_callback=progress_callback)

        # 编码为 base64
        with open(video_path, "rb") as f:
            video_data = f.read()

        video_b64 = base64.b64encode(video_data).decode("utf-8")
        video_url = f"data:video/mp4;base64,{video_b64}"

        print(f"[MouthDetector] 视频批量检测模式: {total_frames} 帧, FPS={fps}")

        if progress_callback:
            progress_callback(0, 1)

        try:
            response = self.client.chat.completions.create(
                model=self.config.qwen_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": video_url, "fps": fps},
                            },
                            {"type": "text", "text": self.VIDEO_DETECT_PROMPT},
                        ],
                    }
                ],
                max_tokens=4096,
                extra_headers={
                    "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}'
                },
            )

            content = response.choices[0].message.content or ""
            print(f"[MouthDetector] API 响应长度: {len(content)}")
            print(f"[MouthDetector] API 响应内容: {content[:500]}...")

            # 注意：VL 模型可能按原始尺寸返回坐标，而不是压缩后的尺寸
            # 尝试使用原始尺寸解析
            vl_width = width
            vl_height = height
            print(f"[MouthDetector] 使用尺寸: {vl_width}x{vl_height} (原始尺寸)")
            bboxes = self._parse_video_response(content, vl_width, vl_height)
            print(f"[MouthDetector] 检测到 {len(bboxes)} 个帧的嘴部位置")

        except Exception as e:
            print(f"[MouthDetector] 视频批量检测失败: {e}")
            print("[MouthDetector] 回退到逐帧检测模式")
            # 清理临时文件
            if temp_video_path and temp_video_path.exists():
                temp_video_path.unlink()
            return self.detect_video(original_video_path, progress_callback=progress_callback)
        finally:
            # 清理临时文件
            if temp_video_path and temp_video_path.exists():
                try:
                    temp_video_path.unlink()
                except Exception:
                    pass

        if progress_callback:
            progress_callback(1, 1)

        # 将检测结果插值到所有帧
        frames = self._interpolate_frames(bboxes, total_frames, video_fps, fps, width, height)

        return MouthTrack(
            fps=video_fps,
            width=width,
            height=height,
            frames=frames,
        )

    def _interpolate_frames(
        self,
        bboxes: list[MouthBBox],
        total_frames: int,
        video_fps: float,
        sample_fps: float,
        width: int,
        height: int,
    ) -> list[dict]:
        """将采样帧的检测结果线性插值到所有帧"""
        if not bboxes:
            # 没有检测结果，返回全无效
            return [
                {"frame": i, "quad": [[0, 0], [0, 0], [0, 0], [0, 0]], "valid": False, "confidence": 0}
                for i in range(total_frames)
            ]

        # 计算采样间隔
        sample_interval = video_fps / sample_fps

        frames = []
        for frame_idx in range(total_frames):
            # 计算在采样帧序列中的浮点位置
            sample_pos = frame_idx / sample_interval

            # 找到前后两个采样帧
            idx_before = int(sample_pos)
            idx_after = min(idx_before + 1, len(bboxes) - 1)
            idx_before = min(idx_before, len(bboxes) - 1)

            bbox_before = bboxes[idx_before]
            bbox_after = bboxes[idx_after]

            # 如果前后帧相同或有一个无效，直接使用
            if idx_before == idx_after or not bbox_before.valid or not bbox_after.valid:
                bbox = bbox_before if bbox_before.valid else bbox_after
                frames.append({
                    "frame": frame_idx,
                    "quad": bbox.to_quad(width, height).tolist(),
                    "valid": bbox.valid,
                    "confidence": bbox.confidence,
                })
                continue

            # 线性插值
            t = sample_pos - idx_before  # 0-1 之间的插值因子
            interpolated_bbox = MouthBBox(
                x1=bbox_before.x1 * (1 - t) + bbox_after.x1 * t,
                y1=bbox_before.y1 * (1 - t) + bbox_after.y1 * t,
                x2=bbox_before.x2 * (1 - t) + bbox_after.x2 * t,
                y2=bbox_before.y2 * (1 - t) + bbox_after.y2 * t,
                confidence=(bbox_before.confidence + bbox_after.confidence) / 2,
                valid=True,
            )

            frames.append({
                "frame": frame_idx,
                "quad": interpolated_bbox.to_quad(width, height).tolist(),
                "valid": True,
                "confidence": interpolated_bbox.confidence,
            })

        return frames

    def detect_video(
        self,
        video_path: str | Path,
        sample_rate: int | None = None,
        progress_callback: callable | None = None,
        max_workers: int = 3,
    ) -> "MouthTrack":
        """
        检测视频中所有帧的嘴部位置（并发 API 调用）

        Args:
            video_path: 视频路径
            sample_rate: 采样率（1=每帧, 5=每5帧）
            progress_callback: 进度回调 fn(current, total)
            max_workers: 并发线程数

        Returns:
            MouthTrack
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        video_path = Path(video_path)
        sample_rate = sample_rate or self.config.mouth_detect_sample_rate

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 1. 收集需要检测的帧
        sample_frames: list[tuple[int, np.ndarray]] = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    sample_frames.append((frame_idx, frame.copy()))

                frame_idx += 1
        finally:
            cap.release()

        print(f"[MouthDetector] 并发检测 {len(sample_frames)} 帧，线程数={max_workers}")

        # 2. 并发检测
        results: dict[int, MouthBBox] = {}
        completed = 0

        def detect_single(args):
            idx, img = args
            return idx, self.detect_frame(img, smooth=False)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(detect_single, sf): sf[0] for sf in sample_frames}

            for future in as_completed(futures):
                idx, bbox = future.result()
                results[idx] = bbox
                completed += 1

                if progress_callback:
                    progress_callback(completed, len(sample_frames))

        # 3. 按顺序整理结果，应用平滑
        self._last_bbox = None
        sorted_indices = sorted(results.keys())
        smoothed_results: dict[int, MouthBBox] = {}

        for idx in sorted_indices:
            bbox = self._smooth_bbox(results[idx])
            smoothed_results[idx] = bbox

        # 4. 插值到所有帧
        frames: list[dict] = []
        last_bbox: MouthBBox | None = None

        for i in range(total_frames):
            if i in smoothed_results:
                last_bbox = smoothed_results[i]

            bbox = last_bbox or MouthBBox.invalid()
            frames.append({
                "frame": i,
                "quad": bbox.to_quad(width, height).tolist(),
                "valid": bbox.valid,
                "confidence": bbox.confidence,
            })

        return MouthTrack(
            fps=fps,
            width=width,
            height=height,
            frames=frames,
        )


@dataclass
class MouthTrack:
    """嘴部轨迹数据"""

    fps: float
    width: int
    height: int
    frames: list[dict]
    ref_sprite_size: tuple[int, int] = (128, 64)
    calibration: dict | None = None

    def save(self, path: str | Path) -> None:
        """保存为 JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "refSpriteSize": list(self.ref_sprite_size),
            "calibration": self.calibration
            or {
                "offset": [0, 0],
                "scale": 1.0,
                "rotation": 0.0,
            },
            "calibrationApplied": False,
            "frames": [
                {
                    "quad": f["quad"],
                    "valid": f["valid"],
                }
                for f in self.frames
            ],
        }

        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

        print(f"[MouthTrack] 保存到: {path}")
        print(f"  帧数: {len(self.frames)}")
        print(f"  FPS: {self.fps}")
        print(f"  尺寸: {self.width}x{self.height}")

    @classmethod
    def load(cls, path: str | Path) -> "MouthTrack":
        """从 JSON 加载"""
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        return cls(
            fps=data["fps"],
            width=data["width"],
            height=data["height"],
            frames=data["frames"],
            ref_sprite_size=tuple(data.get("refSpriteSize", [128, 64])),
            calibration=data.get("calibration"),
        )

    def iter_quads(self) -> Iterator[tuple[int, np.ndarray, bool]]:
        """迭代所有帧的 quad"""
        for f in self.frames:
            yield f.get("frame", 0), np.array(f["quad"], dtype=np.float32), f["valid"]
