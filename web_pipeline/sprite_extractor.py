"""
嘴 PNG 提取模块 - 从视频中自动提取 5 种嘴型 PNG

输出文件：
- open.png: 口开
- closed.png: 口闭
- half.png: 半开
- e.png: 横长（え）
- u.png: 嘟嘴（う）

选帧流程：
1. 按 height/width/aspect_ratio 生成 20 个候补帧
2. 拼成一张图发给 VL 模型
3. VL 判断哪个是哪种嘴型
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from .config import Config
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

    # VL 选择 prompt
    VL_SELECT_PROMPT = """这是从视频中提取的 20 个嘴部候补图片，排列成 4 行 5 列的网格。
每个图片下方标注了编号 (0-19)。

请从这 20 个候补中选择 5 种嘴型：
1. open (张嘴): 嘴巴张开最大的，通常能看到牙齿和口腔内部
2. closed (闭嘴): 嘴巴完全闭合的，只能看到嘴唇线
3. half (半开): 嘴巴微微张开的，介于张嘴和闭嘴之间
4. e (横长): 嘴巴横向张开的，像发"え"音时的嘴型
5. u (嘟嘴): 嘴巴收窄/嘟起的，像发"う"音时的嘴型

要求：
1. 每种嘴型选择一个最合适的候补编号
2. 5 种嘴型必须选择不同的编号（不能重复）
3. 只输出 JSON，不要其他文字

输出格式：
{"open": 编号, "closed": 编号, "half": 编号, "e": 编号, "u": 编号}

示例：
{"open": 3, "closed": 15, "half": 8, "e": 5, "u": 12}"""

    def __init__(
        self,
        config: Optional[Config] = None,
        feather_px: int = 15,
        mask_scale: float = 0.85,
        padding: float = 1.2,
    ):
        """
        Args:
            config: API 配置（用于 VL 选择）
            feather_px: 羽化像素
            mask_scale: 椭圆 mask 缩放
            padding: 输出尺寸 padding
        """
        self.config = config or Config()
        self.feather_px = feather_px
        self.mask_scale = mask_scale
        self.padding = padding

        # 初始化 VL client
        self._client: Optional["OpenAI"] = None

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

    def _get_client(self) -> "OpenAI":
        """获取 OpenAI client"""
        if self._client is None:
            if OpenAI is None:
                raise ImportError("请安装 openai: pip install openai")
            self.config.validate()
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        return self._client

    def _generate_candidates(
        self,
        frames: List[MouthFrameInfo],
        count: int = 20,
    ) -> List[MouthFrameInfo]:
        """
        生成候补帧（像原项目那样按 height/width/aspect_ratio 选择）

        选择策略：
        - 4 枚 height 最大 (open 候补)
        - 4 枚 height 最小 (closed 候补)
        - 4 枚 height 接近中位数 (half 候补)
        - 4 枚 aspect_ratio 最大 (e 候补 - 横长)
        - 4 枚 width 最小 (u 候补 - 窄嘴)
        """
        valid_frames = [f for f in frames if f.valid and f.width > 0 and f.height > 0]

        if len(valid_frames) < count:
            print(f"[SpriteExtractor] 有效帧不足 {count} 个，使用全部 {len(valid_frames)} 个")
            return valid_frames

        # 过滤异常尺寸
        widths = np.array([f.width for f in valid_frames])
        heights = np.array([f.height for f in valid_frames])

        w_q1, w_q3 = np.percentile(widths, [25, 75])
        h_q1, h_q3 = np.percentile(heights, [25, 75])
        w_iqr, h_iqr = w_q3 - w_q1, h_q3 - h_q1

        filtered = [
            f for f in valid_frames
            if (w_q1 - 1.5 * w_iqr) <= f.width <= (w_q3 + 1.5 * w_iqr)
            and (h_q1 - 1.5 * h_iqr) <= f.height <= (h_q3 + 1.5 * h_iqr)
        ]

        if len(filtered) < count:
            filtered = valid_frames

        # 重新计算指标
        heights = np.array([f.height for f in filtered])
        widths = np.array([f.width for f in filtered])
        aspects = widths / np.maximum(heights, 1e-6)

        selected_indices: set[int] = set()
        candidates: List[MouthFrameInfo] = []

        def pick_by_score(scores: np.ndarray, n: int, maximize: bool = True):
            """按分数选择 n 个不重复的帧"""
            order = np.argsort(scores)
            if maximize:
                order = order[::-1]

            picked = 0
            for idx in order:
                if idx not in selected_indices and picked < n:
                    selected_indices.add(idx)
                    candidates.append(filtered[idx])
                    picked += 1
                if picked >= n:
                    break

        # 每类选 4 个
        per_category = count // 5

        # open 候补: height 最大
        pick_by_score(heights, per_category, maximize=True)

        # closed 候补: height 最小
        pick_by_score(heights, per_category, maximize=False)

        # half 候补: height 接近中位数
        median_h = np.median(heights)
        half_scores = -np.abs(heights - median_h)
        pick_by_score(half_scores, per_category, maximize=True)

        # e 候补: aspect_ratio 最大 (横长)
        pick_by_score(aspects, per_category, maximize=True)

        # u 候补: width 最小 (窄嘴)
        pick_by_score(widths, per_category, maximize=False)

        print(f"[SpriteExtractor] 生成了 {len(candidates)} 个候补帧")
        return candidates

    def _create_candidate_grid(
        self,
        video_path: Path,
        candidates: List[MouthFrameInfo],
        thumb_size: int = 100,
        cols: int = 5,
    ) -> np.ndarray:
        """
        创建候补帧的网格图（用于发送给 VL）

        Returns:
            BGR 图像
        """
        rows = (len(candidates) + cols - 1) // cols
        grid_w = cols * thumb_size
        grid_h = rows * (thumb_size + 20)  # +20 for label

        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240  # 浅灰背景

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        try:
            for i, mf in enumerate(candidates):
                row = i // cols
                col = i % cols
                x = col * thumb_size
                y = row * (thumb_size + 20)

                # 读取帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, mf.frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # 裁剪嘴部区域
                quad = mf.quad.astype(np.float32)
                x1, y1 = int(quad[0][0]), int(quad[0][1])
                x2, y2 = int(quad[2][0]), int(quad[2][1])

                # 边界检查
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    roi = frame[y1:y2, x1:x2]

                    # 缩放到 thumb_size
                    roi_h, roi_w = roi.shape[:2]
                    scale = min(thumb_size / roi_w, thumb_size / roi_h) * 0.9
                    new_w = max(1, int(roi_w * scale))
                    new_h = max(1, int(roi_h * scale))
                    roi_resized = cv2.resize(roi, (new_w, new_h))

                    # 居中放置
                    offset_x = (thumb_size - new_w) // 2
                    offset_y = (thumb_size - new_h) // 2
                    grid[y + offset_y:y + offset_y + new_h,
                         x + offset_x:x + offset_x + new_w] = roi_resized

                # 添加编号标签
                label_y = y + thumb_size + 15
                cv2.putText(
                    grid, str(i), (x + thumb_size // 2 - 10, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                )

        finally:
            cap.release()

        return grid

    def _vl_select_from_candidates(
        self,
        grid_image: np.ndarray,
        num_candidates: int,
    ) -> Dict[str, int]:
        """
        调用 VL 模型从候补中选择 5 种嘴型

        Returns:
            {"open": idx, "closed": idx, "half": idx, "e": idx, "u": idx}
        """
        client = self._get_client()

        # 编码图像
        _, buffer = cv2.imencode(".jpg", grid_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64 = base64.b64encode(buffer).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64}"

        print("[SpriteExtractor] 调用 VL 模型选择嘴型...")

        try:
            response = client.chat.completions.create(
                model=self.config.qwen_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": self.VL_SELECT_PROMPT},
                        ],
                    }
                ],
                max_tokens=200,
                extra_headers={
                    "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}'
                },
            )

            content = response.choices[0].message.content or ""
            print(f"[SpriteExtractor] VL 响应: {content}")

            # 解析 JSON
            json_match = re.search(r"\{[^}]+\}", content)
            if not json_match:
                raise ValueError(f"无法解析 VL 响应: {content}")

            result = json.loads(json_match.group())

            # 验证结果
            required_keys = ["open", "closed", "half", "e", "u"]
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"缺少 {key} 的选择")
                idx = result[key]
                if not isinstance(idx, int) or idx < 0 or idx >= num_candidates:
                    raise ValueError(f"{key} 的索引无效: {idx}")

            # 检查是否有重复
            values = list(result.values())
            if len(values) != len(set(values)):
                print("[SpriteExtractor] 警告: VL 选择有重复，尝试去重...")
                # 去重：保留第一个出现的
                used = set()
                for key in required_keys:
                    if result[key] in used:
                        # 找一个未使用的
                        for i in range(num_candidates):
                            if i not in used:
                                result[key] = i
                                break
                    used.add(result[key])

            return result

        except Exception as e:
            print(f"[SpriteExtractor] VL 选择失败: {e}")
            raise

    def _select_5_types_with_vl(
        self,
        frames: List[MouthFrameInfo],
        video_path: Path,
    ) -> SpriteSelection:
        """
        使用 VL 模型选择 5 种嘴型（推荐）
        """
        # 1. 生成候补
        candidates = self._generate_candidates(frames, count=20)

        if len(candidates) < 5:
            raise ValueError(f"候补帧不足 5 个 ({len(candidates)})")

        # 2. 创建候补网格图
        grid = self._create_candidate_grid(video_path, candidates)

        # 3. 调用 VL 选择
        selection = self._vl_select_from_candidates(grid, len(candidates))

        # 4. 转换为 SpriteSelection（候补索引 -> 原始帧索引）
        return SpriteSelection(
            open_idx=candidates[selection["open"]].frame_idx,
            closed_idx=candidates[selection["closed"]].frame_idx,
            half_idx=candidates[selection["half"]].frame_idx,
            e_idx=candidates[selection["e"]].frame_idx,
            u_idx=candidates[selection["u"]].frame_idx,
        )

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
        valid_frames = [f for f in frames if f.valid and f.width > 0 and f.height > 0]

        if len(valid_frames) < 5:
            raise ValueError(f"有效帧不足 5 个 ({len(valid_frames)})")

        # 过滤异常尺寸的帧（使用 IQR 方法）
        widths = np.array([f.width for f in valid_frames])
        heights = np.array([f.height for f in valid_frames])

        # 计算 IQR
        w_q1, w_q3 = np.percentile(widths, [25, 75])
        h_q1, h_q3 = np.percentile(heights, [25, 75])
        w_iqr = w_q3 - w_q1
        h_iqr = h_q3 - h_q1

        # 过滤掉异常值（超出 1.5 * IQR）
        w_lower, w_upper = w_q1 - 1.5 * w_iqr, w_q3 + 1.5 * w_iqr
        h_lower, h_upper = h_q1 - 1.5 * h_iqr, h_q3 + 1.5 * h_iqr

        filtered_frames = [
            f for f in valid_frames
            if w_lower <= f.width <= w_upper and h_lower <= f.height <= h_upper
        ]

        if len(filtered_frames) < 5:
            print(f"[SpriteExtractor] 过滤异常尺寸后帧数不足，使用原始有效帧")
            filtered_frames = valid_frames
        else:
            filtered_count = len(valid_frames) - len(filtered_frames)
            if filtered_count > 0:
                print(f"[SpriteExtractor] 过滤了 {filtered_count} 个异常尺寸的帧")

        valid_frames = filtered_frames

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
        """
        计算统一的输出尺寸

        使用所有有效帧的中位数尺寸（而非最大值），以过滤异常检测结果
        """
        # 收集所有有效帧的尺寸
        valid_frames = [f for f in frames if f.valid and f.width > 0 and f.height > 0]

        if not valid_frames:
            # 回退：使用选择帧的尺寸
            indices = list(selection.as_dict().values())
            idx_to_frame = {f.frame_idx: f for f in frames}
            max_w, max_h = 0.0, 0.0
            for idx in indices:
                if idx in idx_to_frame:
                    f = idx_to_frame[idx]
                    max_w = max(max_w, f.width)
                    max_h = max(max_h, f.height)
            w = int(max_w * self.padding) if max_w > 0 else 100
            h = int(max_h * self.padding) if max_h > 0 else 100
        else:
            # 使用中位数 + 一个标准差作为尺寸，过滤极端异常值
            widths = np.array([f.width for f in valid_frames])
            heights = np.array([f.height for f in valid_frames])

            # 使用 75th percentile（比中位数稍大，但过滤极端值）
            w = int(np.percentile(widths, 75) * self.padding)
            h = int(np.percentile(heights, 75) * self.padding)

            print(f"[SpriteExtractor] 尺寸统计: width median={np.median(widths):.1f}, 75th={np.percentile(widths, 75):.1f}, max={np.max(widths):.1f}")
            print(f"[SpriteExtractor] 尺寸统计: height median={np.median(heights):.1f}, 75th={np.percentile(heights, 75):.1f}, max={np.max(heights):.1f}")

        # 确保偶数且最小值
        w = max(2, w + (w % 2))
        h = max(2, h + (h % 2))

        return w, h

    def extract(
        self,
        video_path: str | Path,
        track: MouthTrack,
        output_dir: str | Path,
        progress_callback: callable | None = None,
        use_vl: bool = True,
    ) -> Dict[str, Path]:
        """
        提取 5 种嘴型 PNG

        Args:
            video_path: 视频路径
            track: 嘴部轨迹
            output_dir: 输出目录
            progress_callback: 进度回调
            use_vl: 是否使用 VL 模型选择嘴型（推荐）

        Returns:
            {"open": path, "closed": path, ...}
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 分析帧（不需要图像内容分析，如果用 VL 选择）
        frames = self._analyze_frames(track, video_path if not use_vl else None)

        # 选择 5 种嘴型
        if use_vl:
            try:
                selection = self._select_5_types_with_vl(frames, video_path)
            except Exception as e:
                print(f"[SpriteExtractor] VL 选择失败，回退到传统方法: {e}")
                frames = self._analyze_frames(track, video_path)  # 需要 openness
                selection = self._select_5_types(frames)
        else:
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
