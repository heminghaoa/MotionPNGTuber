"""
Pipeline - 完整的资产生成流水线

将角色图转换为 Web 端需要的资产：
1. base.mp4 - 循环底片视频
2. mouthless.mp4 - 无嘴底片视频
3. mouth_track.json - 嘴部位置轨迹
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from .config import Config
from .mouth_detector import MouthDetector, MouthTrack
from .mouth_eraser import MouthEraser
from .sprite_extractor import SpriteExtractor
from .video_generator import TaskStatus, VideoGenerator


class PipelineStage(Enum):
    """流水线阶段"""

    INIT = "init"
    GENERATE_VIDEO = "generate_video"
    DETECT_MOUTH = "detect_mouth"
    ERASE_MOUTH = "erase_mouth"
    EXTRACT_SPRITES = "extract_sprites"
    DONE = "done"
    FAILED = "failed"


@dataclass
class PipelineResult:
    """流水线结果"""

    stage: PipelineStage
    success: bool
    base_video_path: Path | None = None
    mouthless_video_path: Path | None = None
    mouth_track_path: Path | None = None
    mouth_assets_dir: Path | None = None
    error_message: str | None = None


@dataclass
class PipelineProgress:
    """流水线进度"""

    stage: PipelineStage
    progress: float  # 0-1
    message: str


class Pipeline:
    """
    完整的资产生成流水线

    用法:
        pipeline = Pipeline(config, output_dir="output/")

        # 从角色图生成完整资产
        result = pipeline.run_from_image(
            image_url="https://...",
            prompt="idle animation"
        )

        # 或者从已有视频处理
        result = pipeline.run_from_video(
            video_path="base.mp4"
        )
    """

    def __init__(
        self,
        config: Config | None = None,
        output_dir: str | Path = "output",
    ):
        self.config = config or Config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化各模块
        self.video_generator = VideoGenerator(self.config)
        self.mouth_detector = MouthDetector(self.config)
        self.mouth_eraser = MouthEraser(self.config)
        self.sprite_extractor = SpriteExtractor(config=self.config)

        # 进度回调
        self._progress_callback: Callable[[PipelineProgress], None] | None = None

    def set_progress_callback(
        self, callback: Callable[[PipelineProgress], None]
    ) -> None:
        """设置进度回调"""
        self._progress_callback = callback

    def _report_progress(
        self, stage: PipelineStage, progress: float, message: str
    ) -> None:
        """报告进度"""
        if self._progress_callback:
            self._progress_callback(
                PipelineProgress(stage=stage, progress=progress, message=message)
            )

    def run_from_image(
        self,
        image_url: str,
        prompt: str | None = None,
        use_loop_mode: bool = True,
    ) -> PipelineResult:
        """
        从角色图运行完整流水线

        Args:
            image_url: 角色图 URL
            prompt: 视频生成提示词
            use_loop_mode: 是否使用循环模式（首尾帧相同）

        Returns:
            PipelineResult
        """
        self._report_progress(PipelineStage.INIT, 0, "初始化...")

        # Step 1: 生成视频
        self._report_progress(PipelineStage.GENERATE_VIDEO, 0, "开始生成视频...")

        if use_loop_mode:
            task = self.video_generator.generate_loop_video(
                image_url=image_url,
                prompt=prompt,
            )
        else:
            task = self.video_generator.generate_from_image(
                image_url=image_url,
                prompt=prompt,
            )

        if not task.task_id:
            return PipelineResult(
                stage=PipelineStage.GENERATE_VIDEO,
                success=False,
                error_message=task.error_message or "创建视频任务失败",
            )

        # 等待视频生成完成
        def video_progress(status: TaskStatus, elapsed: float) -> None:
            progress = min(0.9, elapsed / 300)  # 假设最长5分钟
            self._report_progress(
                PipelineStage.GENERATE_VIDEO,
                progress,
                f"生成视频中... ({status.value}, {elapsed:.0f}s)",
            )

        result = self.video_generator.wait_for_task(
            task.task_id,
            progress_callback=video_progress,
        )

        if result.status != TaskStatus.SUCCEEDED or not result.video_url:
            return PipelineResult(
                stage=PipelineStage.GENERATE_VIDEO,
                success=False,
                error_message=result.error_message or "视频生成失败",
            )

        # 下载视频
        base_video_path = self.output_dir / "base.mp4"
        self.video_generator.download_video(result.video_url, base_video_path)

        self._report_progress(PipelineStage.GENERATE_VIDEO, 1.0, "视频生成完成")

        # 继续处理视频
        return self._process_video(base_video_path)

    def run_from_video(self, video_path: str | Path) -> PipelineResult:
        """
        从已有视频运行流水线（跳过视频生成）

        Args:
            video_path: 输入视频路径

        Returns:
            PipelineResult
        """
        video_path = Path(video_path)
        if not video_path.exists():
            return PipelineResult(
                stage=PipelineStage.INIT,
                success=False,
                error_message=f"视频文件不存在: {video_path}",
            )

        # 复制到输出目录
        base_video_path = self.output_dir / "base.mp4"
        if video_path != base_video_path:
            shutil.copy(video_path, base_video_path)

        return self._process_video(base_video_path)

    def _process_video(self, base_video_path: Path) -> PipelineResult:
        """处理视频（检测嘴部 + 口消し）"""

        # Step 2: 检测嘴部
        self._report_progress(PipelineStage.DETECT_MOUTH, 0, "开始检测嘴部...")

        def detect_progress(current: int, total: int) -> None:
            progress = current / max(1, total)
            self._report_progress(
                PipelineStage.DETECT_MOUTH,
                progress,
                f"检测嘴部... ({current}/{total})",
            )

        try:
            # 根据配置选择检测模式
            if self.config.mouth_detect_batch_mode:
                self._report_progress(
                    PipelineStage.DETECT_MOUTH, 0,
                    "视频批量检测模式（一次 API 调用）..."
                )
                track = self.mouth_detector.detect_video_batch(
                    base_video_path,
                    fps=self.config.mouth_detect_batch_fps,
                    progress_callback=detect_progress,
                )
            else:
                track = self.mouth_detector.detect_video(
                    base_video_path,
                    progress_callback=detect_progress,
                )
        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.DETECT_MOUTH,
                success=False,
                base_video_path=base_video_path,
                error_message=f"嘴部检测失败: {e}",
            )

        # 保存轨迹
        mouth_track_path = self.output_dir / "mouth_track.json"
        track.save(mouth_track_path)

        self._report_progress(PipelineStage.DETECT_MOUTH, 1.0, "嘴部检测完成")

        # Step 3: 口消し
        self._report_progress(PipelineStage.ERASE_MOUTH, 0, "开始口消し...")

        def erase_progress(current: int, total: int) -> None:
            progress = current / max(1, total)
            self._report_progress(
                PipelineStage.ERASE_MOUTH,
                progress,
                f"口消し... ({current}/{total})",
            )

        mouthless_video_path = self.output_dir / "mouthless.mp4"

        try:
            self.mouth_eraser.erase_video(
                base_video_path,
                track,
                mouthless_video_path,
                progress_callback=erase_progress,
            )
        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.ERASE_MOUTH,
                success=False,
                base_video_path=base_video_path,
                mouth_track_path=mouth_track_path,
                error_message=f"口消し失败: {e}",
            )

        self._report_progress(PipelineStage.ERASE_MOUTH, 1.0, "口消し完成")

        # Step 4: 提取嘴 PNG
        self._report_progress(PipelineStage.EXTRACT_SPRITES, 0, "开始提取嘴 PNG...")

        def sprite_progress(current: int, total: int) -> None:
            progress = current / max(1, total)
            self._report_progress(
                PipelineStage.EXTRACT_SPRITES,
                progress,
                f"提取嘴 PNG... ({current}/{total})",
            )

        mouth_assets_dir = self.output_dir / "mouth_assets"

        try:
            self.sprite_extractor.extract(
                base_video_path,
                track,
                mouth_assets_dir,
                progress_callback=sprite_progress,
            )
        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.EXTRACT_SPRITES,
                success=False,
                base_video_path=base_video_path,
                mouthless_video_path=mouthless_video_path,
                mouth_track_path=mouth_track_path,
                error_message=f"嘴 PNG 提取失败: {e}",
            )

        self._report_progress(PipelineStage.EXTRACT_SPRITES, 1.0, "嘴 PNG 提取完成")

        # 完成
        self._report_progress(PipelineStage.DONE, 1.0, "流水线完成")

        return PipelineResult(
            stage=PipelineStage.DONE,
            success=True,
            base_video_path=base_video_path,
            mouthless_video_path=mouthless_video_path,
            mouth_track_path=mouth_track_path,
            mouth_assets_dir=mouth_assets_dir,
        )


def create_web_assets(
    image_url: str | None = None,
    video_path: str | Path | None = None,
    output_dir: str | Path = "output",
    prompt: str | None = None,
    api_key: str | None = None,
) -> PipelineResult:
    """
    便捷函数：生成 Web 端需要的资产

    Args:
        image_url: 角色图 URL（从图生成视频）
        video_path: 已有视频路径（直接处理）
        output_dir: 输出目录
        prompt: 视频生成提示词
        api_key: API Key（可选，默认从环境变量读取）

    Returns:
        PipelineResult
    """
    config = Config()
    if api_key:
        config.api_key = api_key

    pipeline = Pipeline(config, output_dir)

    if video_path:
        return pipeline.run_from_video(video_path)
    elif image_url:
        return pipeline.run_from_image(image_url, prompt)
    else:
        return PipelineResult(
            stage=PipelineStage.INIT,
            success=False,
            error_message="必须提供 image_url 或 video_path",
        )
