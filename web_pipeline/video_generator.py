"""
视频生成模块 - 使用 WAN 模型生成底片视频

支持两种模式：
1. 首帧生视频 (Image-to-Video)
2. 首尾帧生视频 (First-Last Frame) - 更适合循环视频
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import requests

from .config import Config


class TaskStatus(Enum):
    """任务状态"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass
class VideoTask:
    """视频生成任务"""

    task_id: str
    status: TaskStatus
    video_url: str | None = None
    error_message: str | None = None
    usage: dict | None = None


class VideoGenerator:
    """
    使用 WAN 模型生成视频

    用法:
        generator = VideoGenerator(config)

        # 首帧生视频
        task = generator.generate_from_image(
            image_url="https://...",
            prompt="idle animation"
        )

        # 首尾帧生视频（循环）
        task = generator.generate_loop_video(
            image_url="https://...",
            prompt="subtle movement"
        )

        # 等待完成
        result = generator.wait_for_task(task.task_id)
    """

    # 默认提示词
    DEFAULT_PROMPT = (
        "anime character idle animation, subtle breathing, "
        "hair swaying gently, stationary camera, loop-friendly, "
        "no large movements, consistent lighting"
    )

    DEFAULT_NEGATIVE_PROMPT = (
        "blurry, low quality, distorted face, extra limbs, "
        "camera movement, zoom, fast motion, text, watermark"
    )

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.validate()

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
                # 关闭内容审查
                "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}',
            }
        )

    def _create_task(self, url: str, payload: dict) -> VideoTask:
        """创建异步任务"""
        try:
            resp = self.session.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            output = data.get("output", {})
            task_id = output.get("task_id", "")
            status_str = output.get("task_status", "UNKNOWN")

            try:
                status = TaskStatus(status_str)
            except ValueError:
                status = TaskStatus.UNKNOWN

            return VideoTask(
                task_id=task_id,
                status=status,
                error_message=data.get("message"),
            )

        except requests.RequestException as e:
            return VideoTask(
                task_id="",
                status=TaskStatus.FAILED,
                error_message=str(e),
            )

    def _query_task(self, task_id: str) -> VideoTask:
        """查询任务状态"""
        url = self.config.task_query_url(task_id)

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            output = data.get("output", {})
            status_str = output.get("task_status", "UNKNOWN")

            try:
                status = TaskStatus(status_str)
            except ValueError:
                status = TaskStatus.UNKNOWN

            return VideoTask(
                task_id=task_id,
                status=status,
                video_url=output.get("video_url"),
                error_message=output.get("message") or data.get("message"),
                usage=data.get("usage"),
            )

        except requests.RequestException as e:
            return VideoTask(
                task_id=task_id,
                status=TaskStatus.UNKNOWN,
                error_message=str(e),
            )

    def generate_from_image(
        self,
        image_url: str,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        resolution: Literal["480P", "720P", "1080P"] | None = None,
        duration: int | None = None,
    ) -> VideoTask:
        """
        从首帧图像生成视频 (Image-to-Video)

        Args:
            image_url: 首帧图像 URL
            prompt: 提示词
            negative_prompt: 负面提示词
            resolution: 分辨率
            duration: 时长（秒）

        Returns:
            VideoTask
        """
        payload = {
            "model": self.config.wan_i2v_model,
            "input": {
                "prompt": prompt or self.DEFAULT_PROMPT,
                "img_url": image_url,
            },
            "parameters": {
                "resolution": resolution or self.config.video_resolution,
                "duration": duration or self.config.video_duration,
                "prompt_extend": True,
                "watermark": False,
            },
        }

        if negative_prompt:
            payload["input"]["negative_prompt"] = negative_prompt

        return self._create_task(self.config.video_synthesis_url, payload)

    def generate_loop_video(
        self,
        image_url: str,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        resolution: Literal["480P", "720P", "1080P"] | None = None,
    ) -> VideoTask:
        """
        从首尾帧生成循环视频 (First-Last Frame)

        首尾帧使用同一张图，实现完美循环

        Args:
            image_url: 首帧图像 URL（也作为尾帧）
            prompt: 提示词
            negative_prompt: 负面提示词
            resolution: 分辨率

        Returns:
            VideoTask
        """
        payload = {
            "model": self.config.wan_kf2v_model,
            "input": {
                "first_frame_url": image_url,
                "last_frame_url": image_url,  # 首尾帧相同 = 完美循环
                "prompt": prompt or self.DEFAULT_PROMPT,
            },
            "parameters": {
                "resolution": resolution or self.config.video_resolution,
                "prompt_extend": True,
                "watermark": False,
            },
        }

        if negative_prompt or self.DEFAULT_NEGATIVE_PROMPT:
            payload["input"]["negative_prompt"] = (
                negative_prompt or self.DEFAULT_NEGATIVE_PROMPT
            )

        return self._create_task(self.config.image_to_video_url, payload)

    def wait_for_task(
        self,
        task_id: str,
        poll_interval: float | None = None,
        timeout: float | None = None,
        progress_callback: callable | None = None,
    ) -> VideoTask:
        """
        等待任务完成

        Args:
            task_id: 任务 ID
            poll_interval: 轮询间隔（秒）
            timeout: 超时时间（秒）
            progress_callback: 进度回调 fn(status, elapsed_time)

        Returns:
            VideoTask
        """
        poll_interval = poll_interval or self.config.poll_interval
        timeout = timeout or self.config.poll_timeout

        start_time = time.time()
        last_status = TaskStatus.UNKNOWN

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                return VideoTask(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error_message=f"任务超时 ({timeout}s)",
                )

            task = self._query_task(task_id)

            if task.status != last_status:
                last_status = task.status
                if progress_callback:
                    progress_callback(task.status, elapsed)

            if task.status == TaskStatus.SUCCEEDED:
                return task
            elif task.status == TaskStatus.FAILED:
                return task

            time.sleep(poll_interval)

    def download_video(
        self,
        video_url: str,
        output_path: str | Path,
    ) -> Path:
        """
        下载视频到本地

        Args:
            video_url: 视频 URL
            output_path: 输出路径

        Returns:
            输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        resp = requests.get(video_url, stream=True, timeout=300)
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"[VideoGenerator] 下载到: {output_path}")
        return output_path
