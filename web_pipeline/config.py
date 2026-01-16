"""
配置模块 - API 端点和参数配置
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _load_env_file() -> None:
    """加载 .env 文件"""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value


def _get_api_key() -> str:
    """获取 API Key（支持多个环境变量名）"""
    _load_env_file()
    # 优先使用 INTL 版本（新加坡端点）
    return (
        os.getenv("DASHSCOPE_INTL_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or ""
    )


@dataclass
class Config:
    """API 配置"""

    # API Key（从环境变量读取，支持 .env 文件）
    api_key: str = field(default_factory=_get_api_key)

    # 区域端点
    region: Literal["singapore", "beijing", "virginia"] = "singapore"

    # Qwen-VL 配置
    qwen_model: str = "qwen3-vl-plus-2025-12-19"

    # WAN 视频生成配置
    wan_i2v_model: str = "wan2.1-i2v-plus"  # 首帧生视频
    wan_kf2v_model: str = "wan2.1-kf2v-plus"  # 首尾帧生视频

    # 视频参数
    video_resolution: Literal["480P", "720P", "1080P"] = "720P"
    video_duration: int = 5  # 秒

    # 嘴部检测参数
    mouth_detect_batch_mode: bool = False  # 使用视频批量检测模式（一次API调用）
    mouth_detect_batch_fps: float = 2.0  # 批量模式采样帧率 (0.1-10)
    mouth_detect_sample_rate: int = 1  # 逐帧模式采样率: 1=每帧检测
    mouth_smoothing_beta: float = 1.0  # EMA 平滑系数 (1.0=不平滑，直接使用检测值)

    # 口消し参数
    erase_pad: int = 30  # mask 膨胀像素（增大以完全覆盖嘴部）
    erase_inpaint_radius: int = 7  # inpaint 半径
    erase_blend: bool = True  # 边缘融合

    # 轮询配置
    poll_interval: float = 15.0  # 秒
    poll_timeout: float = 600.0  # 超时（秒）

    @property
    def base_url(self) -> str:
        """OpenAI 兼容 API 的 base_url（用于 Qwen-VL）"""
        urls = {
            "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "virginia": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        }
        return urls[self.region]

    @property
    def dashscope_base_url(self) -> str:
        """DashScope API base URL（用于 WAN）"""
        urls = {
            "singapore": "https://dashscope-intl.aliyuncs.com",
            "beijing": "https://dashscope.aliyuncs.com",
            "virginia": "https://dashscope-us.aliyuncs.com",
        }
        return urls[self.region]

    @property
    def video_synthesis_url(self) -> str:
        """WAN 视频生成端点"""
        return f"{self.dashscope_base_url}/api/v1/services/aigc/video-generation/video-synthesis"

    @property
    def image_to_video_url(self) -> str:
        """WAN 首尾帧生视频端点"""
        return f"{self.dashscope_base_url}/api/v1/services/aigc/image2video/video-synthesis"

    def task_query_url(self, task_id: str) -> str:
        """任务查询端点"""
        return f"{self.dashscope_base_url}/api/v1/tasks/{task_id}"

    def validate(self) -> None:
        """验证配置"""
        if not self.api_key:
            raise ValueError(
                "API Key 未设置。请设置环境变量或 .env 文件：\n"
                "  export DASHSCOPE_INTL_API_KEY='your-api-key'  # 新加坡端点\n"
                "  或 export DASHSCOPE_API_KEY='your-api-key'"
            )
