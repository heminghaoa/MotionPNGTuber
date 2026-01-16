"""
web_pipeline - MotionPNGTuber Web 版资产生成流水线

使用大模型（Qwen-VL、WAN）替代传统 CV 依赖，生成 Web 端需要的资产：
- base.mp4: 底片视频（循环）
- mouth_track.json: 嘴部位置轨迹
- mouth_assets/: 嘴 PNG

API 端点（新加坡）：
- Qwen-VL: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
- WAN: https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/...
"""

from .config import Config
from .mouth_detector import MouthDetector, MouthTrack
from .mouth_eraser import MouthEraser
from .video_generator import VideoGenerator
from .sprite_extractor import SpriteExtractor
from .pipeline import Pipeline

__all__ = [
    "Config",
    "MouthDetector",
    "MouthTrack",
    "MouthEraser",
    "VideoGenerator",
    "SpriteExtractor",
    "Pipeline",
]
