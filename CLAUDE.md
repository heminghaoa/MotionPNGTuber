# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MotionPNGTuber is a real-time lip-sync system for animated avatars using MP4 loop videos with mouth sprite overlays. It bridges PNGTuber and Live2D by enabling hair/clothing motion through video while maintaining simple sprite-based mouth animation driven by microphone input.

**v0.2.0**: Web 版重构，使用大模型 (Qwen-VL, WAN) 替代传统 CV 依赖 (MMPose, anime-face-detector)。

## Installation

```bash
# 安装依赖（所有平台通用）
uv sync

# 验证安装
uv run python -c "import cv2; import openai; print('OK')"

# 设置 API Key
export DASHSCOPE_INTL_API_KEY='your-api-key'
# 或创建 .env 文件
```

## Web Pipeline (推荐)

新的 `web_pipeline/` 模块使用大模型 API，无需本地 GPU：

```bash
# 完整流水线：视频 → 检测 → 消嘴 → 提取 PNG
uv run python web_pipeline/test_full_pipeline.py --video your_video.mp4

# 从图片生成（需要公共 URL）
uv run python web_pipeline/test_full_pipeline.py --image "https://example.com/character.png"

# 单独测试各模块
uv run python web_pipeline/test_sprite.py      # 测试 PNG 提取
uv run python web_pipeline/test_wan.py <url>   # 测试视频生成
```

### 输出文件
```
output/
├── base.mp4           # 原始/生成的视频
├── mouthless.mp4      # 无嘴底片视频
├── mouth_track.json   # 嘴部位置轨迹
└── mouth_assets/      # 5 种嘴型 PNG
    ├── open.png
    ├── closed.png
    ├── half.png
    ├── e.png
    └── u.png
```

### Web 播放器
```bash
cd web_player && python -m http.server 8080
# 打开 http://localhost:8080
```

## Architecture

### Web Pipeline (v0.2.0)
```
角色图/视频
    ↓
[mouth_detector.py] Qwen-VL 检测嘴部位置（批量模式：一次API调用）
    ↓
mouth_track.json
    ↓
[mouth_eraser.py] OpenCV inpaint 消除嘴部
    ↓
mouthless.mp4
    ↓
[sprite_extractor.py] 自动提取 5 种嘴型 PNG
    ↓
mouth_assets/
    ↓
[web_player] WebAudio 驱动实时口型
```

### 模块说明
| 模块 | 功能 | 技术 |
|------|------|------|
| `web_pipeline/config.py` | API 配置 | DashScope 新加坡端点 |
| `web_pipeline/mouth_detector.py` | 嘴部检测 | Qwen-VL（支持视频批量） |
| `web_pipeline/mouth_eraser.py` | 口消し | OpenCV inpaint |
| `web_pipeline/sprite_extractor.py` | 嘴 PNG 提取 | 智能选帧 + 透视变换 |
| `web_pipeline/video_generator.py` | 视频生成 | WAN (I2V/KF2V) |
| `web_pipeline/pipeline.py` | 完整流水线 | 统一接口 |
| `web_player/index.html` | Web 播放器 | WebAudio + Canvas |

### GUI Applications (桌面版)
- `mouth_track_gui.py` - 统一工作流 GUI
- `multi_video_live_gui.py` - 多动作切换 GUI
- `mouth_erase_tuner_gui.py` - 口消し参数调节
- `mouth_sprite_extractor_gui.py` - 嘴 PNG 提取 GUI

### Core Utilities
- `lipsync_core.py` - 核心工具类（MouthTrack, BgVideo）
- `realtime_emotion_audio.py` - 音频情绪分类
- `convert_npz_to_json.py` - NPZ ↔ JSON 格式转换

### Data Formats
- `mouth_track.json`: 嘴部轨迹（fps, width, height, frames[{quad, valid}]）
- `mouth_track.npz`: NumPy 格式（兼容旧版）
- Mouth sprites: 5 PNGs with alpha (open, closed, half, e, u)

## Aliyun DashScope API

**模型文档**: https://help.aliyun.com/zh/model-studio/models

### 端点
| 区域 | OpenAI 兼容 | DashScope |
|------|-------------|-----------|
| 新加坡 | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `https://dashscope-intl.aliyuncs.com` |
| 北京 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `https://dashscope.aliyuncs.com` |

### 当前使用的模型
| 功能 | 模型名称 |
|------|----------|
| VL 视觉检测 | `qwen3-vl-plus-2025-12-19` |
| 首帧生视频 (I2V) | `wan2.1-i2v-plus` |
| 首尾帧生视频 (KF2V) | `wan2.1-kf2v-plus` |

### VL 视频输入限制
| 参数 | 限制 |
|------|------|
| 时长 | 2秒 ~ 1小时 |
| 帧数 | 最多 2000 帧 |
| 文件大小 | Base64: 10MB, URL: 2GB |
| FPS | 0.1-10（默认 2.0） |

### 关闭内容审查
```python
headers = {
    "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}'
}
```

### 环境变量
```bash
DASHSCOPE_INTL_API_KEY=your-api-key  # 新加坡端点（推荐）
DASHSCOPE_API_KEY=your-api-key       # 北京端点
```

## Version History

### v0.2.0 (2025-01)
- **Web Pipeline**: 使用大模型替代传统 CV
  - Qwen-VL 替代 anime-face-detector + MMPose
  - OpenCV inpaint 替代云端 inpainting API
  - 视频批量检测模式（一次 API 调用）
- **依赖简化**: 删除 torch, mmcv-full, mmdet, mmpose（约 2GB → 10MB）
- **Web 播放器**: WebAudio 驱动的实时口型

### v0.1.0
- 原始版本，依赖 anime-face-detector + MMPose
