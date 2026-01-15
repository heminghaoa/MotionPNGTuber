# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MotionPNGTuber is a real-time lip-sync system for animated avatars using MP4 loop videos with mouth sprite overlays. It bridges PNGTuber and Live2D by enabling hair/clothing motion through video while maintaining simple sprite-based mouth animation driven by microphone input.

## Common Commands

```bash
# Install dependencies (Windows)
uv sync

# Install dependencies (macOS - requires manual steps, see README)
cp pyproject.macos.toml pyproject.toml
uv venv .venv && uv sync
# Additional manual builds required for mmcv-full, xtcocotools

# Verify installation
uv run python -c "import cv2; import torch; print('OK')"

# Main workflow GUI (one-click solution)
uv run python mouth_track_gui.py

# Multi-motion runtime GUI
uv run python multi_video_live_gui.py

# CLI: Face/mouth detection
uv run python face_track_anime_detector.py --video loop.mp4 --out mouth_track.npz

# CLI: Calibration
uv run python calibrate_mouth_track.py --video loop.mp4 --track mouth_track.npz --sprite open.png

# CLI: Mouth erasure
uv run python auto_erase_mouth.py --video loop.mp4 --track mouth_track_calibrated.npz --out loop_mouthless.mp4

# CLI: Real-time lip-sync
uv run python loop_lipsync_runtime_patched_emotion_auto.py \
  --loop-video loop_mouthless.mp4 \
  --mouth-dir mouth_dir/Char \
  --track mouth_track_calibrated.npz
```

## Architecture

### Core Module
- `lipsync_core.py` - Shared utilities used by all runtime/GUI modules:
  - `MouthTrack`: Manages mouth quad positions from NPZ files with calibration
  - `BgVideo`: Background video playback wrapper
  - Video probing, alpha compositing, perspective warping functions

### Processing Pipeline
```
Video → face_track_anime_detector.py → mouth_track.npz
                                              ↓
                                   calibrate_mouth_track.py
                                              ↓
                                   mouth_track_calibrated.npz
                                              ↓
Video + NPZ → auto_erase_mouth.py → loop_mouthless.mp4
                                              ↓
                            loop_lipsync_runtime_*.py (real-time)
```

### Key Components
- `auto_mouth_track_v2.py` - Quality-based wrapper with multi-attempt retry logic
- `auto_erase_mouth.py` - Auto-tuning mouth erasure with scoring
- `realtime_emotion_audio.py` - Audio-based emotion classification (neutral/happy/angry/sad/excited)

### GUI Applications
- `mouth_track_gui.py` - Unified one-click workflow (detect → calibrate → erase → run)
- `multi_video_live_gui.py` - Multi-motion switching with emotion auto-detection
- `mouth_erase_tuner_gui.py` - Manual mouth erasure parameter tuning
- `mouth_sprite_extractor_gui.py` - Auto-extract 5 mouth sprites from video

### Data Formats
- `mouth_track.npz` / `mouth_track_calibrated.npz`: NumPy archive with quad positions (N,4,2), validity flags, confidence scores, calibration params
- Mouth sprites: 5 PNGs (open.png, closed.png, half.png, e.png, u.png) with alpha transparency
- Session files: `.mouth_track_last_session.json`, `.multi_video_live_session.json`

## Platform Notes

- **Windows**: CUDA 11.7 with pre-built mmcv-full wheel
- **macOS (experimental)**: Apple Silicon requires manual source builds for mmcv-full and xtcocotools; use `pyproject.macos.toml`
- Dependencies: anime-face-detector, MMDetection, MMPose for face/mouth landmark detection

## Aliyun DashScope API (Web Pipeline)

**模型文档**: https://help.aliyun.com/zh/model-studio/models

### 新加坡端点
- OpenAI 兼容: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- DashScope API: `https://dashscope-intl.aliyuncs.com`

### 当前使用的模型
| 功能 | 模型名称 |
|------|----------|
| VL 视觉检测 | `qwen3-vl-plus-2025-12-19` |
| 首帧生视频 (I2V) | `wan2.1-i2v-plus` |
| 首尾帧生视频 (KF2V) | `wan2.1-kf2v-plus` |

### 关闭内容审查
```python
headers = {
    "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}'
}
```

### 环境变量
```bash
DASHSCOPE_INTL_API_KEY=your-api-key  # 新加坡端点
DASHSCOPE_API_KEY=your-api-key       # 北京端点
```
