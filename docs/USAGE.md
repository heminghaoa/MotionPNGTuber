# MotionPNGTuber 使用文档

## 简介

MotionPNGTuber 是一个实时口型同步系统，让你的 2D 角色动起来。

**特点：**
- 使用大模型 API（无需本地 GPU）
- 自动检测嘴部位置
- 自动消除原始嘴部
- 自动提取 5 种嘴型 PNG
- Web 播放器实时口型同步

## 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/heminghaoa/MotionPNGTuber.git
cd MotionPNGTuber

# 安装依赖
pip install uv
uv sync
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```bash
DASHSCOPE_INTL_API_KEY=你的API密钥
```

> 获取 API Key: https://dashscope.console.aliyun.com/

### 3. 运行

```bash
# 处理你的角色视频
uv run python web_pipeline/test_full_pipeline.py --video 你的视频.mp4 --output output/

# 启动 Web 播放器
cd web_player && python -m http.server 8080
```

打开浏览器访问 http://localhost:8080

---

## 详细使用

### 方式一：从视频生成资产

如果你已经有一个角色循环视频（如呼吸动画）：

```bash
uv run python web_pipeline/test_full_pipeline.py \
    --video your_character_loop.mp4 \
    --output output/my_character
```

**输出文件：**
```
output/my_character/
├── base.mp4           # 原始视频（复制）
├── mouthless.mp4      # 无嘴视频（用于播放）
├── mouth_track.json   # 嘴部位置数据
└── mouth_assets/      # 5 种嘴型
    ├── open.png       # 张嘴
    ├── closed.png     # 闭嘴
    ├── half.png       # 半开
    ├── e.png          # え型（横长）
    └── u.png          # う型（嘟嘴）
```

### 方式二：从图片生成视频

如果你只有一张角色图片，可以用 AI 生成循环动画：

```bash
# 需要先上传图片到公共 URL（如 OSS）
uv run python web_pipeline/test_full_pipeline.py \
    --image "https://your-bucket.oss.com/character.png" \
    --output output/my_character
```

> 注意：图片必须是公共可访问的 URL

### 方式三：单独使用各模块

#### 只检测嘴部位置

```python
from web_pipeline import MouthDetector, MouthTrack

detector = MouthDetector()
track = detector.detect_video_batch("video.mp4", fps=2.0)
track.save("mouth_track.json")
```

#### 只消除嘴部

```python
from web_pipeline import MouthEraser, MouthTrack

track = MouthTrack.load("mouth_track.json")
eraser = MouthEraser()
eraser.erase_video("video.mp4", track, "mouthless.mp4")
```

#### 只提取嘴型 PNG

```python
from web_pipeline import SpriteExtractor, MouthTrack

track = MouthTrack.load("mouth_track.json")
extractor = SpriteExtractor()
extractor.extract("video.mp4", track, "mouth_assets/")
```

---

## Web 播放器

### 启动

```bash
cd web_player
python -m http.server 8080
```

### 使用

1. 打开 http://localhost:8080
2. 点击 **Play** 播放视频
3. 点击 **Start Mic** 开启麦克风
4. 对着麦克风说话，口型会实时变化

### 加载自定义资产

1. 将生成的文件复制到 `web_player/` 目录：
   - `mouthless.mp4`
   - `mouth_track.json`
   - `open.png`, `closed.png`, `half.png`, `e.png`, `u.png`

2. 刷新页面，播放器会自动加载

或者使用文件选择器手动选择 `output/` 目录下的文件。

---

## 配置选项

### 检测模式

编辑 `web_pipeline/config.py`：

```python
# 批量模式（推荐，一次 API 调用）
mouth_detect_batch_mode: bool = True
mouth_detect_batch_fps: float = 2.0  # 采样帧率

# 逐帧模式（精度高但慢）
mouth_detect_batch_mode: bool = False
mouth_detect_sample_rate: int = 5  # 每 5 帧检测一次
```

### 口消し参数

```python
erase_pad: int = 10           # mask 膨胀像素
erase_inpaint_radius: int = 5 # inpaint 半径
erase_blend: bool = True      # 边缘融合
```

### 视频生成参数

```python
video_resolution: str = "720P"  # 480P, 720P, 1080P
video_duration: int = 5         # 视频时长（秒）
```

---

## API 说明

### 支持的模型

| 功能 | 模型 | 说明 |
|------|------|------|
| 嘴部检测 | `qwen3-vl-plus-2025-12-19` | 视觉语言模型 |
| 视频生成 | `wan2.1-i2v-plus` | 首帧生视频 |
| 循环视频 | `wan2.1-kf2v-plus` | 首尾帧生视频 |

### API 限制

| 参数 | 限制 |
|------|------|
| 视频时长 | 2秒 ~ 1小时 |
| 视频帧数 | 最多 2000 帧 |
| 文件大小 | Base64: 10MB, URL: 2GB |

### 费用估算

- VL 检测：约 ¥0.01/次（批量模式）
- 视频生成：约 ¥0.5/次

---

## 常见问题

### Q: 检测速度很慢？

A: 确保启用了批量模式：
```python
mouth_detect_batch_mode = True
```
批量模式一次 API 调用处理整个视频，比逐帧模式快 30 倍。

### Q: 视频文件太大无法上传？

A: 系统会自动压缩视频到 10MB 以下。如果仍然失败，可以手动用 ffmpeg 压缩：
```bash
ffmpeg -i input.mp4 -vf "scale=iw*0.5:ih*0.5" -crf 28 output.mp4
```

### Q: API 返回内容审查错误？

A: 系统已自动关闭内容审查。如果仍有问题，检查 API Key 是否正确。

### Q: 嘴部位置检测不准？

A: 尝试以下方法：
1. 确保角色脸部清晰可见
2. 使用更高分辨率的视频
3. 切换到逐帧模式（精度更高）

### Q: Web 播放器无法播放？

A: 检查浏览器控制台错误，确保：
1. 文件路径正确
2. 视频格式为 MP4 (H.264)
3. 允许麦克风权限

---

## 文件格式

### mouth_track.json

```json
{
  "fps": 25.0,
  "width": 1080,
  "height": 1920,
  "frames": [
    {
      "quad": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
      "valid": true
    }
  ]
}
```

### 嘴型 PNG

- 格式：PNG with Alpha
- 尺寸：自动计算（统一尺寸）
- 内容：椭圆 mask + 羽化边缘

---

## 联系方式

- GitHub: https://github.com/heminghaoa/MotionPNGTuber
- Issues: https://github.com/heminghaoa/MotionPNGTuber/issues
