#!/usr/bin/env python3
"""
测试脚本 - 验证 web_pipeline 模块

使用方法:
    # 设置 API Key
    export DASHSCOPE_API_KEY='your-api-key'

    # 测试从已有视频处理（不需要调用 API）
    python -m web_pipeline.test_pipeline --video assets/assets03/loop.mp4

    # 测试完整流程（需要 API）
    python -m web_pipeline.test_pipeline --image-url "https://..."

    # 只测试嘴部检测
    python -m web_pipeline.test_pipeline --test-detect --video assets/assets03/loop.mp4

    # 只测试口消し
    python -m web_pipeline.test_pipeline --test-erase --video assets/assets03/loop.mp4
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np


def test_mouth_detector(video_path: str, num_frames: int = 5) -> bool:
    """测试嘴部检测模块"""
    print("\n=== 测试嘴部检测 ===")

    from web_pipeline.config import Config
    from web_pipeline.mouth_detector import MouthDetector

    try:
        config = Config()
        config.validate()
    except ValueError as e:
        print(f"警告: {e}")
        print("跳过嘴部检测测试（需要 API Key）")
        return False

    detector = MouthDetector(config)

    # 读取视频的几帧
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    print(f"视频: {video_path}")
    print(f"总帧数: {total_frames}, 采样 {num_frames} 帧")

    for i in range(num_frames):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        bbox = detector.detect_frame(frame)
        print(
            f"  帧 {frame_idx}: bbox=[{bbox.x1:.3f}, {bbox.y1:.3f}, {bbox.x2:.3f}, {bbox.y2:.3f}], "
            f"valid={bbox.valid}, conf={bbox.confidence:.2f}"
        )

    cap.release()
    print("嘴部检测测试完成")
    return True


def test_mouth_eraser(video_path: str, output_dir: str = "test_output") -> bool:
    """测试口消し模块（使用模拟的 track）"""
    print("\n=== 测试口消し ===")

    from web_pipeline.mouth_detector import MouthTrack
    from web_pipeline.mouth_eraser import MouthEraser

    # 读取视频信息
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"视频: {video_path}")
    print(f"尺寸: {width}x{height}, FPS: {fps}, 帧数: {total_frames}")

    # 创建模拟的 track（假设嘴在中下部）
    # 实际使用时应该用 MouthDetector 检测
    mock_bbox = [0.35, 0.55, 0.65, 0.70]  # 归一化坐标
    x1, y1, x2, y2 = mock_bbox
    mock_quad = [
        [x1 * width, y1 * height],
        [x2 * width, y1 * height],
        [x2 * width, y2 * height],
        [x1 * width, y2 * height],
    ]

    frames = [{"quad": mock_quad, "valid": True} for _ in range(total_frames)]

    track = MouthTrack(
        fps=fps,
        width=width,
        height=height,
        frames=frames,
    )

    # 执行口消し
    eraser = MouthEraser()
    output_path = Path(output_dir) / "mouthless_test.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(current: int, total: int) -> None:
        if current % 30 == 0 or current == total:
            print(f"  进度: {current}/{total}")

    eraser.erase_video(video_path, track, output_path, progress_callback=progress)

    print(f"输出: {output_path}")
    print("口消し测试完成")
    return True


def test_full_pipeline(
    video_path: str | None = None,
    image_url: str | None = None,
    output_dir: str = "test_output",
) -> bool:
    """测试完整流水线"""
    print("\n=== 测试完整流水线 ===")

    from web_pipeline.pipeline import Pipeline, PipelineProgress, PipelineStage, Config

    config = Config()

    if not image_url and not video_path:
        print("错误: 必须提供 --video 或 --image-url")
        return False

    # 如果只有视频，检查是否需要 API
    if video_path and not image_url:
        try:
            config.validate()
        except ValueError as e:
            print(f"警告: {e}")
            print("将跳过嘴部检测步骤")

    pipeline = Pipeline(config, output_dir)

    def progress_callback(p: PipelineProgress) -> None:
        print(f"  [{p.stage.value}] {p.progress:.0%} - {p.message}")

    pipeline.set_progress_callback(progress_callback)

    if video_path:
        result = pipeline.run_from_video(video_path)
    else:
        result = pipeline.run_from_image(image_url)

    print(f"\n结果:")
    print(f"  成功: {result.success}")
    print(f"  阶段: {result.stage.value}")
    if result.base_video_path:
        print(f"  底片视频: {result.base_video_path}")
    if result.mouth_track_path:
        print(f"  嘴部轨迹: {result.mouth_track_path}")
    if result.mouthless_video_path:
        print(f"  无嘴视频: {result.mouthless_video_path}")
    if result.error_message:
        print(f"  错误: {result.error_message}")

    return result.success


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 web_pipeline 模块")
    parser.add_argument("--video", help="输入视频路径")
    parser.add_argument("--image-url", help="角色图 URL（用于测试视频生成）")
    parser.add_argument("--output-dir", default="test_output", help="输出目录")
    parser.add_argument("--test-detect", action="store_true", help="只测试嘴部检测")
    parser.add_argument("--test-erase", action="store_true", help="只测试口消し")

    args = parser.parse_args()

    if args.test_detect:
        if not args.video:
            print("错误: --test-detect 需要 --video 参数")
            return 1
        success = test_mouth_detector(args.video)
        return 0 if success else 1

    if args.test_erase:
        if not args.video:
            print("错误: --test-erase 需要 --video 参数")
            return 1
        success = test_mouth_eraser(args.video, args.output_dir)
        return 0 if success else 1

    # 完整流水线测试
    success = test_full_pipeline(args.video, args.image_url, args.output_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
