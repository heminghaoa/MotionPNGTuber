#!/usr/bin/env python3
"""
完整流水线测试

用法:
    # 从已有视频测试（不调用 WAN API）
    python test_full_pipeline.py --video assets/assets03/loop.mp4

    # 从图片 URL 生成（需要公共 URL）
    python test_full_pipeline.py --image "https://example.com/character.png"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web_pipeline.pipeline import Pipeline, PipelineStage, PipelineProgress


def main():
    parser = argparse.ArgumentParser(description="完整流水线测试")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="已有视频路径")
    group.add_argument("--image", type=str, help="图片 URL（需要公共 URL）")
    parser.add_argument("--output", type=str, default="output/pipeline_test", help="输出目录")

    args = parser.parse_args()

    output_dir = Path(args.output)
    print(f"输出目录: {output_dir}")

    # 创建流水线
    pipeline = Pipeline(output_dir=output_dir)

    # 设置进度回调
    def progress_callback(p: PipelineProgress):
        print(f"[{p.stage.value}] {p.progress*100:.0f}% - {p.message}")

    pipeline.set_progress_callback(progress_callback)

    # 运行流水线
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"视频文件不存在: {video_path}")
            sys.exit(1)
        print(f"从视频开始: {video_path}")
        result = pipeline.run_from_video(video_path)
    else:
        print(f"从图片开始: {args.image}")
        result = pipeline.run_from_image(args.image)

    # 输出结果
    print("\n" + "=" * 50)
    print(f"结果: {'成功' if result.success else '失败'}")
    print(f"阶段: {result.stage.value}")

    if result.error_message:
        print(f"错误: {result.error_message}")

    if result.base_video_path:
        print(f"底片视频: {result.base_video_path}")
    if result.mouthless_video_path:
        print(f"无嘴视频: {result.mouthless_video_path}")
    if result.mouth_track_path:
        print(f"嘴部轨迹: {result.mouth_track_path}")
    if result.mouth_assets_dir:
        print(f"嘴 PNG 目录: {result.mouth_assets_dir}")


if __name__ == "__main__":
    main()
