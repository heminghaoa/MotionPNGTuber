#!/usr/bin/env python3
"""
测试 WAN 视频生成模块

用法:
    python test_wan.py <image_url>

示例:
    python test_wan.py "https://example.com/character.png"
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web_pipeline.video_generator import VideoGenerator, TaskStatus


def test_wan_video_generation(image_url: str):
    """测试 WAN 视频生成"""
    output_dir = Path(__file__).parent.parent / "output/test_wan"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"图片 URL: {image_url}")
    print(f"输出目录: {output_dir}")

    generator = VideoGenerator()

    # 使用首尾帧生成循环视频
    print("\n[1/3] 创建视频生成任务...")
    task = generator.generate_loop_video(
        image_url=image_url,
        prompt="anime character idle animation, subtle breathing, hair swaying gently, stationary camera",
    )

    if not task.task_id:
        print(f"创建任务失败: {task.error_message}")
        return

    print(f"任务创建成功: {task.task_id}")
    print(f"状态: {task.status.value}")

    # 等待任务完成
    print("\n[2/3] 等待视频生成...")

    def progress_callback(status: TaskStatus, elapsed: float):
        print(f"  状态: {status.value}, 耗时: {elapsed:.0f}s")

    result = generator.wait_for_task(
        task.task_id,
        progress_callback=progress_callback,
    )

    print(f"\n任务完成: {result.status.value}")

    if result.status != TaskStatus.SUCCEEDED:
        print(f"错误: {result.error_message}")
        return

    if not result.video_url:
        print("错误: 没有视频 URL")
        return

    print(f"视频 URL: {result.video_url}")

    # 下载视频
    print("\n[3/3] 下载视频...")
    output_path = output_dir / "generated.mp4"
    generator.download_video(result.video_url, output_path)

    print(f"\n测试完成! 视频保存到: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n请提供图片 URL")
        print("示例: python test_wan.py 'https://example.com/character.png'")
        sys.exit(1)

    image_url = sys.argv[1]
    test_wan_video_generation(image_url)
