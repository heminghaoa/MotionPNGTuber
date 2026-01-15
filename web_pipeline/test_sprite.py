#!/usr/bin/env python3
"""
测试嘴 PNG 提取功能
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from web_pipeline.mouth_detector import MouthDetector, MouthTrack
from web_pipeline.sprite_extractor import SpriteExtractor


def test_sprite_extraction():
    """测试从视频提取嘴 PNG"""
    # 使用已有的测试视频
    video_path = Path(__file__).parent.parent / "assets/assets03/loop.mp4"
    output_dir = Path(__file__).parent.parent / "output/test_sprites"

    print(f"视频: {video_path}")
    print(f"输出: {output_dir}")

    # 检测嘴部
    print("\n[1/2] 检测嘴部...")
    detector = MouthDetector()

    def detect_progress(current: int, total: int):
        if current % 10 == 0 or current == total:
            print(f"  进度: {current}/{total}")

    track = detector.detect_video(video_path, progress_callback=detect_progress)
    print(f"检测完成: {len(track.frames)} 帧")

    # 提取嘴 PNG
    print("\n[2/2] 提取嘴 PNG...")
    extractor = SpriteExtractor()

    def sprite_progress(current: int, total: int):
        print(f"  进度: {current}/{total}")

    results = extractor.extract(video_path, track, output_dir, progress_callback=sprite_progress)

    print("\n提取完成!")
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    test_sprite_extraction()
