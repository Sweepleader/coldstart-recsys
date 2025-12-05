import os
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import imageio_ffmpeg

"""
2025年12月5日 新增工具, 用于将视频转换为wav格式
    需下载的库: 
    pip install imageio_ffmpeg -i https://pypi.tuna.tsinghua.edu.cn/simple
"""

def _convert_job(args):
    src_path, dst_dir, ffmpeg_exe, sample_rate, bitrate, channels = args
    base = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(dst_dir, base + ".wav")
    os.makedirs(dst_dir, exist_ok=True)
    cmd = [
        ffmpeg_exe, "-y", "-i", src_path,
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-b:a", bitrate,
        out_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="MicroLens-50k_videos")
    parser.add_argument("--output", default="MicroLens-50k_audio")
    parser.add_argument("--workers", type=int, default=min(5, (os.cpu_count() or 1)))
    parser.add_argument("--sample-rate", type=int, default=5000)
    parser.add_argument("--bitrate", default="5k")
    parser.add_argument("--channels", type=int, default=1)
    args = parser.parse_args()

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    mp4_files = [f for f in os.listdir(args.input) if f.lower().endswith(".mp4")]
    full_paths = [os.path.join(args.input, f) for f in mp4_files]

    if not full_paths:
        print("No .mp4 files found.")
        return

    tasks = [
        (path, args.output, ffmpeg_exe, args.sample_rate, args.bitrate, args.channels)
        for path in full_paths
    ]

    succeeded = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_convert_job, t) for t in tasks]
        for fut in as_completed(futs):
            try:
                fut.result()
                succeeded += 1
            except Exception:
                failed += 1

    print(f"Done. Succeeded: {succeeded}, Failed: {failed}")

if __name__ == "__main__":
    main()
"""
文件在运行过程中无特殊输出, 运行完成后会显示转换成功和失败的数量
"""