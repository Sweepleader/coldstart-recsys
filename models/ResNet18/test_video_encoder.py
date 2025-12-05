import argparse
import sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

"""
2025 12月5日 测试视频编码器
"""

def load_frames_from_paths(paths):
    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    imgs = [t(Image.open(p).convert('RGB')) for p in paths]
    return torch.stack(imgs, dim=0)


def resolve_paths(args):
    if args.images:
        paths = [Path(p) for p in args.images]
        paths = [p for p in paths if p.exists()]
        if len(paths) == 0:
            raise FileNotFoundError('no valid image paths')
        if len(paths) < args.frames:
            paths = paths + [paths[-1]] * (args.frames - len(paths))
        return [str(p) for p in paths[:args.frames]]
    if args.dir:
        d = Path(args.dir)
        if not d.exists():
            raise FileNotFoundError('directory not found')
        exts = {'.jpg', '.jpeg', '.png'}
        files = sorted([p for p in d.iterdir() if p.suffix.lower() in exts])
        if len(files) == 0:
            raise FileNotFoundError('no images in directory')
        if len(files) < args.frames:
            files = files + [files[-1]] * (args.frames - len(files))
        return [str(p) for p in files[:args.frames]]
    raise ValueError('must provide --images or --dir')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', nargs='*')
    parser.add_argument('--dir')
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--output_dim', type=int, default=128)
    parser.add_argument('--freeze_backbone', action='store_true')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    from models.m3csr.history_model.m3csr_124 import VideoEncoder

    paths = resolve_paths(args)
    frames = load_frames_from_paths(paths)
    frames = frames.unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VideoEncoder(output_dim=args.output_dim, freeze_backbone=args.freeze_backbone).to(device)

    with torch.no_grad():
        out = model(frames.to(device))
    print(out.shape)
    print(out[0, :10])


if __name__ == '__main__':
    main()

