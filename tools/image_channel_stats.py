import math
from pathlib import Path
from PIL import Image
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import time

"""
2025年12月3日 计算图像数据集的通道均值和标准差
    本脚本用于计算数据集图像的通道均值和标准差.

    输入:
        root: 图像数据集的根目录, 包含多个子目录, 每个子目录代表一个类别.
        image_size: 图像的目标大小, 默认为224.
        batch_size: 数据加载器的批量大小, 默认为64.
        num_workers: 数据加载器的工作线程数, 默认为0.
        max_samples: 最大样本数, 用于调试, 默认为None.
    输出:
        mean: 图像数据集的通道均值, 形状为[3].
        std: 图像数据集的通道标准差, 形状为[3].
    
    计算结果用于在m3csr模型的VideoEncoder中ImageNet归一化图像数据集的通道, 以提高模型的训练效果.
"""

class ImageFolderFlat(Dataset):
    def __init__(self, root, image_size=224):
        self.root = Path(root)
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.paths = [p for p in self.root.rglob('*') if p.suffix.lower() in exts]
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)

def compute_channel_stats(root, image_size=224, batch_size=64, num_workers=0, max_samples=None):
    ds = ImageFolderFlat(root, image_size=image_size)
    if max_samples is not None:
        ds = torch.utils.data.Subset(ds, list(range(min(len(ds), max_samples))))
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    sum_c = torch.zeros(3)
    sum_sq_c = torch.zeros(3)
    n_pixels = 0
    total = len(ds)
    processed = 0
    start_t = time.time()
    for x in dl:
        b, c, h, w = x.shape
        sum_c += x.sum(dim=[0, 2, 3])
        sum_sq_c += (x * x).sum(dim=[0, 2, 3])
        n_pixels += b * h * w
        processed += b
        if total:
            pct = processed / total
            bar_len = 30
            filled = int(bar_len * pct)
            bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'
            elapsed = time.time() - start_t
            speed = processed / max(elapsed, 1e-9)
            eta = (total - processed) / max(speed, 1e-9)
            msg = f"\r{bar} {processed}/{total} {pct*100:5.1f}%  {speed:6.1f}/s  ETA {eta:6.1f}s"
            sys.stdout.write(msg)
            sys.stdout.flush()
    if total:
        sys.stdout.write("\n")
    mean = (sum_c / n_pixels)
    std = (sum_sq_c / n_pixels - mean ** 2).clamp_min(1e-12).sqrt()
    return mean.tolist(), std.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=r'F:\DataSets\MicroLens-50k\MicroLens-50k_frames_interval_1_number_5\MicroLens-50k_frames_interval_1_number_5')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    mean, std = compute_channel_stats(
        root=args.root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    print('mean:', mean)
    print('std:', std)

if __name__ == '__main__':
    main()

"""
执行命令: python image_channel_stats.py --root (数据集根目录) --num_workers (工作线程数, 大小取决于cpu内核)

输出结果:
    mean: [0.441645085811615, 0.42157307267189026, 0.40526244044303894]
    std: [0.2907397747039795, 0.2775101959705353, 0.2843948006629944]

    代码输出的多次结果平均误差小于0.0001. 采用小数点3位保留不会产生过多影响.

与经典的ImageNet数据集的通道均值和标准差不同, 本脚本计算的通道均值和标准差是针对特定数据集的. 在使用中请根据具体数据集的情况进行调整.
经典的ImageNet数据集的通道均值和标准差为:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

    与MicroLens-50k视频切片帧的通道均值和标准差 > ±0.02, 故修改参数.
"""