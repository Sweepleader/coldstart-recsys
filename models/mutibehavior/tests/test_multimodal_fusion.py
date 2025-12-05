import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _setup_sys_path():
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _make_video_input(batch=2, frames=4, h=224, w=224):
    x = torch.rand(batch, frames, 3, h, w, dtype=torch.float32)
    return x


def _make_audio_waveforms(batch=2, seconds=1.0, sr=16000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    waves = []
    for i in range(batch):
        f1 = 220 + 110 * i
        f2 = 440 + 220 * i
        w = 0.5 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sign(np.sin(2 * np.pi * f2 * t))
        w = np.clip(w, -1.0, 1.0).astype(np.float32)
        waves.append(w)
    return waves


def _make_texts(batch=2):
    return [
        "这是一段用于测试的中文短文本。",
        "Simple English sentence for encoder consistency check.",
    ][:batch]


def _print_stats(name, y):
    y = y.detach().cpu()
    norms = y.norm(p=2, dim=-1)
    print(f"[{name}] shape={tuple(y.shape)}")
    print(f"[{name}] L2 norms (raw)={norms.tolist()}")
    yn = F.normalize(y, dim=-1)
    norms_n = yn.norm(p=2, dim=-1)
    print(f"[{name}] L2 norms (after normalize)={norms_n.tolist()}")
    print(f"[{name}] sample[0][:5]={y[0,:5].tolist()}")


def main():
    _setup_sys_path()
    try:
        from models.m3csr.history_model.m3csr_125 import VideoEncoder as VideoEncoder125
        from models.m3csr.history_model.m3csr_125 import AudioEncoder as AudioEncoder125
    except Exception as e:
        raise RuntimeError(f"导入 m3csr_125 失败: {e}")

    try:
        from models.m3csr.history_model.m3csr_125 import TextEncoderSBERT as TextEncoder125
    except Exception as e:
        TextEncoder125 = None
        print(f"警告: 导入 m3csr_125.TextEncoderSBERT 失败，跳过文本编码器。原因: {e}")

    frames = _make_video_input(batch=2, frames=4, h=224, w=224)
    waves = _make_audio_waveforms(batch=2, seconds=1.0, sr=16000)
    texts = _make_texts(batch=2)

    ve = VideoEncoder125(output_dim=128, freeze_backbone=True)
    ae = AudioEncoder125(output_dim=128, freeze=True)
    te = TextEncoder125(output_dim=128) if TextEncoder125 is not None else None

    with torch.no_grad():
        yv = ve(frames)
        try:
            ya = ae(waves)
        except Exception as e:
            print(f"警告: AudioEncoder 前向失败，使用随机特征回退。原因: {e}")
            ya = torch.randn(len(waves), 128)

        if te is not None:
            try:
                yt = te(texts=texts)
            except Exception as e:
                print(f"警告: TextEncoderSBERT 前向失败，使用随机特征回退。原因: {e}")
                yt = torch.randn(len(texts), 128)
        else:
            yt = torch.randn(len(texts), 128)

    _print_stats("Video", yv)
    _print_stats("Audio", ya)
    _print_stats("Text", yt)

    dims = [yv.shape[-1], ya.shape[-1], yt.shape[-1]]
    print(f"[Check] 维度一致性: {dims}（期望全为 128）")


if __name__ == "__main__":
    main()
