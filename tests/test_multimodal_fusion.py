import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _setup_sys_path():
    root = _repo_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _make_video_input(batch=1, frames=4, h=224, w=224):
    return torch.rand(batch, frames, 3, h, w, dtype=torch.float32)


def _make_audio_waveforms(batch=1, seconds=1.0, sr=16000):
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

    frames_dir = _repo_root() / 'models' / 'mutibehavior' / 'test_file' / 'picture'
    try:
        from PIL import Image
        import torchvision.transforms as T
        imgs = []
        for p in sorted(frames_dir.glob('*.jpg')):
            img = Image.open(p).convert('RGB')
            x = T.Compose([T.Resize((224, 224)), T.ToTensor()])(img)
            imgs.append(x)
        if len(imgs) >= 4:
            x = torch.stack(imgs[:4], dim=0).unsqueeze(0)
            frames = x
        else:
            frames = _make_video_input(batch=1, frames=4, h=224, w=224)
    except Exception:
        frames = _make_video_input(batch=1, frames=4, h=224, w=224)

    audio_path = _repo_root() / 'models' / 'mutibehavior' / 'test_file' / 'bus_chatter.wav'
    audio_inputs = [str(audio_path)]
    use_array = False
    audio_arr = None
    audio_sr = 16000
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(str(audio_path))
        if hasattr(data, 'dtype') and data.dtype == np.int16:
            arr = (data.astype(np.float32)) / 32768.0
        else:
            arr = data.astype(np.float32)
        audio_arr = [arr]
        audio_sr = int(sr)
        use_array = True
    except Exception:
        try:
            import soundfile as sf
            data, sr = sf.read(str(audio_path), dtype='float32')
            audio_arr = [data.astype(np.float32)]
            audio_sr = int(sr)
            use_array = True
        except Exception:
            use_array = False

    txt_path = _repo_root() / 'models' / 'mutibehavior' / 'test_file' / 'text.txt'
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        texts = lines[:2] if lines else _make_texts(batch=2)
    except Exception:
        texts = _make_texts(batch=2)

    ve = VideoEncoder125(output_dim=128, freeze_backbone=True)
    ae = AudioEncoder125(output_dim=128, freeze=True)
    te = TextEncoder125(output_dim=128) if TextEncoder125 is not None else None

    with torch.no_grad():
        yv = ve(frames)
        try:
            if use_array and audio_arr is not None:
                ya = ae(audio_arr, sample_rate=audio_sr)
            else:
                ya = ae(audio_inputs)
        except Exception as e:
            print(f"警告: AudioEncoder 前向失败，使用随机特征回退。原因: {e}")
            ya = torch.randn(len(audio_inputs), 128)

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

"""
输出结果:
    
[Video] shape=(1, 128)
[Video] L2 norms (raw)=[1.0]
[Video] L2 norms (after normalize)=[1.0]
[Video] sample[0][:5]=[0.13246163725852966, 0.07867936044931412, -0.07994518429040909, 0.1286083459854126, -0.2283448874950409]
[Audio] shape=(1, 128)
[Audio] L2 norms (raw)=[0.9999998807907104]
[Audio] L2 norms (after normalize)=[1.0]
[Audio] sample[0][:5]=[0.044482916593551636, -0.13161899149417877, 0.023601843044161797, -0.07109308242797852, 0.16867418587207794]
[Text] shape=(1, 128)
[Text] L2 norms (raw)=[0.9999999403953552]
[Text] L2 norms (after normalize)=[1.0]
[Text] sample[0][:5]=[0.035908132791519165, 0.05619512498378754, 0.017329175025224686, 0.021952763199806213, -0.06361228972673416]
[Check] 维度一致性: [128, 128, 128]（期望全为 128）


由于使用数据为本地固定数据, 所以输出结果不变。可复现。
"""
