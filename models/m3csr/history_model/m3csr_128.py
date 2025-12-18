import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from pathlib import Path
import os

"""
M3-CSR 12月8日 更新说明

"""

# ------------------------------------------------------------
# 1. 视频帧编码器: 使用 ResNet18
# ------------------------------------------------------------
class VideoEncoder(nn.Module):
    """
    视频编码器
    - 使用预训练 ResNet18 对每帧提取特征, 再对时间维度做平均池化得到视频级表示。
    参数:
        output_dim: 输出特征维度
        freeze_backbone: 是否冻结除分类头(fc)之外的参数, 便于稳定微调。
    预训练:
        优先加载 ImageNet 预训练权重, 若不可用则回退为随机初始化。
    归一化:
        按 ImageNet 均值/方差对输入进行通道归一化, 建议输入值域为 [0,1]。
    输入/输出:
        输入 frames 形状 [B, F, 3, H, W]; 输出形状 [B, output_dim]。
    """
    def __init__(self, output_dim=128, freeze_backbone=False):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights
            # 使用 ImageNet 预训练权重初始化 ResNet18
            base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            base = models.resnet18(weights=None)
        in_features = base.fc.in_features # 读取最后一层的输入特征维度
        base.fc = nn.Linear(in_features, output_dim) # 将最后一层替换为新的全连接层, 输出特定的维度 output_dim
        if freeze_backbone:
            # 冻结除分类头(fc)之外的所有参数
            for name, p in base.named_parameters():
                if not name.startswith("fc"):
                    p.requires_grad = False
        self.model = base
        self.norm = nn.LayerNorm(output_dim)
        # ImageNet 通道均值与方差, 作为 buffer 以确保训练/推理一致且随模型保存
        # 此处使用 MicroLens-50k 视频切片帧的通道均值与方差, 参数选用详见/tools/image_channel_stats.py
        self.register_buffer("mean", torch.tensor([0.442, 0.422, 0.405]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.291, 0.278, 0.284]).view(1, 3, 1, 1))

    def forward(self, frames):
        """
        参数:
            frames: [B, F, 3, H, W]
        过程:
            1) 合并时间维以逐帧编码
            2) 按 ImageNet 统计做通道归一化
            3) 通过 ResNet18 提取每帧特征
            4) 对时间维做平均池化得到视频特征
        返回:
            [B, D]
        """
        B, T, C, H, W = frames.size()
        # 1) 合并时间维度, 逐帧送入骨干网络
        x = frames.reshape(B * T, C, H, W)
        # 2) 通道归一化到 ImageNet 统计
        x = (x - self.mean) / self.std
        # 3) 骨干网络编码
        feat = self.model(x)
        # 4) 帧级特征做平均池化, 得到视频级表示
        feat = feat.reshape(B, T, -1).mean(1)
        feat = self.norm(feat)
        feat = F.normalize(feat, dim=-1)
        return feat


# ------------------------------------------------------------
# 2. 音频编码器: VGGish
# ------------------------------------------------------------
class AudioEncoder(nn.Module):
    def __init__(self, output_dim=128, freeze=True):
        super().__init__()
        # 加载预训练 VGGish 编码器 (优先本地, 失败再远程; 无网则离线回退)
        self.model = None
        try:
            local_hub = Path(__file__).resolve().parents[3] / 'models' / 'VGGish' / 'hub' / 'harritaylor_torchvggish_master'
            if local_hub.exists():
                self.model = torch.hub.load(str(local_hub), 'vggish', source='local')
            else:
                self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
            self.model.eval()
        except Exception:
            self.model = None
        self.model_ok = self.model is not None
        # 冻结骨干以贴合“中台不可训练嵌入”的使用方式
        if freeze and self.model_ok:
            for p in self.model.parameters():
                p.requires_grad = False
        # VGGish 输出为 128 维嵌入 (常见为按时间帧堆叠的 [T, 128])
        in_dim = 128
        # 投影到下游统一维度 (dim); 若一致则用恒等映射
        self.proj = nn.Linear(in_dim, output_dim) if output_dim != in_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_inputs, sample_rate: int = 16000):
        """
        参数:
            audio_inputs: List[str] 或 List[np.ndarray] 或 List[Tensor]
        过程:
            1) 通过 VGGish 将音频转为嵌入序列 [T, 128]
            2) 沿时间维做均值聚合得到 [128]
            3) 将量化的 0-255 值归一到 [0,1], 并做样本内标准化 (z-score)
            4) 线性投影到下游维度, 并做 L2 归一化以统一模态尺度
        返回:
            [B, output_dim]
        """
        feats = []
        for inp in audio_inputs:
            with torch.no_grad():
                if self.model_ok:
                    if isinstance(inp, str):
                        emb = self.model.forward(inp)
                    elif isinstance(inp, np.ndarray):
                        emb = self.model.forward(inp, fs=sample_rate)
                    elif isinstance(inp, torch.Tensor):
                        arr = inp.detach().cpu().numpy()
                        emb = self.model.forward(arr, fs=sample_rate)
                    else:
                        raise ValueError('Unsupported audio input type')
                else:
                    # 离线回退: 使用 FFT 幅度的前 128 维作为特征
                    if isinstance(inp, torch.Tensor):
                        arr = inp.detach().cpu().numpy().astype(np.float32)
                    elif isinstance(inp, np.ndarray):
                        arr = inp.astype(np.float32)
                    else:
                        raise ValueError('Offline mode requires ndarray or Tensor')
                    if arr.ndim > 1:
                        arr = arr.mean(axis=1)
                    mag = np.abs(np.fft.rfft(arr))
                    vec = np.log1p(mag[:128])
                    emb = torch.from_numpy(vec.astype(np.float32))
            # emb 可能是 numpy 或 Tensor; 形状通常为 [T, 128]
            x = emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
            # 沿时间维聚合为 [128]
            if x.dim() == 2:
                x = x.mean(0)
            orig_dtype = x.dtype
            x = x.to(torch.float32)
            mx = x.max()
            mn = x.min()
            if (orig_dtype == torch.uint8) or (mn.item() >= 0.0 and mx.item() <= 255.0 and mx.item() > 1.5):
                x = x / 255.0
            feats.append(x)
        # 堆叠为批次 [B, 128]
        x = torch.stack(feats, dim=0)
        # 设备对齐
        dev = self.proj.weight.device if isinstance(self.proj, nn.Linear) else self.norm.weight.device
        if x.device != dev:
            x = x.to(dev)
        if self.norm.weight.device != dev:
            self.norm = self.norm.to(dev)
        y = self.proj(x)
        y = self.norm(y)
        # L2 归一化, 提升与图像/文本模态融合时的稳定性
        y = F.normalize(y, dim=-1)
        return y


# ------------------------------------------------------------
# 3. 文本编码器: sentence-BERT
# ------------------------------------------------------------
class TextEncoderSBERT(nn.Module):
    def __init__(self, output_dim=128, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', proj_trainable=True, try_online=False):
        super().__init__()
        self.output_dim = output_dim
        self.ok = True
        offline_env = os.environ.get('HF_HUB_OFFLINE') == '1' or os.environ.get('TRANSFORMERS_OFFLINE') == '1'
        if try_online and not offline_env:
            try:
                from sentence_transformers import SentenceTransformer
                self.sbert = SentenceTransformer(model_name)
                self.sbert.eval()
                sbert_dim = self.sbert.get_sentence_embedding_dimension()
                self.proj = nn.Linear(sbert_dim, output_dim)
                if not proj_trainable:
                    for p in self.proj.parameters():
                        p.requires_grad = False
                self.norm = nn.LayerNorm(output_dim)
            except Exception:
                self.ok = False
                self.sbert = None
                self.proj = nn.Identity()
                self.norm = nn.LayerNorm(output_dim)
        else:
            self.ok = False
            self.sbert = None
            self.proj = nn.Identity()
            self.norm = nn.LayerNorm(output_dim)

    def forward(self, texts=None, embeddings=None):
        if embeddings is not None:
            x = embeddings
        else:
            if not self.ok:
                vecs = []
                for t in texts:
                    seed = abs(hash(t)) & 0xFFFFFFFF
                    rng = np.random.default_rng(seed)
                    v = rng.standard_normal(self.output_dim).astype(np.float32)
                    vecs.append(torch.from_numpy(v))
                x = torch.stack(vecs, dim=0)
            else:
                np_emb = self.sbert.encode(texts, convert_to_numpy=True)
                x = torch.tensor(np_emb, dtype=torch.float32)
        if isinstance(self.proj, nn.Linear):
            device = self.proj.weight.device
        else:
            device = self.norm.weight.device
        x = x.to(device)
        y = self.proj(x)
        y = self.norm(y)
        y = F.normalize(y, dim=-1)
        return y


# ------------------------------------------------------------
# 4. 行为序列塔（GRU / Transformer）
# ------------------------------------------------------------
class BehaviorTower(nn.Module):
    def __init__(self, num_items, emb_dim=128):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

    def forward(self, seq_items):
        """
        seq_items: [B, Seq]
        """
        emb = self.item_emb(seq_items)
        _, h = self.gru(emb)
        return h.squeeze(0)


# ------------------------------------------------------------
# 5. CSR 冷启动塔（内容塔 + CF 塔）
# ------------------------------------------------------------
class CSRTower(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.fc = nn.Linear(dim * 2, dim)
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, content_vec, cf_vec):
        fusion = torch.cat([content_vec, cf_vec], dim=-1)

        gate = torch.sigmoid(self.gate(fusion))
        out = gate * content_vec + (1 - gate) * cf_vec
        out = self.fc(out)

        return out


# ------------------------------------------------------------
# 6. M3-CSR 多模态
# ------------------------------------------------------------
class M3CSR(nn.Module):
    def __init__(self, num_items, vocab_size, dim=128, use_sbert=False, use_vggish_hub=False):
        super().__init__()
        # 多模态编码器
        self.video_encoder = VideoEncoder(dim)
        self.audio_encoder = AudioEncoder(dim)
        self.text_encoder = TextEncoderSBERT(dim)

        # 行为塔
        self.behavior_tower = BehaviorTower(num_items, dim)

        # CF item embedding
        self.cf_emb = nn.Embedding(num_items, dim)

        # CSR 冷启动融合
        self.csr_tower = CSRTower(dim)

    def forward(self, seq_items, frames, audio_mel=None, text_ids=None, item_id=None, texts=None, text_emb=None, audio_paths=None):
        """
        seq_items: 行为序列             [B, seq]
        frames:    视频帧               [B, 6, 3, 224, 224]
        audio_mel: Mel 频谱             [B, 1, 128, 128]
        text_ids:  文本序列             [B, L]
        item_id:   目标 item id         [B]
        """

        # 1. 内容塔：多模态融合
        v = self.video_encoder(frames)
        a = self.audio_encoder(audio_mel) if audio_paths is None else self.audio_encoder(audio_paths)
        if texts is not None or text_emb is not None:
            t = self.text_encoder(texts=texts, embeddings=text_emb)
        else:
            t = self.text_encoder(text_ids)

        content_vec = (v + a + t) / 3

        # 2. 协同过滤塔
        cf_vec = self.cf_emb(item_id)

        # 3. CSR 冷启动融合塔
        item_vec = self.csr_tower(content_vec, cf_vec)

        # 4. 行为塔（用户序列）
        user_vec = self.behavior_tower(seq_items)

        # 5. 打分
        score = (user_vec * item_vec).sum(-1)
        return score
