import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

"""
M3-CSR 12月3 模型更新
    对于 TextEncoder:
        - 使用 SBERT (all-MiniLM-L6-v2) 并线性投影到 dim
        - 使用 SBERT 时在 forward 传入 texts (字符串列表), 依赖 sentence-transformers

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
        B, F, C, H, W = frames.size()
        # 1) 合并时间维度, 逐帧送入骨干网络
        x = frames.reshape(B * F, C, H, W)
        # 2) 通道归一化到 ImageNet 统计
        x = (x - self.mean) / self.std
        # 3) 骨干网络编码
        feat = self.model(x)
        # 4) 帧级特征做平均池化, 得到视频级表示
        feat = feat.reshape(B, F, -1).mean(1)
        return feat


# ------------------------------------------------------------
# 2. 音频编码器: VGGish
# ------------------------------------------------------------
class AudioEncoder(nn.Module):
    def __init__(self, output_dim=128, freeze=True):
        super().__init__()
        # 加载预训练 VGGish 编码器 (torch.hub)
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        # 冻结骨干以贴合“中台不可训练嵌入”的使用方式
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        # VGGish 输出为 128 维嵌入 (常见为按时间帧堆叠的 [T, 128])
        in_dim = 128
        # 投影到下游统一维度 (dim); 若一致则用恒等映射
        self.proj = nn.Linear(in_dim, output_dim) if output_dim != in_dim else nn.Identity()

    def forward(self, audio_paths):
        """
        参数:
            audio_paths: List[str] 音频文件路径列表, 每个元素对应一个样本。
        过程:
            1) 通过 VGGish 将音频转为嵌入序列 [T, 128]
            2) 沿时间维做均值聚合得到 [128]
            3) 将量化的 0-255 值归一到 [0,1], 并做样本内标准化 (z-score)
            4) 线性投影到下游维度, 并做 L2 归一化以统一模态尺度
        返回:
            [B, output_dim]
        """
        feats = []
        for p in audio_paths:
            emb = self.model.forward(p)
            # emb 可能是 numpy 或 Tensor; 形状通常为 [T, 128]
            x = emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
            # 沿时间维聚合为 [128]
            if x.dim() == 2:
                x = x.mean(0)
            # 转为 float32 并缩放到 [0,1]
            x = x.to(torch.float32) / 255.0
            feats.append(x)
        # 堆叠为批次 [B, 128]
        x = torch.stack(feats, dim=0)
        # 样本内标准化 (z-score), 统一不同音频的幅值尺度
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
        # 设备对齐后投影到下游维度
        if isinstance(self.proj, nn.Linear):
            x = x.to(self.proj.weight.device)
        y = self.proj(x)
        # L2 归一化, 提升与图像/文本模态融合时的稳定性
        y = F.normalize(y, dim=-1)
        return y


# ------------------------------------------------------------
# 3. 文本编码器: sentence-BERT
# ------------------------------------------------------------
class TextEncoderSBERT(nn.Module):
    def __init__(self, output_dim=128, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', proj_trainable=True):
        super().__init__()
        self.output_dim = output_dim
        self.ok = True
        try:
            from sentence_transformers import SentenceTransformer
            self.sbert = SentenceTransformer(model_name)
            sbert_dim = self.sbert.get_sentence_embedding_dimension()
            self.proj = nn.Linear(sbert_dim, output_dim)
            if not proj_trainable:
                for p in self.proj.parameters():
                    p.requires_grad = False
        except Exception:
            self.ok = False
            self.sbert = None
            self.proj = nn.Identity()
    def forward(self, texts=None, embeddings=None):
        if embeddings is not None:
            x = embeddings
        else:
            if not self.ok:
                raise RuntimeError('SBERT not available; please install sentence-transformers')
            np_emb = self.sbert.encode(texts, convert_to_numpy=True)
            x = torch.tensor(np_emb, dtype=torch.float32)
        if isinstance(self.proj, nn.Linear):
            x = x.to(self.proj.weight.device)
        return self.proj(x)


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
