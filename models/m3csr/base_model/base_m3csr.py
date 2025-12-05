import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ------------------------------------------------------------
# 1. 视频帧编码器: 使用 ResNet18
# ------------------------------------------------------------
class VideoEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        base = models.resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, output_dim)
        self.model = base

    def forward(self, frames):
        """
        frames: [B, num_frames, 3, H, W]
        """
        B, F, C, H, W = frames.size()
        frames = frames.view(B * F, C, H, W)
        feat = self.model(frames)            # [B*F, D]
        feat = feat.view(B, F, -1).mean(1)   # [B, D]
        return feat


# ------------------------------------------------------------
# 2. 音频编码器: CNN + Mel-Spectrogram
# ------------------------------------------------------------
class AudioEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.linear = nn.Linear(256, output_dim)

    def forward(self, mel):
        """
        mel: [B, 1, 128, 128]
        """
        x = self.cnn(mel)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# ------------------------------------------------------------
# 3. 文本编码器: Embedding + 平均池化
# ------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)

class TextEncoderSBERT(nn.Module):
    def __init__(self, output_dim=128, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.output_dim = output_dim
        self.ok = True
        try:
            from sentence_transformers import SentenceTransformer
            self.sbert = SentenceTransformer(model_name)
            sbert_dim = self.sbert.get_sentence_embedding_dimension()
            self.proj = nn.Linear(sbert_dim, output_dim)
        except Exception:
            self.ok = False
            self.sbert = None
            self.proj = nn.Identity()
    def forward(self, texts):
        if not self.ok:
            raise RuntimeError('SBERT not available; please install sentence-transformers')
        emb = torch.tensor(self.sbert.encode(texts, convert_to_numpy=True), dtype=torch.float32)
        return self.proj(emb)

    def forward(self, text_ids):
        """
        text_ids: [B, L]
        """
        emb = self.emb(text_ids)       # [B, L, D]
        return emb.mean(1)             # [B, D]


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
# 6. M3-CSR 多模态版
# ------------------------------------------------------------
class M3CSR(nn.Module):
    def __init__(self, num_items, vocab_size, dim=128, use_sbert=False):
        super().__init__()
        # 多模态编码器
        self.video_encoder = VideoEncoder(dim)
        self.audio_encoder = AudioEncoder(dim)
        self.text_encoder = TextEncoder(vocab_size, dim) if not use_sbert else TextEncoderSBERT(dim)

        # 行为塔
        self.behavior_tower = BehaviorTower(num_items, dim)

        # CF item embedding
        self.cf_emb = nn.Embedding(num_items, dim)

        # CSR 冷启动融合
        self.csr_tower = CSRTower(dim)

    def forward(self, seq_items, frames, audio_mel, text_ids=None, item_id=None, texts=None):
        """
        seq_items: 行为序列             [B, seq]
        frames:    视频帧               [B, 6, 3, 224, 224]
        audio_mel: Mel 频谱             [B, 1, 128, 128]
        text_ids:  文本序列             [B, L]
        item_id:   目标 item id         [B]
        """

        # 1. 内容塔：多模态融合
        v = self.video_encoder(frames)
        a = self.audio_encoder(audio_mel)
        t = self.text_encoder(texts) if texts is not None else self.text_encoder(text_ids)

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
