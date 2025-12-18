import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os  # Added os import
import sys
from pathlib import Path
from torchvision import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 1. 视频编码器 (标准化实现)
# ------------------------------------------------------------------------------
class VideoEncoder(nn.Module):
    def __init__(self, output_dim=128, freeze_backbone=False):
        super().__init__()
        # 加载 ResNet34
        try:
            from torchvision.models import ResNet34_Weights
            weights = ResNet34_Weights.DEFAULT
            base = models.resnet34(weights=weights)
        except ImportError:
            # 旧版本 torchvision 的回退方案
            base = models.resnet34(pretrained=True)
        except Exception as e:
            logger.warning(f"加载 ResNet34 权重失败: {e}. 使用随机初始化。")
            base = models.resnet34(weights=None)

        # 替换全连接层
        in_features = base.fc.in_features
        base.fc = nn.Linear(in_features, output_dim)

        if freeze_backbone:
            for name, p in base.named_parameters():
                if not name.startswith("fc"):
                    p.requires_grad = False
        
        self.model = base
        self.norm = nn.LayerNorm(output_dim)
        
        # 封面权重融合参数 (可学习)
        # 初始化为 0.5 (sigmoid(0) = 0.5)
        self.cover_gate = nn.Parameter(torch.zeros(1))
        
        # 标准 ImageNet 归一化常数
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, frames, cover=None):
        """
        参数:
            frames: [B, T, C, H, W] 视频帧序列
            cover:  [B, C, H, W] 封面图 (可选)
        返回:
            [B, output_dim]
        """
        if frames is None and cover is None:
            return None
            
        # 1. 处理视频帧
        frame_feat = None
        if frames is not None:
            B, T, C, H, W = frames.size()
            x = frames.reshape(B * T, C, H, W)
            x = (x - self.mean) / self.std
            feat = self.model(x) # [B*T, D]
            frame_feat = feat.reshape(B, T, -1).mean(dim=1) # [B, D]
        
        # 2. 处理封面图
        cover_feat = None
        if cover is not None:
            # cover: [B, C, H, W]
            c = (cover - self.mean) / self.std
            cover_feat = self.model(c) # [B, D]
            
        # 3. 融合逻辑
        if frame_feat is not None and cover_feat is not None:
            # 动态权重融合
            alpha = torch.sigmoid(self.cover_gate) # (0, 1)
            # 假设封面更重要，或者通过学习决定
            # 这里实现: alpha * cover + (1-alpha) * frames
            final_feat = alpha * cover_feat + (1 - alpha) * frame_feat
        elif frame_feat is not None:
            final_feat = frame_feat
        elif cover_feat is not None:
            final_feat = cover_feat
        else:
            return None
        
        # 输出归一化
        final_feat = self.norm(final_feat)
        final_feat = F.normalize(final_feat, dim=-1)
        
        return final_feat

# ------------------------------------------------------------------------------
# 2. 鲁棒音频编码器
# ------------------------------------------------------------------------------
class RobustAudioEncoder(nn.Module):
    def __init__(self, output_dim=128, input_channels=1):
        super().__init__()
        
        # 轻量级 2D CNN 处理梅尔频谱图 (例如 64xN)
        # 假设输入梅尔频谱图大约为 [B, 1, 64, 100+]
        self.backbone = nn.Sequential(
            # 卷积块 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # [32, 32, T/2]
            
            # 卷积块 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # [64, 16, T/4]
            
            # 卷积块 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # 全局平均池化 -> [B, 128, 1, 1]
        )
        
        self.fc = nn.Linear(128, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_mel):
        """
        参数:
            audio_mel: Tensor [B, 1, F, T] 或 [B, D] (预计算特征)。
                        如果为 None，返回零向量。
        """
        device = self.fc.weight.device
        
        # 处理缺失模态
        if audio_mel is None:
            return None 

        # 确保输入在正确的设备上 (虽然 DataLoader 应该处理这个)
        if audio_mel.device != device:
            audio_mel = audio_mel.to(device)

        if audio_mel.dim() == 4:
            x = self.backbone(audio_mel)
            x = x.flatten(1) # [B, 128]
        elif audio_mel.dim() == 2:
            # 假设是预计算的嵌入
            x = audio_mel
            # 可选: 如果维度不匹配则投影
            if x.shape[1] != 128:
                # 在实际场景中，我们可以在这里添加一个投影层
                pass
        else:
            raise ValueError(f"不支持的音频输入形状 {audio_mel.shape}。")
            
        out = self.fc(x)
        out = self.norm(out)  # LayerNorm
        out = F.normalize(out, p=2, dim=-1)  # 关键修复：L2归一化
        
        return out


# ------------------------------------------------------------------------------
# 2.5 VGGish 音频编码器 (新增)
# ------------------------------------------------------------------------------
class VGGishAudioEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        
        # 动态添加 VGGish 路径到 sys.path
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        vggish_path = project_root / 'models' / 'VGGish' / 'hub' / 'harritaylor_torchvggish_master'
        
        if str(vggish_path) not in sys.path:
            sys.path.append(str(vggish_path))
            logger.info(f"Added VGGish path: {vggish_path}")
            
        try:
            from torchvggish import vggish
            import hubconf
        except ImportError as e:
            logger.error(f"Failed to import VGGish from {vggish_path}: {e}")
            raise e
            
        # 初始化 VGGish
        # preprocess=False: 假设输入是 Log Mel Spectrogram [B, 1, F, T]
        # postprocess=True: 使用 PCA + Quantization 产生 128D Embedding
        # 强制 device='cpu' 以保持与父模块一致 (父模块默认在CPU)。
        # 如果父模块后续被移动到 CUDA，我们需要在 forward 中动态处理。
        self.vggish = vggish.VGGish(urls=hubconf.model_urls, device=torch.device('cpu'), pretrained=True, preprocess=False, postprocess=True)
        
        # 投影层
        if output_dim != 128:
            self.fc = nn.Linear(128, output_dim)
        else:
            self.fc = nn.Identity()
            
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_mel):
        """
        参数:
            audio_mel: Tensor [B, 1, F, T] (F=64) 或 [B, D]
        """
        if audio_mel is None:
            return None
            
        # 动态检测当前参数所在的设备
        # 因为 VGGish 内部维护了一个 self.device 属性，它不会自动随 .cuda() 更新
        # 所以我们需要手动同步它
        current_device = next(self.vggish.parameters()).device
        self.vggish.device = current_device
        
        # 此时 self.norm 应该也已经在正确的设备上了
        device = current_device
        
        # 确保输入在正确设备
        if audio_mel.device != device:
            audio_mel = audio_mel.to(device)
            
        if audio_mel.dim() == 2:
            x = audio_mel
        elif audio_mel.dim() == 4:
            # [B, 1, 64, T] -> VGGish 需要 [N, 1, 96, 64] (Time=96, Freq=64)
            # Permute to [B, 1, T, 64]
            x = audio_mel.permute(0, 1, 3, 2)
            
            B, C, T, Freq = x.shape
            target_T = 96
            
            if T < target_T:
                pad_amount = target_T - T
                # Pad time dimension
                # x is [B, 1, T, 64] -> last dim is 64. 
                # pad format for F.pad on 4D: (left, right, top, bottom) for last 2 dims?
                # Actually for 4D (N,C,H,W), pad is (W_l, W_r, H_t, H_b).
                # Here H=T, W=64.
                # We want to pad H (Time).
                x = F.pad(x, (0, 0, 0, pad_amount))
                out = self.vggish(x)
                if out.dim() == 1:
                    out = out.unsqueeze(0)
            else:
                # Split along T
                x_sq = x.squeeze(1) # [B, T, 64]
                chunks = x_sq.split(target_T, dim=1)
                
                chunk_embs = []
                for chunk in chunks:
                    if chunk.shape[1] < target_T:
                        pad = target_T - chunk.shape[1]
                        # chunk: [B, T', 64]
                        # pad T' (H)
                        # pad format (W_l, W_r, H_t, H_b)
                        chunk = F.pad(chunk, (0, 0, 0, pad))
                    
                    # Add channel dim: [B, 1, 96, 64]
                    inp = chunk.unsqueeze(1)
                    emb = self.vggish(inp)
                    if emb.dim() == 1:
                        emb = emb.unsqueeze(0)
                    chunk_embs.append(emb)
                
                # Stack and mean
                out = torch.stack(chunk_embs).mean(dim=0)
        else:
            raise ValueError(f"Unsupported audio shape: {audio_mel.shape}")
            
        out = self.fc(out)
        out = self.norm(out)
        out = F.normalize(out, p=2, dim=-1)
        
        return out


# ------------------------------------------------------------------------------
# 3. 鲁棒文本编码器
# ------------------------------------------------------------------------------
class RobustTextEncoder(nn.Module):
    def __init__(self, output_dim=128, model_name_or_path='F:/coldstart-recsys/models/SBERT/all-MiniLM-L6-v2', 
                    vocab_size=50000, embed_dim=300):
        super().__init__()
        self.output_dim = output_dim
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name_or_path)
            self.sbert_dim = self.model.get_sentence_embedding_dimension()
            self.proj = nn.Linear(self.sbert_dim, output_dim)
            self.use_sbert = True
        except Exception as e:
            logger.warning(f"SBERT加载失败 ({e})，使用轻量级词嵌入+BiLSTM")
            self.use_sbert = False
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, output_dim//2, 
                                bidirectional=True, batch_first=True)
            self.proj = nn.Identity()
        
        self.norm = nn.LayerNorm(output_dim)  # LayerNorm
    
    def forward(self, texts):
        if self.use_sbert:
            # SBERT输出通常已归一化，但投影后需要重新归一化
            # 关键修复：解决 RuntimeError: Inference tensors cannot be saved for backward
            # 不要在 with torch.no_grad() 块内部获取 tensor 后再在块外部用于梯度计算
            # 即使 detach() 有时也不够，因为某些 SBERT 版本内部可能设置了全局状态
            # 最稳妥的方式：完全在这个块内完成计算，然后 clone() + detach() 出来
            with torch.no_grad():
                embeddings = self.model.encode(texts, convert_to_tensor=True, 
                                                device=self.proj.weight.device, show_progress_bar=False)
                # 强制断开计算图并克隆数据，确保得到一个纯净的 Leaf Tensor
                embeddings = embeddings.clone().detach()
            
            # 现在 embeddings 是一个普通的 Tensor，可以作为输入传给需要梯度的层
            out = self.proj(embeddings)
            out = self.norm(out)
            # 关键修复：添加L2归一化
            out = F.normalize(out, p=2, dim=-1)
        else:
            # 回退方案处理
            if isinstance(texts, list):
                # [简化的回退逻辑，实际应更完善]
                batch_vecs = []
                for text in texts:
                    char_indices = [ord(c) % 1000 for c in text[:50]]
                    indices = torch.tensor(char_indices, device=self.embed.weight.device)
                    emb = self.embed(indices).mean(0)
                    batch_vecs.append(emb)
                x = torch.stack(batch_vecs, dim=0)
                lstm_out, _ = self.lstm(x.unsqueeze(1))
                x = lstm_out.squeeze(1)
            else:
                x = texts
                
            out = self.proj(x)
            out = self.norm(out)
            # 关键修复：回退方案同样需要归一化
            out = F.normalize(out, p=2, dim=-1)
        
        return out

class ShortVideoTextEncoder(nn.Module):
    """
    专为短视频标题优化的文本编码器
    处理流程：特殊符号/标签分离 -> 主干编码 -> 特征增强
    处理的标题类似于: Gu long song gaga bad, even three hundred years old are not let go. # comic # Two Yuan # My destiny villain
        是标准的短视频UGC(用户生成内容)风格标题
    """
    def __init__(self, output_dim=128, sbert_model_name='F:/coldstart-recsys/models/SBERT/all-MiniLM-L6-v2'):
        super().__init__()
        
        self.use_sbert = True
        try:
            # 1. 加载主干SBERT模型（核心）
            from sentence_transformers import SentenceTransformer
            
            # 检查是否是本地绝对路径且是否存在
            if os.path.isabs(sbert_model_name) and not os.path.exists(sbert_model_name):
                logger.warning(f"指定的本地 SBERT 路径 {sbert_model_name} 不存在，尝试使用默认模型名 'all-MiniLM-L6-v2' 从 HuggingFace 下载/加载。")
                sbert_model_name = 'all-MiniLM-L6-v2'
                
            self.sbert = SentenceTransformer(sbert_model_name)
            sbert_dim = self.sbert.get_sentence_embedding_dimension()
        except ImportError:
            logger.warning("SBERT库未找到，ShortVideoTextEncoder 将回退到简单的 Hash Embedding 模式。")
            self.use_sbert = False
            sbert_dim = 384 # 假设默认维度
            self.vocab_size = 10000
            self.embed = nn.Embedding(self.vocab_size, sbert_dim)
        except Exception as e:
            logger.warning(f"SBERT加载失败 ({e})，ShortVideoTextEncoder 将回退到简单的 Hash Embedding 模式。")
            self.use_sbert = False
            sbert_dim = 384
            self.vocab_size = 10000
            self.embed = nn.Embedding(self.vocab_size, sbert_dim)
        
        # 2. 标签特征提取器（专门处理#标签#）
        self.hashtag_encoder = nn.Linear(sbert_dim, 64)  # 将标签序列编码为紧凑特征
        
        # 3. 最终投影层（融合主文和标签特征）
        self.fusion_proj = nn.Sequential(
            nn.Linear(sbert_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)

    def _preprocess_text(self, texts):
        """
        预处理：分离主文本和话题标签
        """
        main_texts = []
        hashtag_texts = []
        
        for text in texts:
            # 简单正则分离标签和正文（实际应用需更健壮）
            import re
            hashtags = re.findall(r'#([^#]+)#', text)
            main = re.sub(r'#[^#]+#', '', text).strip()
            
            main_texts.append(main if main else text)  # 保底
            # 将所有标签拼接成一个字符串送入编码器
            hashtag_texts.append(' '.join(hashtags))
            
        return main_texts, hashtag_texts
    
    def _get_embedding(self, texts):
        device = self.hashtag_encoder.weight.device
        if self.use_sbert:
            # 如果使用 SBERT (SentenceTransformer 或 Transformers)
            if hasattr(self, 'use_transformers_direct') and self.use_transformers_direct:
                # 使用原生 transformers
                inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
                outputs = self.model(**inputs)
                # Mean Pooling - Take attention mask into account for correct averaging
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            else:
                # 使用 sentence_transformers
                # 注意: encode() 可能会自动使用 CUDA，但我们的模块可能在 CPU 上
                embeddings = self.sbert.encode(texts, convert_to_tensor=True)
            
            # 关键修复: 
            # 1. 确保 tensor 在正确的设备上 (与 hashtag_encoder 一致)
            # 2. clone().detach() 断开梯度图，避免 "Inference tensors cannot be saved for backward" 错误
            return embeddings.to(device).clone().detach()
        else:
            # 简单的回退：Hash Embedding + Mean Pooling
            embeddings = []
            for text in texts:
                if not text:
                    # 空文本处理
                    emb = torch.zeros(self.embed.embedding_dim, device=device)
                else:
                    # 简单的 hash 映射
                    indices = torch.tensor([hash(w) % self.vocab_size for w in text.split()], device=device)
                    emb = self.embed(indices).mean(dim=0)
                embeddings.append(emb)
            return torch.stack(embeddings)

    def forward(self, texts):
        # 步骤1: 预处理分离
        main_texts, hashtag_texts = self._preprocess_text(texts)
        
        with torch.no_grad():
            # 步骤2: 分别编码主文本和标签
            # main_emb = self.sbert.encode(main_texts, convert_to_tensor=True)
            # hashtag_emb = self.sbert.encode(hashtag_texts, convert_to_tensor=True)
            main_emb = self._get_embedding(main_texts)
            hashtag_emb = self._get_embedding(hashtag_texts)
        
        # 步骤3: 标签特征压缩
        hashtag_feat = self.hashtag_encoder(hashtag_emb)  # [B, 64]

        
        # 步骤4: 特征融合
        combined = torch.cat([main_emb, hashtag_feat], dim=-1)  # [B, sbert_dim+64]
        out = self.fusion_proj(combined)
        out = self.norm(out)
        out = F.normalize(out, p=2, dim=-1)  # L2归一化
        
        return out

# ------------------------------------------------------------------------------
# 4. 基于注意力的多模态融合
# ------------------------------------------------------------------------------
class MultiModalAttentionFusion(nn.Module):
    """
    加权融合模块。
    用注意力机制为每个模态计算一个权重，根据权重加权融合特征。
    
    优先级:
    - 视觉 (视频): 短视频中重要性最高。
    - 文本: 中等重要性。
    - 音频: 视上下文而定。
    """
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        
        # 注意力网络
        # 根据内容为每个模态计算一个权重
        self.attn_net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False) 
        )
        
        # 可学习的偏置，用于强制 "视觉 > 文本 > 音频" 的先验初始化
        # 顺序: [Video, Audio, Text]
        
        # 我们希望权重大概是: Video=0.5, Text=0.3, Audio=0.2
        # Logits: Video=1.0, Text=0.5, Audio=0.0 -> Softmax 会处理比例
        # self.modality_bias = nn.Parameter(torch.tensor([1.0, 0.0, 0.5])) # Video=0.5, Audio=0.2, Text=0.3
        
        # 或者我们可以用相等的权重初始化
        self.modality_bias = nn.Parameter(torch.zeros(3)) # [Video, Audio, Text]

    def forward(self, video_vec, audio_vec, text_vec):
        """
        参数:
            video_vec: [B, D]
            audio_vec: [B, D]
            text_vec:  [B, D]
        """
        # 统一处理：所有None转为零向量
        batch_size = None
        
        # 确定batch_size（从第一个非None的向量）
        for vec in [video_vec, audio_vec, text_vec]:
            if vec is not None:
                batch_size = vec.shape[0]
                break
        
        if batch_size is None:
            raise ValueError("所有模态输入都为None")
        
        # 设备
        device = self.attn_net[0].weight.device
        
        # 替换None为零向量
        if video_vec is None:
            video_vec = torch.zeros(batch_size, self.dim, device=device)
        if audio_vec is None:
            audio_vec = torch.zeros(batch_size, self.dim, device=device)  
        if text_vec is None:
            text_vec = torch.zeros(batch_size, self.dim, device=device)
        # 堆叠: [B, 3, D] -> (Video, Audio, Text)
        
        stack = torch.stack([video_vec, audio_vec, text_vec], dim=1)
        
        # 1. 基于内容的注意力分数
        # [B, 3, D] -> [B, 3, 1]
        scores = self.attn_net(stack)
        
        # 2. 添加模态偏置 (先验知识)
        # scores: [B, 3, 1] + [3]
        scores = scores + self.modality_bias.view(1, 3, 1)
        
        # 3. Softmax 归一化
        weights = F.softmax(scores, dim=1) # [B, 3, 1]
        
        # 4. 加权求和
        fused = (stack * weights).sum(dim=1) # [B, D]
        
        return fused, weights.squeeze(-1)

# ------------------------------------------------------------------------------
# 5. M3CSR 模态编码器 (Modality Encoder)
# ------------------------------------------------------------------------------
class ModalityEncoder(nn.Module):
    """
    M3CSR 核心组件：结合原始模态特征和聚类ID嵌入。
    
    设计意图说明：
    虽然所有模态共用同一个全局 Cluster ID，但每个模态编码器维护自己独立的 Embedding 表。
    这意味着：Cluster ID=5 在视频空间对应的向量与在音频空间对应的向量是不同的。
    这允许模型学习同一语义类别在不同模态下的特定表达 (Modality-Specific Representation)。
    """
    def __init__(self, input_dim, output_dim, num_clusters=100, modality_name="unknown"):
        super().__init__()
        self.input_dim = input_dim
        self.modality_name = modality_name
        
        # 独立聚类嵌入表 (Modality-Specific Cluster Embedding Table)
        # 即便 ID 相同，不同模态查询到的 embedding 不同
        self.cluster_emb = nn.Embedding(num_clusters, input_dim)
        
        # 融合层 (Combine modality-specific feature and cluster embedding)
        # 使用 Concat + MLP 的方式融合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, feature, cluster_ids):
        """
        参数:
            feature: [B, D] 原始模态特征
            cluster_ids: [B] 聚类ID
        """
        device = self.cluster_emb.weight.device
        
        # 获取聚类嵌入
        if cluster_ids is None:
            # 如果没有提供cluster_ids，使用全0或其他默认策略
            # 这里为了健壮性，假设ID=0为默认/未知类别
            B = feature.shape[0] if feature is not None else 1
            cluster_ids = torch.zeros(B, dtype=torch.long, device=device)
            
        c_emb = self.cluster_emb(cluster_ids) # [B, D]
        
        # 处理特征缺失的情况
        if feature is None:
            feature = torch.zeros_like(c_emb)
        
        # 融合
        combined = torch.cat([feature, c_emb], dim=-1) # [B, 2*D]
        out = self.fusion_mlp(combined) # [B, Out_D]
        
        return out

# ------------------------------------------------------------------------------
# 6. M3CSR 完整多模态编码器
# ------------------------------------------------------------------------------
class M3CSR_MultiModalEncoder(nn.Module):
    """
    完整的 M3CSR 多模态编码器实现。
    
    结构:
    1. 基础编码器 (Base Encoders): 提取原始 Video/Audio/Text 特征 (完全冻结)。
    2. 模态编码器 (Modality Encoders): 结合原始特征与聚类ID，注入协同语义 (可训练)。
    3. 多模态融合 (Fusion): 注意力加权融合 (可训练)。
    """
    def __init__(self, num_clusters=1000, unified_dim=128, use_vggish=True):
        super().__init__()
        
        # 1. 基础特征提取器 (Base Encoders)
        # 注意: 按照 M3CSR，这些通常是预训练好并冻结的
        self.video_encoder = VideoEncoder(output_dim=unified_dim, freeze_backbone=True)
        
        # 音频编码器选择
        if use_vggish:
            logger.info("M3CSR: 使用 VGGishAudioEncoder")
            self.audio_encoder = VGGishAudioEncoder(output_dim=unified_dim)
        else:
            logger.info("M3CSR: 使用 RobustAudioEncoder")
            self.audio_encoder = RobustAudioEncoder(output_dim=unified_dim)
            
        # 使用专门的短视频标题编码器
        self.text_encoder = ShortVideoTextEncoder(output_dim=unified_dim)
        
        # 冻结基础编码器 (确保所有子模块都被冻结，包括BN层和Dropout行为)
        self._freeze_module(self.video_encoder, "VideoEncoder")
        self._freeze_module(self.audio_encoder, "AudioEncoder")
        self._freeze_module(self.text_encoder, "TextEncoder")
        
        # 2. M3CSR 模态编码器 (Modality Encoders) - 可训练
        # 为每个模态实例化独立的编码器，维护独立的 Cluster Embedding 表
        self.video_modality_encoder = ModalityEncoder(unified_dim, unified_dim, num_clusters, modality_name="video")
        self.audio_modality_encoder = ModalityEncoder(unified_dim, unified_dim, num_clusters, modality_name="audio")
        self.text_modality_encoder = ModalityEncoder(unified_dim, unified_dim, num_clusters, modality_name="text")
        
        # 3. 多模态融合 (Fusion) - 可训练
        self.fusion = MultiModalAttentionFusion(dim=unified_dim)

    def _freeze_module(self, module, name="Module"):
        """
        深度冻结模块：
        1. 设置 requires_grad = False
        2. 设置 eval() 模式 (固定 BatchNorm 和 Dropout)
        """
        for param in module.parameters():
            param.requires_grad = False
        
        # 关键：切换到 eval 模式，防止 BN 层更新 running_stats 和 Dropout 随机失活
        module.eval()
        
        # 验证冻结状态
        any_trainable = any(p.requires_grad for p in module.parameters())
        if any_trainable:
            logger.warning(f"警告: {name} 未能完全冻结!")
        else:
            logger.info(f"{name} 已成功冻结 (Parameters frozen + Eval mode).")

    def train(self, mode=True):
        """
        重写 train 方法，确保基础编码器永远保持在 eval 模式
        """
        super().train(mode)
        # 强制基础编码器保持 eval
        self.video_encoder.eval()
        self.audio_encoder.eval()
        self.text_encoder.eval()
        return self
            
    def forward(self, frames, audio_mel, texts, cluster_ids=None, cover=None):
        """
        参数:
            frames: [B, T, C, H, W]
            audio_mel: [B, 1, F, T]
            texts: List[str] or [B, ...]
            cluster_ids: [B] (可选) 如果提供，则执行增强流程；如果不提供，可以仅返回基础特征。
            cover: [B, C, H, W] 封面图
        """
        # 1. 提取基础特征 (No Grad)
        with torch.no_grad():
            base_v = self.video_encoder(frames, cover)
            base_a = self.audio_encoder(audio_mel)
            base_t = self.text_encoder(texts)
        
        # 2. 如果没有提供 Cluster ID，则无法进行模态增强，返回基础融合特征
        # 这对于推断阶段生成 ID 很有用
        if cluster_ids is None:
            # 简单融合，用于聚类
            # 注意：这里我们还没有模态特定的增强，只能做简单的融合
            # 临时使用 fusion 模块（虽然它设计上是融合增强后的特征，但维度兼容）
            fused, weights = self.fusion(base_v, base_a, base_t)
            return fused, weights, (base_v, base_a, base_t)
            
        # 3. 注入聚类语义 (Modality Encoding)
        # 此时梯度开始流动
        video_repr = self.video_modality_encoder(base_v, cluster_ids)
        audio_repr = self.audio_modality_encoder(base_a, cluster_ids)
        text_repr = self.text_modality_encoder(base_t, cluster_ids)
        
        # 4. 融合
        fused, weights = self.fusion(video_repr, audio_repr, text_repr)
        
        return fused, weights

    def get_base_embeddings(self, frames, audio_mel, texts, cover=None):
        """
        辅助方法：获取基础模态特征，用于聚类生成 ID
        """
        with torch.no_grad():
            base_v = self.video_encoder(frames, cover)
            base_a = self.audio_encoder(audio_mel)
            base_t = self.text_encoder(texts)
        return base_v, base_a, base_t

