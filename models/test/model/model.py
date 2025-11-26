# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTower(nn.Module):
    """
    简单的双塔模型。

    - 用户塔：embedding 表将 user_id 映射为稠密向量
    - 物品塔：embedding 表将 item_id 映射为稠密向量

    匹配分数：在 L2 归一化后，用户向量与物品向量的点积。
    L2 归一化可稳定点积的尺度，有助于训练稳定性。
    对于多模态特征，物品塔可以替换为一个 MLP，
    将预提取的特征（例如 CLIP）映射到相同的嵌入维度。
    """
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        # 小型 MLP 变换；通常有助于提升模型容量
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), 
            nn.ReLU()
            )
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), 
            nn.ReLU()
            )
        # 初始化嵌入参数
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, user_ids, item_ids):
        """
        计算批量用户-物品对的打分。
        user_ids: LongTensor (B,)
        item_ids: LongTensor (B,) 或 (B, N)，若为每个用户评分多个物品（按需展平）
        返回：scores (B,) 或 (B, N)
        """
        u = self.user_emb(user_ids)        # (B, D)
        i = self.item_emb(item_ids)        # (B, D) 或 (B, N, D)
        u = self.user_mlp(u)               # (B, D)
        i = self.item_mlp(i)               # (B, D)
        # 归一化
        u = F.normalize(u, p=2, dim=-1)
        i = F.normalize(i, p=2, dim=-1)
        # 点积
        scores = (u * i).sum(dim=-1)
        return scores

    def get_user_embedding(self, user_ids):
        with torch.no_grad():
            u = self.user_emb(user_ids)
            u = self.user_mlp(u)
            return F.normalize(u, dim=-1)

    def get_item_embedding(self, item_ids):
        with torch.no_grad():
            i = self.item_emb(item_ids)
            i = self.item_mlp(i)
            return F.normalize(i, dim=-1)
