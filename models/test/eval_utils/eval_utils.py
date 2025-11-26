# eval_utils.py
import numpy as np
import torch
import random

def evaluate(model, test_interactions, num_items, device='cpu', k=10, num_neg=100):
    """
    使用负采样评估 Recall@k 和 NDCG@k。

    对每个测试对 (user, pos_item)：
      - 采样 `num_neg` 个不等于正样本的负样本（实际应排除所有训练集正样本）
      - 计算正样本与负样本的得分
      - 排序并检查正样本是否位于前 k

    这是近似的离线评估协议，但在推荐系统评估中非常常用。
    """
    model.eval()
    recalls = []
    ndcgs = []
    with torch.no_grad():
        for (u, pos) in test_interactions:
            # 采样负样本（简单均匀采样）
            negs = []
            while len(negs) < num_neg:
                nid = random.randrange(num_items)
                if nid == pos:
                    continue
                negs.append(nid)
            items = [pos] + negs
            users = torch.tensor([u]*len(items), dtype=torch.long, device=device)
            items_t = torch.tensor(items, dtype=torch.long, device=device)
            scores = model(users, items_t).cpu().numpy()
            # 分数越高越好
            rank_indices = np.argsort(-scores)
            # 正样本的位置（原始索引为 0）
            pos_rank = np.where(rank_indices == 0)[0][0]
            if pos_rank < k:
                recalls.append(1.0)
                ndcg = 1.0 / np.log2(pos_rank + 2)  # 单个相关项的 DCG 贡献
                ndcgs.append(ndcg)
            else:
                recalls.append(0.0)
                ndcgs.append(0.0)
    return float(np.mean(recalls)), float(np.mean(ndcgs))
