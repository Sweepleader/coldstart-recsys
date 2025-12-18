import torch
import torch.nn as nn


class BehaviorTowerImproved(nn.Module):
    def __init__(self, num_items, emb_dim=128, dropout=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

    def forward(self, seq_items, lengths=None, pool="last", decay=None):
        x = self.item_emb(seq_items)
        if decay is not None:
            T = x.size(1)
            w = torch.arange(T - 1, -1, -1, device=x.device).float()
            w = torch.pow(torch.tensor(decay, dtype=x.dtype, device=x.device), w).view(1, T, 1)
            x = x * w
        x = self.dropout(x)
        if pool == "mean" and lengths is not None:
            m = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            m = m.float().unsqueeze(-1)
            s = (x * m).sum(1)
            d = m.sum(1).clamp_min(1.0)
            return s / d
        _, h = self.gru(x)
        return h.squeeze(0)

    def item_vectors(self, item_ids):
        return self.item_emb(item_ids)

