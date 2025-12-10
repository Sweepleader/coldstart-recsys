import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BehaviorTowerImproved(nn.Module):
    def __init__(self, num_items, emb_dim=128, dropout=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

    def forward(self, seq_items, lengths=None, pool="last", decay=None):
        x = self.item_emb(seq_items)
        w = None
        if decay is not None:
            T = x.size(1)
            w = torch.arange(T - 1, -1, -1, device=x.device).float()
            w = torch.pow(torch.tensor(decay, dtype=x.dtype, device=x.device), w).view(1, T, 1)
            x = x * w
        x = self.dropout(x)
        if pool == "mean":
            if lengths is None:
                if w is None:
                    return x.mean(1)
                d = w.sum(1).clamp_min(1.0)
                return x.sum(1) / d
            m = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            m = m.float().unsqueeze(-1)
            s = (x * m).sum(1)
            if w is None:
                d = m.sum(1).clamp_min(1.0)
            else:
                d = (w * m).sum(1).clamp_min(1.0)
            return s / d
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
        else:
            _, h = self.gru(x)
        return h.squeeze(0)

    def item_vectors(self, item_ids):
        return self.item_emb(item_ids)

