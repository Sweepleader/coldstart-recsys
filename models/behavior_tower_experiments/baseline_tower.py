import torch
import torch.nn as nn


class BehaviorTowerBaseline(nn.Module):
    def __init__(self, num_items, emb_dim=128):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

    def forward(self, seq_items):
        emb = self.item_emb(seq_items)
        _, h = self.gru(emb)
        return h.squeeze(0)

    def item_vectors(self, item_ids):
        return self.item_emb(item_ids)

