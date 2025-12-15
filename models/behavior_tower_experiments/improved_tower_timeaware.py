import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class TimeAwareAttentionBiGRU(nn.Module):
    """
    Paper Proposal Model:
    Bi-Directional GRU + Time Interval Embeddings + Self-Attention
    """
    def __init__(self, num_items, emb_dim=128, time_bins=128, dropout=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.time_emb = nn.Embedding(time_bins, emb_dim)
        
        # Input: Item_Emb (dim) + Time_Emb (dim) = 2 * dim
        self.input_proj = nn.Linear(emb_dim * 2, emb_dim)
        
        # Bi-Directional GRU
        # Output: 2 * dim (Forward + Backward)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
        # Attention Mechanism: Query-Based Attention
        # Use a learnable query vector to attend to the sequence
        self.attn_query = nn.Parameter(torch.randn(1, emb_dim * 2))
        self.attn_fc = nn.Linear(emb_dim * 2, 1, bias=False) 
        
        # Item Projection: Map item embedding (dim) to match user vector (2*dim)
        # This replaces the simple cat([v, v]) hack
        self.item_proj = nn.Linear(emb_dim, emb_dim * 2)
        
    def forward(self, seq_items, time_gaps, lengths=None):
        # 1. Prepare Inputs
        x = self.item_emb(seq_items)       # [B, L, D]
        t = self.time_emb(time_gaps)       # [B, L, D]
        
        # Fusion: Additive
        xt = x + t
        xt = self.dropout(xt)
        
        # 2. Bi-GRU Encoding
        if lengths is not None:
            # Ensure lengths are on CPU for pack_padded_sequence
            lengths_cpu = lengths.cpu()
            packed = pack_padded_sequence(xt, lengths_cpu, batch_first=True, enforce_sorted=False)
            out_packed, hidden = self.gru(packed)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
            
            # Extract Last States from hidden: [2, B, D] -> [B, 2D]
            # hidden layout: [num_layers*num_directions, batch, hidden_size]
            # We assume 1 layer bidirectional
            last_h = torch.cat([hidden[-2], hidden[-1]], dim=1) # [B, 2D]
        else:
            out, hidden = self.gru(xt)
            last_h = torch.cat([hidden[-2], hidden[-1]], dim=1)
            
        # 3. Query-Based Attention Pooling
        # Query: last_h or global query? Let's use last_h as query to attend to history
        # Attention Score = tanh(W * h_i + U * q)
        
        # Simplified Dot-Product Attention with Query
        # Key = out [B, L, 2D]
        # Query = last_h [B, 2D]
        
        # [B, L, 2D] * [B, 2D, 1] -> [B, L, 1]
        attn_scores = torch.bmm(out, last_h.unsqueeze(2)) 
        
        if lengths is not None:
            mask = torch.arange(out.size(1), device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), -1e9)
            
        attn_weights = torch.softmax(attn_scores, dim=1)
        attention_vector = (out * attn_weights).sum(dim=1) # [B, 2D]
        
        # 4. Residual Connection: Attention + Last State
        user_vector = attention_vector + last_h
        
        return user_vector

    def item_vectors(self, item_ids):
        # Learned projection instead of simple concat
        v = self.item_emb(item_ids)
        return self.item_proj(v) 

    def forward_with_import(self, *args, **kwargs):
        import torch.nn.functional as F
        return self.forward(*args, **kwargs)

# -----------------------------------------------------------------------------
class TimeAwareAttentionBiGRU_regional(nn.Module):
    """
    Paper Proposal Model:
    Bi-Directional GRU + Time Interval Embeddings + Self-Attention
    """
    def __init__(self, num_items, emb_dim=128, time_bins=128, dropout=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.time_emb = nn.Embedding(time_bins, emb_dim)
        
        # Input: Item_Emb (dim) + Time_Emb (dim) = 2 * dim
        self.input_proj = nn.Linear(emb_dim * 2, emb_dim)
        
        # Bi-Directional GRU
        # Output: 2 * dim (Forward + Backward)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
        # Attention Mechanism
        # 将 Bi-GRU 的输出序列聚合为一个 User Vector
        self.attn_fc = nn.Linear(emb_dim * 2, 1) # Simple Attention Score
        
    def forward(self, seq_items, time_gaps, lengths=None):
        # 1. Prepare Inputs
        x = self.item_emb(seq_items)       # [B, L, D]
        t = self.time_emb(time_gaps)       # [B, L, D]
        
        # Fusion: Concat + Project (Better than simple add)
        xt = torch.cat([x, t], dim=-1)     # [B, L, 2D]
        xt = self.input_proj(xt)           # [B, L, D]
        xt = F.relu(xt)
        xt = self.dropout(xt)
        
        # 2. Bi-GRU Encoding
        if lengths is not None:
            packed = pack_padded_sequence(xt, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, _ = self.gru(packed)
            # Unpack: [B, L, 2D]
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, _ = self.gru(xt)
            
        # 3. Time-Aware Attention Pooling
        # Attention Score: [B, L, 1]
        attn_scores = self.attn_fc(out) 
        
        # Mask padding (if lengths provided)
        if lengths is not None:
            mask = torch.arange(out.size(1), device=out.device).unsqueeze(0) < lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), -1e9)
            
        attn_weights = torch.softmax(attn_scores, dim=1) # [B, L, 1]
        
        # Weighted Sum
        user_vector = (out * attn_weights).sum(dim=1) # [B, 2D]
        
        return user_vector

    def item_vectors(self, item_ids):
        # Item vectors for matching need to match user_vector dimension (2D)
        # We project simple item embeddings to 2D
        v = self.item_emb(item_ids)
        # Hack: adapt dimension. Ideally should train a separate projection
        return torch.cat([v, v], dim=-1) 

    def forward_with_import(self, *args, **kwargs):
        import torch.nn.functional as F
        return self.forward(*args, **kwargs)

# -----------------------------------------------------------------------------
class BehaviorTowerTimeAware(nn.Module):
    def __init__(self, num_items, emb_dim=128, time_bins=12, dropout=0.1):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.time_emb = nn.Embedding(time_bins, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

    def forward(self, seq_items, lengths=None, time_gaps=None, pool="last", decay=None):
        x = self.item_emb(seq_items)
        if time_gaps is not None:
            t = self.time_emb(time_gaps)
            x = x + t
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
