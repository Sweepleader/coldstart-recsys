import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SASRec(nn.Module):
    def __init__(self, num_items, emb_dim=64, num_heads=2, num_layers=2, dropout=0.2, max_len=50):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=num_heads, 
            dim_feedforward=emb_dim*4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Pre-Norm usually converges faster
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(emb_dim)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, seq_items, seq_lens, seq_times=None):
        # seq_items: [B, L]
        batch_size, seq_len = seq_items.size()
        
        # Create masks
        # Padding mask: [B, L] (True for padding/0)
        padding_mask = (seq_items == 0)
        
        # Causal mask: [L, L] (Upper triangular is -inf)
        # Ensure we don't attend to future tokens
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(seq_items.device)
        
        # Embeddings
        # Positions: [0, 1, 2, ..., L-1]
        positions = torch.arange(seq_len, device=seq_items.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.item_emb(seq_items) + self.pos_emb(positions)
        x = self.emb_dropout(x)
        
        # Transformer
        # Note: src_key_padding_mask takes True for positions to IGNORE
        x = self.transformer_encoder(
            x, 
            mask=causal_mask, 
            src_key_padding_mask=padding_mask
        )
        
        x = self.norm(x)
        
        # Gather the last valid item for each sequence as the user representation
        # seq_lens: [B] or [B, 1]
        last_indices = (seq_lens - 1).clamp(min=0).view(-1, 1, 1).expand(-1, -1, x.size(-1))
        # Gather: [B, 1, D] -> [B, D]
        user_emb = torch.gather(x, 1, last_indices).squeeze(1)
        
        return F.normalize(user_emb, p=2, dim=-1)

    def item_vectors(self, item_ids):
        out = self.item_emb(item_ids)
        return F.normalize(out, p=2, dim=-1)

class TimeSASRec(SASRec):
    def __init__(self, num_items, emb_dim=64, num_heads=2, num_layers=2, dropout=0.2, max_len=50, num_time_buckets=64, fusion_type='concat'):
        super().__init__(num_items, emb_dim, num_heads, num_layers, dropout, max_len)
        self.fusion_type = fusion_type
        self.num_time_buckets = num_time_buckets
        self.num_heads = num_heads
        
        if fusion_type == 'attention_bias':
            # Learn a scalar bias for each time bucket per head
            self.time_bias_emb = nn.Embedding(num_time_buckets, num_heads)
            # Initialize to 0 to start with no bias
            nn.init.constant_(self.time_bias_emb.weight, 0.0)
            # No time_emb needed for input
        else:
            self.time_emb = nn.Embedding(num_time_buckets, emb_dim)
            if fusion_type == 'concat':
                # Project concatenated [item, time] back to emb_dim
                self.fusion_layer = nn.Sequential(
                    nn.Linear(emb_dim * 2, emb_dim),
                    nn.GELU(), # Use GELU for better performance in Transformers
                    nn.Dropout(dropout)
                )
                # Re-init weights for the fusion layer
                for m in self.fusion_layer:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
                        nn.init.constant_(m.bias, 0)
    
    def time_to_bucket(self, t):
        # t is tensor
        # log2 bucketing
        t = t.float()
        return (torch.log2(t + 1).long()).clamp(0, self.num_time_buckets - 1)

    def forward(self, seq_items, seq_lens, seq_times):
        # seq_items: [B, L]
        # seq_times: [B, L] (Raw timestamps, padded with 0)
        batch_size, seq_len = seq_items.size()
        
        # Create masks
        padding_mask = (seq_items == 0)
        # Causal mask: [L, L] (Upper triangular is True)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(seq_items.device)
        
        # Embeddings
        positions = torch.arange(seq_len, device=seq_items.device).unsqueeze(0).expand(batch_size, -1)
        item_emb = self.item_emb(seq_items)
        
        if self.fusion_type == 'attention_bias':
            x = item_emb + self.pos_emb(positions)
            x = self.emb_dropout(x)
            
            # Compute Time Distance Matrix [B, L, L]
            t_i = seq_times.unsqueeze(2) # [B, L, 1]
            t_j = seq_times.unsqueeze(1) # [B, 1, L]
            dist = torch.abs(t_i - t_j)
            dist_buckets = self.time_to_bucket(dist)
            
            # Bias: [B, L, L, num_heads] -> [B, num_heads, L, L]
            bias = self.time_bias_emb(dist_buckets).permute(0, 3, 1, 2)
            # Flatten to [B*num_heads, L, L]
            bias = bias.reshape(batch_size * self.num_heads, seq_len, seq_len)
            
            # Base Attention Mask (Causal)
            attn_mask = torch.zeros((seq_len, seq_len), device=seq_items.device)
            attn_mask = attn_mask.masked_fill(causal_mask, float('-inf'))
            
            # Expand to batch*heads
            attn_mask = attn_mask.unsqueeze(0).repeat(batch_size * self.num_heads, 1, 1)
            
            # Add Time Bias
            attn_mask = attn_mask + bias
            
            # Pass to transformer
            # Use PyTorch's src_key_padding_mask handling
            x = self.transformer_encoder(
                x, 
                mask=attn_mask, 
                src_key_padding_mask=padding_mask
            )
            
        else:
            # 3. Compute differences for time intervals
            # Shift seq_times to get diffs
            # Diffs: [B, L] (diff between t_i and t_{i-1})
            # For first item, diff is undefined/0.
            # We want: diff[i] = t[i] - t[i-1].
            
            # Mask out invalid diffs (e.g. padding to first item)
            # With Right Padding: [t1, t2, 0] -> diffs should mask 0 (t1-?), 2 (0-t2). Keep 1 (t2-t1).
            # padding_mask: [F, F, T].
            # We want mask: [T, F, T].
            
            zeros = torch.zeros((batch_size, 1), dtype=torch.long, device=seq_items.device)
            # Use original seq_times for diff, assuming 0 is pad
            diffs = seq_times[:, 1:] - seq_times[:, :-1]
            diffs = torch.cat([zeros, diffs], dim=1) # [B, L]
            
            # Create diff_mask
            # Mask position 0 (start)
            # Mask positions where padding_mask is True (padding)
            diff_mask = padding_mask.clone()
            diff_mask[:, 0] = True 
            
            diffs[diff_mask] = 0
            
            seq_intervals = self.time_to_bucket(diffs)
            time_emb = self.time_emb(seq_intervals)
            
            if self.fusion_type == 'concat':
                x = torch.cat([item_emb, time_emb], dim=-1)
                x = self.fusion_layer(x)
                x = x + self.pos_emb(positions)
            else:
                x = item_emb + self.pos_emb(positions) + time_emb
                
            x = self.emb_dropout(x)
            x = self.transformer_encoder(
                x, 
                mask=causal_mask, 
                src_key_padding_mask=padding_mask
            )
        
        x = self.norm(x)
        
        # Gather
        last_indices = (seq_lens - 1).clamp(min=0).view(-1, 1, 1).expand(-1, -1, x.size(-1))
        user_emb = torch.gather(x, 1, last_indices).squeeze(1)
        
        return F.normalize(user_emb, p=2, dim=-1)
