import torch
import torch.nn as nn
from models.behavior_tower_experiments.sasrec_model import TimeSASRec

def test_model():
    model = TimeSASRec(num_items=100, emb_dim=32, num_heads=2, num_layers=2, fusion_type='attention_bias')
    model.eval()
    
    bs = 2
    seq_len = 5
    
    seq_items = torch.randint(1, 100, (bs, seq_len))
    seq_lens = torch.tensor([5, 5])
    seq_times = torch.randint(0, 1000, (bs, seq_len))
    
    print("Input items:", seq_items)
    print("Input times:", seq_times)
    
    try:
        out = model(seq_items, seq_lens, seq_times)
        print("Output shape:", out.shape)
        print("Output stats:", out.mean().item(), out.std().item())
        print("Contains NaNs:", torch.isnan(out).any().item())
        
        # Check scores against random items
        items = torch.arange(100)
        item_embs = model.item_vectors(items)
        scores = out @ item_embs.T
        print("Scores shape:", scores.shape)
        print("Scores stats:", scores.mean().item(), scores.std().item())
        
        # Check if scores are identical
        print("Scores[0] variance:", scores[0].var().item())
        
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
