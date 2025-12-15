import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# =========================
# Utils
# =========================

def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import torch.nn.functional as F

# =========================
# Model
# =========================

try:
    from sasrec_model import SASRec, TimeSASRec
except ImportError:
    from models.behavior_tower_experiments.sasrec_model import SASRec, TimeSASRec

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, factor_num=32, num_layers=3, dropout=0.0):
        super().__init__()
        self.factor_num = factor_num
        self.num_layers = num_layers
        
        # GMF
        self.embed_user_GMF = nn.Embedding(num_users, factor_num)
        self.embed_item_GMF = nn.Embedding(num_items, factor_num)
        
        # MLP
        self.embed_user_MLP = nn.Embedding(num_users, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(num_items, factor_num * (2 ** (num_layers - 1)))
        
        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
        
        # Prediction
        self.predict_layer = nn.Linear(factor_num * 2, 1)
        
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)
        
        concat = torch.cat((output_GMF, output_MLP), -1)
        prediction = self.predict_layer(concat)
        return prediction.view(-1)

class BehaviorTowerGRU(nn.Module):
    def __init__(self, num_items, emb_dim=64, dropout=0.2, padding_idx=0):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, seq_items, seq_lens):
        x = self.item_emb(seq_items)
        x = self.dropout(x)
        x = pack_padded_sequence(
            x, seq_lens.cpu().squeeze(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(x)
        h = h.squeeze(0)
        out = self.proj(h)
        return F.normalize(out, p=2, dim=-1) # Add L2 normalization

    def item_vectors(self, item_ids):
        out = self.proj(self.item_emb(item_ids))
        return F.normalize(out, p=2, dim=-1) # Add L2 normalization


# =========================
# Dataset
# =========================

class NCFTrainDataset(Dataset):
    def __init__(self, user_seq, num_items, num_negatives=4):
        self.features = []
        self.labels = []
        
        # user_seq is dict: user -> items
        for u, items in user_seq.items():
            # For checking existence efficiently
            item_set = set(items)
            for i in items:
                # Positive
                self.features.append([u, i])
                self.labels.append(1.0)
                
                # Negatives
                for _ in range(num_negatives):
                    neg = random.randint(1, num_items - 1)
                    while neg in item_set:
                        neg = random.randint(1, num_items - 1)
                    self.features.append([u, neg])
                    self.labels.append(0.0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user, item = self.features[idx]
        label = self.labels[idx]
        return torch.LongTensor([user]), torch.LongTensor([item]), torch.FloatTensor([label])

class TrainDataset(Dataset):
    def __init__(self, user_sequences, user_times, num_items, max_len=50, slide_window_step=1):
        self.num_items = num_items
        self.max_len = max_len
        self.samples = []
        
        # Data Augmentation: Generate multiple samples per user using sliding window
        for u, hist in user_sequences.items():
            hist_t = user_times[u]
            if len(hist) < 2:
                continue
            
            # Start from index 1 (predicting 2nd item using 1st item)
            # Use step to control data size if needed
            for i in range(1, len(hist), slide_window_step):
                seq = hist[:i]
                seq_t = hist_t[:i]
                pos = hist[i]
                
                # Only keep the last max_len items for efficiency
                seq = seq[-max_len:]
                seq_t = seq_t[-max_len:]
                
                self.samples.append((seq, seq_t, pos, hist))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, seq_t, pos, hist = self.samples[idx]
        
        neg = random.randint(1, self.num_items - 1)
        while neg in hist: # Simple negative sampling
            neg = random.randint(1, self.num_items - 1)

        seq_len = len(seq)
        
        pad_len = self.max_len - seq_len
        pad_seq = seq + [0] * pad_len
        # Pass raw timestamps, padded with 0
        pad_times = seq_t + [0] * pad_len

        return (
            torch.LongTensor(pad_seq),
            torch.LongTensor([seq_len]),
            torch.LongTensor(pad_times),
            torch.LongTensor([pos]),
            torch.LongTensor([neg]),
        )


class NCFTestDataset(Dataset):
    def __init__(self, user_sequences, test_items, num_items, neg_num=99):
        self.data = []
        
        for u in user_sequences:
            if u not in test_items:
                continue
                
            gt = test_items[u]
            # Need to exclude all training items for this user when sampling negatives
            train_items = set(user_sequences[u])
            
            negatives = set()
            while len(negatives) < neg_num:
                neg = random.randint(1, num_items - 1)
                if neg != gt and neg not in train_items:
                    negatives.add(neg)
                    
            candidates = [gt] + list(negatives)
            self.data.append((u, gt, candidates))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, gt, candidates = self.data[idx]
        return (
            torch.LongTensor([u]),
            torch.LongTensor([gt]),
            torch.LongTensor(candidates)
        )

class TestDataset(Dataset):
    def __init__(self, user_sequences, user_times, test_items, num_items, max_len=50, neg_num=99):
        self.data = []
        self.num_items = num_items
        self.max_len = max_len
        self.neg_num = neg_num

        for u in user_sequences:
            if u not in test_items:
                continue
                
            hist = user_sequences[u]
            hist_t = user_times[u]
            gt = test_items[u]

            negatives = set()
            while len(negatives) < neg_num:
                neg = random.randint(1, num_items - 1)
                if neg != gt:
                    negatives.add(neg)

            candidates = [gt] + list(negatives)
            random.shuffle(candidates) # Shuffle to avoid position bias
            
            seq = hist[-max_len:]
            seq_t = hist_t[-max_len:]
            seq_len = len(seq)
            
            pad_len = self.max_len - seq_len
            pad_seq = seq + [0] * pad_len
            pad_times = seq_t + [0] * pad_len
            
            self.data.append((pad_seq, seq_len, pad_times, gt, candidates))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, seq_len, pad_times, gt, candidates = self.data[idx]
        return (
            torch.LongTensor(seq),
            torch.LongTensor([seq_len]),
            torch.LongTensor(pad_times),
            torch.LongTensor([gt]),
            torch.LongTensor(candidates),
        )


# =========================
# Metrics
# =========================

def evaluate_seq(model, dataloader, device, k=10):
    model.eval()
    HR, Recall, NDCG = [], [], []

    with torch.no_grad():
        for seq, lens, seq_times, gt, candidates in dataloader:
            seq, lens = seq.to(device), lens.to(device)
            seq_times = seq_times.to(device)
            gt = gt.to(device)
            candidates = candidates.to(device)
            
            # Check if model accepts seq_times
            if isinstance(model, TimeSASRec):
                user_vec = model(seq, lens, seq_times)
            else:
                user_vec = model(seq, lens)
                
            item_vecs = model.item_vectors(candidates)

            scores = torch.bmm(item_vecs, user_vec.unsqueeze(-1)).squeeze(-1)
            topk = scores.topk(k, dim=1).indices

            for i in range(len(gt)):
                gt_idx = (candidates[i] == gt[i]).nonzero(as_tuple=True)[0].item()
                rank_list = topk[i].cpu().numpy()

                hit = int(gt_idx in rank_list)
                HR.append(hit)
                Recall.append(hit)

                if hit:
                    rank = np.where(rank_list == gt_idx)[0][0]
                    NDCG.append(1 / np.log2(rank + 2))
                else:
                    NDCG.append(0)

    return np.mean(HR), np.mean(Recall), np.mean(NDCG)

def evaluate_ncf(model, dataloader, device, k=10):
    model.eval()
    HR, Recall, NDCG = [], [], []

    with torch.no_grad():
        for u, gt, candidates in dataloader:
            # u: [B, 1]
            # candidates: [B, 100]
            u = u.to(device)
            candidates = candidates.to(device)
            gt = gt.to(device)
            
            # Expand u to match candidates: [B, 100]
            batch_size, num_cand = candidates.size()
            u_expanded = u.expand(-1, num_cand)
            
            # Flatten to feed into model
            u_flat = u_expanded.contiguous().view(-1)     # [B*100]
            c_flat = candidates.view(-1)     # [B*100]
            
            scores = model(u_flat, c_flat)   # [B*100]
            scores = scores.view(batch_size, num_cand) # [B, 100]
            
            topk = scores.topk(k, dim=1).indices

            for i in range(batch_size):
                # gt is at index 0 in candidates
                gt_item = gt[i].item()
                # Find where gt is in candidates
                cands = candidates[i].cpu().numpy()
                gt_idx = np.where(cands == gt_item)[0][0]
                
                rank_list = topk[i].cpu().numpy()
                hit = int(gt_idx in rank_list)
                HR.append(hit)
                Recall.append(hit)

                if hit:
                    rank = np.where(rank_list == gt_idx)[0][0]
                    NDCG.append(1 / np.log2(rank + 2))
                else:
                    NDCG.append(0)

    return np.mean(HR), np.mean(Recall), np.mean(NDCG)


# =========================
# Data Loader
# =========================

def load_ml100k_leave_one_out(data_path):
    cols = ["user", "item", "rating", "timestamp"]
    df = pd.read_csv(data_path, sep="\t", names=cols)

    # Filter by rating if needed (optional, but standard NCF uses all interactions as implicit feedback)
    # df = df[df.rating >= 4] 

    # Sort by timestamp to ensure sequence order
    df = df.sort_values(['user', 'timestamp'])

    user_seq = {}
    user_seq_times = {}
    test_item = {}
    
    # Re-map user and item IDs to be contiguous and 0-based if necessary
    # But ML-100k IDs are 1-based, fitting our padding_idx=0 logic perfectly if max_id is small enough.
    # Max item ID is 1682. 
    
    num_items = df['item'].max() + 1
    num_users = df['user'].max() + 1

    for u, group in df.groupby('user'):
        items = group['item'].tolist()
        times = group['timestamp'].tolist()
        
        # Need at least 2 items to have a train sequence and a test item
        if len(items) < 2:
            continue
            
        # Leave-one-out split
        test_item[u] = items[-1]
        user_seq[u] = items[:-1]
        user_seq_times[u] = times[:-1]

    return user_seq, user_seq_times, test_item, num_items, num_users


# =========================
# Loss
# =========================

def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.mean(F.logsigmoid(pos_score - neg_score))
    return loss

# =========================
# Main
# =========================

def main(args):
    set_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    user_seq, user_seq_times, test_item, num_items, num_users = load_ml100k_leave_one_out(
        args.data_path
    )

    print(f"#Users: {num_users}")
    print(f"#Items: {num_items}")
    print(f"#Train Interactions: {sum(len(v) for v in user_seq.values())}")

    if args.model == 'ncf':
        # Switch to NeuMF
        model = NeuMF(
            num_users=num_users,
            num_items=num_items,
            factor_num=32,
            num_layers=3,
            dropout=0.0
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Test Dataset is static
        test_ds = NCFTestDataset(user_seq, test_item, num_items)
        test_loader = DataLoader(test_ds, batch_size=100)

        for epoch in range(1, args.epochs + 1):
            # Resample negatives every epoch
            train_ds = NCFTrainDataset(user_seq, num_items, num_negatives=4)
            train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
            
            model.train()
            total_loss = 0

            for u, i, label in train_loader:
                u, i, label = u.to(device), i.to(device), label.to(device)
                
                # Squeeze because Dataset returns [1] tensors
                prediction = model(u.squeeze(), i.squeeze())
                loss = criterion(prediction, label.squeeze())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            hr, recall, ndcg = evaluate_ncf(model, test_loader, device)

            print(
                f"Epoch {epoch:02d} | "
                f"Loss {total_loss / len(train_loader):.4f} | "
                f"HR@10 {hr:.4f} | "
                f"Recall@10 {recall:.4f} | "
                f"NDCG@10 {ndcg:.4f}"
            )

    elif args.model in ['sasrec', 'gru', 'time_sasrec']:
        if args.model == 'sasrec':
            model = SASRec(num_items=num_items, emb_dim=64, num_heads=2, num_layers=2, dropout=0.1).to(device)
        elif args.model == 'time_sasrec':
            model = TimeSASRec(num_items=num_items, emb_dim=64, num_heads=2, num_layers=2, dropout=0.1, fusion_type=args.fusion_type).to(device)
        else:
            model = BehaviorTowerGRU(num_items=num_items, emb_dim=64, dropout=0.1).to(device)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Test Dataset
        test_ds = TestDataset(user_seq, user_seq_times, test_item, num_items)
        test_loader = DataLoader(test_ds, batch_size=100, shuffle=False)
        
        # Train Dataset
        train_ds = TrainDataset(user_seq, user_seq_times, num_items, max_len=50)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0
            
            for seq, len_seq, seq_times, pos, neg in train_loader:
                seq, len_seq = seq.to(device), len_seq.to(device)
                seq_times = seq_times.to(device)
                pos, neg = pos.to(device), neg.to(device)
                
                if args.model == 'time_sasrec':
                    user_emb = model(seq, len_seq, seq_times)
                else:
                    user_emb = model(seq, len_seq)
                    
                pos_emb = model.item_vectors(pos.squeeze())
                neg_emb = model.item_vectors(neg.squeeze())
                
                loss = bpr_loss(user_emb, pos_emb, neg_emb)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            hr, recall, ndcg = evaluate_seq(model, test_loader, device)
            
            print(
                f"Epoch {epoch:02d} | "
                f"Loss {total_loss / len(train_loader):.4f} | "
                f"HR@10 {hr:.4f} | "
                f"Recall@10 {recall:.4f} | "
                f"NDCG@10 {ndcg:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='F:/coldstart-recsys/data/ml-100k/u.data')
    parser.add_argument('--item_path', type=str,
                        default='F:/coldstart-recsys/data/ml-100k/u.item')
    parser.add_argument('--model', type=str, default='ncf', choices=['ncf', 'sasrec', 'gru', 'time_sasrec'],
                        help='Model to train: ncf, sasrec, gru, time_sasrec')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--fusion_type', type=str, default='concat', choices=['concat', 'add', 'attention_bias'],
                        help='Fusion type for TimeSASRec')
    args = parser.parse_args()
    main(args)
