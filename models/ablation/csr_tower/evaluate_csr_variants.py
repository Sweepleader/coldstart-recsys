import argparse
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

def _setup_sys_path():
    here = Path(__file__).resolve()
    # repo root: .../coldstart-recsys
    try:
        root = here.parents[3]
    except Exception:
        root = here.parent
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)

_setup_sys_path()


def load_titles(item_path):
    df = pd.read_csv(item_path, sep='|', header=None, encoding='latin-1', engine='python')
    titles = df.iloc[:, 1].astype(str).tolist()
    return titles, int(df.iloc[:, 0].max())

def build_histories(data_path, num_users):
    df = pd.read_csv(data_path, sep='\t', names=['user','item','rating','ts'], engine='python')
    df['user'] = df['user'].astype(int) - 1
    df['item'] = df['item'].astype(int) - 1
    hist = {u: [] for u in range(num_users)}
    for _, r in df.iterrows():
        hist[int(r['user'])].append(int(r['item']))
    return hist, df

def train_behavior_tower(train_df, num_items, emb_dim=128, epochs=3, batch_size=256, seed=42):
    import math
    import random
    from models.behavior_tower_experiments.baseline_tower import BehaviorTowerBaseline
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tower = BehaviorTowerBaseline(num_items, emb_dim=emb_dim).to(device)
    user_pos = {}
    for _, r in train_df.iterrows():
        u = int(r['user']); i = int(r['item'])
        user_pos.setdefault(u, set()).add(i)
    triples = []
    rng = np.random.default_rng(seed)
    for _, r in train_df.iterrows():
        u = int(r['user']); pos = int(r['item'])
        neg = int(rng.integers(0, num_items))
        while neg in user_pos[u]:
            neg = int(rng.integers(0, num_items))
        triples.append((u, pos, neg))
    optimizer = torch.optim.Adam(tower.parameters(), lr=1e-3)
    histories = {u: [] for u in range(int(train_df['user'].max()) + 1)}
    for _, r in train_df.iterrows():
        histories[int(r['user'])].append(int(r['item']))
    def bpr_step(batch):
        users, pos_items, neg_items = zip(*batch)
        seq_tensors = []
        for u in users:
            seq = histories.get(u, [])
            if len(seq) == 0:
                seq = [0]
            seq = seq[-50:]
            seq_tensors.append(torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0))
        uvecs = []
        tower.train()
        for st in seq_tensors:
            uv = tower(st)
            uvecs.append(uv)
        uvec = torch.stack(uvecs, dim=0).squeeze(1)
        pos_t = torch.tensor(pos_items, dtype=torch.long, device=device)
        neg_t = torch.tensor(neg_items, dtype=torch.long, device=device)
        ivec_pos = tower.item_vectors(pos_t)
        ivec_neg = tower.item_vectors(neg_t)
        uvec = F.normalize(uvec, dim=-1)
        ivec_pos = F.normalize(ivec_pos, dim=-1)
        ivec_neg = F.normalize(ivec_neg, dim=-1)
        pos_scores = (uvec * ivec_pos).sum(-1)
        neg_scores = (uvec * ivec_neg).sum(-1)
        x = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(x) + 1e-8).mean()
        return loss
    num_batches = math.ceil(len(triples) / batch_size)
    for _ in range(epochs):
        for b in range(num_batches):
            s = b * batch_size
            e = min(len(triples), s + batch_size)
            batch = triples[s:e]
            optimizer.zero_grad()
            loss = bpr_step(batch)
            loss.backward()
            optimizer.step()
    return tower

def encode_content_titles(titles, dim=128):
    from models.m3csr.history_model.m3csr_125 import TextEncoderSBERT
    enc = TextEncoderSBERT(dim)
    with torch.no_grad():
        x = enc(texts=titles)
    return F.normalize(x, dim=-1)

def evaluate_csr_variant(tower, histories, test_df, item_titles, content_vecs, variant):
    from models.ablation.csr_tower.variants import SoftmaxCSRTower, SigmoidCSRTower
    device = next(tower.parameters()).device
    content_vecs = content_vecs.to(device)
    if variant == 'softmax':
        csr = SoftmaxCSRTower(dim=content_vecs.size(1), temperature=1.0, out_norm=True).to(device)
    else:
        csr = SigmoidCSRTower(dim=content_vecs.size(1), out_norm=False).to(device)
    recalls = []
    ndcgs = []
    for _, r in test_df.iterrows():
        u = int(r['user']) - 1
        pos = int(r['item']) - 1
        seq = histories.get(u, [])
        if len(seq) == 0:
            continue
        seq = seq[-50:]
        seq_t = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        uvec = tower(seq_t)
        uvec = F.normalize(uvec, dim=-1)
        items = [pos]
        while len(items) < 101:
            nid = np.random.randint(0, len(item_titles))
            if nid == pos:
                continue
            items.append(nid)
        items_t = torch.tensor(items, dtype=torch.long, device=device)
        cf_vec = tower.item_vectors(items_t)
        cf_vec = F.normalize(cf_vec, dim=-1)
        cont_t = content_vecs[items_t]
        ivec = csr(cont_t, cf_vec)
        ivec = F.normalize(ivec, dim=-1)
        scores = (uvec * ivec).sum(-1).detach().cpu().numpy()
        order = scores.argsort()[::-1]
        pos_idx = np.where(order == 0)[0]
        rank = int(pos_idx[0]) if len(pos_idx) > 0 else len(order)
        if rank < 10:
            recalls.append(1.0)
            ndcgs.append(1.0 / float(np.log2(rank + 2.0)))
        else:
            recalls.append(0.0)
            ndcgs.append(0.0)
    if len(recalls) == 0:
        return 0.0, 0.0
    return float(sum(recalls) / len(recalls)), float(sum(ndcgs) / len(ndcgs))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_path', type=str, default='F:/coldstart-recsys/data/ml-100k/u1.base')
    p.add_argument('--test_path', type=str, default='F:/coldstart-recsys/data/ml-100k/u1.test')
    p.add_argument('--item_path', type=str, default='F:/coldstart-recsys/data/ml-100k/u.item')
    p.add_argument('--emb_dim', type=int, default=128)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_csv', type=str, default='models/ablation/csr_tower/results.csv')
    args = p.parse_args()
    titles, num_items = load_titles(args.item_path)
    train_df = pd.read_csv(args.train_path, sep='\t', names=['user','item','rating','ts'], engine='python')
    num_users = int(train_df['user'].max())
    histories, train_df = build_histories(args.train_path, num_users)
    tower = train_behavior_tower(train_df, num_items, emb_dim=args.emb_dim, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    content_vecs = encode_content_titles(titles, dim=args.emb_dim)
    test_df = pd.read_csv(args.test_path, sep='\t', names=['user','item','rating','ts'], engine='python')
    r1, n1 = evaluate_csr_variant(tower, histories, test_df, titles, content_vecs, variant='softmax')
    r2, n2 = evaluate_csr_variant(tower, histories, test_df, titles, content_vecs, variant='sigmoid')
    df = pd.DataFrame([{'variant':'softmax','recall@10':r1,'ndcg@10':n1},{'variant':'sigmoid','recall@10':r2,'ndcg@10':n2}])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
