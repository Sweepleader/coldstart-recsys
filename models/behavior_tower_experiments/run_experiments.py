import argparse
import random
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
import math


def load_num_items(item_path):
    try:
        df = pd.read_csv(item_path, sep='|', header=None, encoding='latin-1', engine='python')
        return int(df.iloc[:, 0].max())
    except Exception:
        return None


def build_histories(data_path, num_users, sort_by_ts=True):
    df = pd.read_csv(data_path, sep='\t', names=['user','item','rating','ts'], engine='python')
    df['user'] = df['user'].astype(int) - 1
    df['item'] = df['item'].astype(int) - 1
    if sort_by_ts:
        df = df.sort_values(['user','ts'])
    hist = {u: [] for u in range(num_users)}
    for _, r in df.iterrows():
        hist[int(r['user'])].append(int(r['item']))
    return hist


def build_histories_with_ts(data_path, num_users, sort_by_ts=True):
    df = pd.read_csv(data_path, sep='\t', names=['user','item','rating','ts'], engine='python')
    df['user'] = df['user'].astype(int) - 1
    df['item'] = df['item'].astype(int) - 1
    if sort_by_ts:
        df = df.sort_values(['user','ts'])
    items = {u: [] for u in range(num_users)}
    times = {u: [] for u in range(num_users)}
    for _, r in df.iterrows():
        u = int(r['user']); i = int(r['item']); t = int(r['ts'])
        items[u].append(i)
        times[u].append(t)
    return items, times


def evaluate_tower(tower, histories, test_df, num_items, device='cpu', k=10, num_neg=100, max_len=50, pool='last', decay=None, histories_ts=None):
    recalls = []
    ndcgs = []
    for _, r in test_df.iterrows():
        u = int(r['user']) - 1
        pos = int(r['item']) - 1
        seq = histories.get(u, [])
        if len(seq) == 0:
            continue
        seq = seq[-max_len:]
        seq_t = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        if histories_ts is not None:
            ts_list = histories_ts.get(u, [])
            tb = []
            xs = ts_list[-max_len:] if ts_list is not None else []
            for t in xs:
                sec = int(t)
                dsec = sec % (24 * 3600)
                hour = dsec // 3600
                tb.append(int(hour // 2))
            if len(tb) == 0:
                tb = [0] * seq_t.size(1)
            tb_t = torch.tensor(tb, dtype=torch.long, device=device).unsqueeze(0)
            lengths = torch.tensor([seq_t.size(1)], dtype=torch.long, device=device)
            if hasattr(tower, 'forward_with_import'): # Hack to detect TimeAwareAttentionBiGRU or similar
                 uvec = tower(seq_t, lengths=lengths, time_gaps=tb_t)
            elif pool == 'mean':
                uvec = tower(seq_t, lengths=lengths, time_gaps=tb_t, pool='mean', decay=decay)
            else:
                uvec = tower(seq_t, lengths=lengths, time_gaps=tb_t, pool='last', decay=decay)
        else:
            if pool == 'mean':
                lengths = torch.tensor([seq_t.size(1)], dtype=torch.long, device=device)
                uvec = tower(seq_t, lengths=lengths, pool='mean', decay=decay)
            else:
                uvec = tower(seq_t)
        items = [pos]
        while len(items) < num_neg + 1:
            nid = random.randrange(num_items)
            if nid == pos:
                continue
            items.append(nid)
        items_t = torch.tensor(items, dtype=torch.long, device=device)
        ivec = tower.item_vectors(items_t)
        uvec = F.normalize(uvec, dim=-1)
        ivec = F.normalize(ivec, dim=-1)
        scores = (uvec.unsqueeze(1) * ivec).sum(-1).squeeze(0).detach().cpu().numpy()
        order = scores.argsort()[::-1]
        pos_idx = np.where(order == 0)[0]
        rank = int(pos_idx[0]) if len(pos_idx) > 0 else len(order)
        if rank < k:
            recalls.append(1.0)
            ndcgs.append(1.0 / float(np.log2(rank + 2.0)))
        else:
            recalls.append(0.0)
            ndcgs.append(0.0)
    if len(recalls) == 0:
        return 0.0, 0.0
    return float(sum(recalls) / len(recalls)), float(sum(ndcgs) / len(ndcgs))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_path', type=str, default='F:/coldstart-recsys/data/ml-100k/u1.base')
    p.add_argument('--test_path', type=str, default='F:/coldstart-recsys/data/ml-100k/u1.test')
    p.add_argument('--item_path', type=str, default='F:/coldstart-recsys/data/ml-100k/u.item')
    p.add_argument('--emb_dim', type=int, default=128)
    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--num_neg', type=int, default=100)
    p.add_argument('--tower', type=str, default='baseline', choices=['baseline','twolayer','bidirectional','twotwo','improved','timeaware','timeaware_attention','baseline_timeaware'])
    p.add_argument('--pool', type=str, default='last', choices=['last','mean'])
    p.add_argument('--decay', type=float, default=None)
    p.add_argument('--decay_start', type=float, default=None)
    p.add_argument('--decay_end', type=float, default=None)
    p.add_argument('--decay_schedule', type=str, default='none', choices=['none','linear','cosine'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_csv', type=str, default=None)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--neg', type=int, default=1)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--freeze_item_emb', action='store_true')
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def main():
    args = parse_args()
    set_seed(args.seed)
    item_df = pd.read_csv(args.item_path, sep='|', header=None, encoding='latin-1', engine='python')
    num_items = int(item_df.iloc[:, 0].max())
    train_df = pd.read_csv(args.train_path, sep='\t', names=['user','item','rating','ts'], engine='python')
    num_users = int(train_df['user'].max())
    num_users = num_users
    histories_items, histories_ts = build_histories_with_ts(args.train_path, num_users)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from baseline_tower import BehaviorTowerBaseline, BehaviorTowerBaselineTwoLayer, BehaviorTowerBaselineBidirectional, BehaviorTowerBaselineTwoLayerBidirectional, BehaviorTowerBaselineTimeAware
    from improved_tower import BehaviorTowerImproved
    from improved_tower_timeaware import BehaviorTowerTimeAware, TimeAwareAttentionBiGRU
    builders = {
        'baseline': lambda: BehaviorTowerBaseline(num_items, emb_dim=args.emb_dim).to(device),
        'twotwo': lambda: BehaviorTowerBaselineTwoLayerBidirectional(num_items, emb_dim=args.emb_dim, dropout=0.1).to(device),
        'twolayer': lambda: BehaviorTowerBaselineTwoLayer(num_items, emb_dim=args.emb_dim, dropout=0.1).to(device),
        'bidirectional': lambda: BehaviorTowerBaselineBidirectional(num_items, emb_dim=args.emb_dim, dropout=0.1).to(device),
        'improved': lambda: BehaviorTowerImproved(num_items, emb_dim=args.emb_dim, dropout=0.1).to(device),
        'timeaware': lambda: BehaviorTowerTimeAware(num_items, emb_dim=args.emb_dim, dropout=0.1).to(device),
        'timeaware_attention': lambda: TimeAwareAttentionBiGRU(num_items, emb_dim=args.emb_dim, dropout=0.1).to(device),
        'baseline_timeaware': lambda: BehaviorTowerBaselineTimeAware(num_items, emb_dim=args.emb_dim, dropout=0.1).to(device),
    }
    tower = builders[args.tower]()

    if args.freeze_item_emb:
        try:
            for p in tower.item_emb.parameters():
                p.requires_grad = False
        except Exception:
            pass

    # -------- 训练行为塔（BPR）--------
    train_df = pd.read_csv(args.train_path, sep='\t', names=['user','item','rating','ts'], engine='python')
    train_df['user'] = train_df['user'].astype(int) - 1
    train_df['item'] = train_df['item'].astype(int) - 1
    # 用户正例集合
    user_pos = {}
    for _, r in train_df.iterrows():
        u = int(r['user']); i = int(r['item'])
        user_pos.setdefault(u, set()).add(i)

    triples = []
    rng = np.random.default_rng(args.seed)
    for _, r in train_df.iterrows():
        u = int(r['user']); pos = int(r['item'])
        for _ in range(args.neg):
            neg = int(rng.integers(0, num_items))
            while neg in user_pos[u]:
                neg = int(rng.integers(0, num_items))
            triples.append((u, pos, neg))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, tower.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    def to_time_buckets(ts_list, max_len):
        if ts_list is None or len(ts_list) == 0:
            return [0]
        xs = ts_list[-max_len:]
        buckets = []
        for t in xs:
            sec = int(t)
            dsec = sec % (24 * 3600)
            hour = dsec // 3600
            buckets.append(int(hour // 2))
        return buckets

    def bpr_step(batch, current_decay=None):
        users, pos_items, neg_items = zip(*batch)
        # 构造每个用户的最近序列
        seq_tensors = []
        for u in users:
            seq = histories_items.get(u, [])
            if len(seq) == 0:
                seq = [0]
            seq = seq[-args.max_len:]
            seq_tensors.append(torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0))
        # 逐个前向，避免 padding 影响
        uvecs = []
        tower.train()
        for idx, st in enumerate(seq_tensors):
            if args.tower in ['timeaware', 'baseline_timeaware', 'timeaware_attention']:
                lengths = torch.tensor([st.size(1)], dtype=torch.long, device=device)
                tb = to_time_buckets(histories_ts.get(users[idx], []), args.max_len)
                tb_t = torch.tensor(tb, dtype=torch.long, device=device).unsqueeze(0)
                
                if args.tower == 'timeaware_attention':
                    uv = tower(st, time_gaps=tb_t, lengths=lengths)
                elif args.pool == 'mean':
                    uv = tower(st, lengths=lengths, time_gaps=tb_t, pool='mean', decay=current_decay)
                else:
                    uv = tower(st, lengths=lengths, time_gaps=tb_t, pool='last', decay=current_decay)
            elif args.tower == 'improved':
                lengths = torch.tensor([st.size(1)], dtype=torch.long, device=device)
                if args.pool == 'mean':
                    uv = tower(st, lengths=lengths, pool='mean', decay=current_decay)
                else:
                    uv = tower(st, lengths=lengths, pool='last', decay=current_decay)
            else:
                uv = tower(st)
            uvecs.append(uv)
        uvec = torch.stack(uvecs, dim=0).squeeze(1)  # [B, D]
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

    num_batches = math.ceil(len(triples) / args.batch_size)

    def compute_decay(epoch_idx):
        if args.decay_schedule == 'none':
            return args.decay
        start = args.decay_start if args.decay_start is not None else args.decay
        end = args.decay_end if args.decay_end is not None else args.decay
        if start is None or end is None:
            return args.decay
        t = 0.0
        if args.epochs > 1:
            t = float(epoch_idx - 1) / float(args.epochs - 1)
        if args.decay_schedule == 'linear':
            return start + (end - start) * t
        # cosine
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t))
    for epoch in range(1, args.epochs + 1):
        current_decay = compute_decay(epoch)
        total = 0.0
        for b in range(num_batches):
            s = b * args.batch_size
            e = min(len(triples), s + args.batch_size)
            batch = triples[s:e]
            optimizer.zero_grad()
            loss = bpr_step(batch, current_decay=current_decay)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        avg = total / len(triples)
        print(f"Epoch {epoch}\tAvgLoss={avg:.6f}")
        # per-epoch evaluation for profiling (matches run_profile parser)
        test_df = pd.read_csv(args.test_path, sep='\t', names=['user','item','rating','ts'], engine='python')
        recall, ndcg = evaluate_tower(tower, histories_items, test_df, num_items, device=device, k=args.k, num_neg=args.num_neg, max_len=args.max_len, pool=args.pool, decay=(current_decay if args.tower in ['improved','timeaware','baseline_timeaware'] else None), histories_ts=(histories_ts if args.tower in ['timeaware','baseline_timeaware', 'timeaware_attention'] else None))
        print(f"Eval Recall@{args.k}={recall:.4f} NDCG@{args.k}={ndcg:.4f}")
    test_df = pd.read_csv(args.test_path, sep='\t', names=['user','item','rating','ts'], engine='python')
    final_decay = compute_decay(args.epochs)
    recall, ndcg = evaluate_tower(tower, histories_items, test_df, num_items, device=device, k=args.k, num_neg=args.num_neg, max_len=args.max_len, pool=args.pool, decay=(final_decay if args.tower in ['improved','timeaware','baseline_timeaware'] else None), histories_ts=(histories_ts if args.tower in ['timeaware','baseline_timeaware', 'timeaware_attention'] else None))
    print(f"Recall@{args.k}={recall:.4f} NDCG@{args.k}={ndcg:.4f}")
    if args.save_csv:
        row = [{
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'tower': args.tower,
            'pool': args.pool,
            'decay': (final_decay if (args.tower == 'improved' and final_decay is not None) else (args.decay if args.decay is not None else '')),
            'emb_dim': args.emb_dim,
            'max_len': args.max_len,
            'k': args.k,
            'num_neg': args.num_neg,
            'recall': recall,
            'ndcg': ndcg,
            'seed': args.seed,
        }]
        df = pd.DataFrame(row)
        df.to_csv(args.save_csv, mode='a', index=False, header=not os.path.exists(args.save_csv))


if __name__ == '__main__':
    main()
