import argparse
import random
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
import math
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from functools import lru_cache
import warnings
from typing import Dict, List, Tuple, Optional

# 忽略特定警告
warnings.filterwarnings('ignore')

# -------------------- 数据加载和预处理函数 --------------------

def load_num_items(item_path: str) -> Optional[int]:
    """加载物品数量"""
    try:
        df = pd.read_csv(item_path, sep='|', header=None, encoding='latin-1', engine='python')
        return int(df.iloc[:, 0].max())
    except Exception as e:
        print(f"Error loading items: {e}")
        return None


def build_histories_with_ts(
    data_path: str, 
    num_users: int, 
    sort_by_ts: bool = True
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """构建用户历史序列和时间信息"""
    try:
        df = pd.read_csv(
            data_path, 
            sep='\t', 
            names=['user', 'item', 'rating', 'ts'], 
            engine='python'
        )
        df['user'] = df['user'].astype(int) - 1
        df['item'] = df['item'].astype(int) - 1
        
        if sort_by_ts:
            df = df.sort_values(['user', 'ts'])
        
        items = {u: [] for u in range(num_users)}
        times = {u: [] for u in range(num_users)}
        
        for _, r in df.iterrows():
            u = int(r['user'])
            i = int(r['item'])
            t = int(r['ts'])
            items[u].append(i)
            times[u].append(t)
        
        return items, times
    except Exception as e:
        print(f"Error building histories: {e}")
        return {}, {}


def time_to_bucket(ts_list: List[int], max_len: int = 50) -> List[int]:
    """将时间戳转换为时间桶（每2小时一个桶）"""
    buckets = []
    for t in ts_list[-max_len:]:
        sec = int(t)
        dsec = sec % (24 * 3600)
        hour = dsec // 3600
        buckets.append(int(hour // 2))
    return buckets if buckets else [0]


# -------------------- 数据集类 --------------------

class TripleDataset(Dataset):
    """训练数据集类，用于BPR损失"""
    
    def __init__(
        self, 
        train_df: pd.DataFrame,
        histories_items: Dict[int, List[int]],
        histories_ts: Optional[Dict[int, List[int]]],
        num_items: int,
        max_len: int = 50,
        neg_per_pos: int = 1,
        seed: int = 42
    ):
        self.train_df = train_df
        self.histories_items = histories_items
        self.histories_ts = histories_ts
        self.num_items = num_items
        self.max_len = max_len
        self.neg_per_pos = neg_per_pos
        self.rng = np.random.default_rng(seed)
        
        # 构建用户正样本集合
        self.user_pos = {}
        for _, r in train_df.iterrows():
            u = int(r['user'])
            i = int(r['item'])
            self.user_pos.setdefault(u, set()).add(i)
        
        # 预计算所有训练三元组
        self.triples = self._generate_triples()
    
    def _generate_triples(self) -> List[Tuple[int, int, int]]:
        """生成训练三元组（用户，正样本，负样本）"""
        triples = []
        for _, r in self.train_df.iterrows():
            u = int(r['user'])
            pos = int(r['item'])
            for _ in range(self.neg_per_pos):
                neg = int(self.rng.integers(0, self.num_items))
                while neg in self.user_pos[u]:
                    neg = int(self.rng.integers(0, self.num_items))
                triples.append((u, pos, neg))
        return triples
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        u, pos, neg = self.triples[idx]
        
        # 获取用户历史序列
        seq = self.histories_items.get(u, [])
        if len(seq) == 0:
            seq = [0]
        seq = seq[-self.max_len:]
        
        # 获取时间信息
        time_buckets = [0]
        if self.histories_ts is not None:
            ts_list = self.histories_ts.get(u, [])
            time_buckets = time_to_bucket(ts_list, self.max_len)
        
        return {
            'user_id': u,
            'sequence': np.array(seq, dtype=np.int64),
            'pos_item': pos,
            'neg_item': neg,
            'time_buckets': np.array(time_buckets, dtype=np.int64),
            'seq_length': len(seq)
        }


# -------------------- 评估函数（批量版本） --------------------

@torch.no_grad()
def evaluate_tower_batch(
    tower,
    histories_items: Dict[int, List[int]],
    test_df: pd.DataFrame,
    num_items: int,
    device: str = 'cpu',
    k: int = 10,
    num_neg: int = 100,
    max_len: int = 50,
    batch_size: int = 1024,
    pool: str = 'last',
    decay: Optional[float] = None,
    histories_ts: Optional[Dict[int, List[int]]] = None,
    tower_type: str = 'baseline'  # 新增：传递模型类型
) -> Tuple[float, float]:
    """批量评估模型性能"""
    tower.eval()
    
    recalls = []
    ndcgs = []
    
    # 准备测试数据
    test_users = (test_df['user'].astype(int) - 1).tolist()
    test_items = (test_df['item'].astype(int) - 1).tolist()
    
    # 批量处理测试数据
    for batch_start in range(0, len(test_users), batch_size):
        batch_end = min(batch_start + batch_size, len(test_users))
        batch_users = test_users[batch_start:batch_end]
        batch_pos_items = test_items[batch_start:batch_end]
        
        # 准备批量数据
        batch_seqs = []
        batch_times = []
        batch_lengths = []
        
        for u in batch_users:
            seq = histories_items.get(u, [])
            if len(seq) == 0:
                seq = [0]
            seq = seq[-max_len:]
            batch_seqs.append(torch.tensor(seq, dtype=torch.long))
            batch_lengths.append(len(seq))
            
            if histories_ts is not None and tower_type in ['timeaware', 'baseline_timeaware', 'timeaware_attention', 'timeaware_attention2']:
                ts_list = histories_ts.get(u, [])
                time_buckets = time_to_bucket(ts_list, max_len)
                batch_times.append(torch.tensor(time_buckets, dtype=torch.long))
            else:
                batch_times.append(torch.zeros(len(seq), dtype=torch.long))
        
        # 填充序列
        seqs = pad_sequence(batch_seqs, batch_first=True, padding_value=0).to(device)
        lengths = torch.tensor(batch_lengths, dtype=torch.long).to(device)
        times_t = pad_sequence(batch_times, batch_first=True, padding_value=0).to(device)
        
        # 根据模型类型计算用户向量
        if tower_type in ['timeaware_attention', 'timeaware_attention2']:
            # 这两个模型有特殊的前向传播方法
            uvec = tower(seqs, lengths=lengths, time_gaps=times_t)
        elif tower_type in ['timeaware', 'baseline_timeaware']:
            # 时间感知模型
            if pool == 'mean':
                uvec = tower(seqs, lengths=lengths, time_gaps=times_t, pool='mean', decay=decay)
            else:
                uvec = tower(seqs, lengths=lengths, time_gaps=times_t, pool='last', decay=decay)
        elif tower_type == 'improved':
            # 改进模型
            if pool == 'mean':
                uvec = tower(seqs, lengths=lengths, pool='mean', decay=decay)
            else:
                uvec = tower(seqs, lengths=lengths, pool='last', decay=decay)
        else:
            # 基础模型（baseline, twolayer, bidirectional, twotwo）
            uvec = tower(seqs)
        
        uvec = F.normalize(uvec, dim=-1)
        
        # 为每个测试样本评估
        for i, (uvec_i, pos) in enumerate(zip(uvec, batch_pos_items)):
            # 采样负样本
            neg_samples = set()
            while len(neg_samples) < num_neg:
                neg = random.randrange(num_items)
                if neg != pos:
                    neg_samples.add(neg)
            
            items = [pos] + list(neg_samples)
            items_t = torch.tensor(items, dtype=torch.long, device=device)
            
            # 计算物品向量
            if hasattr(tower, 'item_vectors'):
                ivec = tower.item_vectors(items_t)
            else:
                ivec = tower.item_emb(items_t)
            
            # ivec = F.normalize(ivec, dim=-1)
            
            # 计算分数
            scores = (uvec_i.unsqueeze(0) * ivec).sum(-1)
            scores_np = scores.detach().cpu().numpy()
            
            # 计算排名
            order = scores_np.argsort()[::-1]
            pos_idx = np.where(order == 0)[0]
            rank = int(pos_idx[0]) if len(pos_idx) > 0 else len(order)
            
            # 计算指标
            if rank < k:
                recalls.append(1.0)
                ndcgs.append(1.0 / float(np.log2(rank + 2.0)))
            else:
                recalls.append(0.0)
                ndcgs.append(0.0)
    
    if len(recalls) == 0:
        return 0.0, 0.0
    
    return float(sum(recalls) / len(recalls)), float(sum(ndcgs) / len(ndcgs))


# -------------------- 训练函数 --------------------

def train_epoch(
    tower,
    dataloader,
    optimizer,
    device,
    epoch,
    args,
    histories_ts=None,
    scaler=None
):
    """训练一个epoch"""
    tower.train()
    total_loss = 0.0
    total_samples = 0
    
    # 计算当前epoch的衰减值
    current_decay = compute_decay(epoch, args)
    
    # 使用进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        # 准备数据
        seqs = batch['sequence'].to(device)
        lengths = batch['seq_length'].to(device)
        pos_items = batch['pos_item'].to(device)
        neg_items = batch['neg_item'].to(device)
        time_buckets = batch['time_buckets'].to(device) if histories_ts is not None else None
        
        optimizer.zero_grad()
        
        # 混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # 前向传播
                # 根据模型类型调用
                if tower_type in ['timeaware_attention', 'timeaware_attention2']:
                    uvec = tower(seqs, lengths=lengths, time_gaps=time_buckets)
                elif tower_type in ['timeaware', 'baseline_timeaware']:
                    if pool == 'mean':
                        uvec = tower(seqs, lengths=lengths, time_gaps=time_buckets, pool='mean', decay=current_decay)
                    else:
                        uvec = tower(seqs, lengths=lengths, time_gaps=time_buckets, pool='last', decay=current_decay)
                elif tower_type == 'improved':
                    if pool == 'mean':
                        uvec = tower(seqs, lengths=lengths, pool='mean', decay=current_decay)
                    else:
                        uvec = tower(seqs, lengths=lengths, pool='last', decay=current_decay)
                else:
                    # 基础模型
                    uvec = tower(seqs)
                
                # 计算物品向量
                if hasattr(tower, 'item_vectors'):
                    ivec_pos = tower.item_vectors(pos_items)
                    ivec_neg = tower.item_vectors(neg_items)
                else:
                    ivec_pos = tower.item_emb(pos_items)
                    ivec_neg = tower.item_emb(neg_items)
                
                # ========== 关键修改：添加归一化，与评估一致 ==========
                uvec = F.normalize(uvec, dim=-1)
                ivec_pos = F.normalize(ivec_pos, dim=-1)
                ivec_neg = F.normalize(ivec_neg, dim=-1)
                # ===================================================
                
                # BPR损失
                pos_scores = (uvec * ivec_pos).sum(-1)
                neg_scores = (uvec * ivec_neg).sum(-1)
                x = pos_scores - neg_scores
                loss = -torch.log(torch.sigmoid(x) + 1e-8).mean()
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 不使用混合精度
            # 前向传播
            if args.tower in ['timeaware', 'baseline_timeaware', 'timeaware_attention', 'timeaware_attention2']:
                if args.tower in ['timeaware_attention', 'timeaware_attention2']:
                    uvec = tower(seqs, lengths=lengths, time_gaps=time_buckets)
                elif args.pool == 'mean':
                    uvec = tower(seqs, lengths=lengths, time_gaps=time_buckets, pool='mean', decay=current_decay)
                else:
                    uvec = tower(seqs, lengths=lengths, time_gaps=time_buckets, pool='last', decay=current_decay)
            elif args.tower == 'improved':
                if args.pool == 'mean':
                    uvec = tower(seqs, lengths=lengths, pool='mean', decay=current_decay)
                else:
                    uvec = tower(seqs, lengths=lengths, pool='last', decay=current_decay)
            else:
                uvec = tower(seqs)
            
            # 计算物品向量
            if hasattr(tower, 'item_vectors'):
                ivec_pos = tower.item_vectors(pos_items)
                ivec_neg = tower.item_vectors(neg_items)
            else:
                ivec_pos = tower.item_emb(pos_items)
                ivec_neg = tower.item_emb(neg_items)
            
            # ========== 关键修改：添加归一化，与评估一致 ==========
            uvec = F.normalize(uvec, dim=-1)
            ivec_pos = F.normalize(ivec_pos, dim=-1)
            ivec_neg = F.normalize(ivec_neg, dim=-1)
            # ===================================================
            
            # BPR损失
            pos_scores = (uvec * ivec_pos).sum(-1)
            neg_scores = (uvec * ivec_neg).sum(-1)
            x = pos_scores - neg_scores
            loss = -torch.log(torch.sigmoid(x) + 1e-8).mean()
            
            # 反向传播
            loss.backward()
            optimizer.step()
        
        # 累积损失
        batch_size = len(pos_items)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # 更新进度条
        pbar.set_postfix({
            'loss': loss.item(),
            'avg_loss': total_loss / total_samples
        })
    
    return total_loss / total_samples


# -------------------- 辅助函数 --------------------

def compute_decay(epoch_idx: int, args) -> Optional[float]:
    """计算衰减值"""
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
    elif args.decay_schedule == 'cosine':
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t))
    
    return args.decay


def collate_fn(batch):
    """自定义collate函数，用于DataLoader"""
    user_ids = torch.tensor([item['user_id'] for item in batch], dtype=torch.long)
    sequences = [torch.tensor(item['sequence'], dtype=torch.long) for item in batch]
    pos_items = torch.tensor([item['pos_item'] for item in batch], dtype=torch.long)
    neg_items = torch.tensor([item['neg_item'] for item in batch], dtype=torch.long)
    time_buckets = [torch.tensor(item['time_buckets'], dtype=torch.long) for item in batch]
    seq_lengths = torch.tensor([item['seq_length'] for item in batch], dtype=torch.long)
    
    # 填充序列
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_times = pad_sequence(time_buckets, batch_first=True, padding_value=0)
    
    return {
        'user_id': user_ids,
        'sequence': padded_seqs,
        'pos_item': pos_items,
        'neg_item': neg_items,
        'time_buckets': padded_times,
        'seq_length': seq_lengths
    }


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------- 主函数 --------------------

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练和评估行为塔模型')
    # parser.add_argument('--dataset', type=str, default='ml-100k', help='数据集名称')
    
    # 数据路径
    parser.add_argument('--train_path', type=str, default='F:/coldstart-recsys/data/ml-100k/u1.base')
    parser.add_argument('--test_path', type=str, default='F:/coldstart-recsys/data/ml-100k/u1.test')
    parser.add_argument('--item_path', type=str, default='F:/coldstart-recsys/data/ml-100k/u.item')
    
    # 模型参数
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--tower', type=str, default='baseline',
                       choices=['baseline', 'twolayer', 'bidirectional', 'twotwo',
                                'improved', 'timeaware', 'timeaware_attention',
                                'timeaware_attention2', 'baseline_timeaware'])
    parser.add_argument('--pool', type=str, default='last', choices=['last', 'mean'])
    parser.add_argument('--freeze_item_emb', action='store_true')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--neg', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    
    # 评估参数
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--num_neg', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=1024)
    
    # 衰减参数
    parser.add_argument('--decay', type=float, default=None)
    parser.add_argument('--decay_start', type=float, default=None)
    parser.add_argument('--decay_end', type=float, default=None)
    parser.add_argument('--decay_schedule', type=str, default='none',
                       choices=['none', 'linear', 'cosine'])
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_csv', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=10)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载数据（只加载一次）
    try:
        item_df = pd.read_csv(args.item_path, sep='|', header=None, 
                             encoding='latin-1', engine='python')
        num_items = int(item_df.iloc[:, 0].max())
        print(f"Number of items: {num_items}")
        
        train_df = pd.read_csv(args.train_path, sep='\t', 
                              names=['user', 'item', 'rating', 'ts'], 
                              engine='python')
        train_df['user'] = train_df['user'].astype(int) - 1
        train_df['item'] = train_df['item'].astype(int) - 1
        
        num_users = int(train_df['user'].max()) + 1
        print(f"Number of users: {num_users}")
        
        # 构建历史序列
        histories_items, histories_ts = build_histories_with_ts(
            args.train_path, num_users
        )
        print(f"Built histories for {len(histories_items)} users")
        
        # 加载测试数据
        test_df = pd.read_csv(args.test_path, sep='\t', 
                             names=['user', 'item', 'rating', 'ts'], 
                             engine='python')
        print(f"Test samples: {len(test_df)}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 导入模型
    try:
        from baseline_tower import (
            BehaviorTowerBaseline, 
            BehaviorTowerBaselineTwoLayer, 
            BehaviorTowerBaselineBidirectional, 
            BehaviorTowerBaselineTwoLayerBidirectional, 
            BehaviorTowerBaselineTimeAware
        )
        from improved_tower import BehaviorTowerImproved
        from improved_tower_timeaware import (
            BehaviorTowerTimeAware, 
            TimeAwareAttentionBiGRU, 
            TimeAwareAttentionBiGRU_regional
        )
    except ImportError as e:
        print(f"Error importing models: {e}")
        return
    
    # 模型构建器
    builders = {
        'baseline': lambda: BehaviorTowerBaseline(num_items, emb_dim=args.emb_dim),
        'twotwo': lambda: BehaviorTowerBaselineTwoLayerBidirectional(
            num_items, emb_dim=args.emb_dim, dropout=0.1
        ),
        'twolayer': lambda: BehaviorTowerBaselineTwoLayer(
            num_items, emb_dim=args.emb_dim, dropout=0.1
        ),
        'bidirectional': lambda: BehaviorTowerBaselineBidirectional(
            num_items, emb_dim=args.emb_dim, dropout=0.1
        ),
        'improved': lambda: BehaviorTowerImproved(
            num_items, emb_dim=args.emb_dim, dropout=0.1
        ),
        'timeaware': lambda: BehaviorTowerTimeAware(
            num_items, emb_dim=args.emb_dim, dropout=0.1
        ),
        'timeaware_attention': lambda: TimeAwareAttentionBiGRU(
            num_items, emb_dim=args.emb_dim, dropout=0.1
        ),
        'timeaware_attention2': lambda: TimeAwareAttentionBiGRU_regional(
            num_items, emb_dim=args.emb_dim, dropout=0.1
        ),
        'baseline_timeaware': lambda: BehaviorTowerBaselineTimeAware(
            num_items, emb_dim=args.emb_dim, dropout=0.1
        ),
    }
    
    # 创建模型
    if args.tower not in builders:
        print(f"Unknown tower type: {args.tower}")
        return
    
    tower = builders[args.tower]().to(device)
    print(f"Created {args.tower} tower with {sum(p.numel() for p in tower.parameters()):,} parameters")
    
    # 冻结物品嵌入
    if args.freeze_item_emb and hasattr(tower, 'item_emb'):
        for p in tower.item_emb.parameters():
            p.requires_grad = False
        print("Froze item embeddings")
    
    # 创建数据集和数据加载器
    dataset = TripleDataset(
        train_df=train_df,
        histories_items=histories_items,
        histories_ts=histories_ts,
        num_items=num_items,
        max_len=args.max_len,
        neg_per_pos=args.neg,
        seed=args.seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() // 2),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Training samples: {len(dataset)}")
    print(f"DataLoader batches: {len(dataloader)}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, tower.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device == 'cuda') else None
    
    # 训练循环
    best_recall = 0.0
    best_ndcg = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # 训练
        avg_loss = train_epoch(
            tower=tower,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            histories_ts=histories_ts if args.tower in ['timeaware', 'baseline_timeaware', 
                                                       'timeaware_attention', 'timeaware_attention2'] else None,
            scaler=scaler
        )
        
        print(f"Average loss: {avg_loss:.6f}")
        
        # 评估
        print("Evaluating...")
        recall, ndcg = evaluate_tower_batch(
            tower=tower,
            histories_items=histories_items,
            test_df=test_df,
            num_items=num_items,
            device=device,  
            k=args.k,
            num_neg=args.num_neg,
            max_len=args.max_len,
            batch_size=args.eval_batch_size,
            pool=args.pool,
            decay=compute_decay(epoch, args) if args.tower in ['improved', 'timeaware', 'baseline_timeaware'] else None,
            histories_ts=histories_ts if args.tower in ['timeaware', 'baseline_timeaware', 
                                                    'timeaware_attention', 'timeaware_attention2'] else None,
            tower_type=args.tower  # 新增参数
        )
        
        print(f"Recall@{args.k}: {recall:.4f}, NDCG@{args.k}: {ndcg:.4f}")
        
        # 保存最佳模型
        if recall > best_recall or (recall == best_recall and ndcg > best_ndcg):
            best_recall = recall
            best_ndcg = ndcg
            torch.save(tower.state_dict(), f'best_model_{args.tower}.pth')
            print(f"Saved best model with Recall@{args.k}: {recall:.4f}")
    
    # 最终评估
    print(f"\n{'='*50}")
    print("Final Results:")
    print(f"{'='*50}")
    
    final_decay = compute_decay(args.epochs, args)
    final_recall, final_ndcg = evaluate_tower_batch(
        tower=tower,
        histories_items=histories_items,
        test_df=test_df,
        num_items=num_items,
        device=device,
        k=args.k,
        num_neg=args.num_neg,
        max_len=args.max_len,
        batch_size=args.eval_batch_size,
        pool=args.pool,
        decay=final_decay if args.tower in ['improved', 'timeaware', 'baseline_timeaware'] else None,
        histories_ts=histories_ts if args.tower in ['timeaware', 'baseline_timeaware', 
                                                   'timeaware_attention', 'timeaware_attention2'] else None
    )
    
    print(f"Final Recall@{args.k}: {final_recall:.4f}")
    print(f"Final NDCG@{args.k}: {final_ndcg:.4f}")
    print(f"Best Recall@{args.k}: {best_recall:.4f}")
    print(f"Best NDCG@{args.k}: {best_ndcg:.4f}")
    
    # 保存结果到CSV
    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        
        result = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'tower': args.tower,
            'pool': args.pool,
            'decay': final_decay if (args.tower in ['improved', 'timeaware', 'baseline_timeaware'] and final_decay is not None) else '',
            'emb_dim': args.emb_dim,
            'max_len': args.max_len,
            'k': args.k,
            'num_neg': args.num_neg,
            'final_recall': final_recall,
            'final_ndcg': final_ndcg,
            'best_recall': best_recall,
            'best_ndcg': best_ndcg,
            'seed': args.seed,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
        }
        
        df = pd.DataFrame([result])
        file_exists = os.path.exists(args.save_csv)
        df.to_csv(args.save_csv, mode='a', index=False, header=not file_exists)
        print(f"Results saved to {args.save_csv}")


if __name__ == '__main__':
    main()