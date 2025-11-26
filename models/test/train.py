# train.py
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd

from data_utils import ML100KDataset, load_num_items
from model import TwoTower
from eval_utils import evaluate

def bpr_loss(pos_scores, neg_scores, reduction='mean'):
    """
    BPR pairwise loss implementation.

    For each user u, positive item i and negative item j:
      maximize sigmoid(s(u,i) - s(u,j))
      Equivalent loss: -log(sigmoid(pos - neg))
    BPR is a pairwise ranking loss commonly used for implicit feedback.
    """
    x = pos_scores - neg_scores
    loss = -torch.log(torch.sigmoid(x) + 1e-8)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/ml-100k/u1.base',
                        help='path to training file (u1.base)')
    parser.add_argument('--test_path', type=str, default='data/ml-100k/u1.test',
                        help='path to test file (u1.test)')
    parser.add_argument('--item_path', type=str, default='data/ml-100k/u.item',
                        help='path to u.item (optional for num_items)')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--neg', type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    num_items = load_num_items(args.item_path)
    if num_items is None:
        num_items = 1682  # fallback for MovieLens 100K

    # Dataset and DataLoader
    train_ds = ML100KDataset(args.data_path, num_items, negative_sampling=args.neg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Load test interactions (list of (user, pos_item))
    test_df = pd.read_csv(args.test_path, sep='\\t', names=['user','item','rating','ts'], engine='python')
    test_interactions = [(int(r['user'])-1, int(r['item'])-1) for _, r in test_df.iterrows()]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TwoTower(train_ds.num_users, num_items, emb_dim=args.emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            if args.neg == 1:
                users, pos_items, neg_items = batch
            else:
                users, pos_items, neg_items = batch  # neg_items is a list of lists
            users = users.long().to(device)
            pos_items = pos_items.long().to(device)
            # handle neg_items which could be list or tensor
            if isinstance(neg_items, list) or (not torch.is_tensor(neg_items)):
                neg_items = torch.tensor(neg_items).long().to(device)
            else:
                neg_items = neg_items.long().to(device)

            optimizer.zero_grad()
            pos_scores = model(users, pos_items)
            neg_scores = model(users, neg_items)
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * users.size(0)

        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch}\tAvgLoss={avg_loss:.6f}")

        # Evaluate on test set each epoch (using sampled negative eval)
        model.eval()
        recall10, ndcg10 = evaluate(model, test_interactions, num_items, device=device, k=10, num_neg=100)
        print(f"Eval Recall@10={recall10:.4f}  NDCG@10={ndcg10:.4f}")

if __name__ == '__main__':
    main()
