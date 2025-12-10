import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd


PROFILES = {
    'baseline': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'baseline', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--neg', '1', '--k', '10', '--num_neg', '100', '--seed', '42'
        ],
        'name': 'baseline_last'
    },
    'timeaware_attention_bigru': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'timeaware_attention', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42'
        ],
        'name': 'timeaware_attention_bigru'
    },
}
    'twolayer': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'twolayer', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--neg', '1', '--k', '10', '--num_neg', '100', '--seed', '42'
        ],
        'name': 'twolayer'
    },
    'bidirectional': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'bidirectional', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--neg', '1', '--k', '10', '--num_neg', '100', '--seed', '42'
        ],
        'name': 'bidirectional'
    },
    'twotwo': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'twotwo', '--pool', 'last', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42'
        ],
        'name': 'twotwo'
    },
    'improved_last': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'improved', '--pool', 'last', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42'
        ],
        'name': 'improved_last'
    },
    'improved_mean': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'improved', '--pool', 'mean', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42'
        ],
        'name': 'improved_mean'
    },
    'improved_mean_decay': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'improved', '--pool', 'mean', '--decay', '0.9', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42'
        ],
        'name': 'improved_mean_decay'
    },
    'improved_last_linear_decay': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'improved', '--pool', 'last', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42', '--decay_schedule', 'linear', '--decay_start', '0.995', '--decay_end', '0.99'
        ],
        'name': 'improved_last_linear_decay'
    },
    'timeaware_last': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'timeaware', '--pool', 'last', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42'
        ],
        'name': 'timeaware_last'
    },
    'timeaware_mean': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'timeaware', '--pool', 'mean', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42'
        ],
        'name': 'timeaware_mean'
    },
    'timeaware_mean_decay': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'timeaware', '--pool', 'mean', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42', '--decay_schedule', 'linear', '--decay_start', '0.995', '--decay_end', '0.99'
        ],
        'name': 'timeaware_mean_decay'
    },
    'baseline_timeaware_last': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'baseline_timeaware', '--pool', 'last', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42'
        ],
        'name': 'baseline_timeaware_last'
    },
    'baseline_timeaware_mean': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'baseline_timeaware', '--pool', 'mean', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42'
        ],
        'name': 'baseline_timeaware_mean'
    },
    'baseline_timeaware_last_linear_decay': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'baseline_timeaware', '--pool', 'last', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42', '--decay_schedule', 'linear', '--decay_start', '0.995', '--decay_end', '0.99'
        ],
        'name': 'baseline_timeaware_last_linear_decay'
    },
    'timeaware_last_linear_decay': {
        'cmd': [
            'python', 'models/behavior_tower_experiments/run_experiments.py',
            '--tower', 'timeaware', '--pool', 'last', '--emb_dim', '128', '--epochs', '50', '--lr', '1e-3', '--batch_size', '256', '--seed', '42', '--decay_schedule', 'linear', '--decay_start', '0.995', '--decay_end', '0.99'
        ],
        'name': 'timeaware_last_linear_decay'
    },
}


CHECK_EPOCHS = [3,5,7,10,13,15,17,20,23,25,27,30,33,35,37,40,43,45,47,50]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', type=str, required=True, choices=list(PROFILES.keys()))
    p.add_argument('--out_csv', type=str, default='models/behavior_tower_experiments/result/profile_results.csv')
    return p.parse_args()


def main():
    args = parse_args()
    prof = PROFILES[args.profile]
    cmd = prof['cmd']
    # run training once (40 epochs) — run_experiments.py already trains and prints per-epoch loss
    print('Running:', ' '.join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout
    print(stdout)
    # parse per-epoch loss and metrics by pairing lines
    records = {}
    current_epoch = None
    for ln in stdout.splitlines():
        s = ln.strip()
        if s.startswith('Epoch') and 'AvgLoss=' in s:
            try:
                parts = s.split()  # ['Epoch', 'N', 'AvgLoss=VAL'] or with tab
                ep = int(parts[1])
                loss_val = float(s.split('AvgLoss=')[1])
                current_epoch = ep
                rec = records.get(ep, {'epoch': ep})
                rec['avg_loss'] = loss_val
                records[ep] = rec
            except Exception:
                current_epoch = None
        elif ('Recall@10=' in s) and ('NDCG@10=' in s):
            try:
                left = s.split('Eval Recall@10=')[1] if 'Eval Recall@10=' in s else s.split('Recall@10=')[1]
                toks = left.split('NDCG@10=')
                recall_val = float(toks[0].strip())
                ndcg_val = float(toks[1].strip())
                if current_epoch is not None:
                    rec = records.get(current_epoch, {'epoch': current_epoch})
                    rec['recall@10'] = recall_val
                    rec['ndcg@10'] = ndcg_val
                    records[current_epoch] = rec
            except Exception:
                pass
    # final metrics (optional)
    final_recall = None
    final_ndcg = None
    for ln in stdout.splitlines()[::-1]:
        s = ln.strip()
        if ('Recall@10=' in s) and ('NDCG@10=' in s):
            try:
                left = s.split('Eval Recall@10=')[1] if 'Eval Recall@10=' in s else s.split('Recall@10=')[1]
                toks = left.split('NDCG@10=')
                final_recall = float(toks[0].strip())
                final_ndcg = float(toks[1].strip())
                break
            except Exception:
                break
    # build rows for required epochs only
    rows = []
    for ep in CHECK_EPOCHS:
        rec = records.get(ep, {'epoch': ep})
        rec['profile'] = prof['name']
        rec['final_recall@10'] = final_recall
        rec['final_ndcg@10'] = final_ndcg
        rows.append(rec)
    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print('Saved results to', str(out))
    for ep in CHECK_EPOCHS:
        r = records.get(ep)
        if r is not None and ('recall@10' in r) and ('ndcg@10' in r):
            print(f"Epoch {ep}: Recall@10={r['recall@10']:.4f} NDCG@10={r['ndcg@10']:.4f}")


if __name__ == '__main__':
    main()
