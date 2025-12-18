import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_result_csvs(result_dir: Path):
    rows = []
    for p in sorted(result_dir.glob('results_*.csv')):
        try:
            df = pd.read_csv(p)
            m = re.search(r'_(\d+)epochs', p.name)
            ep = int(m.group(1)) if m else None
            df['epochs'] = ep
            df['file'] = p.name
            rows.append(df)
        except Exception:
            continue
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def plot_metrics(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if 'tower' in df.columns and 'epochs' in df.columns:
        for metric in ['recall', 'ndcg']:
            plt.figure()
            for name, g in df.groupby('tower'):
                g2 = g.sort_values('epochs')
                plt.plot(g2['epochs'], g2[metric], marker='o', label=str(name))
            plt.xlabel('epochs')
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f'{metric}_vs_epochs.png')
            plt.close()
    plt.figure()
    df2 = df.copy()
    df2 = df2[df2['epochs'].notna()]
    df2 = df2.sort_values(['epochs','tower'])
    df2['label'] = df2['tower'].astype(str) + '_' + df2['epochs'].astype(int).astype(str)
    plt.bar(df2['label'], df2['recall'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('recall')
    plt.tight_layout()
    plt.savefig(out_dir / 'recall_bars.png')
    plt.close()
    plt.figure()
    plt.bar(df2['label'], df2['ndcg'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('ndcg')
    plt.tight_layout()
    plt.savefig(out_dir / 'ndcg_bars.png')
    plt.close()


def plot_losses(loss_csv: Path, out_dir: Path):
    if not loss_csv.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(loss_csv)
    plt.figure()
    plt.plot(df['epoch'], df['loss'], marker='o')
    plt.xlabel('epoch')
    plt.ylabel('avg_loss')
    plt.tight_layout()
    plt.savefig(out_dir / 'loss_curve.png')
    plt.close()


def main():
    root = Path(__file__).resolve().parent
    result_dir = root / 'result'
    plots_dir = result_dir / 'plots'
    df = load_result_csvs(result_dir)
    plot_metrics(df, plots_dir)
    loss_csv = result_dir / 'baseline_losses_13epochs.csv'
    plot_losses(loss_csv, plots_dir)


if __name__ == '__main__':
    main()
