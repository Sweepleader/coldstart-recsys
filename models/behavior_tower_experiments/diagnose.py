# diagnose.py
import pandas as pd
import numpy as np
import torch

def main():
    # 1. 检查数据
    print("=== 数据检查 ===")
    item_df = pd.read_csv('F:/coldstart-recsys/data/ml-100k/u.item', 
                         sep='|', header=None, encoding='latin-1', engine='python')
    num_items = int(item_df.iloc[:, 0].max())
    print(f"物品文件中的最大物品ID: {num_items}")
    
    train_df = pd.read_csv('F:/coldstart-recsys/data/ml-100k/u1.base', 
                          sep='\t', names=['user','item','rating','ts'])
    num_users = int(train_df['user'].max())
    print(f"训练集中的最大用户ID: {num_users}")
    print(f"训练集中的最大物品ID: {train_df['item'].max()}")
    
    # 2. 检查历史构建
    from run_experiments import build_histories_with_ts
    histories_items, histories_ts = build_histories_with_ts(
        'F:/coldstart-recsys/data/ml-100k/u1.base', num_users
    )
    
    print(f"\n=== 历史序列检查 ===")
    print(f"构建了 {len(histories_items)} 个用户的历史")
    
    # 检查几个用户
    for u in [0, 1, 2]:
        if u in histories_items:
            print(f"用户 {u}: {len(histories_items[u])} 个交互")
    
    # 3. 检查测试集
    test_df = pd.read_csv('F:/coldstart-recsys/data/ml-100k/u1.test', 
                         sep='\t', names=['user','item','rating','ts'])
    print(f"\n=== 测试集检查 ===")
    print(f"测试集大小: {len(test_df)}")
    
    # 检查测试用户是否在训练集中
    test_users = set(test_df['user'] - 1)
    train_users = set(range(num_users))
    
    missing = test_users - train_users
    print(f"测试集中不在训练集中的用户数: {len(missing)}")
    
    if len(missing) > 0:
        print(f"前5个缺失用户: {list(missing)[:5]}")
        # 这些用户会被跳过，影响评估结果
    
    # 4. 简单模拟评估
    print(f"\n=== 评估模拟 ===")
    total_samples = 0
    skipped = 0
    
    for _, r in test_df.iterrows():
        u = int(r['user']) - 1
        if u not in histories_items or len(histories_items[u]) == 0:
            skipped += 1
        total_samples += 1
    
    print(f"总测试样本: {total_samples}")
    print(f"因无历史跳过的样本: {skipped}")
    print(f"有效评估样本: {total_samples - skipped}")
    print(f"跳过比例: {skipped/total_samples:.2%}")

if __name__ == '__main__':
    main()