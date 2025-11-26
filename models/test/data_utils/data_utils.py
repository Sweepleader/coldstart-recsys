# data_utils.py
# MovieLens-100K 数据集加载器（支持 u.data / u1.base 格式）
# 生成 BPR 训练所需的 (用户, 正例物品, 负例物品) 三元组

import random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ML100KDataset(Dataset):
    """
    生成 (用户, 正例物品, 负例物品) 三元组的数据集类。

    预期文件格式：每行 -> 用户ID  物品ID  评分  时间戳（制表符分隔）
    文件使用 1-based 编号（从1开始）；本类在内部会转换为 0-based 索引（从0开始）。

    为简化实现，本版本采用均匀负采样策略。
    """
    def __init__(self, data_path, num_items=None, negative_sampling=1):
        # 读取交互数据，转换为 (用户, 物品) 列表格式
        df = pd.read_csv(data_path, sep='\t', names=['user','item','rating','ts'], engine='python')
        # 转换为 0-based 索引（减去1）
        self.interactions = [(int(r['user'])-1, int(r['item'])-1) for _, r in df.iterrows()]
        # 构建用户的正例物品集合，用于快速负采样（避免采样到正例）
        self.user_pos = defaultdict(set)
        for u, i in self.interactions:
            self.user_pos[u].add(i)
        # 推断用户数量（取最大用户ID +1，因已转为0-based）
        self.num_users = max(u for u,_ in self.interactions) + 1
        # 物品数量可通过参数指定（来自 u.item 文件）或从数据中推断
        if num_items is None:
            self.num_items = max(i for _,i in self.interactions) + 1
        else:
            self.num_items = num_items
        # 每个正例对应的负采样数量（默认1）
        self.negative_sampling = negative_sampling


    def __len__(self):
        # 数据集长度 = 交互数据的条数
        return len(self.interactions)
        

    def __getitem__(self, idx):
        """
        返回值：
          当 negative_sampling==1 时：(用户ID, 正例物品ID, 负例物品ID)
          否则：(用户ID, 正例物品ID, [负例物品ID1, 负例物品ID2, ...])
        """
        u, pos = self.interactions[idx]
        negs = []
        # 按指定数量进行负采样
        for _ in range(self.negative_sampling):
            # 随机采样一个物品作为候选负例
            neg = random.randrange(self.num_items)
            # 确保负例不在该用户的正例集合中（重新采样直到满足条件）
            while neg in self.user_pos[u]:
                neg = random.randrange(self.num_items)
            negs.append(neg)
        # 根据负采样数量返回不同格式
        if self.negative_sampling == 1:
            return u, pos, negs[0]
        else:
            return u, pos, negs

def load_num_items(item_path):
    """
    解析 MovieLens 数据集的 u.item 文件，获取物品总数。
    u.item 文件使用 '|' 作为分隔符，物品ID位于第一列。
    """
    try:
        # 读取 u.item 文件（指定编码为 latin-1 以处理特殊字符）
        df = pd.read_csv(item_path, sep='|', header=None, encoding='latin-1', engine='python')
        # 取第一列（物品ID）的最大值作为物品总数
        max_id = df.iloc[:,0].max()
        return int(max_id)
    except Exception:
        # 若读取失败（文件不存在/格式错误），返回 None
        return None
