# 双塔推荐模型（MovieLens-100K）基础实现总结

## 概览
- 本目录提供一个简洁、可运行的双塔（Two-Tower）推荐模型基线，结合 MovieLens-100K 数据集进行训练与评估。
- 模型使用用户塔与物品塔的嵌入向量，经小型 MLP 与 L2 归一化后，通过点积计算匹配分数；训练采用 BPR 成对排序损失；评估采用负采样近似的 Recall@k / NDCG@k。

## 代码结构
- `data_utils/data_utils.py`
  - 读取交互数据并构造 `ML100KDataset`（`models/test/data_utils/data_utils.py:12-66`）。
  - 简单均匀负采样；`load_num_items` 解析 `u.item` 获取物品总数（`models/test/data_utils/data_utils.py:68-81`）。
- `model/model.py`
  - `TwoTower` 模型：用户与物品 `Embedding` + 小型 `MLP` + `F.normalize`，点积得分（`models/test/model/model.py:6-45`）。
  - 提供用户/物品向量导出（`models/test/model/model.py:47-57`）。
- `eval_utils/eval_utils.py`
  - 负采样评估 `evaluate`：为每个正例采样若干负例，计算并排名，统计指标（`models/test/eval_utils/eval_utils.py:6-44`）。
- `train.py`
  - `bpr_loss` 与训练/评估主流程（`models/test/train.py:12-20`, `models/test/train.py:63-94`）。

## 数据集与路径
- 期望文件位置（默认参数可直接运行）：
  - 训练：`data/ml-100k/u1.base`
  - 测试：`data/ml-100k/u1.test`
  - 物品：`data/ml-100k/u.item`
- 数据文件为制表符分隔，原始 ID 为 1-based，代码内部转换为 0-based（`models/test/data_utils/data_utils.py:16-18, 23-26`）。

## 训练配置
- 关键参数（来源于 `argparse`）：`emb_dim=64`、`batch_size=1024`、`epochs=3`、`lr=1e-3`、`neg=1`（`models/test/train.py:38-43`）。
- 设备选择：自动使用 `cuda` 或 `cpu`（`models/test/train.py:59`）。
- 优化器：Adam（`models/test/train.py:61`）。

## 评估说明
- 负采样评估（近似离线）：每个测试对采样 `num_neg=100` 个负例，计算正例与负例分数并排名；`k=10`（`models/test/eval_utils/eval_utils.py:6-16, 21-44`）。
- 指标：
  - `Recall@k`：正例是否位于前 k。
  - `NDCG@k`：正例的折损增益（单一相关项场景下为 `1/log2(rank+2)`）。

## 运行示例与结果
- 运行：
  - `python models/test/train.py`
  - 或指定参数：`python models/test/train.py --data_path data/ml-100k/u1.base --test_path data/ml-100k/u1.test --item_path data/ml-100k/u.item --emb_dim 64 --batch_size 1024 --epochs 3 --lr 1e-3 --neg 1`
- 示例日志（用户提供）：
  - `Epoch 1  AvgLoss=0.563942`，`Recall@10=0.3775`，`NDCG@10=0.1780`
  - `Epoch 2  AvgLoss=0.500776`，`Recall@10=0.3945`，`NDCG@10=0.1894`
  - `Epoch 3  AvgLoss=0.499099`，`Recall@10=0.4098`，`NDCG@10=0.2022`
- 结论：损失下降、指标提升，训练与评估流程正常，模型作为基线是“完整且可用”的。

## 已知局限与改进方向
- 多负样本维度对齐：`TwoTower.forward` 文档支持 `(B, N, D)` 输入，但当前点积在 `i` 为 3D 时需要将 `u` 扩展到 `(B, N, D)` 再求和（现默认 `neg=1` 不触发）。
- 评估近似性：负采样评估未严格排除训练集所有正例，指标为近似估计；可改为对用户的全物品排名或更严格的过滤。
- 复现实验：建议统一随机种子并保存 `args`/模型权重，便于复现与对比。
- 正则化与建模：可加入权重衰减、dropout、更深 MLP，或引入多模态/内容特征。

## 快速定位
- 数据集类：`models/test/data_utils/data_utils.py:12-66`
- 模型：`models/test/model/model.py:6-45`
- 评估：`models/test/eval_utils/eval_utils.py:6-44`
- 训练入口：`models/test/train.py:45-97`

## 致谢
- 数据集：MovieLens-100K（GroupLens Research）。
