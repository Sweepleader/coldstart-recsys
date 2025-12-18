import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
import csv
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

from multimodal_encoder_1216 import (
    VideoEncoder, 
    RobustAudioEncoder, 
    RobustTextEncoder, 
    MultiModalAttentionFusion,
    M3CSR_MultiModalEncoder
)

# ------------------------------------------------------------------------------
# 稳定性设置
# ------------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------------------------
# 可视化函数
# ------------------------------------------------------------------------------
def visualize_clusters(features, cluster_ids, item_ids, raw_titles, save_path="cluster_viz.png"):
    """
    使用 PCA 将特征降维到 2D 并绘制散点图。
    features: [N, dim]
    cluster_ids: [N]
    """
    # 转换为 numpy
    if isinstance(features, torch.Tensor):
        X = features.cpu().numpy()
    else:
        X = features
        
    if isinstance(cluster_ids, torch.Tensor):
        c_ids = cluster_ids.cpu().numpy()
    else:
        c_ids = cluster_ids
        
    # PCA 降维
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 10))
    
    # 定义颜色映射
    # 假设 k=3, 我们用 distinct colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 绘制散点
    unique_clusters = np.unique(c_ids)
    for cid in unique_clusters:
        mask = (c_ids == cid)
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1], 
            label=f'Cluster {cid}', 
            s=200, 
            alpha=0.8,
            edgecolors='w', 
            linewidth=2
        )
        
    # 添加标签
    for i, uid in enumerate(item_ids):
        x, y = X_2d[i, 0], X_2d[i, 1]
        
        # 处理标题，防止过长
        title = raw_titles[i]
        # 简单的截断逻辑
        display_title = f"ID:{uid}\n"
        if "#" in title:
            # 优先显示 tag
            tags = [t for t in title.split() if t.startswith("#")]
            display_title += " ".join(tags[:2])
        else:
            display_title += title[:15] + "..."
            
        plt.annotate(
            display_title, 
            (x, y), 
            xytext=(10, 10), 
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
    plt.title("M3CSR Multi-modal Feature Clustering (PCA 2D Projection)", fontsize=16)
    plt.xlabel(f"PCA Component 1 (Var: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"PCA Component 2 (Var: {pca.explained_variance_ratio_[1]:.2f})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n[Visualization] 聚类分布图已保存至: {save_path}")
    plt.close()

# ------------------------------------------------------------------------------
# 数据加载辅助函数
# ------------------------------------------------------------------------------

def load_video_frames_for_id(base_dir, item_id, num_frames=5):
    """
    为指定 ID 加载视频帧。
    在 base_dir 中寻找 {id}-{seq}.png 或 {id}-{seq}.jpg
    返回: [1, T, 3, H, W]
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    frames = []
    # 尝试找到对应的帧文件
    # 假设文件名格式为: item_id-seq.ext (例如 1-1.png)
    
    for i in range(1, num_frames + 1):
        # 尝试 png 和 jpg
        found = False
        for ext in ['png', 'jpg', 'jpeg']:
            fname = f"{item_id}-{i}.{ext}"
            fpath = os.path.join(base_dir, fname)
            if os.path.exists(fpath):
                try:
                    img = Image.open(fpath).convert('RGB')
                    frames.append(transform(img))
                    found = True
                    break
                except Exception as e:
                    print(f"  [Error] 加载帧 {fpath} 失败: {e}")
        
        if not found:
            # 如果中间某帧缺失，用全黑填充，或者复用上一帧
            # 这里简单用全黑
            frames.append(torch.zeros(3, 224, 224))
            
    if not frames:
        return torch.randn(1, num_frames, 3, 224, 224)
        
    return torch.stack(frames).unsqueeze(0)

def load_cover_for_id(base_dir, item_id):
    """
    加载封面: base_dir/{item_id}.jpg
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # 尝试多种扩展名
    for ext in ['jpg', 'png', 'jpeg']:
        path = os.path.join(base_dir, f"{item_id}.{ext}")
        if os.path.exists(path):
            try:
                img = Image.open(path).convert('RGB')
                return transform(img).unsqueeze(0)
            except:
                pass
                
    # 默认黑色
    return torch.zeros(1, 3, 224, 224)

def load_audio_for_id(base_dir, item_id):
    """
    加载音频: base_dir/{item_id}.wav
    """
    path = os.path.join(base_dir, f"{item_id}.wav")
    
    # 如果没有 wav，尝试查找 mp4 并提取 (这里仅做路径检查，不实际做复杂的 ffmpeg 提取，除非必要)
    # 为简单起见，如果找不到 wav，回退到随机/静音
    
    if os.path.exists(path):
        try:
            import torchaudio
            import torchaudio.transforms as T
            
            # 抑制 torchaudio 警告
            import warnings
            warnings.filterwarnings("ignore")

            waveform, sample_rate = torchaudio.load(path)
            
            if sample_rate != 16000:
                resampler = T.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            mel_transform = T.MelSpectrogram(sample_rate=16000, n_mels=64)
            mel_spec = mel_transform(waveform)
            
            target_time = 100
            current_time = mel_spec.shape[2]
            
            if current_time > target_time:
                mel_spec = mel_spec[:, :, :target_time]
            elif current_time < target_time:
                pad_amount = target_time - current_time
                pad = torch.zeros(1, 64, pad_amount)
                mel_spec = torch.cat([mel_spec, pad], dim=2)
                
            return mel_spec.unsqueeze(0) # [1, 1, 64, 100]
            
        except Exception as e:
            print(f"  [Warning] 音频加载失败 {path}: {e}")
            
    # 随机噪声回退
    return torch.randn(1, 1, 64, 100)

def load_titles(csv_path):
    """
    读取 titles.csv 返回 {id: title}
    """
    titles = {}
    if not os.path.exists(csv_path):
        print(f"警告: 找不到 titles.csv: {csv_path}")
        return titles
        
    try:
        # 使用 utf-8-sig 处理可能的 BOM
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            # 规范化 header: 去除空格
            if reader.fieldnames:
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
                
            for row in reader:
                # 再次清理 row keys (以防万一)
                clean_row = {k.strip(): v for k, v in row.items() if k}
                
                if 'item' in clean_row and 'title' in clean_row:
                    try:
                        uid = int(clean_row['item'])
                        titles[uid] = clean_row['title']
                    except ValueError:
                        pass
    except Exception as e:
        print(f"读取 titles.csv 失败: {e}")
        
    return titles

# ------------------------------------------------------------------------------
# 聚类算法 (自动选择最佳 K)
# ------------------------------------------------------------------------------
def perform_kmeans_auto(features, seed=42):
    """
    自动选择最佳 K 值的 K-Means 聚类 (基于 Silhouette Score)
    features: [N, dim]
    返回: labels [N], best_k
    """
    # 1. 转换为 Numpy
    if isinstance(features, torch.Tensor):
        X = features.cpu().numpy()
    else:
        X = features
        
    # 2. L2 归一化
    X_norm = normalize(X, norm='l2')
    
    N = X_norm.shape[0]
    
    # 3. 遍历寻找最佳 K
    best_score = -1.0
    best_k = 2
    best_labels = None
    
    # K 的范围：从 2 到 N-1 (Silhouette Score 需要至少 2 个簇，且不能每个样本一簇)
    max_k = min(N - 1, 6) # 对于只有7个样本，尝试到 6 即可
    
    if max_k < 2:
        # 样本太少，强制分为 2 类或直接返回全 0
        print("样本太少，无法自动选择 K，默认 k=2")
        kmeans = KMeans(n_clusters=2, random_state=seed, n_init=50)
        labels = kmeans.fit_predict(X_norm)
        return torch.tensor(labels, dtype=torch.long), 2
        
    print(f"自动寻找最佳 K 值 (范围 2-{max_k})...")
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=50)
        labels = kmeans.fit_predict(X_norm)
        
        score = silhouette_score(X_norm, labels)
        print(f"  k={k}: Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            
    print(f"✅ 选定最佳 K 值: {best_k} (Score: {best_score:.4f})")
    
    return torch.tensor(best_labels, dtype=torch.long), best_k

# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------
def main():
    # 固定随机种子
    set_seed(42)
    
    print("="*60)
    print("M3CSR 多模态处理流程 (Batch Processing - Auto Clustering)")
    print("="*60)
    
    # 1. 路径配置
    base_dir = "f:\\coldstart-recsys\\models\\mutibehavior\\test_file"
    frame_dir = os.path.join(base_dir, "test_frames_interval_1_number_5")
    cover_dir = os.path.join(base_dir, "test_covers")
    title_path = os.path.join(base_dir, "titles.csv")
    
    # 2. 准备数据 ID 列表 (1-7)
    item_ids = list(range(1, 8))
    
    # 加载标题映射
    title_map = load_titles(title_path)
    
    # 3. 初始化模型
    print("\n[初始化模型]")
    num_clusters = 5 # 假设聚成 5 类 (样本少，设小一点)
    model = M3CSR_MultiModalEncoder(num_clusters=num_clusters, unified_dim=256)
    model.eval()
    
    # 4. 收集数据 & 阶段 1: 基础特征提取
    print("\n[阶段 1: 数据加载与基础特征提取]")
    
    batch_data = [] # 存储 (video, audio, text, cover, item_id)
    base_embeddings = []
    
    for uid in item_ids:
        print(f"正在处理 Item ID: {uid} ...")
        
        # 加载各模态
        vid_input = load_video_frames_for_id(frame_dir, uid)
        aud_input = load_audio_for_id(base_dir, uid) # 假设 wav 在 test_file 根目录
        cov_input = load_cover_for_id(cover_dir, uid)
        
        # 文本
        raw_text = title_map.get(uid, "Unknown content")
        txt_input = [raw_text]
        
        # 存储输入以便阶段 3 使用
        batch_data.append({
            'id': uid,
            'video': vid_input,
            'audio': aud_input,
            'cover': cov_input,
            'text': txt_input,
            'raw_text': raw_text
        })
        
        # 推理 (不带 ID)
        with torch.no_grad():
            # 返回: fused, weights, bases
            base_fused, _, _ = model(vid_input, aud_input, txt_input, cluster_ids=None, cover=cov_input)
            base_embeddings.append(base_fused)
            
    # 堆叠所有基础特征 [N, dim]
    all_base_features = torch.cat(base_embeddings, dim=0) # [7, 128]
    print(f"\n所有 Item 基础特征提取完毕: {all_base_features.shape}")
    
    # 5. 阶段 2: 全局聚类 (Global Clustering)
    print("\n[阶段 2: 全局语义聚类 (模拟 Offline 过程)]")
    # 使用 K-Means 将 items 分组
    
    # 自动选择最佳 K
    cluster_assignments, best_k = perform_kmeans_auto(all_base_features) 
    print("聚类结果:")
    
    # 收集标题用于可视化
    all_raw_titles = [item['raw_text'] for item in batch_data]
    
    for idx, cid in enumerate(cluster_assignments):
        uid = batch_data[idx]['id']
        title = batch_data[idx]['raw_text'][:30] + "..."
        print(f"  Item {uid} ({title}) -> Cluster {cid.item()}")
        
    # 可视化
    viz_path = os.path.join(base_dir, "cluster_viz.png")
    visualize_clusters(all_base_features, cluster_assignments, item_ids, all_raw_titles, save_path=viz_path)
        
    # 6. 阶段 3: 协同语义增强 (Final Representation)
    print("\n[阶段 3: 协同语义增强 (M3CSR Final Inference)]")
    
    for idx, item in enumerate(batch_data):
        uid = item['id']
        cid = cluster_assignments[idx].unsqueeze(0) # [1]
        
        with torch.no_grad():
            final_vec, weights = model(
                item['video'], 
                item['audio'], 
                item['text'], 
                cid, 
                cover=item['cover']
            )
            
        w = weights.tolist()[0]
        # 打印详细结果
        print(f"\nItem {uid} 最终增强向量: {final_vec.shape}")
        print(f"  模态权重: Video={w[0]:.3f}, Audio={w[1]:.3f}, Text={w[2]:.3f}")
        
        # 简单判定主导模态
        modes = ['Video', 'Audio', 'Text']
        max_idx = w.index(max(w))
        print(f"  主导模态: {modes[max_idx]}")

if __name__ == "__main__":
    main()


'''

运行: python test_real_data.py
输出:
============================================================
M3CSR 多模态处理流程 (Batch Processing - Auto Clustering)
============================================================

[初始化模型]
INFO:multimodal_encoder_1216:M3CSR: 使用 VGGishAudioEncoder
INFO:multimodal_encoder_1216:Added VGGish path: F:\coldstart-recsys\models\VGGish\hub\harritaylor_torchvggish_master
INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: cuda:0
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: F:/coldstart-recsys/models/SBERT/all-MiniLM-L6-v2
INFO:multimodal_encoder_1216:VideoEncoder 已成功冻结 (Parameters frozen + Eval mode).
INFO:multimodal_encoder_1216:AudioEncoder 已成功冻结 (Parameters frozen + Eval mode).
INFO:multimodal_encoder_1216:TextEncoder 已成功冻结 (Parameters frozen + Eval mode).

[阶段 1: 数据加载与基础特征提取]
正在处理 Item ID: 1 ...
Batches:   0%|                                                        | 0/1 [00:00<?, ?it/s]D:\CodeTools\Anaconda\envs\torch\lib\site-packages\transformers\models\bert\modeling_bert.py:413: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:555.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
Batches: 100%|████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.26it/s]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 107.99it/s]
正在处理 Item ID: 2 ...
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 147.95it/s]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 128.70it/s]
正在处理 Item ID: 3 ...
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 166.41it/s]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 166.20it/s] 
正在处理 Item ID: 4 ...
Batches: 100%|████████████████████████████████████████████████| 1/1 [00:00<00:00, 98.00it/s]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 120.09it/s] 
正在处理 Item ID: 5 ...
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 100.45it/s]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 113.95it/s] 
正在处理 Item ID: 6 ...
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 126.95it/s]
Batches: 100%|████████████████████████████████████████████████| 1/1 [00:00<00:00, 89.10it/s] 
正在处理 Item ID: 7 ...
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 156.05it/s]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 117.15it/s] 

所有 Item 基础特征提取完毕: torch.Size([7, 256])

[阶段 2: 全局语义聚类 (模拟 Offline 过程)]
自动寻找最佳 K 值 (范围 2-6)...
  k=2: Silhouette Score = 0.1800
  k=3: Silhouette Score = 0.1753
  k=4: Silhouette Score = 0.1317
  k=5: Silhouette Score = 0.0420
  k=6: Silhouette Score = 0.0048
✅ 选定最佳 K 值: 2 (Score: 0.1800)
聚类结果:
  Item 1 (🎧 ASMR Cats Grooming  #asmr #A...) -> Cluster 0
  Item 2 (Rain Sound On Window with Thun...) -> Cluster 1
  Item 3 (Relaxing Snowfall - Sound of L...) -> Cluster 1
  Item 4 (This is a text document for te...) -> Cluster 1
  Item 5 (Dare you to laugh?! #cat #kitt...) -> Cluster 0
  Item 6 (Oh no, fallen into a human tra...) -> Cluster 0
  Item 7 (Sonar Cat #cat #cute #kitty #k...) -> Cluster 0

[Visualization] 聚类分布图已保存至: f:\coldstart-recsys\models\mutibehavior\test_file\cluster_viz.png

[阶段 3: 协同语义增强 (M3CSR Final Inference)]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 102.80it/s]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 117.76it/s]

Item 1 最终增强向量: torch.Size([1, 256])
  模态权重: Video=0.460, Audio=0.288, Text=0.252
  主导模态: Video
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 119.98it/s]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 100.63it/s]

Item 2 最终增强向量: torch.Size([1, 256])
  模态权重: Video=0.284, Audio=0.235, Text=0.481
  主导模态: Text
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 117.56it/s]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 116.54it/s]

Item 3 最终增强向量: torch.Size([1, 256])
  模态权重: Video=0.283, Audio=0.234, Text=0.483
  主导模态: Text
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 133.23it/s]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 122.72it/s]

Item 4 最终增强向量: torch.Size([1, 256])
  模态权重: Video=0.281, Audio=0.236, Text=0.483
  主导模态: Text
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 122.37it/s]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 114.04it/s] 

Item 5 最终增强向量: torch.Size([1, 256])
  模态权重: Video=0.464, Audio=0.287, Text=0.249
  主导模态: Video
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 112.58it/s]
Batches: 100%|████████████████████████████████████████████████| 1/1 [00:00<00:00, 82.78it/s]

Item 6 最终增强向量: torch.Size([1, 256])
  模态权重: Video=0.465, Audio=0.286, Text=0.249
  主导模态: Video
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 139.55it/s]
Batches: 100%|███████████████████████████████████████████████| 1/1 [00:00<00:00, 137.27it/s] 

Item 7 最终增强向量: torch.Size([1, 256])
  模态权重: Video=0.464, Audio=0.287, Text=0.249
  主导模态: Video

'''