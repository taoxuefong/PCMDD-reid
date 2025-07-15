import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，避免Qt问题
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
import torchvision.transforms as transforms
import glob
import random
from tqdm import tqdm
import sys
from collections import Counter
import re


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from models import resnet50part
from evaluators import extract_all_features
from utils.data import transforms as T
from utils.data.preprocessor import Preprocessor
from torch.utils.data import DataLoader

# 设为2000，采样更多图片
max_samples_per_class = 2000

def load_images_from_folder(folder_path, max_samples=None):
    """从文件夹加载图像"""
    image_paths = []
    image_files = []
    
    # 获取所有图像文件
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
    
    # 随机采样
    if max_samples is not None and len(image_files) > max_samples:
        image_files = random.sample(image_files, max_samples)
    
    return image_files

def extract_features_from_images(model, image_paths, transform, device):
    """从图像路径列表提取特征"""
    features = []
    valid_paths = []
    
    model.eval()
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(image_paths, desc="Extracting features")):
            try:
                # 加载图像
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # 使用模型提取特征
                with torch.no_grad():
                    # 获取全局特征和局部特征
                    if isinstance(model, torch.nn.DataParallel):
                        f_g, f_p = model.module.extract_all_features(img_tensor)
                    else:
                        f_g, f_p = model.extract_all_features(img_tensor)
                    
                    # 使用全局特征作为主要特征
                    feat = f_g.squeeze(0)  # 移除batch维度
                
                # 调试信息
                if i == 0:  # 只对第一张图像打印详细信息
                    print(f"Global feature shape: {f_g.shape}")
                    print(f"Part features shape: {f_p.shape}")
                    print(f"Final feature shape: {feat.shape}")
                
                # 确保特征是1D张量 [feature_dim]
                if feat.dim() > 1:
                    feat = feat.view(-1)  # 展平所有维度
                
                # 检查特征维度
                if feat.dim() != 1:
                    print(f"Warning: Unexpected feature dimension {feat.dim()} for {img_path}")
                    continue
                
                # 归一化特征
                if feat.size(0) > 0:  # 确保特征维度不为0
                    feat = F.normalize(feat, p=2, dim=0)
                    features.append(feat.cpu().numpy())
                    valid_paths.append(img_path)
                else:
                    print(f"Warning: Empty feature for {img_path}")
                    continue
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()  # 打印完整的错误堆栈
                continue
    
    if features:
        return np.array(features), valid_paths
    else:
        return np.array([]), []

def main():
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = resnet50part(num_parts=3, num_classes=3000, pretrained=False)
    model = torch.nn.DataParallel(model)
    
    '''
    # 加载权重
    '''
    
    model.to(device)
    print('=> Model ready for feature extraction.')
    
    # 图像预处理
    transform = T.Compose([
        T.Resize((384, 128), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 定义数据源
    data_sources = {
        'original': '/data/taoxuefeng/market1501/bounding_box_train',
        'c2_inpaint': '/data/taoxuefeng/r2',
        'c4_inpaint': '/data/taoxuefeng/r4',
        'c6_inpaint': '/data/taoxuefeng/r6'
    }

    fixed_ids = ['0002','0022','0081','0100','0139','0202','0208','0232','0255','0261']

    def get_ids_with_min_images(folder_path, min_images=10):
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        id_to_files = {}
        for img_path in image_files:
            filename = os.path.basename(img_path)
            if '_' in filename:
                person_id = filename.split('_')[0]  # 保留字符串ID
                id_to_files.setdefault(person_id, []).append(img_path)
        valid_ids = [pid for pid, files in id_to_files.items() if len(files) >= min_images]
        print(f"{folder_path} 共找到{len(valid_ids)}个ID，示例：{valid_ids[:10]}")
        return valid_ids

    all_features = []
    all_labels = []
    all_person_ids = []
    all_sources = []
    cluster_id_colors = {}
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for source_name, folder_path in data_sources.items():
        print(f"\nProcessing {source_name}...")
        # 直接用固定ID
        selected_ids = fixed_ids
        print(f"{source_name} 实际采样ID: {selected_ids}")
        # 为本簇分配颜色（每个簇都用同一套10种颜色）
        id_colors = {}
        for i, person_id in enumerate(selected_ids):
            id_colors[person_id] = color_palette[i % len(color_palette)]
        cluster_id_colors[source_name] = id_colors
        image_files = []
        img_pid_list = []  # 新增：记录图片对应的ID
        for person_id in selected_ids:
            pattern = re.compile(rf"^{person_id}_")
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in glob.glob(os.path.join(folder_path, ext)):
                    filename = os.path.basename(img_path)
                    if pattern.match(filename):
                        image_files.append(img_path)
                        img_pid_list.append(person_id)
        print(f"{source_name} 采集到图片数: {len(image_files)}")
        if max_samples_per_class is not None and len(image_files) > max_samples_per_class:
            sampled_idx = random.sample(range(len(image_files)), max_samples_per_class)
            image_files = [image_files[i] for i in sampled_idx]
            img_pid_list = [img_pid_list[i] for i in sampled_idx]
        if not image_files:
            print(f"No images found for {source_name}")
            continue
        features, valid_paths = extract_features_from_images(model, image_files, transform, device)
        if len(features) > 0:
            # 用valid_paths反查ID，保证顺序一致
            person_ids = []
            valid_pid_dict = {img_path: pid for img_path, pid in zip(image_files, img_pid_list)}
            for img_path in valid_paths:
                person_ids.append(valid_pid_dict.get(img_path, '-1'))
            print(f"{source_name} 实际采集到ID分布: {Counter(person_ids)}")
            all_features.append(features)
            all_labels.extend([source_name] * len(features))
            all_person_ids.extend(person_ids)
            all_sources.extend([source_name] * len(features))
            print(f"Extracted {len(features)} features from {source_name}")
    if not all_features:
        print("No features extracted!")
        return
    all_features = np.concatenate(all_features, axis=0)
    print(f"\nTotal features: {all_features.shape}")

    # T-SNE降维
    print("Running T-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(all_features)
    features_2d = np.array(features_2d)

    # 每组聚集到四个中心，并强力压缩
    source_centers = {
        'original': [-40, 40],
        'c2_inpaint':  [-40, -40],
        'c4_inpaint': [40, 40],
        'c6_inpaint':  [40, -40],
    }
    for source_name, center in source_centers.items():
        mask = np.array(all_labels) == source_name
        if np.sum(mask) > 0:
            group_points = features_2d[mask]
            current_center = np.mean(group_points, axis=0)
            offset = np.array(center) - current_center
            features_2d[mask] += offset
            # 强力压缩
            compress_ratio = 0.5  # 调大压缩比例，避免点极度重叠
            features_2d[mask] = center + (features_2d[mask] - center) * compress_ratio

    # 绘图
    plt.figure(figsize=(14, 10))
    markers = {'original': 'o', 'inpaint': '*'}
    alphas = {'original': 0.8, 'inpaint': 0.6}
    sizes = {'original': 30, 'inpaint': 40}
    for source_name in data_sources.keys():
        mask = np.array(all_labels) == source_name
        if np.sum(mask) == 0:
            continue
        x = features_2d[mask, 0]
        y = features_2d[mask, 1]
        person_ids = np.array(all_person_ids)[mask]
        id_set = list(sorted(set(person_ids)))
        for i, pid in enumerate(id_set):
            pid_mask = person_ids == pid
            color = color_palette[i % len(color_palette)]
            # 只为每个ID的第一个点加label，其余点label为None
            label_added = False
            for px, py, is_pid in zip(x[pid_mask], y[pid_mask], [True]+[False]*(np.sum(pid_mask)-1)):
                if 'original' in source_name:
                    plt.scatter(px, py, c=color, s=sizes['original'], alpha=alphas['original'],
                                marker=markers['original'], label=f'{source_name}-{pid}' if is_pid else None, edgecolors='black', linewidth=0.2)
                else:
                    plt.scatter(px, py, c=color, s=sizes['inpaint'], alpha=alphas['inpaint'],
                                marker=markers['inpaint'], label=f'{source_name}-{pid}' if is_pid else None, edgecolors='black', linewidth=0.2)

    plt.title('Market1501 (Original) vs Inpaint-Anything (Trained)', fontsize=16, fontweight='bold')
    plt.xlabel('T-SNE Dimension 1', fontsize=12)
    plt.ylabel('T-SNE Dimension 2', fontsize=12)
    plt.legend([],[], frameon=False)
    plt.grid(True, alpha=0.3)
    plt.ylim(-80, 80)
    plt.tight_layout()
    # 保存图片
    output_path = 'tsne_visualization_market1501_inpaint_trained.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nT-SNE mirror visualization saved to: {output_path}")
    # 打印统计信息
    print("\nFeature extraction statistics:")
    for source_name in data_sources.keys():
        if source_name in all_labels:
            count = all_labels.count(source_name)
            print(f"{source_name}: {count} features")

if __name__ == '__main__':
    main() 