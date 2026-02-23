import os
import json
import h5py
import random
import argparse
from collections import defaultdict
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Generate train/valid JSON with Stratified Split (View + Multi-Label)")
    
    parser.add_argument("--root", type=str, default="data", help="Dataset root path")
    parser.add_argument("--valid_ratio", type=float, default=0.2, help="Validation set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    return parser.parse_args()

def safe_get_scalar(dataset_val):
    """提取单个标量 (用于 View ID)"""
    if isinstance(dataset_val, (int, np.integer)):
        return int(dataset_val)
    if isinstance(dataset_val, np.ndarray):
        return int(dataset_val.flatten()[0])
    if isinstance(dataset_val, (list, tuple)):
        return int(dataset_val[0])
    return int(dataset_val)

def safe_get_label_tuple(dataset_val):
    """
    将 (7,) 的 numpy label 转换为 tuple，作为字典的 key
    例如: array([0, 1, 0...]) -> (0, 1, 0...)
    """
    if isinstance(dataset_val, np.ndarray):
        # 展平并转为 tuple，确保 hashable
        return tuple(dataset_val.flatten().tolist())
    elif isinstance(dataset_val, list):
        return tuple(dataset_val)
    else:
        # 如果万一是标量
        return (int(dataset_val),)

def stratified_split_by_view_and_multilabel(filenames, views, labels, valid_ratio=0.2):
    """
    根据 (View, Label_Combination) 进行分层
    """
    group_dict = defaultdict(list)

    # 1. 分组
    for name, view, label in zip(filenames, views, labels):
        v_int = safe_get_scalar(view)
        l_tuple = safe_get_label_tuple(label)  # 转为 tuple 才能做字典 key
        
        key = (v_int, l_tuple)
        group_dict[key].append(name)

    train_files = []
    valid_files = []

    print(f"\n[Info] Splitting data with Stratified Sampling (View + Label Combination)")
    print(f"Validation Ratio: {valid_ratio:.0%}")
    print("-" * 120)
    print(f"{'View':<6} | {'Label Combination (Length 7)':<35} | {'Total':<8} | {'Train':<8} | {'Valid':<8} | {'Valid %':<8}")
    print("-" * 120)

    # 排序 keys (tuple 默认可排序)
    sorted_keys = sorted(group_dict.keys())

    for key in sorted_keys:
        view, l_tuple = key
        files = group_dict[key]
        n_total = len(files)
        
        n_valid = int(n_total * valid_ratio)
        
        # 随机打乱
        random.shuffle(files)
        
        current_valid = files[:n_valid]
        current_train = files[n_valid:]
        
        valid_files.extend(current_valid)
        train_files.extend(current_train)
        
        # 简单的格式化 label tuple 为字符串以便打印
        label_str = str(l_tuple).replace(" ", "")
        actual_ratio = len(current_valid) / n_total if n_total > 0 else 0
        
        print(f"{view:<6} | {label_str:<35} | {n_total:<8} | {len(current_train):<8} | {len(current_valid):<8} | {actual_ratio:<8.1%}")

    print("-" * 120)
    print(f"{'ALL':<44} | {len(filenames):<8} | {len(train_files):<8} | {len(valid_files):<8} | {len(valid_files)/len(filenames):<8.1%}")
    print("-" * 120 + "\n")
    
    return train_files, valid_files

if __name__ == "__main__":
    args = get_args()

    dataset_root_path = args.root
    images_dir_path = os.path.join(dataset_root_path, 'images')
    labels_dir_path = os.path.join(dataset_root_path, 'labels')

    if not os.path.exists(images_dir_path) or not os.path.exists(labels_dir_path):
        print(f"Error: Data paths not found.\nImages: {images_dir_path}\nLabels: {labels_dir_path}")
        exit(1)

    all_image_filenames = [name for name in os.listdir(images_dir_path) if name.endswith('.h5')]
    all_labeled_filenames = [name.replace('_label', '') for name in os.listdir(labels_dir_path) if name.endswith('.h5')]
    all_unlabeled_filenames = [name for name in all_image_filenames if name not in all_labeled_filenames]

    print(f"Reading info from {len(all_labeled_filenames)} labeled files...")
    all_views = []
    all_labels = []

    # 读取数据
    for filename in all_labeled_filenames:
        img_path = os.path.join(images_dir_path, filename)
        with h5py.File(img_path, 'r') as f:
            all_views.append(f['view'][:]) 

        lbl_path = os.path.join(labels_dir_path, filename.replace('.h5', '_label.h5'))
        with h5py.File(lbl_path, 'r') as f:
            # 直接读取整个 (7,) 数组，不转 int
            all_labels.append(f['label'][:]) 

    random.seed(args.seed)

    # 执行基于组合的分层划分
    train_files, valid_files = stratified_split_by_view_and_multilabel(
        all_labeled_filenames, 
        all_views, 
        all_labels, 
        valid_ratio=args.valid_ratio
    )

    # -------------------------------------------------
    # 保存 JSON
    # -------------------------------------------------
    def create_dataset_list(filenames):
        dataset_list = []
        for name in filenames:
            img_path = os.path.abspath(os.path.join(images_dir_path, name))
            lbl_path = os.path.abspath(os.path.join(labels_dir_path, name.replace('.h5', '_label.h5')))
            
            with h5py.File(img_path, 'r') as f:
                view_id = safe_get_scalar(f['view'][:])

            dataset_list.append({
                'image': img_path,
                'label': lbl_path,
                'view_id': view_id
            })
        return dataset_list

    valid_dataset_list = create_dataset_list(valid_files)
    train_labeled_dataset_list = create_dataset_list(train_files)

    train_unlabeled_dataset_list = []
    for name in all_unlabeled_filenames:
        img_path = os.path.abspath(os.path.join(images_dir_path, name))
        with h5py.File(img_path, 'r') as f:
            view_id = safe_get_scalar(f['view'][:])
            
        train_unlabeled_dataset_list.append({
            'image': img_path,
            'label': None,
            'view_id': view_id
        })

    print(f"Saving JSON files to {dataset_root_path}...")
    with open(os.path.join(dataset_root_path, 'train_labeled.json'), 'w') as f:
        json.dump(train_labeled_dataset_list, f, indent=4)
    with open(os.path.join(dataset_root_path, 'train_unlabeled.json'), 'w') as f:
        json.dump(train_unlabeled_dataset_list, f, indent=4)
    with open(os.path.join(dataset_root_path, 'valid.json'), 'w') as f:
        json.dump(valid_dataset_list, f, indent=4)
        
    print("Done.")