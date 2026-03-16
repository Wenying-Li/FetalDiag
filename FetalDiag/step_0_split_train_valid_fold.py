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
    """将 (7,) 的 numpy label 转换为 tuple，作为字典的 key"""
    if isinstance(dataset_val, np.ndarray):
        return tuple(dataset_val.flatten().tolist())
    elif isinstance(dataset_val, list):
        return tuple(dataset_val)
    else:
        return (int(dataset_val),)

def stratified_split_by_view_and_multilabel(filenames, views, labels, valid_ratio=0.2):
    """根据 (View, Label_Combination) 进行分层"""
    group_dict = defaultdict(list)
    name_to_label = {}  

    # 1. 分组录入
    for name, view, label in zip(filenames, views, labels):
        v_int = safe_get_scalar(view)
        l_tuple = safe_get_label_tuple(label)
        
        key = (v_int, l_tuple)
        group_dict[key].append(name)
        name_to_label[name] = l_tuple  

    train_files = []
    valid_files = []

    # 2. 初始按比例分配
    sorted_keys = sorted(group_dict.keys())
    for key in sorted_keys:
        files = group_dict[key].copy()
        n_total = len(files)
        
        n_valid = int(n_total * valid_ratio)
        
        # 【核心优化】：只要某个组合总数 >= 2，但按比例算下来是 0，强行给验证集分 1 个
        # 这样能避免绝大多数少数类在第一轮分配时 Validation 变为 0
        if n_valid == 0 and n_total >= 2:
            n_valid = 1
            
        random.shuffle(files)
        
        current_valid = files[:n_valid]
        current_train = files[n_valid:]
        
        valid_files.extend(current_valid)
        train_files.extend(current_train)

    # 3. 全局后处理检查 (应对极端情况，例如某个类别的所有组合 total 都等于 1)
    adjustment_logs = []
    if filenames:
        num_classes = len(name_to_label[filenames[0]])
        val_tp_counts = [0] * num_classes
        
        for f in valid_files:
            lbl = name_to_label[f]
            for c in range(num_classes):
                if lbl[c] > 0:
                    val_tp_counts[c] += 1
        
        missing_classes = [c for c in range(num_classes) if val_tp_counts[c] == 0]
        
        if missing_classes:
            for c in missing_classes:
                if val_tp_counts[c] > 0:
                    continue
                found = False
                for i, f in enumerate(train_files):
                    lbl = name_to_label[f]
                    if lbl[c] > 0:
                        train_files.pop(i)
                        valid_files.append(f)
                        for idx in range(num_classes):
                            if lbl[idx] > 0:
                                val_tp_counts[idx] += 1
                        adjustment_logs.append(f"Moved 1 sample to Valid for globally missing Class {c} (Label: {lbl})")
                        found = True
                        break
                if not found:
                    adjustment_logs.append(f"[Warning] Could not find ANY sample for Class {c} in Train set!")

    # =========================================================================
    # 4. 打印最终的统计结果 (此时划分已全部落锤固定)
    # =========================================================================
    print(f"\n[Info] Splitting data with Stratified Sampling (View + Label Combination)")
    print(f"Validation Ratio: {valid_ratio:.0%}")
    
    if adjustment_logs:
        print("-" * 120)
        print("[Post-process Adjustments]")
        for log in adjustment_logs:
            print("  -> " + log)
            
    print("-" * 120)
    print(f"{'View':<6} | {'Label Combination (Length 7)':<35} | {'Total':<8} | {'Train':<8} | {'Valid':<8} | {'Valid %':<8}")
    print("-" * 120)

    final_valid_set = set(valid_files)

    for key in sorted_keys:
        view, l_tuple = key
        files = group_dict[key]
        n_total = len(files)
        
        # 根据最终的 valid_files 集合倒推实际属于 Validation 的数量
        current_valid_count = sum(1 for f in files if f in final_valid_set)
        current_train_count = n_total - current_valid_count
        
        label_str = str(l_tuple).replace(" ", "")
        actual_ratio = current_valid_count / n_total if n_total > 0 else 0
        
        print(f"{view:<6} | {label_str:<35} | {n_total:<8} | {current_train_count:<8} | {current_valid_count:<8} | {actual_ratio:<8.1%}")

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