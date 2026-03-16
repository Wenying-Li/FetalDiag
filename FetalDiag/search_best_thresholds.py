import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.fetus_eval import FETUSEvalDataset
from model.unet import UNet
from model.Echocare import Echocare_UniMatch
from util.utils import masked_metrics_with_threshold_search

DEFAULT_CLS_ALLOWED: Dict[int, List[int]] = {
    0: [0, 1],
    1: [0, 2, 3],
    2: [4, 5],
    3: [2, 5, 6],
}

def build_model(args, device):
    if args.model == "unet":
        model = UNet(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            view_num_classes=args.view_num_classes,
        )
    elif args.model == "echocare":
        model = Echocare_UniMatch(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            view_num_classes=args.view_num_classes,
            ssl_checkpoint=None,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model.to(device)

def load_checkpoint_strict(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

def _load_json_arg(s: Optional[str]):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s.startswith("{") or s.startswith("["):
        return json.loads(s)
    with open(s, "r", encoding="utf-8") as f:
        return json.load(f)

def load_cls_allowed(arg: Optional[str], default: Dict[int, List[int]]) -> Dict[int, List[int]]:
    raw = _load_json_arg(arg)
    if raw is None:
        return default
    if not isinstance(raw, dict):
        raise ValueError("--cls-allowed must be a JSON object: {view_id:[cls_ids...]}")
    out: Dict[int, List[int]] = {}
    for k, v in raw.items():
        kk = int(k)
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"cls_allowed[{k}] must be a list.")
        out[kk] = [int(x) for x in v]
    return out

@torch.inference_mode()
def run_threshold_search(model, loader, device, args, cls_allowed):
    model.eval()

    y_true_all, y_prob_all, views_all = [], [], []

    use_amp = args.amp and (device.type == "cuda")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    for batch in tqdm(loader, total=len(loader), desc="Collect"):
        # FETUSEvalDataset returns: image, view, mask, label, image_h5_file
        image, view, mask, label, _ = batch

        image = image.to(device, non_blocking=True)
        view = view.to(device, non_blocking=True).long().view(-1)
        label = label.to(device, non_blocking=True).float()

        image_rs = F.interpolate(
            image, (args.resize_target, args.resize_target),
            mode="bilinear", align_corners=False
        )

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            _, pred_class_out = model(image_rs)
            pred_prob = torch.sigmoid(pred_class_out)

        y_true_all.append(label.cpu().numpy())
        y_prob_all.append(pred_prob.cpu().numpy())
        views_all.append(view.cpu().numpy())

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    views_all = np.concatenate(views_all, axis=0)

    metrics = masked_metrics_with_threshold_search(
        y_true_all, y_prob_all, views_all, cls_allowed
    )

    return metrics

def parse_args():
    p = argparse.ArgumentParser("Search Best Thresholds (match inference)")
    p.add_argument("--valid-json", type=str, default="./data/valid.json")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--model", type=str, default="unet", choices=["unet", "echocare"])

    p.add_argument("--resize-target", type=int, default=256)
    p.add_argument("--seg-num-classes", type=int, default=15)
    p.add_argument("--cls-num-classes", type=int, default=7)
    p.add_argument("--view-num-classes", type=int, default=4)

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--gpu", type=str, default="0")

    p.add_argument("--cls-allowed", type=str, default=None,
                   help="JSON string or .json path: {view_id:[cls_ids...]}")

    p.add_argument("--amp", action="store_true", help="enable autocast")
    p.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    cls_allowed = load_cls_allowed(args.cls_allowed, DEFAULT_CLS_ALLOWED)

    model = build_model(args, device)
    load_checkpoint_strict(model, args.ckpt, device)

    dataset = FETUSEvalDataset(args.valid_json)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    metrics = run_threshold_search(model, loader, device, args, cls_allowed)

    best_thr = metrics["per_class_best_thr"]
    best_f1 = metrics["per_class_f1@best"]

    print("Best thresholds per class:")
    print(", ".join([f"{t:.2f}" for t in best_thr]))
    print("Best F1 per class:")
    print(", ".join([f"{f:.4f}" for f in best_f1]))

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()