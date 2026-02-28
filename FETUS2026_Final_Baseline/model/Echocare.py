import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
from monai.networks.nets.swin_unetr import SwinTransformer, WindowAttention


class _LoRA_qkv(nn.Module):
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v, r: int, alpha: float = 1.0):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.r = r
        self.alpha = alpha
        self.scale = alpha / float(r)

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += self.scale * new_q
        qkv[:, :, -self.dim:] += self.scale * new_v
        return qkv


class SwinUNETR_Seg(nn.Module):
    """
    EchoCare-style SwinUNETR (2D) segmentation backbone:
    - encoder: SwinTransformer (in_chans=3)
    - decoder: UNETR-style blocks
    """

    def __init__(
        self,
        seg_num_classes: int,
        ssl_checkpoint: str = None,
        in_chans: int = 3,
        r: int = 5,
        alpha: float = 5.0,
    ):
        super().__init__()

        self.Swin_encoder = SwinTransformer(
            in_chans=in_chans,
            embed_dim=128,
            window_size=[8, 8],
            patch_size=[2, 2],
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            use_checkpoint=True,
            spatial_dims=2,
            use_v2=True,
        )

        if ssl_checkpoint is not None:
            model_dict = torch.load(ssl_checkpoint, map_location="cpu")
            if isinstance(model_dict, dict) and "state_dict" in model_dict:
                model_dict = model_dict["state_dict"]
            if isinstance(model_dict, dict) and "mask_token" in model_dict:
                model_dict.pop("mask_token")
            msg = self.Swin_encoder.load_state_dict(model_dict, strict=False)
            print("missing:", len(msg.missing_keys))
            print("unexpected:", len(msg.unexpected_keys))
            print("example missing:", msg.missing_keys[:20])
            print("example unexpected:", msg.unexpected_keys[:20])
            print("Using pretrained self-supervised Swin backbone weights!")

        for _, module in self.Swin_encoder.named_modules():
            if isinstance(module, WindowAttention):
                old_qkv = module.qkv
                dim = old_qkv.in_features
                w_a_linear_q = nn.Linear(dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, dim, bias=False)
                w_a_linear_v = nn.Linear(dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, dim, bias=False)
                module.qkv = _LoRA_qkv(
                    module.qkv,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                    r=r,
                    alpha=alpha,
                )

        self.reset_parameters()

        for name, p in self.Swin_encoder.named_parameters():
            if "linear_a" in name or "linear_b" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        spatial_dims = 2
        encode_feature_size = 128
        decode_feature_size = 64
        norm_name = "instance"

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=encode_feature_size,
            out_channels=decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * encode_feature_size,
            out_channels=2 * decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * encode_feature_size,
            out_channels=4 * decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * encode_feature_size,
            out_channels=8 * decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * encode_feature_size,
            out_channels=16 * decode_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * decode_feature_size,
            out_channels=8 * decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * decode_feature_size,
            out_channels=4 * decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * decode_feature_size,
            out_channels=2 * decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * decode_feature_size,
            out_channels=decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=decode_feature_size,
            out_channels=decode_feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=decode_feature_size,
            out_channels=seg_num_classes,
        )

        self.bottleneck_dim = 16 * decode_feature_size

    def encode(self, x3):
        hidden_states_out = self.Swin_encoder(x3)
        enc0 = self.encoder1(x3)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        enc4 = self.encoder5(hidden_states_out[3])
        dec4 = self.encoder10(hidden_states_out[4])
        return enc0, enc1, enc2, enc3, enc4, dec4

    def decode(self, enc0, enc1, enc2, enc3, enc4, dec4):
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, _LoRA_qkv):
                nn.init.kaiming_uniform_(m.linear_a_q.weight, a=math.sqrt(5))
                nn.init.zeros_(m.linear_b_q.weight)
                nn.init.kaiming_uniform_(m.linear_a_v.weight, a=math.sqrt(5))
                nn.init.zeros_(m.linear_b_v.weight)


class Echocare_UniMatch(nn.Module):
    """
    Echocare-style UniMatch model with disease-specific segmentation priors.

    Design goals:
      1) keep the original image classification branch as the dominant signal;
      2) derive only detached, low-dimensional anatomy statistics from segmentation;
      3) use disease-specific priors only for labels that have plausible structural correlates;
      4) inject priors as lightweight residual logit corrections to reduce negative transfer.

    Forward compatibility:
      - forward(x, need_fp=False)
      - forward(x, need_fp=False, view_ids=view_tensor)
      - need_fp=True returns (seg_main, seg_fp), (cls_main, cls_fp)
    """

    # view ids are 0-based in the training pipeline
    DEFAULT_SEG_ALLOWED = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],      # 4CH
        1: [0, 1, 2, 4, 8],               # LVOT
        2: [0, 6, 8, 9, 10, 11, 12],      # RVOT
        3: [0, 9, 12, 13, 14],            # 3VT
    }

    DEFAULT_CLS_ALLOWED = {
        0: [0, 1],     # 4CH
        1: [0, 2, 3],  # LVOT
        2: [4, 5],     # RVOT
        3: [2, 5, 6],  # 3VT
    }

    # segmentation label ids (0 is background)
    SEG = {
        "LA": 1,
        "LV": 2,
        "RA": 3,
        "RV": 4,
        "HEART": 5,
        "DAO": 6,
        "THOR": 7,
        "AAO": 8,
        "MPA": 9,
        "LPA": 10,
        "RPA": 11,
        "SVC": 12,
        "ARCH": 13,
        "TRACH": 14,
    }

    # Classes that keep image-dominant prediction only.
    IMG_ONLY_CLASSES = (1, 3)
    # Classes that receive disease-specific structural residual corrections.
    STRUCTURAL_CLASSES = (0, 2, 4, 5, 6)

    def __init__(
        self,
        in_chns: int,
        seg_class_num: int,
        cls_class_num: int,
        view_num_classes: int = 4,
        ssl_checkpoint: str = None,
        seg_allowed: Optional[Dict[int, List[int]]] = None,
        cls_allowed: Optional[Dict[int, List[int]]] = None,
        struct_hidden_dim: int = 128,
    ):
        super().__init__()

        if in_chns == 3:
            self.in_adapter = nn.Identity()
            in_chans = 3
        else:
            self.in_adapter = nn.Conv2d(1, 3, 1, bias=False)
            with torch.no_grad():
                self.in_adapter.weight.zero_()
                self.in_adapter.weight[:, 0, 0, 0] = 1.0
            in_chans = 3

        self.seg_net = SwinUNETR_Seg(
            seg_num_classes=seg_class_num,
            ssl_checkpoint=ssl_checkpoint,
            in_chans=in_chans,
        )

        self.seg_class_num = seg_class_num
        self.cls_class_num = cls_class_num
        self.view_num_classes = view_num_classes
        self.num_fg_classes = max(seg_class_num - 1, 0)

        bottleneck_dim = self.seg_net.bottleneck_dim
        hidden_dim = 512

        self.cls_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, cls_class_num),
        )

        self.fp_dropout = nn.Dropout2d(0.5)

        self.seg_allowed = self._sanitize_seg_allowed(seg_allowed or self.DEFAULT_SEG_ALLOWED)
        self.cls_allowed = self._sanitize_cls_allowed(cls_allowed or self.DEFAULT_CLS_ALLOWED)

        # disease-specific feature dimensions
        self.struct_feature_dims = {
            0: 10,  # Ventricular Septal Defect
            2: 9,   # Aortic Hypoplasia
            4: 10,  # Double Outlet Right Ventricle
            5: 10,  # Pulmonary Valve Stenosis
            6: 8,   # Right Aortic Arch
        }

        self.struct_heads = nn.ModuleDict()
        for cls_idx in self.STRUCTURAL_CLASSES:
            if cls_idx < self.cls_class_num:
                feat_dim = self.struct_feature_dims[cls_idx]
                self.struct_heads[str(cls_idx)] = nn.Sequential(
                    nn.Linear(bottleneck_dim + feat_dim, struct_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(struct_hidden_dim, 1),
                )

        # Per-class residual strength, initialized weak and stable.
        self.prior_logit_scale = nn.Parameter(torch.full((cls_class_num,), -2.1972246))  # sigmoid ~= 0.10

    def _sanitize_seg_allowed(self, seg_allowed: Dict[int, List[int]]) -> Dict[int, List[int]]:
        clean = {}
        for view_id, cls_ids in seg_allowed.items():
            filtered = []
            for cls_id in cls_ids:
                cls_id = int(cls_id)
                if 0 <= cls_id < self.seg_class_num:
                    filtered.append(cls_id)
            clean[int(view_id)] = filtered
        return clean

    def _sanitize_cls_allowed(self, cls_allowed: Dict[int, List[int]]) -> Dict[int, List[int]]:
        clean = {}
        for view_id, cls_ids in cls_allowed.items():
            filtered = []
            for cls_id in cls_ids:
                cls_id = int(cls_id)
                if 0 <= cls_id < self.cls_class_num:
                    filtered.append(cls_id)
            clean[int(view_id)] = filtered
        return clean

    def _pool_embed(self, feat_bchw: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feat_bchw, 1).flatten(1)

    def _flatten_view_ids(self, view_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if view_ids is None:
            return None
        view_ids = view_ids.reshape(-1)
        return view_ids.to(dtype=torch.long)

    def _build_seg_fg_mask(self, batch_size: int, device: torch.device, view_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.num_fg_classes == 0:
            return None
        view_ids = self._flatten_view_ids(view_ids)
        if view_ids is None or view_ids.numel() != batch_size:
            return None
        view_ids = view_ids.to(device=device)

        mask = torch.zeros(batch_size, self.num_fg_classes, device=device, dtype=torch.float32)
        for b in range(batch_size):
            v = int(view_ids[b].item())
            allowed = self.seg_allowed.get(v, list(range(1, self.seg_class_num)))
            for cls_id in allowed:
                if 1 <= cls_id < self.seg_class_num:
                    mask[b, cls_id - 1] = 1.0
        return mask.unsqueeze(-1).unsqueeze(-1)

    def _build_cls_valid_mask(self, batch_size: int, device: torch.device, view_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        view_ids = self._flatten_view_ids(view_ids)
        if view_ids is None or view_ids.numel() != batch_size:
            return None
        view_ids = view_ids.to(device=device)

        mask = torch.zeros(batch_size, self.cls_class_num, device=device, dtype=torch.float32)
        for b in range(batch_size):
            v = int(view_ids[b].item())
            for cls_id in self.cls_allowed.get(v, []):
                if 0 <= cls_id < self.cls_class_num:
                    mask[b, cls_id] = 1.0
        return mask

    def _prepare_seg_probs(self, seg_logits: torch.Tensor, view_ids: Optional[torch.Tensor]) -> torch.Tensor:
        # Force float32 on the prior/statistics path to avoid fp16 overflow on large spatial sums.
        probs = torch.softmax(seg_logits.detach().float(), dim=1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        if self.num_fg_classes > 0:
            fg_mask = self._build_seg_fg_mask(probs.shape[0], probs.device, view_ids)
            if fg_mask is not None:
                probs = probs.clone()
                probs[:, 1:, :, :] = probs[:, 1:, :, :] * fg_mask.float()
        return probs

    def _spatial_stats(self, prob_map: torch.Tensor):
        """
        prob_map: [B, H, W]
        returns area, cx, cy, sx, sy, conf (all [B])
        """
        prob_map = prob_map.float()
        bsz, h, w = prob_map.shape
        eps = 1e-6

        mass = prob_map.sum(dim=(1, 2))
        mass = torch.nan_to_num(mass, nan=0.0, posinf=float(h * w), neginf=0.0)
        area = mass / float(h * w)

        xs = torch.linspace(0.0, 1.0, steps=w, device=prob_map.device, dtype=torch.float32).view(1, 1, w)
        ys = torch.linspace(0.0, 1.0, steps=h, device=prob_map.device, dtype=torch.float32).view(1, h, 1)

        denom = mass.clamp_min(eps)
        cx_num = (prob_map * xs).sum(dim=(1, 2))
        cy_num = (prob_map * ys).sum(dim=(1, 2))
        cx = cx_num / denom
        cy = cy_num / denom

        cx = torch.where(mass > eps, cx, torch.full_like(cx, 0.5))
        cy = torch.where(mass > eps, cy, torch.full_like(cy, 0.5))

        sx = (prob_map * (xs - cx.view(bsz, 1, 1)).pow(2)).sum(dim=(1, 2)) / denom
        sy = (prob_map * (ys - cy.view(bsz, 1, 1)).pow(2)).sum(dim=(1, 2)) / denom
        conf = prob_map.amax(dim=(1, 2))

        area = torch.nan_to_num(area, nan=0.0, posinf=1.0, neginf=0.0)
        cx = torch.nan_to_num(cx, nan=0.5, posinf=1.0, neginf=0.0)
        cy = torch.nan_to_num(cy, nan=0.5, posinf=1.0, neginf=0.0)
        sx = torch.nan_to_num(sx, nan=0.0, posinf=1.0, neginf=0.0)
        sy = torch.nan_to_num(sy, nan=0.0, posinf=1.0, neginf=0.0)
        conf = torch.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0)
        return area, cx, cy, sx, sy, conf

    def _distance(self, ax: torch.Tensor, ay: torch.Tensor, bx: torch.Tensor, by: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((ax - bx).pow(2) + (ay - by).pow(2) + 1e-8)

    def _ratio(self, num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        return num / (den + 1e-6)

    def _collect_seg_stats(self, probs: torch.Tensor):
        stats = {}
        for cls_id in range(1, self.seg_class_num):
            area, cx, cy, sx, sy, conf = self._spatial_stats(probs[:, cls_id, :, :])
            stats[cls_id] = {
                "area": area,
                "cx": cx,
                "cy": cy,
                "sx": sx,
                "sy": sy,
                "conf": conf,
            }
        return stats

    def _extract_disease_features(self, seg_logits: torch.Tensor, view_ids: Optional[torch.Tensor] = None) -> Dict[int, torch.Tensor]:
        """
        Build detached, disease-specific, low-dimensional anatomy descriptors.
        Features are intentionally hand-crafted and weakly coupled.
        """
        if self.num_fg_classes == 0:
            return {cls_idx: seg_logits.new_zeros(seg_logits.shape[0], dim) for cls_idx, dim in self.struct_feature_dims.items()}

        probs = self._prepare_seg_probs(seg_logits, view_ids=view_ids)
        stats = self._collect_seg_stats(probs)
        S = self.SEG

        def get(cls_name: str, key: str) -> torch.Tensor:
            cls_id = S[cls_name]
            if cls_id >= self.seg_class_num:
                return seg_logits.new_zeros(seg_logits.shape[0])
            return stats[cls_id][key]

        # Frequently used stats
        area_la, cx_la, cy_la, sx_la, sy_la, conf_la = [get("LA", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_lv, cx_lv, cy_lv, sx_lv, sy_lv, conf_lv = [get("LV", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_ra, cx_ra, cy_ra, sx_ra, sy_ra, conf_ra = [get("RA", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_rv, cx_rv, cy_rv, sx_rv, sy_rv, conf_rv = [get("RV", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_heart, cx_heart, cy_heart, sx_heart, sy_heart, conf_heart = [get("HEART", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_dao, cx_dao, cy_dao, sx_dao, sy_dao, conf_dao = [get("DAO", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_thor, cx_thor, cy_thor, sx_thor, sy_thor, conf_thor = [get("THOR", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_aao, cx_aao, cy_aao, sx_aao, sy_aao, conf_aao = [get("AAO", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_mpa, cx_mpa, cy_mpa, sx_mpa, sy_mpa, conf_mpa = [get("MPA", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_lpa, cx_lpa, cy_lpa, sx_lpa, sy_lpa, conf_lpa = [get("LPA", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_rpa, cx_rpa, cy_rpa, sx_rpa, sy_rpa, conf_rpa = [get("RPA", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_svc, cx_svc, cy_svc, sx_svc, sy_svc, conf_svc = [get("SVC", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_arch, cx_arch, cy_arch, sx_arch, sy_arch, conf_arch = [get("ARCH", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]
        area_trach, cx_trach, cy_trach, sx_trach, sy_trach, conf_trach = [get("TRACH", k) for k in ("area", "cx", "cy", "sx", "sy", "conf")]

        features: Dict[int, torch.Tensor] = {}

        # Class 0: Ventricular Septal Defect
        features[0] = torch.stack([
            area_lv,
            area_rv,
            area_la,
            area_ra,
            self._ratio(area_rv, area_lv),
            self._ratio(area_ra, area_la),
            self._distance(cx_lv, cy_lv, cx_rv, cy_rv),
            self._distance(cx_la, cy_la, cx_ra, cy_ra),
            area_heart,
            sx_heart + sy_heart,
        ], dim=1)

        # Class 2: Aortic Hypoplasia
        features[2] = torch.stack([
            area_aao,
            area_dao,
            area_arch,
            conf_aao,
            conf_dao,
            conf_arch,
            self._ratio(area_aao, area_dao),
            self._ratio(area_arch, area_trach),
            self._distance(cx_aao, cy_aao, cx_dao, cy_dao),
        ], dim=1)

        # Class 4: Double Outlet Right Ventricle
        features[4] = torch.stack([
            area_rv,
            area_aao,
            area_mpa,
            self._distance(cx_rv, cy_rv, cx_aao, cy_aao),
            self._distance(cx_rv, cy_rv, cx_mpa, cy_mpa),
            self._distance(cx_aao, cy_aao, cx_mpa, cy_mpa),
            cx_aao - cx_rv,
            cy_aao - cy_rv,
            cx_mpa - cx_rv,
            cy_mpa - cy_rv,
        ], dim=1)

        # Class 5: Pulmonary Valve Stenosis
        features[5] = torch.stack([
            area_mpa,
            area_lpa,
            area_rpa,
            conf_mpa,
            conf_lpa,
            conf_rpa,
            self._ratio(area_mpa, area_aao),
            self._ratio(area_lpa + area_rpa, area_mpa),
            sx_mpa + sy_mpa,
            self._distance(cx_mpa, cy_mpa, cx_aao, cy_aao),
        ], dim=1)

        # Class 6: Right Aortic Arch
        features[6] = torch.stack([
            area_arch,
            area_trach,
            conf_arch,
            conf_trach,
            cx_arch - cx_trach,
            cy_arch - cy_trach,
            self._distance(cx_arch, cy_arch, cx_trach, cy_trach),
            self._ratio(area_arch, area_trach),
        ], dim=1)

        for cls_idx, feat in list(features.items()):
            feat = feat.float()
            feat = torch.nan_to_num(feat, nan=0.0, posinf=10.0, neginf=-10.0)
            feat = feat.clamp(min=-10.0, max=10.0)
            features[cls_idx] = feat

        return features

    def _apply_class_specific_priors(
        self,
        base_logits: torch.Tensor,
        embed: torch.Tensor,
        seg_logits: torch.Tensor,
        view_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_logits = base_logits.float()
        embed = embed.float()
        seg_logits = seg_logits.float()

        final_logits = base_logits.clone()
        cls_valid_mask = self._build_cls_valid_mask(base_logits.shape[0], base_logits.device, view_ids)
        disease_feats = self._extract_disease_features(seg_logits, view_ids=view_ids)

        for cls_idx, feat in disease_feats.items():
            if cls_idx >= self.cls_class_num:
                continue
            if cls_valid_mask is not None and torch.all(cls_valid_mask[:, cls_idx] == 0):
                continue

            key = str(cls_idx)
            if key not in self.struct_heads:
                continue

            head = self.struct_heads[key]
            feat = torch.nan_to_num(feat.float(), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            delta = head(torch.cat([embed, feat], dim=1)).squeeze(1)
            delta = torch.nan_to_num(delta, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            alpha = torch.sigmoid(self.prior_logit_scale[cls_idx]).float()
            if cls_valid_mask is not None:
                delta = delta * cls_valid_mask[:, cls_idx]
            final_logits[:, cls_idx] = final_logits[:, cls_idx] + alpha * delta

        final_logits = torch.nan_to_num(final_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        return final_logits

    def forward(self, x, need_fp: bool = False, view_ids: Optional[torch.Tensor] = None):
        """
        Returns logits (not sigmoid probabilities), consistent with BCEWithLogits usage.

        Args:
            x:        input image tensor.
            need_fp:  if True, returns main/fp split for UniMatch false-positive branch.
            view_ids: optional view labels (0-based). If provided, segmentation-derived
                      disease priors are masked by view-valid segmentation and class sets.
        """
        x3 = self.in_adapter(x)
        enc0, enc1, enc2, enc3, enc4, dec4 = self.seg_net.encode(x3)

        if need_fp:
            p_enc0 = torch.cat([enc0, self.fp_dropout(enc0)], dim=0)
            p_enc1 = torch.cat([enc1, self.fp_dropout(enc1)], dim=0)
            p_enc2 = torch.cat([enc2, self.fp_dropout(enc2)], dim=0)
            p_enc3 = torch.cat([enc3, self.fp_dropout(enc3)], dim=0)
            p_enc4 = torch.cat([enc4, self.fp_dropout(enc4)], dim=0)
            p_dec4 = torch.cat([dec4, self.fp_dropout(dec4)], dim=0)

            seg_logits = self.seg_net.decode(p_enc0, p_enc1, p_enc2, p_enc3, p_enc4, p_dec4)
            embed = self._pool_embed(p_dec4)
            base_logits = self.cls_decoder(embed)

            view_ids_fp = None
            if view_ids is not None:
                view_ids = self._flatten_view_ids(view_ids)
                view_ids_fp = torch.cat([view_ids, view_ids], dim=0)

            with torch.autocast(device_type=x.device.type, enabled=False):
                cls_logits = self._apply_class_specific_priors(
                    base_logits=base_logits.float(),
                    embed=embed.float(),
                    seg_logits=seg_logits.float(),
                    view_ids=view_ids_fp,
                )
            return seg_logits.chunk(2), cls_logits.chunk(2)

        seg_logits = self.seg_net.decode(enc0, enc1, enc2, enc3, enc4, dec4)
        embed = self._pool_embed(dec4)
        base_logits = self.cls_decoder(embed)
        with torch.autocast(device_type=x.device.type, enabled=False):
            cls_logits = self._apply_class_specific_priors(
                base_logits=base_logits.float(),
                embed=embed.float(),
                seg_logits=seg_logits.float(),
                view_ids=view_ids,
            )
        return seg_logits, cls_logits
