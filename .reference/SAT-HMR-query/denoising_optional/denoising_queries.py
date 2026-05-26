"""Optional denoising query utilities extracted from SAT-HMR.

SAT-HMR source reference:
- models/dn_components.py: prepare_for_cdn, dn_post_process
"""

from typing import Dict, List, Tuple

import torch
from torch import nn

try:
    from reference_query_mechanism.query_init.query_initializer import inverse_sigmoid
except ImportError:
    from query_init.query_initializer import inverse_sigmoid


def prepare_denoising_queries(
    targets: List[Dict[str, torch.Tensor]],
    dn_cfg: Dict,
    num_queries: int,
    hidden_dim: int,
    dn_encoder: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Prepare noisy GT queries that are prepended before regular queries.

    Args:
        targets: list of target dictionaries.
        dn_cfg: denoising configuration.
        num_queries: number of regular matching queries.
        hidden_dim: decoder hidden dimension.
        dn_encoder: label embedding or parameter encoder.

    Returns:
        input_query_tgt: `(B, pad_size, C)` denoising content queries.
        input_query_bbox: `(B, pad_size, 4)` denoising reference boxes in inverse-sigmoid space.
        attn_mask: `(pad_size + num_queries, pad_size + num_queries)` bool mask.
        dn_meta: metadata needed for post-processing.
    """
    device = targets[0]["boxes"].device
    dn_number = dn_cfg["dn_number"]
    box_noise_scale = dn_cfg.get("box_noise_scale", 0.4)
    tgt_noise_scale = dn_cfg.get("tgt_noise_scale", 0.0)
    tgt_embed_type = dn_cfg.get("tgt_embed_type", "labels")

    known = [torch.ones_like(target["labels"]) for target in targets]
    batch_size = len(known)
    known_num = [int(mask.sum()) for mask in known]

    if max(known_num) == 0:
        dn_number = 1
    elif dn_number >= 100:
        dn_number = dn_number // (max(known_num) * 2)
    elif dn_number < 1:
        dn_number = 1

    boxes = torch.cat([target["boxes"] for target in targets])
    batch_idx = torch.cat([torch.full_like(target["labels"].long(), i) for i, target in enumerate(targets)])
    known_indice = torch.nonzero(torch.cat(known)).view(-1)
    known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
    known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)

    single_pad = max(known_num)
    pad_size = single_pad * 2 * dn_number
    positive_idx = torch.arange(len(boxes), device=device).long().unsqueeze(0).repeat(dn_number, 1)
    positive_idx += (torch.arange(dn_number, device=device) * len(boxes) * 2).long().unsqueeze(1)
    positive_idx = positive_idx.flatten()
    negative_idx = positive_idx + len(boxes)

    known_bbox = boxes.repeat(2 * dn_number, 1)
    known_bbox_expand = known_bbox.clone()
    if box_noise_scale > 0:
        known_bbox_xyxy = torch.zeros_like(known_bbox)
        known_bbox_xyxy[:, :2] = known_bbox[:, :2] - known_bbox[:, 2:] / 2
        known_bbox_xyxy[:, 2:] = known_bbox[:, :2] + known_bbox[:, 2:] / 2

        diff = torch.zeros_like(known_bbox)
        diff[:, :2] = known_bbox[:, 2:] / 2
        diff[:, 2:] = known_bbox[:, 2:] / 2

        rand_sign = torch.randint_like(known_bbox, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
        rand_part = torch.rand_like(known_bbox)
        rand_part[negative_idx] += 1.0
        rand_part *= rand_sign
        known_bbox_xyxy = known_bbox_xyxy + rand_part * diff * box_noise_scale
        known_bbox_xyxy = known_bbox_xyxy.clamp(min=0.0, max=1.0)
        known_bbox_expand[:, :2] = (known_bbox_xyxy[:, :2] + known_bbox_xyxy[:, 2:]) / 2
        known_bbox_expand[:, 2:] = known_bbox_xyxy[:, 2:] - known_bbox_xyxy[:, :2]

    input_bbox_embed = inverse_sigmoid(known_bbox_expand)

    if tgt_embed_type == "labels":
        labels = torch.cat([target["labels"] for target in targets])
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_labels_expanded = known_labels.clone()
        if tgt_noise_scale > 0:
            chosen = torch.nonzero(torch.rand_like(known_labels_expanded.float()) < tgt_noise_scale).view(-1)
            new_label = torch.randint_like(chosen, 0, dn_cfg["dn_labelbook_size"])
            known_labels_expanded.scatter_(0, chosen, new_label)
        input_tgt_embed = dn_encoder(known_labels_expanded.long().to(device=device))
    elif tgt_embed_type == "params":
        poses = torch.cat([target["poses"] for target in targets])
        betas = torch.cat([target["betas"] for target in targets])
        params = torch.cat([poses, betas], dim=-1)
        known_params = params.repeat(2 * dn_number, 1)
        known_params_expanded = known_params.clone()
        if tgt_noise_scale > 0:
            rand_sign = torch.randint_like(known_params, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_params)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_params_expanded = known_params_expanded + rand_part * tgt_noise_scale
        input_tgt_embed = dn_encoder(known_params_expanded.to(device=device))
    else:
        raise ValueError("tgt_embed_type must be 'labels' or 'params'.")

    input_query_tgt = torch.zeros((batch_size, pad_size, hidden_dim), device=device)
    input_query_bbox = torch.zeros((batch_size, pad_size, 4), device=device)

    if known_num:
        map_known_indice = torch.cat([torch.arange(num) for num in known_num]).to(device=device)
        map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        input_query_tgt[(known_bid.long(), map_known_indice)] = input_tgt_embed
        input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

    tgt_size = pad_size + num_queries
    attn_mask = torch.zeros((tgt_size, tgt_size), dtype=torch.bool, device=device)
    attn_mask[pad_size:, :pad_size] = True
    for group_idx in range(dn_number):
        start = single_pad * 2 * group_idx
        end = single_pad * 2 * (group_idx + 1)
        attn_mask[start:end, end:pad_size] = True
        attn_mask[start:end, :start] = True

    dn_meta = {"pad_size": pad_size, "num_dn_group": dn_number}
    return input_query_tgt, input_query_bbox, attn_mask, dn_meta


def split_denoising_outputs(outputs: Dict[str, torch.Tensor], dn_meta: Dict) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Split denoising outputs from regular matching-query outputs.

    This generic version works for output tensors whose query dimension is dim=1,
    e.g. `(B, Q, C)`, or dim=2 for stacked decoder layers `(L, B, Q, C)`.
    """
    pad_size = dn_meta.get("pad_size", 0)
    if pad_size <= 0:
        return outputs, dn_meta

    regular_outputs = {}
    known_outputs = {}
    for key, value in outputs.items():
        if not torch.is_tensor(value):
            regular_outputs[key] = value
            continue
        if value.ndim >= 4:
            known_outputs[key] = value[:, :, :pad_size]
            regular_outputs[key] = value[:, :, pad_size:]
        elif value.ndim >= 3:
            known_outputs[key] = value[:, :pad_size]
            regular_outputs[key] = value[:, pad_size:]
        else:
            regular_outputs[key] = value

    dn_meta["output_known"] = known_outputs
    return regular_outputs, dn_meta
