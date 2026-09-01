#!/usr/bin/env python3
"""Standalone benchmark: RGB + dataset GT K -> NLF -> optional Pose V2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_hmr4d_smpl_metrics import box_iou_cxcywh, extract_gt_smpl, move_to_device
from scripts.eval.evaluate_nlf_pose_stabilizer_v2_metrics import MetricTotals, metric_values
from vggt_omega.data import HMR4DSupportEvalDataset, hmr4d_eval_collate_fn
from vggt_omega.data.geometry import resolve_image_size_config
from vggt_omega.integrations.nlf_smpl_provider import NLFSMPLProvider
from vggt_omega.models import PoseStabilizerConfig, PoseTemporalStabilizer
from vggt_omega.models.smpl_layer import SMPLLayer
from vggt_omega.tracking.smpl_track_assigner import BaseSMPLTrackAssigner
from vggt_omega.training.config import deep_update, load_yaml_config, require_path


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({args.device}) but unavailable")
    device = torch.device(args.device)
    cfg = deep_update(load_yaml_config(args.path_config), load_yaml_config(args.config))
    data = build_dataset(cfg, args)
    nlf = build_nlf(cfg).to(device).eval()
    smpl = SMPLLayer(require_path(cfg, "assets.smpl_model_dir")).to(device).eval()
    stabilizer = load_stabilizer(args.temporal_checkpoint, device) if args.temporal_checkpoint else None
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    records = [r for r in data.records if not args.sequence_filter or args.sequence_filter.lower() in r.vid.lower() or args.sequence_filter.lower() in str(r.label.get("vname", "")).lower()]
    if not records: raise ValueError("No support records matched sequence_filter")
    print_manifest(args, cfg, data, records, stabilizer, device)
    totals = {"nlf_base": MetricTotals(), "nlf_pose_temporal": MetricTotals()}
    rows: list[dict[str, Any]] = []
    coverage = {"records": 0, "metric_frames": 0, "temporal_applied_frames": 0}
    for record_idx, record in enumerate(data.records):
        if record not in records: continue
        indices = [i for i, item in enumerate(data._index) if item[0] == record_idx]
        result = infer_record(data, indices, nlf, cfg, device, args)
        refined, applied = temporal_refine(result, stabilizer)
        valid = result["eval_mask"] & (result["query"] >= 0)
        if not bool(valid.any()):
            print(f"[skip] {record.vid}: no matched NLF detections", flush=True); continue
        q = result["query"][valid]
        base = metric_values(result["pose"][valid, q], result["betas"][valid, q], result["transl"][valid, q], result["gt_pose"][valid], result["gt_betas"][valid], result["gt_transl"][valid], smpl)
        temp = metric_values(refined[valid], result["betas"][valid, q], result["transl"][valid, q], result["gt_pose"][valid], result["gt_betas"][valid], result["gt_transl"][valid], smpl)
        for name, values in (("nlf_base", base), ("nlf_pose_temporal", temp)):
            for key, value in values.items(): totals[name].add(key, value.mean(), int(value.numel()))
        valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
        for local, frame in enumerate(valid_idx.tolist()):
            rows.append({"vid": record.vid, "vname": str(record.label.get("vname", "")), "frame": frame, "query": int(result["query"][frame]), "temporal_applied": int(applied[frame]), **{f"base_{k[:-2]}_mm": float(v[local].detach().cpu()*1000) for k,v in base.items()}, **{f"temporal_{k[:-2]}_mm": float(v[local].detach().cpu()*1000) for k,v in temp.items()}})
        coverage["records"] += 1; coverage["metric_frames"] += int(valid.sum()); coverage["temporal_applied_frames"] += int(applied[valid].sum())
        print(f"[record] {record.vid}: frames={int(valid.sum())} temporal={int(applied[valid].sum())}", flush=True)
    summary = {"benchmark": "nlf_pose_temporal_gt_intrinsics", "dataset": args.dataset, "input_protocol": "RGB + dataset GT intrinsics -> NLF internal detector -> optional PoseTemporalStabilizerV2; VGGT/HSI/TRSTR absent", "metric_protocol": "SMPL-24 pelvis-aligned PA-MPJPE/MPJPE/PVE", "nlf_checkpoint": str(cfg.get("checkpoints", {}).get("nlf_smpl", "")), "temporal_checkpoint": args.temporal_checkpoint or None, "coverage": {**coverage, "temporal_applied_rate": coverage["temporal_applied_frames"] / max(coverage["metric_frames"], 1)}}
    for name, metric in totals.items(): summary[name] = metric.summary()
    (output / f"{args.dataset}_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_rows(output / f"{args.dataset}_rows.csv", rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--dataset", choices=["3dpw","emdb1"], required=True); p.add_argument("--config", default="benchmarks/nlf_pose_temporal/config.yaml"); p.add_argument("--path-config", default="configs/path.yaml"); p.add_argument("--frames-root", default=""); p.add_argument("--support-root", default=""); p.add_argument("--temporal-checkpoint", default=""); p.add_argument("--output-dir", default="outputs/eval/nlf_pose_temporal"); p.add_argument("--sequence-filter", default=""); p.add_argument("--device", default="cuda:0"); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--num-workers", type=int, default=2); return p.parse_args()


def build_dataset(cfg: dict[str,Any], args: argparse.Namespace) -> HMR4DSupportEvalDataset:
    support = args.support_root or require_path(cfg, "datasets.threedpw_hmr4d_support_root" if args.dataset=="3dpw" else "datasets.emdb_hmr4d_support_root")
    frames = args.frames_root or require_path(cfg,"datasets.hmr4d_eval_frames_root")
    size,res = resolve_image_size_config(cfg.get("data",{})); return HMR4DSupportEvalDataset(dataset=args.dataset,support_root=support,frames_root=frames,sequence_length=1,stride=1,image_size=size,image_resolution=res,resize_mode=str(cfg.get("data",{}).get("resize_mode","balanced")),max_humans=int(cfg.get("data",{}).get("max_humans",20)),patch_size=int(cfg.get("model",{}).get("patch_size",16)))


def build_nlf(cfg: dict[str,Any]) -> NLFSMPLProvider:
    m=cfg.get("model",{}); return NLFSMPLProvider(model_path=str(cfg.get("checkpoints",{}).get("nlf_smpl","")),third_party_root=str(cfg.get("third_party",{}).get("nlf_root","third_party/nlf")),model_name=str(m.get("nlf_model_name","smpl")),use_detector=True,require_boxes=False,internal_batch_size=int(m.get("nlf_internal_batch_size",128)),num_aug=int(m.get("nlf_num_aug",1)),detector_threshold=float(m.get("nlf_detector_threshold",.3)),detector_nms_iou_threshold=float(m.get("nlf_detector_nms_iou_threshold",.7)),max_detections=int(m.get("nlf_max_detections",150)))


def load_stabilizer(path: str, device: torch.device) -> PoseTemporalStabilizer:
    ckpt=torch.load(path,map_location=device,weights_only=False); model=PoseTemporalStabilizer(PoseStabilizerConfig(**ckpt["model_config"])).to(device); model.load_state_dict(ckpt["model_state"],strict=True); model.eval(); return model


def infer_record(dataset: HMR4DSupportEvalDataset, indices: list[int], nlf: NLFSMPLProvider, cfg: dict[str,Any], device: torch.device, args: argparse.Namespace) -> dict[str,torch.Tensor]:
    loader=DataLoader(Subset(dataset,indices),batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,collate_fn=hmr4d_eval_collate_fn)
    parts: dict[str,list[torch.Tensor]]={k:[] for k in ("pred_pose_6d","pred_betas","pred_transl_cam","pred_confs","pred_boxes")}; gt_parts={k:[] for k in ("poses","betas","transl")}; boxes=[]; masks=[]; evals=[]
    for batch in loader:
        batch=move_to_device(batch,device); pred=nlf.forward_with_intrinsics(batch["images"],batch["K_scal3r"],max_humans=int(cfg.get("data",{}).get("max_humans",20))); gt=extract_gt_smpl(batch["eval_label"],device)
        for k in parts: parts[k].append(pred[k][:,0])
        for k in gt_parts: gt_parts[k].append(gt[k][:,0])
        boxes.append(batch["gt_boxes"][:,0,0]); masks.append(batch["boxes_mask"][:,0,0]); evals.append(batch["eval_mask"][:,0])
    out={"pose":torch.cat(parts["pred_pose_6d"]),"betas":torch.cat(parts["pred_betas"]),"transl":torch.cat(parts["pred_transl_cam"]),"confs":torch.cat(parts["pred_confs"]),"boxes":torch.cat(parts["pred_boxes"]),"gt_pose":torch.cat(gt_parts["poses"]),"gt_betas":torch.cat(gt_parts["betas"]),"gt_transl":torch.cat(gt_parts["transl"]),"gt_box":torch.cat(boxes),"gt_box_mask":torch.cat(masks).bool(),"eval_mask":torch.cat(evals).bool()}
    out["track"]=BaseSMPLTrackAssigner().assign(out["boxes"].unsqueeze(0),out["betas"].unsqueeze(0),out["transl"].unsqueeze(0),out["confs"].unsqueeze(0),query_mask=(out["confs"][...,0]>0).unsqueeze(0)); out["query"]=select_queries(out); return out


def select_queries(x:dict[str,torch.Tensor])->torch.Tensor:
    qs=[]
    for t in range(x["pose"].shape[0]):
        valid=x["confs"][t,:,0]>0
        if not bool(valid.any()): qs.append(-1); continue
        if bool(x["gt_box_mask"][t]):
            iou=box_iou_cxcywh(x["boxes"][t],x["gt_box"][t].reshape(1,4)).reshape(-1).masked_fill(~valid,-1); qs.append(int(iou.argmax()) if float(iou.max())>=0 else -1)
        else: qs.append(int(x["confs"][t,:,0].masked_fill(~valid,-1).argmax()))
    return torch.tensor(qs,device=x["pose"].device)


def temporal_refine(x:dict[str,torch.Tensor], model:PoseTemporalStabilizer|None)->tuple[torch.Tensor,torch.Tensor]:
    refined=x["pose"].clone(); applied=torch.zeros(x["pose"].shape[0],dtype=torch.bool,device=refined.device)
    if model is None: return refined,applied
    ids=x["track"]["assigned_track_ids"][0]; mask=x["track"]["assigned_track_mask"][0]
    for track_id in torch.unique(ids[mask]).tolist():
        same=(ids==track_id)&mask; counts=same.sum(1); valid=counts==1; slots=same.long().argmax(1); starts=range(max(0,ids.shape[0]-8))
        windows=[]; wm=[]; center=[]
        for s in starts: windows.append(x["pose"][torch.arange(s,s+9,device=ids.device),slots[s:s+9]]); wm.append(valid[s:s+9]); center.append(s+4)
        if not windows: continue
        for offset in range(0,len(windows),128):
            out=model(torch.stack(windows[offset:offset+128]),torch.stack(wm[offset:offset+128])); good=out["context_valid"][:,4]
            for j,ok in enumerate(good.tolist()):
                if ok: refined[center[offset+j],slots[center[offset+j]]]=out["refined_pose_6d"][j,4]; applied[center[offset+j]]=True
    q=x["query"]; picked=refined[torch.arange(refined.shape[0],device=refined.device),q.clamp_min(0)]; return picked,applied


def write_rows(path:Path,rows:list[dict[str,Any]])->None:
    fields=[
        "vid", "vname", "frame", "query", "temporal_applied",
        "base_pa_mpjpe_mm", "base_mpjpe_mm", "base_pve_mm", "base_cam_mpjpe_no_align_mm", "base_cam_pve_no_align_mm",
        "temporal_pa_mpjpe_mm", "temporal_mpjpe_mm", "temporal_pve_mm", "temporal_cam_mpjpe_no_align_mm", "temporal_cam_pve_no_align_mm",
    ]
    with path.open("w",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def print_manifest(args:argparse.Namespace,cfg:dict[str,Any],data:HMR4DSupportEvalDataset,records:list[Any],model:PoseTemporalStabilizer|None,device:torch.device)->None:
    print("========== NLF pose-temporal benchmark =========="); print(f"device: {device}"); print(f"input: RGB + dataset GT intrinsics -> NLF; VGGT absent"); print(f"NLF checkpoint: {cfg.get('checkpoints',{}).get('nlf_smpl')}"); print(f"V2 checkpoint: {args.temporal_checkpoint or 'disabled (base only)'}"); print(f"records: {len(records)}, frames root: {data.frames_root}")


if __name__=="__main__": main()
