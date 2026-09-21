#!/usr/bin/env python3
"""Export measured single-image GroundedHuman teaser elements, without model edits.

Run through export_groundedhuman_teaser_assets.sh on the Linux server. Inference
matches the existing coarse/residual/clip-depth cascade for S=1; temporal consensus
is unavailable. Hooks observe only, and are removed even if inference fails.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
from scripts.vis.teaser_asset_utils import (
    TEAL, AMBER, CORAL, PURPLE, INK, json_write, project, unproject, regional_samples,
    attention_pool_weights, token_neighborhood, local_geometry_samples, write_ply, font, mesh_layer,
    draw_arrow, draw_samples_crop, contact_sheet,
)


def project_path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/vis_groundedhuman_teaser_sofa.yaml')
    for name in ('image', 'output-dir', 'checkpoint', 'scale-checkpoint', 'device'):
        p.add_argument('--'+name)
    p.add_argument('--person-rank', type=int)
    p.add_argument('--preflight-only', action='store_true')
    return p.parse_args()


def preflight(args):
    import yaml
    settings = yaml.safe_load(project_path(args.config).read_text(encoding='utf-8'))
    for key in ('image','output_dir','checkpoint','scale_checkpoint','device','person_rank'):
        if getattr(args,key) is not None:
            settings[key] = getattr(args,key)
    output = project_path(settings['output_dir'])
    if not output.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Teaser outputs must be below the repository outputs/ directory')
    path_cfg = yaml.safe_load(project_path(settings['path_config']).read_text(encoding='utf-8'))
    required = {key: project_path(settings[key]) for key in ('image','model_config','checkpoint','scale_checkpoint')}
    for key in ('vggt_baseline', 'nlf_smpl'):
        value = path_cfg.get('checkpoints',{}).get(key)
        if not value:
            raise ValueError(f'Missing checkpoints.{key} in {settings["path_config"]}')
        required[key] = project_path(value)
    body_dir = path_cfg.get('assets',{}).get('smpl_model_dir')
    if not body_dir:
        raise ValueError('Missing assets.smpl_model_dir in path config')
    missing = [str(path) for path in required.values() if not path.is_file()]
    if not project_path(body_dir).is_dir():
        missing.append(str(project_path(body_dir)))
    if missing:
        raise FileNotFoundError('Prepare these server resources (no fallback/random heads):\n'+'\n'.join(missing))
    if int(settings['person_rank']) < 0 or int(settings['render_scale']) < 1:
        raise ValueError('person_rank must be nonnegative and render_scale positive')
    if int(settings['scene_stride']) < 1 or float(settings['max_scene_depth']) <= 0:
        raise ValueError('scene_stride and max_scene_depth must be positive')
    output.mkdir(parents=True,exist_ok=True)
    json_write(output/'preflight.json', {'status':'resources_present','input':str(required['image']),
        'resources':{k:str(v) for k,v in required.items()},'output':str(output)})
    return settings, output


def np_cpu(value):
    return value.detach().cpu().numpy().copy()


def load_head_checkpoint(model, path, prefixes):
    """Load only requested heads; scale overlay must never overwrite TRSTR/baseline."""
    import torch
    from scripts.vis.visualize_smpl_inference import extract_state_dict
    state = extract_state_dict(torch.load(path, map_location='cpu', weights_only=False))
    expected = {k:v for k,v in model.state_dict().items() if k.startswith(prefixes)}
    missing = [k for k in expected if k not in state]
    mismatch = [k for k,v in expected.items() if k in state and v.shape != state[k].shape]
    if missing or mismatch:
        raise RuntimeError(f'Incomplete/incompatible head checkpoint {path}: missing={missing[:12]}, shape={mismatch[:12]}')
    if not expected:
        raise RuntimeError(f'No model parameters match {prefixes}')
    model.load_state_dict({k:state[k] for k in expected},strict=False)
    return {'path':str(path),'loaded_tensors':len(expected),'prefixes':list(prefixes)}


class Capture:
    def __init__(self, model, query):
        self.model, self.query = model, query
        self.attention, self.probes, self.logvars, self.person_gates = [], [], [], []
        self.handles=[]; self.logits=None

    def __enter__(self):
        hsi=self.model.hsi_refinement_head; head=self.model.hsi_trstr_head
        self.handles.append(hsi.blocks[-1].cross_attn.register_forward_hook(self.cross_hook))
        self.handles.append(head.scene_probe.attention_score.register_forward_hook(self.logit_hook))
        self.handles.append(head.scene_probe.register_forward_hook(self.probe_hook,with_kwargs=True))
        self.handles.append(head.logvar_head.register_forward_hook(
            lambda m,a,o: self.logvars.append(np_cpu(o[self.query].clamp(-4,4)))))
        self.handles.append(head.person_gate_head.register_forward_hook(
            lambda m,a,o: self.person_gates.append(np_cpu(o[self.query].sigmoid()))))
        return self

    def __exit__(self,*exc):
        for handle in self.handles:
            handle.remove()

    def cross_hook(self,module,args,output):
        import torch
        # Direct forward bypasses this hook, leaves actual model output untouched.
        sl=slice(self.query*24,(self.query+1)*24)
        with torch.no_grad():
            replay,weights=module.forward(args[0][sl],args[1][sl],args[2][sl],
                                          need_weights=True,average_attn_weights=False)
        error=float((replay-output[0][sl]).abs().max().cpu())
        if not torch.isfinite(weights).all() or not torch.allclose(replay,output[0][sl],atol=1e-4,rtol=1e-3):
            raise RuntimeError(f'Attention replay differs from actual forward: max abs {error}')
        self.attention.append({'weights':np_cpu(weights[:,:,0,:]),'replay_max_abs':error})

    def logit_hook(self,module,args,output):
        self.logits=np_cpu(output[self.query,...,0])

    def probe_hook(self,module,args,kwargs,output):
        import torch
        from vggt_omega.models.geometry.regional_scene_probe import _rasterize_human_depth_and_owner
        if kwargs['centers'].shape[0] != 1:
            raise ValueError('This exporter supports exactly one image (B=S=1)')
        with torch.no_grad():
            hd,owner=_rasterize_human_depth_and_owner(
                kwargs['vertices_by_frame'],kwargs['intrinsics_by_frame'],kwargs['person_valid'],
                kwargs['image_size_hw'],tuple(kwargs['depth_by_frame'].shape[-2:]),module.human_depth_dilation_px)
        self.probes.append(dict(
            centers=np_cpu(kwargs['centers'][0,self.query]),
            representatives=np_cpu(kwargs['representatives'][0,self.query]),
            vertices=np_cpu(kwargs['vertices_by_frame'][0,self.query]),
            depth=np_cpu(kwargs['depth_by_frame'][0]),human_depth=np_cpu(hd[0]),owner=np_cpu(owner[0]),
            intrinsics=np_cpu(kwargs['intrinsics_by_frame'][0]),
            image_hw=np.array(kwargs['image_size_hw']),
            logits=self.logits.copy(),valid_ratios=np_cpu(output['valid_ratios'][self.query]),
            region_valid=np_cpu(output['region_valid'][self.query]),
            adaptive_radius=np_cpu(output['adaptive_radius'][self.query]),
            tokens=np_cpu(output['tokens'][self.query]),
        ))


@contextmanager
def without_trstr(model):
    head=model.hsi_trstr_head;model.hsi_trstr_head=None
    try:
        yield
    finally:
        model.hsi_trstr_head=head


def run_inference(settings, output):
    import torch
    from scripts.train.train_smpl import build_model, load_yaml_config
    from scripts.vis.visualize_smpl_inference import load_vggt_baseline_for_camera, load_image
    from scripts.vis.export_paper_architecture_real_assets import compute_coarse_hypotheses
    from vggt_omega.training.config import deep_update
    from vggt_omega.models.heads.hsi_refinement_head import _canonical_depth
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device=torch.device(settings['device'])
    config=deep_update(load_yaml_config(str(project_path(settings['path_config']))),
                       load_yaml_config(str(project_path(settings['model_config']))))
    mc=config['model']
    if mc.get('hsi_trstr_enable_temporal') or mc.get('hsi_enable_temporal_momentum'):
        raise ValueError('Single-image teaser export requires spatial-only inference')
    model=build_model(config).to(device).eval()
    load_vggt_baseline_for_camera(model,config,device)
    audits=[load_head_checkpoint(model,project_path(settings['checkpoint']),('hsi_trstr_head.',)),
            load_head_checkpoint(model,project_path(settings['scale_checkpoint']),('hsi_refinement_head.',))]
    if model.hsi_refinement_head.use_affine_depth_for_transl:
        raise ValueError('Use the accepted inference config: this exporter expects coarse-depth HSI without pre-affine translation')
    resolution=int(config['data'].get('image_resolution',512))
    images,original=load_image(project_path(settings['image']),resolution)
    images=images.to(device);hw=tuple(images.shape[-2:])
    rgb=Image.fromarray((np_cpu(images[0,0]).transpose(1,2,0)*255).round().astype(np.uint8))
    original.save(output/'input_original.png');rgb.save(output/'input_processed.png')
    print('[teaser] VGGT/NLF base pass',flush=True)
    with torch.no_grad(), without_trstr(model):
        first=model(images)
    conf=np_cpu(first['pred_confs'][0,0,:,0]);order=np.argsort(-conf)
    people=[int(q) for q in order if conf[q] >= settings['confidence_threshold']]
    rank=int(settings['person_rank'])
    if rank >= len(people):
        raise RuntimeError(f'Only {len(people)} confident people; cannot select person_rank={rank}')
    query=people[rank];head=model.hsi_trstr_head;hsi=model.hsi_refinement_head
    with torch.no_grad():
        vertices=head._decode_vertices(first['pred_pose_6d'].float(),first['pred_betas'].float(),first['pred_transl_cam'].float())
    raw=_canonical_depth(first['depth']).float()
    hypotheses=compute_coarse_hypotheses(vertices[people],raw[0,0],first['pose_enc'],
        int(settings['coarse_anchor_stride']),float(settings['coarse_scale_min']),float(settings['coarse_scale_max']))
    valid=hypotheses['in_range']
    if valid.sum()<settings['coarse_min_anchor_pixels']:
        raise RuntimeError('Insufficient human-surface scale correspondences; cannot export a claimed metric result')
    coarse_scale=float(np.median(hypotheses['ratio'][valid]));coarse=raw*coarse_scale
    override={key:first[key] for key in ('pred_pose_6d','pred_poses','pred_betas','pred_transl_cam','pred_confs','pred_boxes','pred_cam','base_pred_transl_cam') if key in first}
    print(f'[teaser] selected query={query}, coarse scale={coarse_scale:.6g}; HSI residual pass',flush=True)
    with Capture(model,query) as capture, torch.no_grad():
        with without_trstr(model):
            final=model(images,smpl_override_outputs=override,hsi_depth_override=coarse,
                        hsi_depth_is_metric=True,hsi_geometry_mode='smpl_coarse_metric')
        residual=final['hsi_scene_scale'].float();bias=final['hsi_scene_depth_bias'].float()
        metric=coarse*residual[...,None]+bias[...,None]
        if not torch.isfinite(metric).any() or not torch.isfinite(residual).all() or not torch.isfinite(bias).all():
            raise RuntimeError('Invalid calibrated depth/scale')
        # Important: pass composed depth explicitly, as the sequence cascade does.
        final['hsi_translation_depth']=metric
        final['hsi_scene_scale']=residual*coarse_scale
        final['hsi_scene_depth_bias']=bias
        print('[teaser] TRSTR on fully calibrated metric depth',flush=True)
        final.update(head(predictions=final,depth=metric,pose_enc=final['pose_enc'],
                          image_size_hw=hw,depth_is_metric=True))
        anchors=hsi._anchors_cam(first['pred_pose_6d'].float(),first['pred_betas'].float(),first['pred_transl_cam'].float())
        extrinsics,k=encoding_to_camera(first['pose_enc'],image_size_hw=hw,build_intrinsics=True)
    for key in ('pred_pose_6d','pred_betas','pred_transl_cam'):
        if not torch.equal(first[key],final[key]):
            raise RuntimeError(f'Frozen base output changed during cascade: {key}')
    torch.testing.assert_close(first['pose_enc'],final['pose_enc'],atol=1e-5,rtol=1e-4)
    torch.testing.assert_close(raw,_canonical_depth(final['depth']).float(),atol=1e-5,rtol=1e-4)
    if len(capture.probes)!=head.num_iters+1 or len(capture.logvars)!=head.num_iters:
        raise RuntimeError('Unexpected TRSTR hook sequence; do not mislabel iteration/final samples')
    faces=np.asarray(head.smpl.faces,dtype=np.int64)
    data=dict(query=query,confidence=float(conf[query]),rgb=rgb,hw=hw,k=np_cpu(k[0,0]),
              extrinsics=np_cpu(extrinsics[0,0]),vertices_base=np_cpu(vertices[query]),faces=faces,
              anchors=np_cpu(anchors[0,0,query]),raw=np_cpu(raw[0,0]),coarse=np_cpu(coarse[0,0]),
              metric=np_cpu(metric[0,0]),coarse_scale=coarse_scale,residual_scale=float(residual.flatten()[0].cpu()),
              depth_bias=float(bias.flatten()[0].cpu()),hypotheses=hypotheses,capture=capture,
              head=head,hsi=hsi,first=first,config=config,audits=audits,
              group_names=head.region_bank.group_names,group_ids=np_cpu(head.region_bank.region_group_ids),
              vertex_region_ids=np_cpu(head.region_bank.vertex_region_ids),
              representative_indices=np_cpu(head.region_bank.representative_indices))
    data['predictions']={key:np_cpu(value) for key,value in final.items()
                         if key.startswith('hsi_trstr_') and isinstance(value,torch.Tensor)}
    return data


def export_hsi(data, settings, output):
    import torch
    from vggt_omega.models.heads.hsi_refinement_head import _estimate_depth_normals, _local_nearest_scene_probe
    directory=output/'anchors';directory.mkdir(exist_ok=True)
    hsi=data['hsi'];query=data['query'];first=data['first'];device=first['pred_transl_cam'].device
    with torch.no_grad():
        all_anchors=hsi._anchors_cam(first['pred_pose_6d'].float(),first['pred_betas'].float(),first['pred_transl_cam'].float())
        k=torch.as_tensor(data['k'],device=device)[None]
        depth=torch.as_tensor(data['coarse'],device=device)[None,None]
        uv=torch.as_tensor(project(np_cpu(all_anchors),data['k']),device=device,dtype=torch.float32)
        uv_depth=uv*uv.new_tensor([depth.shape[-1]/data['hw'][1],depth.shape[-2]/data['hw'][0]])
        depth_height,depth_width=depth.shape[-2:]
        depth_normals=_estimate_depth_normals(depth,k,height=depth_height,width=depth_width)
        nearest,normals=_local_nearest_scene_probe(depth,depth_normals,all_anchors,
            uv_depth,k,data['hw'],hsi.probe_window)
    # This accepted config uses local_nearest for affine scale; avoid silently wrong provenance.
    if hsi.affine_probe_mode!='local_nearest' or float(hsi.probe_blend)!=1.0:
        raise ValueError('Teaser geometry export expects local_nearest with probe_blend=1')
    nearest=np_cpu(nearest[0,0,query]);normals=np_cpu(normals[0,0,query])
    cells,positions,grid=token_neighborhood(data['anchors'],data['k'],data['hw'],data['raw'].shape,hsi.scene_window)
    attention=data['capture'].attention[-1]
    weights=attention['weights'].mean(1)
    if weights.shape!=(24,hsi.scene_window**2):
        raise RuntimeError(f'Unexpected anchor attention shape {weights.shape}')
    np.savez_compressed(directory/'all_anchors.npz',anchors_cam=data['anchors'],projected_uv=project(data['anchors'],data['k']),
        nearest_scene_cam=nearest,nearest_scene_normal=normals,offset=nearest-data['anchors'],
        token_cells=cells,token_centers_uv=positions,attention_per_head=attention['weights'],attention_mean=weights)
    rgb=data['rgb'];w,h=rgb.size;scale=int(settings['render_scale'])
    layer=Image.new('RGBA',(w*scale,h*scale));ld=ImageDraw.Draw(layer)
    for uv in project(data['anchors'],data['k']):
        if np.isfinite(uv).all():
            x,y=uv*scale;r=4*scale;ld.ellipse((x-r,y-r,x+r,y+r),fill=CORAL,outline='white',width=scale)
    layer.save(output/'layers/anchors_rgba.png')
    from scripts.vis.create_hsi_paper_ply_elements import MeshBuilder, add_uv_sphere, add_cylinder
    markers=MeshBuilder();links=MeshBuilder()
    for a,b in zip(data['anchors'],nearest):
        if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a[2],b[2])<=0:continue
        add_uv_sphere(markers,a,.012,CORAL,rings=6,segments=10)
        add_uv_sphere(links,b,.008,AMBER,rings=6,segments=10)
        if np.linalg.norm(a-b)>1e-7:add_cylinder(links,a,b,.0025,AMBER,sections=8)
    markers.write(output/'geometry/body_anchors.ply');links.write(output/'geometry/anchor_nearest_scene_links.ply')
    selected=[]
    for a in settings['anchor_indices']:
        if not 0<=a<24:raise ValueError(f'Invalid anchor index {a}')
        anchor_uv=project(data['anchors'][a],data['k'])
        if not np.isfinite(anchor_uv).all() or not (0<=anchor_uv[0]<w and 0<=anchor_uv[1]<h):continue
        pixel_size=np.array([w/grid[1],h/grid[0]])
        bounds=np.r_[positions[a].min(0)-pixel_size*.65,positions[a].max(0)+pixel_size*.65]
        size=600
        bg=rgb.transform((size,size),Image.Transform.EXTENT,tuple(bounds),resample=Image.Resampling.BILINEAR)
        ov=Image.new('RGBA',bg.size);draw=ImageDraw.Draw(ov)
        factor=np.array([size,size])/(bounds[2:]-bounds[:2])
        def point(p):return tuple((np.asarray(p)-bounds[:2])*factor)
        for p,weight in zip(positions[a],weights[a]):
            x,y=point(p);strength=float(weight/max(weights[a].max(),1e-8))
            draw.rectangle([point(p-pixel_size/2),point(p+pixel_size/2)],outline=(*TEAL,210),width=2)
            draw.line([point(anchor_uv),(x,y)],fill=(*CORAL,int(40+215*strength)),width=max(1,int(1+4*strength)))
            r=5+6*strength;draw.ellipse((x-r,y-r,x+r,y+r),fill=(*AMBER,255),outline='white')
        x,y=point(anchor_uv);draw.ellipse((x-8,y-8,x+8,y+8),fill=CORAL,outline='white')
        stem=directory/f'anchor_{a:02d}'
        bg.save(stem.with_name(stem.name+'_background.png'));ov.save(stem.with_name(stem.name+'_overlay.png'))
        Image.alpha_composite(bg.convert('RGBA'),ov).save(stem.with_suffix('.png'))
        candidates=local_geometry_samples(data['anchors'][a],data['coarse'],data['k'],data['hw'],hsi.probe_window)
        nearest_index=int(candidates['nearest_index'])
        json_write(stem.with_suffix('.json'),{'anchor_index':a,'image_crop_xyxy':bounds.tolist(),
            'grid_hw':list(grid),'attention_mean':weights[a].tolist(),'token_cells':cells[a].tolist(),
            'geometry_probe_window_depth_px':int(hsi.probe_window),'geometry_has_valid_sample':nearest_index>=0,
            'nearest_scene_cam_m':nearest[a].tolist() if np.isfinite(nearest[a]).all() else None,
            'attention_source':'last HSI block, final affine-prediction iteration; heads averaged',
            'attention_replay_max_abs':attention['replay_max_abs']})
        if nearest_index>=0:
            np.testing.assert_allclose(candidates['scene_xyz'][nearest_index],nearest[a],atol=2e-5,rtol=1e-5)
        np.savez_compressed(directory/f'anchor_{a:02d}_geometry_search.npz',**candidates)
        # Geometry search is its own asset, not a falsely labelled 3x3 token neighborhood.
        c=candidates['center_uv'];r=hsi.probe_window//2+1
        bb=(float(round(c[0])-r-.5),float(round(c[1])-r-.5),float(round(c[0])+r+.5),float(round(c[1])+r+.5))
        geo_bg=rgb.resize((data['coarse'].shape[1],data['coarse'].shape[0])).transform(
            (480,480),Image.Transform.EXTENT,bb,resample=Image.Resampling.BILINEAR)
        geo_ov=Image.new('RGBA',(480,480));gd=ImageDraw.Draw(geo_ov);pitch=480/(2*r+1)
        for j,uvp in enumerate(candidates['raw_xy']):
            x,y=(uvp-np.array(bb[:2]))*pitch
            color=AMBER if j==nearest_index else (100,130,145)
            rr=8 if j==nearest_index else 4
            if candidates['valid'][j]:gd.ellipse((x-rr,y-rr,x+rr,y+rr),fill=color,outline='white')
            else:gd.line((x-rr,y-rr,x+rr,y+rr),fill=(150,150,150),width=2)
        x,y=(c-np.array(bb[:2]))*pitch
        gd.line((x-9,y,x+9,y),fill=CORAL,width=3);gd.line((x,y-9,x,y+9),fill=CORAL,width=3)
        geo_bg.save(directory/f'anchor_{a:02d}_geometry_background.png')
        geo_ov.save(directory/f'anchor_{a:02d}_geometry_overlay.png')
        Image.alpha_composite(geo_bg.convert('RGBA'),geo_ov).save(directory/f'anchor_{a:02d}_geometry_search.png')
        selected.append(a)
    return selected


def export_regions(data, settings, output):
    head=data['head'];probe=head.scene_probe;capture=data['capture'];query=data['query'];pred=data['predictions']
    directory=output/'regions';directory.mkdir(exist_ok=True)
    votes=pred['hsi_trstr_iteration_region_vote'][:,0,0,query]
    gates=pred['hsi_trstr_iteration_region_gate'][:,0,0,query,:,0]
    valid=pred['hsi_trstr_iteration_region_valid'][:,0,0,query]
    translations=pred['hsi_trstr_iteration_transl'][:,0,0,query]
    logvars=np.stack(capture.logvars)[...,0]
    person_gates=np.stack(capture.person_gates).reshape(-1)
    weights=valid*gates*np.exp(-logvars)
    aggregates=(votes*weights[...,None]).sum(1)/np.maximum(weights.sum(1,keepdims=True),1e-5)
    updates=np.clip(aggregates,-head.max_person_delta_m,head.max_person_delta_m)*person_gates[:,None]
    np.testing.assert_allclose(np.diff(translations,axis=0),updates,atol=2e-6,rtol=1e-4)
    np.savez_compressed(directory/'translation_iterations.npz',translation=translations,vote=votes,
        gate=gates,logvar=logvars,weight=weights,valid=valid,person_gate=person_gates,update=updates,
        group_ids=data['group_ids'],vertex_region_ids=data['vertex_region_ids'],representative_indices=data['representative_indices'])
    # Last update's INPUT regions, not the post-update final probe: votes and geometry align.
    iteration=head.num_iters-1;snapshot=capture.probes[iteration]
    chosen=[]
    for group in settings['region_groups']:
        if group not in data['group_names']:raise ValueError(f'Unknown region group {group}')
        ids=np.flatnonzero(data['group_ids']==data['group_names'].index(group))
        candidates=[]
        for r in ids:
            uv=project(snapshot['centers'][r],data['k'])
            if np.isfinite(uv).all() and (snapshot['centers'][r,2]>0) and 0<=uv[0]<data['hw'][1] and 0<=uv[1]<data['hw'][0]:
                # Prefer valid mixed evidence, not merely large corrections or a flattering visual.
                ratios=snapshot['valid_ratios'][r].reshape(-1,2)
                score=float(np.minimum(ratios[:,0],ratios[:,1]).max())
                candidates.append((bool(valid[iteration,r]),score,float(weights[iteration,r]),int(r)))
        chosen.extend(item[-1] for item in sorted(candidates,reverse=True)[:int(settings['regions_per_group'])])
    if not chosen:raise RuntimeError('No requested body regions project into the input image')
    records=[]
    for it,snap in enumerate(capture.probes):
        np.savez_compressed(directory/f'probe_state_{it:02d}.npz',**snap)
    for r in chosen:
        group=data['group_names'][data['group_ids'][r]]
        samples=regional_samples(snapshot['centers'][r],snapshot['representatives'][r],snapshot['depth'],
            snapshot['human_depth'],snapshot['owner'],snapshot['intrinsics'],snapshot['image_hw'],query,
            probe.fixed_patch_sizes,probe.adaptive_radius_max,probe.annulus_width,probe.human_depth_tolerance_m)
        np.testing.assert_allclose(samples['valid_ratios'],snapshot['valid_ratios'][r],atol=1e-6)
        if int(samples['adaptive_radius'])!=int(snapshot['adaptive_radius'][r]):
            raise RuntimeError('Exported adaptive support differs from the model')
        samples['pool_weights']=attention_pool_weights(snapshot['logits'][r],samples['channel_masks'])
        stem=directory/f'{group}_r{r:02d}'
        np.savez_compressed(stem.with_suffix('.npz'),**samples,center_cam=snapshot['centers'][r],
                            representative_cam=snapshot['representatives'][r],vote_cam=votes[iteration,r],
                            reliability_weight=weights[iteration,r])
        status='valid' if valid[iteration,r] else 'UNSUPPORTED'
        draw_samples_crop(data['rgb'],samples,snapshot['depth'].shape,stem,
                          f'{group} r{r:02d} | update {iteration+1} input | {status}')
        for si,name in enumerate(probe.scale_names):
            draw_samples_crop(data['rgb'],samples,snapshot['depth'].shape,
                stem.with_name(stem.name+'_'+name),name,support_index=si,size=480)
            # 3D points are true backprojections; windows themselves remain in image space.
            for channel,color in [('self_surface',TEAL),('environment',AMBER)]:
                keep=samples['supports'][si]&samples[channel]&np.isfinite(samples['scene_xyz']).all(-1)
                write_ply(output/'geometry'/f'{group}_r{r:02d}_{name}_{channel}.ply',samples['scene_xyz'][keep],color)
        records.append({'region':int(r),'group':group,'iteration_input':iteration,'valid':bool(valid[iteration,r]),
                        'adaptive_radius_px':int(samples['adaptive_radius']),'weight':float(weights[iteration,r]),
                        'vote_cam_m':votes[iteration,r].tolist(),'sample_stem':str(stem.relative_to(output))})
    np.testing.assert_allclose(data['vertices_base']+translations[-1]-translations[0],capture.probes[-1]['vertices'],atol=2e-5)
    json_write(directory/'selection.json',{'selection':'valid, then mixed self/environment support, then reliability; not error reduction',
        'regions':records,'root_delta_m':(translations[-1]-translations[0]).tolist(),
        'all_region_weights_in':'translation_iterations.npz','last_update_input_index':iteration})
    return records, translations, weights, votes


def export_scene_and_layers(data,settings,output,translations,weights,votes):
    rgb=data['rgb'];k=data['k'];w,h=rgb.size;scale=int(settings['render_scale']);faces=data['faces']
    base=data['vertices_base'];refined=data['capture'].probes[-1]['vertices']
    layers=output/'layers';geometry=output/'geometry';pred=data['predictions']
    base_layer,base_z=mesh_layer(base,faces,k,rgb.size,CORAL,scale)
    human,refined_z=mesh_layer(refined,faces,k,rgb.size,TEAL,scale)
    base_layer.save(layers/'human_base_rgba.png');human.save(layers/'human_refined_rgba.png')
    write_ply(geometry/'human_base.ply',base,CORAL,faces);write_ply(geometry/'human_refined.ply',refined,TEAL,faces)
    snap=data['capture'].probes[-1];depth=data['metric'];dh,dw=depth.shape
    y,x=np.mgrid[:dh,:dw];xy=np.stack([x,y],-1)
    xyz=unproject(xy,depth,k,data['hw'],depth.shape)
    valid=np.isfinite(xyz).all(-1)&(depth>1e-5)&(depth<float(settings['max_scene_depth']))
    # Geometry ownership matches the model. It is not an inpainted clean background.
    human_pixels=np.isfinite(snap['human_depth'])&(np.abs(depth-snap['human_depth'])<=data['head'].scene_probe.human_depth_tolerance_m)
    env=valid&~human_pixels
    color=np.asarray(rgb.resize((dw,dh),Image.Resampling.BILINEAR))
    stride=int(settings['scene_stride']);sparse=np.zeros_like(env);sparse[::stride,::stride]=True
    write_ply(geometry/'scene_metric_all.ply',xyz[valid&sparse],color[valid&sparse])
    write_ply(geometry/'scene_metric_environment.ply',xyz[env&sparse],color[env&sparse])
    # Keep raw scale-ambiguous scene distinct from the calibrated metric scene.
    raw_xyz=unproject(xy,data['raw'],k,data['hw'],depth.shape)
    raw_valid=np.isfinite(raw_xyz).all(-1)&(data['raw']>1e-5)&sparse
    write_ply(geometry/'scene_raw_scale_ambiguous.ply',raw_xyz[raw_valid],color[raw_valid])
    env_rgba=np.dstack([color,np.where(env,255,0).astype(np.uint8)])
    env_image=Image.fromarray(env_rgba).resize(human.size,Image.Resampling.NEAREST)
    env_image.save(layers/'environment_rgba.png')
    Image.fromarray((human_pixels*255).astype(np.uint8)).save(layers/'geometry_human_mask.png')
    # Occlusion is a display layer only; body RGBA above remains complete for compositing.
    depth_up=np.asarray(Image.fromarray(depth).resize(human.size,Image.Resampling.NEAREST))
    env_up=np.asarray(Image.fromarray(env).resize(human.size,Image.Resampling.NEAREST))
    human_arr=np.array(human);occluded=env_up&(depth_up+.02<refined_z)
    human_arr[occluded,3]=0
    visible_human=Image.fromarray(human_arr);visible_human.save(layers/'human_scene_occluded_rgba.png')
    overlay=Image.alpha_composite(rgb.resize(human.size).convert('RGBA'),visible_human)
    overlay.save(output/'human_scene_overlay.png')
    white=Image.new('RGBA',human.size,'white')
    scene_composite=Image.alpha_composite(Image.alpha_composite(white,env_image),visible_human)
    scene_composite.save(output/'human_scene_white.png')
    from scripts.vis.create_hsi_paper_ply_elements import MeshBuilder, add_arrow
    region_input=data['capture'].probes[-2]['centers'];last_weights=weights[-1]
    order=np.argsort(-last_weights)[:12]
    for gain in dict.fromkeys([1.0,float(settings['arrow_display_gain'])]):
        arrow_layer=Image.new('RGBA',human.size);draw=ImageDraw.Draw(arrow_layer);mesh=MeshBuilder()
        for r in order:
            if last_weights[r]<=0:continue
            a=region_input[r];b=a+votes[-1,r]*gain
            if min(a[2],b[2])<=0:continue
            uv=project(np.stack([a,b]),k)*scale
            opacity=int(80+175*last_weights[r]/max(last_weights.max(),1e-8))
            draw_arrow(draw,*uv,(*PURPLE,opacity),width=max(2,scale))
            add_arrow(mesh,a,b,.0025,PURPLE)
        # Actual net root update (base -> final); a 1x zero update has no arrow.
        a=base.mean(0);b=a+(translations[-1]-translations[0])*gain
        if min(a[2],b[2])>0:
            draw_arrow(draw,*(project(np.stack([a,b]),k)*scale),(*TEAL,255),width=max(3,2*scale))
            add_arrow(mesh,a,b,.006,TEAL)
        suffix=f'{gain:g}x'
        arrow_layer.save(layers/f'translation_arrows_{suffix}_rgba.png');mesh.write(geometry/f'translation_arrows_{suffix}.ply')
        combined=Image.alpha_composite(scene_composite,arrow_layer)
        dd=ImageDraw.Draw(combined);dd.text((18,18),f'Translation arrows: {suffix} display gain',font=font(20*scale),fill=INK)
        combined.save(output/f'translation_preview_{suffix}.png')
    np.savez_compressed(output/'camera_geometry.npz',intrinsics=k,extrinsics_raw=data['extrinsics'],
        depth_raw=data['raw'],depth_coarse=data['coarse'],depth_metric=data['metric'],
        base_vertices=base,refined_vertices=refined,faces=faces,processed_hw=np.array(data['hw']),
        source_hw=np.array([Image.open(project_path(settings['image'])).height,Image.open(project_path(settings['image'])).width]))


def main():
    args=parse_args();settings,output=preflight(args)
    if args.preflight_only:
        print(f'[teaser] resource preflight passed: {output}/preflight.json');return
    if (output/'manifest.json').exists():
        raise FileExistsError(f'Completed export already exists: {output}. Choose another OUTPUT_DIR to preserve it.')
    if any(p.name!='preflight.json' for p in output.iterdir()):
        raise FileExistsError(f'Output directory contains a previous partial export: {output}. Use a fresh OUTPUT_DIR.')
    for name in ('layers','geometry','anchors','regions'):(output/name).mkdir(exist_ok=True)
    data=run_inference(settings,output)
    anchors=export_hsi(data,settings,output)
    regions,translations,weights,votes=export_regions(data,settings,output)
    print('[teaser] rendering transparent layers and exporting geometry',flush=True)
    export_scene_and_layers(data,settings,output,translations,weights,votes)
    np.savez_compressed(output/'scale_correspondences.npz',**data['hypotheses'])
    items=[('Processed input',data['rgb']),('Calibrated scene + human',Image.open(output/'human_scene_white.png')),
           ('Base human',Image.open(output/'layers/human_base_rgba.png'))]
    items += [(f'Anchor {a}: local token attention',Image.open(output/f'anchors/anchor_{a:02d}.png')) for a in anchors[:3]]
    items += [(f'{r["group"]}: r{r["region"]}',Image.open(output/(r['sample_stem']+'.png'))) for r in regions]
    contact_sheet(items,output/'asset_contact_sheet.png')
    manifest={'status':'complete','input':str(project_path(settings['image'])),
        'input_sha256':hashlib.sha256(project_path(settings['image']).read_bytes()).hexdigest(),
        'settings':settings,'checkpoint_audit':data['audits'],'selected_query':data['query'],'confidence':data['confidence'],
        'scale':{'coarse':data['coarse_scale'],'residual':data['residual_scale'],
                 'effective':data['coarse_scale']*data['residual_scale'],'depth_bias_m':data['depth_bias']},
        'anchors_exported':anchors,'regions':regions,'root_delta_cam_m':(translations[-1]-translations[0]).tolist(),
        'coordinate_contract':{'geometry':'camera coordinates: x right, y down, z forward; metric meters except scene_raw',
          'intrinsics':'processed RGB pixels; original image uses different resize/crop coordinates',
          'anchor_window':'3x3 FEATURE TOKENS, not 3x3 depth pixels',
          'regional_window':'depth pixels, Chebyshev square support and square annulus',
          'region_snapshots':'probe_state_00..num_iters-1 are update inputs; last is final geometry audit',
          'camera_extrinsics_raw':'stored for provenance; translation still scale-ambiguous, not a metric world transform'},
        'validation':{'pose_shape_preserved':True,'regional_masks_match_live_valid_ratios':True,
                      'region_weight_aggregation_matches_actual_updates':True,
                      'attention_replay_max_abs':max(a['replay_max_abs'] for a in data['capture'].attention)},
        'limitations':['Single image: no temporal consensus or unseen-side geometry.',
                      'Contact is a prediction, not a hard physical guarantee.',
                      'Environment mask is depth/SMPL geometry based; no person removal inpainting.',
                      'Arrow gain affects explicitly named display layers only; NPZ/PLY body positions remain measured.'],
        'files':sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())}
    json_write(output/'manifest.json',manifest)
    print(json.dumps({'output':str(output),'manifest':str(output/'manifest.json'),'regions':len(regions),'anchors':len(anchors)},indent=2))


if __name__=='__main__':
    main()
