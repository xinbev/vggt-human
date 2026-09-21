#!/usr/bin/env python3
"""Synthetic export checks; --with-torch also compares to the live model sampler.

These fixtures are NEVER presented as results from sofa.png.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scripts.vis.teaser_asset_utils import (
    regional_samples, attention_pool_weights, token_neighborhood, local_geometry_samples, unproject,
    mesh_layer, draw_samples_crop, write_ply, json_write,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='outputs/debug/groundedhuman_teaser_smoke')
    p.add_argument('--with-torch',action='store_true')
    args=p.parse_args();out=Path(args.output_dir).resolve()
    if not out.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Output must stay in outputs/')
    out.mkdir(parents=True,exist_ok=True)
    k=np.array([[60,0,30],[0,80,20],[0,0,1]],np.float32)
    hw=(40,60);shape=(20,30)
    center=np.array([0,0,2],np.float32)
    reps=np.tile(center,(8,1));reps[0,0]+=.10;reps[1,1]-=.10
    depth=np.full(shape,2,np.float32);hd=np.full(shape,np.inf,np.float32)
    owner=np.full(shape,-1,np.int64)
    hd[8:14,13:17]=2;owner[8:14,13:17]=0
    hd[10,18]=2;owner[10,18]=1
    depth[10,12]=np.nan
    sample=regional_samples(center,reps,depth,hd,owner,k,hw,0)
    assert sample['supports'][0].sum()==9
    assert sample['supports'][1].sum()==49
    radius=int(sample['adaptive_radius'])
    assert sample['supports'][2].sum()==(2*radius+1)**2
    assert not (sample['supports'][2]&sample['supports'][3]).any()
    assert sample['supports'][3].sum()==(2*(radius+2)+1)**2-(2*radius+1)**2
    assert not (sample['self_surface']&sample['environment']).any()
    assert not (sample['other_human']&sample['channel_masks'].any((0,1))).any()
    assert sample['other_human'].sum()==1
    assert sample['self_surface'].any() and sample['environment'].any()
    # Invalid pixels remain excluded, rather than converted to black/near-zero geometry.
    assert not sample['valid'][np.all(sample['raw_xy']==[12,10],axis=-1)].any()
    weights=attention_pool_weights(np.zeros(len(sample['valid'])),sample['channel_masks'])
    nonempty=sample['channel_masks'].any(-1)
    np.testing.assert_allclose(weights.sum(-1)[nonempty],1,atol=1e-6)
    assert (weights[~nonempty]==0).all()
    empty=regional_samples(center,reps,np.full(shape,np.nan,np.float32),hd,owner,k,hw,0)
    assert not empty['valid'].any() and not empty['valid_ratios'].any()
    assert not attention_pool_weights(np.zeros(len(empty['valid'])),empty['channel_masks']).any()
    # Border support: clamping must not fabricate duplicate valid observations.
    edge_center=unproject(np.array([0,0]),np.array(2),k,hw,shape)
    edge=regional_samples(edge_center,np.tile(edge_center,(8,1)),depth,hd,owner,k,hw,0)
    assert (edge['supports'][0]&edge['valid']).sum()==4
    assert edge['valid_ratios'][1]==4/9
    geometry=local_geometry_samples(center,depth,k,hw,9)
    assert len(geometry['raw_xy'])==81 and int(geometry['nearest_index'])>=0
    np.testing.assert_allclose(geometry['scene_xyz'][int(geometry['nearest_index'])],center)
    no_geometry=local_geometry_samples(center,np.full(shape,np.nan),k,hw,9)
    assert int(no_geometry['nearest_index'])==-1
    cells,uv,grid=token_neighborhood(np.array([[0,0,2],[-1,-1,2]],np.float32),k,(64,96),(64,96))
    assert cells.shape==(2,9,2) and grid==(4,6)
    assert len(np.unique(cells[1],axis=0))<9 # current HSI border gather repeats cells
    # Camera projection respects rectangular source/depth size and off-center K.
    xyz=unproject(np.array([[4,6],[20,14]]),np.array([2,4]),k,hw,shape)
    np.testing.assert_allclose(xyz[:,0],(np.array([8,40])-30)*np.array([2,4])/60)
    # Visible front triangle must occlude the farther triangle regardless of face order.
    verts=np.array([[-.25,-.25,1],[.25,-.25,1],[0,.25,1],[-.5,-.5,2],[.5,-.5,2],[0,.5,2]],np.float32)
    faces=np.array([[0,1,2],[3,4,5]])
    rendered,z=mesh_layer(verts,faces,k,(60,40),scale=2)
    assert rendered.mode=='RGBA' and rendered.getbbox() is not None
    assert abs(float(z[40,60])-1)<1e-6 and np.asarray(rendered)[0,0,3]==0
    rendered.save(out/'synthetic_mesh_rgba.png')
    bg=Image.new('RGB',(60,40),(238,239,240))
    draw_samples_crop(bg,sample,shape,out/'synthetic_sampling','SYNTHETIC ONLY - mask validation')
    write_ply(out/'synthetic_mesh.ply',verts,(34,163,170),faces)
    torch_report=check_live_probe(out) if args.with_torch else {'executed':False,'reason':'NumPy/Pillow-only local smoke'}
    json_write(out/'summary.json',{'status':'passed','fixture':'synthetic, not sofa.png',
        'checks':['support counts','annulus exclusion','self/other/environment separation','invalid depth',
                  'out-of-frame support denominator','empty masked attention','HSI border repetition',
                  'non-square camera coordinates','z-buffer ordering','RGBA and PLY export'],
        'torch':torch_report})
    print(f'[ok] teaser asset smoke: {out}/summary.json')


def check_live_probe(out):
    import torch
    from vggt_omega.models.geometry.regional_scene_probe import RegionalSceneProbe, _rasterize_human_depth_and_owner
    from vggt_omega.models.heads.hsi_refinement_head import _estimate_depth_normals, _local_nearest_scene_probe
    # Two overlapping people, front and rear depths, and a rectangular depth plane.
    torch.manual_seed(0)
    module=RegionalSceneProbe(token_dim=8,adaptive_radius_max=4,annulus_width=2,
                               human_depth_dilation_px=1,human_depth_tolerance_m=.1).eval()
    k=torch.tensor([[[36.,0,15],[0,42,11],[0,0,1]]])
    hw=(24,32);dhw=(12,16)
    centers=torch.tensor([[[[0.,0.,2.],[.12,.10,2.]],[[0.,0.,1.],[.05,0.,1.]]]])
    reps=centers[:,:,:,None,:].repeat(1,1,1,8,1)
    reps[:,:,:,0,0]+=.12
    vertices=reps.reshape(1,2,16,3)
    depth=torch.full((1,*dhw),2.)
    depth[:,5:7,7:9]=1.;depth[:,0,0]=float('nan')
    people=torch.tensor([[True,True]])
    with torch.no_grad():
        # Exercise the HSI normal/search API on a rectangular depth plane before
        # expensive real-image inference. This needs neither SMPL files nor ckpts.
        hsi_depth=torch.full((1,1,*dhw),2.)
        normals=_estimate_depth_normals(hsi_depth,k,height=dhw[0],width=dhw[1])
        assert normals.shape==(1,1,*dhw,3)
        assert normals.dtype==hsi_depth.dtype and normals.device==hsi_depth.device
        anchor=torch.tensor([[[[[0.,0.,2.]]]]])
        projected=torch.tensor([[[[[7.5,5.5]]]]])
        nearest,nearest_normal=_local_nearest_scene_probe(hsi_depth,normals,anchor,projected,k,hw,9)
        assert nearest.shape==nearest_normal.shape==anchor.shape
        assert torch.isfinite(nearest).all() and torch.isfinite(nearest_normal).all()
        torch.testing.assert_close(nearest[...,2],torch.full_like(nearest[...,2],2.))
        live=module(centers,reps,vertices,depth,k,people,hw)
        hd,owner=_rasterize_human_depth_and_owner(vertices,k,people,hw,dhw,1)
    for q in range(2):
        for r in range(2):
            exported=regional_samples(centers[0,q,r].numpy(),reps[0,q,r].numpy(),depth[0].numpy(),
                hd[0].numpy(),owner[0].numpy(),k[0].numpy(),hw,q,radius_max=4,tolerance=.1)
            np.testing.assert_allclose(exported['valid_ratios'],live['valid_ratios'][q,r].numpy(),atol=1e-6)
            assert int(exported['adaptive_radius'])==int(live['adaptive_radius'][q,r])
    return {'executed':True,'device':'cpu','hsi_normal_and_nearest_probe':'passed',
            'comparison':'live RegionalSceneProbe, four region/person combinations'}


if __name__=='__main__':main()
