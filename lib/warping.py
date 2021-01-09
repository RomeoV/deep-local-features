
from lib.megadepth_dataset import *
from externals.d2net.lib.utils import *

import numpy as np
import matplotlib.pyplot as plt

import torch
from pathlib import Path 


def uv_to_pos(uv):
    return torch.cat([uv[1, :].view(1, -1), uv[0, :].view(1, -1)], dim=0)
    
def warp(pos1,
        depth1, intrinsics1, pose1, bbox1,
        depth2, intrinsics2, pose2, bbox2):
    device = pos1.device

    Z1, pos1, ids = interpolate_depth(pos1, depth1) #Z1 is depth of pixel

    # COLMAP convention
    u1 = pos1[1, :] + bbox1[1] + .5 #bbox are offsets
    v1 = pos1[0, :] + bbox1[0] + .5

    X1 = (u1 - intrinsics1[0, 2]) * (Z1 / intrinsics1[0, 0])
    Y1 = (v1 - intrinsics1[1, 2]) * (Z1 / intrinsics1[1, 1])

    XYZ1_hom = torch.cat([
        X1.view(1, -1),
        Y1.view(1, -1),
        Z1.view(1, -1),
        torch.ones(1, Z1.size(0), device=device)], dim=0) #World coordinates of 2D points in image1
    
    XYZ2_hom = torch.chain_matmul(pose2, torch.inverse(pose1), XYZ1_hom) #reprojected world coordinates of 2D points in image1
    XYZ2 = XYZ2_hom[: -1, :] / XYZ2_hom[-1, :].view(1, -1)

    uv2_hom = torch.matmul(intrinsics2, XYZ2)
    uv2 = uv2_hom[: -1, :] / uv2_hom[-1, :].view(1, -1)

    u2 = uv2[0, :] - bbox2[1] - .5
    v2 = uv2[1, :] - bbox2[0] - .5
    uv2 = torch.cat([u2.view(1, -1),  v2.view(1, -1)], dim=0)

    annotated_depth, pos2, new_ids = interpolate_depth(uv_to_pos(uv2), depth2)

    ids = ids[new_ids]
    pos1 = pos1[:, new_ids]
    estimated_depth = XYZ2[2, new_ids]

    inlier_mask = torch.abs(estimated_depth - annotated_depth) < 0.05 #optimize this parameter??

    ids = ids[inlier_mask]
    if ids.size(0) == 0:
        raise EmptyTensorError

    pos2 = pos2[:, inlier_mask]
    pos1 = pos1[:, inlier_mask]

    return pos1, pos2, ids

def interpolate_depth(pos, depth):
    #Simple bilinear interpolation with validation depth check!
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    h, w = depth.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Valid depth: Only depth values >0 allowed (otherwise structure would be behind camera point!)
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    ids = ids[valid_depth]
    if ids.size(0) == 0:
        raise EmptyTensorError

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    print(w_top_left.device, depth.device, depth[i_top_left, j_top_left].device)
    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]

if __name__=="__main__":
    megadepth_path = Path("/mnt/c/Users/phill/polybox/Deep Learning/MegaDepthDataset/")
    scene_info_path = megadepth_path / "scene_info/"
    scene_list = megadepth_path / "valid_scenes.txt"

    dataset = MegaDepthDataset(base_path = megadepth_path, scene_info_path=scene_info_path, scene_list_path=scene_list)
    correspondence = dataset[10]
    idx = 0

    depth1 = correspondence['depth1'][idx] # [h1, w1]???
    intrinsics1 = correspondence['intrinsics1'][idx]  # [3, 3]
    pose1 = correspondence['pose1'][idx].view(4, 4)  # [4, 4]
    bbox1 = correspondence['bbox1'][idx]  # [2]

    depth2 = correspondence['depth2'][idx]
    intrinsics2 = correspondence['intrinsics2'][idx]
    pose2 = correspondence['pose2'][idx].view(4, 4)
    bbox2 = correspondence['bbox2'][idx]

    fmap_pos1 = grid_positions(32, 32, device) #feature positions, 2x(32*32)=2x1024 [y,x]-format -> [[0,0],[0,1], ...[32,32]]
    pos1 = upscale_positions(fmap_pos1, scaling_steps=3) # feature positions in 256x256, [y,x]-format -> [[0,0],[0,11.5], ...[256,256]]
    #ids: matching ids in sequence (256*256)
    #default pos1 has ids [0, ..., 1024]
    # now ids says which of these are valid based on relative transformation between them, e.g.
    # [5, 28, 32, ...,1020]
    # so a legal correspondence would be pos1[:,5]<-->pos2[:,0]
    pos1, pos2, ids = warp(pos1,
        depth1, intrinsics1, pose1, bbox1,
        depth2, intrinsics2, pose2, bbox2)

    """ Valid Correspondences"""
    fmap_pos1 = fmap_pos1[:, ids] #uv-positions on 32x32 grid, but in list format 2xlen(ids)
    fmap_pos2 = torch.round(
        downscale_positions(pos2, scaling_steps=scaling_steps)
    ).long() ##corresponding positions