
import os
import torch
import numpy as np
import open3d as o3d

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

mse2psnr = lambda x : -10. * torch.log(x) \
            / torch.log(torch.tensor([10.], device=x.device))

def safe_path(path):
    if os.path.exists(path):
        return path
    else:
        os.mkdir(path)
        return path

def load_mem_data(mem):
    poses = mem.pose
    R, T = (poses[:, :3, :3]), poses[:, :3, -1]
    R, T = R, -(T[: ,None ,:] @ R)[: ,0]
    return mem.pts, mem.image, mem.K, R, T, poses, mem.mask

def get_rays(H, W, K, c2w):
    device = c2w.device
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device),
            torch.linspace(0, H-1, H, device=device))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0],
            -(j-K[1][2])/K[1][1], -torch.ones_like(i, device=device)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def remove_outlier(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size=0.010)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.array(pcd.points)[np.array(ind)]


def grad_loss(output, gt):
    def one_grad(shift):
        ox = output[shift:] - output[:-shift]
        oy = output[:, shift:] - output[:, :-shift]
        gx = gt[shift:] - gt[:-shift]
        gy = gt[:, shift:] - gt[:, :-shift]
        loss = (ox - gx).abs().mean() + (oy - gy).abs().mean()
        return loss
    loss = (one_grad(1) + one_grad(2) + one_grad(3)) / 3.
    return loss


def set_seed(seed=0):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
