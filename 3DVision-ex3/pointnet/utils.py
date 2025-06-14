import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
import numpy as np
from PIL import Image


def rotation_mats(axis, angles):
    """
        Generates rotation matrices for a given axis and angles.
        Args:
            axis (str): Axis of rotation ('x', 'y', or 'z').
            angles (torch.Tensor): Angles in radians of shape (B,).
        Returns:
            torch.Tensor: Rotation matrices of shape (B, 3, 3).
    """
    
    rot_mats = torch.zeros(angles.shape[0], 3, 3, dtype=torch.float32).to(angles.device)
    if axis == "x":
        rot_mats[:, 0, 0] = 1.0
        rot_mats[:, 1, 1] = torch.cos(angles)
        rot_mats[:, 1, 2] = -torch.sin(angles)
        rot_mats[:, 2, 1] = torch.sin(angles)
        rot_mats[:, 2, 2] = torch.cos(angles)
    elif axis == "y":
        rot_mats[:, 0, 0] = torch.cos(angles)
        rot_mats[:, 0, 2] = torch.sin(angles)
        rot_mats[:, 1, 1] = 1.0
        rot_mats[:, 2, 0] = -torch.sin(angles)
        rot_mats[:, 2, 2] = torch.cos(angles)
    elif axis == "z":
        rot_mats[:, 0, 0] = torch.cos(angles)
        rot_mats[:, 0, 1] = -torch.sin(angles)
        rot_mats[:, 1, 0] = torch.sin(angles)
        rot_mats[:, 1, 1] = torch.cos(angles)
        rot_mats[:, 2, 2] = 1.0
    else:
        raise ValueError("Invalid axis: {}".format(axis))
    
    return rot_mats
    


def randomize(pcs, up_axis="z", noise_std=0.02):
    """
    Randomly rotates the point clouds around the y-axis (up) and adds pointwise random noise.
    
    Args:
        pcs (torch.Tensor): The input point clouds of shape (B, N, 3).
        noise_std (float): Standard deviation of the Gaussian noise.
    
    Returns:
        torch.Tensor: Noisy point cloud.
    """
    
    B = pcs.shape[0]
    angle = torch.rand(B) * 2 * np.pi  # Random angle in radians
    
    rot_mats = rotation_mats(up_axis, angle).to(pcs.device)  # Get rotation matrices for the specified axis

    points_r = torch.bmm(pcs, rot_mats.transpose(1,2))  # Rotate the points
    noise = torch.randn_like(points_r) * noise_std
    points_r += noise
    return points_r

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def viz_seg (verts, labels, path, device):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1,labels.shape[0],3))

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i])

    sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
    rend = (rend * 255).astype(np.uint8)

    imageio.mimsave(path, rend, fps=15)

    

def viz_cor(verts1, verts2, path, device):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    N = verts1.shape[0]
    assert N == verts2.shape[0], "Both point clouds must have the same number of points."
    image_size=512
    background_color=(1, 1, 1)
    colors = (verts1[:, [0, 2, 1]].clone() + 1) / 2

    # Construct various camera viewpoints
    dist = 3
    elev = 25
    azim = 0
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    obj_rot_angles = torch.arange(0, 2 * np.pi, step=np.pi / 30, device=device).to(verts1.device)  # 30 angles for a full rotation
    obj_rot_mats = rotation_mats("y", obj_rot_angles)  # Rotate around y-axis
    
    verts1 = torch.bmm(verts1.unsqueeze(0).repeat(obj_rot_mats.shape[0], 1, 1), obj_rot_mats.transpose(1,2))  # Rotate verts1
    verts2 = torch.bmm(verts2.unsqueeze(0).repeat(obj_rot_mats.shape[0], 1, 1), obj_rot_mats.transpose(1,2))  # Rotate verts2
    
    verts1 = verts1 + torch.tensor([-0.75, 0, 0])
    verts2 = verts2 + torch.tensor([0.75, 0, 0])
    
    all_verts = torch.cat([verts1, verts2], dim=1)  # Concatenate the two point clouds
    
    labels = torch.arange(N).to(verts1.device)
    all_labels = torch.cat([labels, labels], dim=0)
    all_colors = colors[all_labels].unsqueeze(0).to(torch.float)
    all_colors = all_colors.repeat(obj_rot_mats.shape[0], 1, 1)

    point_cloud = pytorch3d.structures.Pointclouds(points=all_verts, features=all_colors).to(device)
    
    renderer = get_points_renderer(image_size=image_size, background_color=background_color, radius=0.01, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy()
    rend = (rend * 255).astype(np.uint8)

    imageio.mimsave(path, rend, fps=15)