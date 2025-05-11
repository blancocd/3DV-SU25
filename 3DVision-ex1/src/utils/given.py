import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh


def load_pointcloud(file_path):
    """
    Returns: Nx3 array of points
    """
    points = trimesh.load(file_path).vertices
    return points

def load_cube():
    points = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                      [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    points = points / 10
    points += np.array([0.5, 0.5, 0.5])
    return points

def draw_camera_frustum(ax, R, t, K, image_width=640, image_height=480, scale=0.1):
    """
    Draw a camera frustum in the 3D plot
    
    Args:
        ax: Matplotlib 3D axis
        R: 3x3 rotation matrix (camera extrinsic)
        t: 3x1 translation vector (camera extrinsic)
        K: 3x3 intrinsic matrix
        scale: Size of the camera frustum
    """
    C = - R.T @ t
    ax.scatter(C[0], C[1], C[2], color='black', s=50, label='Camera Center')
    cam_x = R[0]
    cam_y = R[1]
    cam_z = R[2]
    axis_length = scale
    ax.quiver(C[0], C[1], C[2], cam_x[0], cam_x[1], cam_x[2], color='red', length=axis_length, normalize=True)
    ax.quiver(C[0], C[1], C[2], cam_y[0], cam_y[1], cam_y[2], color='green', length=axis_length, normalize=True)
    ax.quiver(C[0], C[1], C[2], cam_z[0], cam_z[1], cam_z[2], color='blue', length=axis_length, normalize=True)
    fx = K[0, 0] * image_width
    fy = K[1, 1] * image_height
    cx = K[0, 2] * image_width
    cy = K[1, 2] * image_height
    z_plane = scale
    corners_cam = np.array([
        [0, 0, z_plane],
        [image_width, 0, z_plane],
        [image_width, image_height, z_plane],
        [0, image_height, z_plane],
    ])
    corners_cam[:, 0] = (corners_cam[:, 0] - cx) * z_plane / fx
    corners_cam[:, 1] = (corners_cam[:, 1] - cy) * z_plane / fy
    corners_world = corners_cam @ R + C
    for corner in corners_world:
        ax.plot([C[0], corner[0]], [C[1], corner[1]], [C[2], corner[2]], 'gray', alpha=0.5)
    ax.plot([corners_world[0, 0], corners_world[1, 0], corners_world[2, 0], corners_world[3, 0], corners_world[0, 0]],
            [corners_world[0, 1], corners_world[1, 1], corners_world[2, 1], corners_world[3, 1], corners_world[0, 1]],
            [corners_world[0, 2], corners_world[1, 2], corners_world[2, 2], corners_world[3, 2], corners_world[0, 2]],
            'gray', alpha=0.8)
    
    
    
def visualize_projection(points_3d, points_2d, mask, K, R, t, colors=None, image_width=640, image_height=480):
    """
    Visualize the 3D point cloud and its 2D projection
    
    Args:
        points_3d: Nx3 array of 3D points
        points_2d: Nx2 array of projected 2D points
        mask: Boolean array indicating which points are visible
        K: 3x3 intrinsic camera matrix 
        R: 3x3 rotation matrix (camera extrinsic)
        t: 3x1 translation vector (camera extrinsic)
        colors: Nx3 array of RGB colors (optional)
        image_width: Width of the image plane in pixels
        image_height: Height of the image plane in pixels
    """
    fig = plt.figure(figsize=(15, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    
    if colors is not None:
        ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                    c=colors, s=2, alpha=0.5)
    else:
        ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                    c='blue', s=2, alpha=0.5)
    
    draw_camera_frustum(ax1, R, t, K, image_width, image_height)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Point Cloud and Camera')

    max_range = np.array([
        points_3d[:, 0].max() - points_3d[:, 0].min(),
        points_3d[:, 1].max() - points_3d[:, 1].min(),
        points_3d[:, 2].max() - points_3d[:, 2].min(),
        2*t[0], 2*t[1], 2*t[2]
    ]).max() / 2.0
    
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax2 = fig.add_subplot(122)
    
    ax2.set_xlim(0, image_width)
    ax2.set_ylim(0, image_height)
    ax2.invert_yaxis()

    if np.any(mask):
        ax2.scatter(points_2d[mask, 0], points_2d[mask, 1], c=colors[mask] if colors is not None else 'red', s=2, alpha=0.8)
    
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.set_title('2D Projection')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
def visualize_multiple_projections(points_3d, points_2d_list, mask_list, projection_names, K, R, t, image_width=640, image_height=480):
    """
    Visualize the 3D point cloud and multiple 2D projections side by side
    
    Args:
        points_3d: Nx3 array of 3D points
        points_2d_list: List of Nx2 arrays, each containing a different 2D projection
        mask_list: List of boolean arrays indicating which points are visible for each projection
        projection_names: List of strings with names for each projection
        K: 3x3 intrinsic camera matrix 
        R: 3x3 rotation matrix (camera extrinsic)
        t: 3x1 translation vector (camera extrinsic)
        image_width: Width of the image plane in pixels
        image_height: Height of the image plane in pixels
    """
    num_projections = len(points_2d_list)
    fig = plt.figure(figsize=(5 * (num_projections + 1), 5))

    ax_3d = fig.add_subplot(1, num_projections + 1, 1, projection='3d')
    
    ax_3d.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=2, alpha=0.5)
    
    draw_camera_frustum(ax_3d, R, t, K, image_width, image_height)
    
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D Point Cloud and Camera')
    
    max_range = np.array([
        points_3d[:, 0].max() - points_3d[:, 0].min(),
        points_3d[:, 1].max() - points_3d[:, 1].min(),
        points_3d[:, 2].max() - points_3d[:, 2].min(),
        2*t[0], 2*t[1], 2*t[2]
    ]).max() / 2.0
    
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    
    for i in range(num_projections):
        ax = fig.add_subplot(1, num_projections + 1, i + 2)
        
        points_2d = points_2d_list[i]
        mask = mask_list[i]
        name = projection_names[i] if i < len(projection_names) else f"Projection {i+1}"
        
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        if np.any(mask):
            ax.scatter(points_2d[mask, 0], points_2d[mask, 1], c='blue', s=2, alpha=0.8)
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)' if i == 0 else '')
        ax.set_title(name)
        ax.invert_xaxis()
        ax.yaxis.tick_right()
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()