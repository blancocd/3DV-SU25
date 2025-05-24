import numpy as np
from PIL import Image, ImageDraw
import trimesh
import pyvista as pv
import cv2 as cv
from superpoint_superglue_deployment import Matcher, SuperPointHandler
import matplotlib.pyplot as plt

def load_scene_params(scene_id):
    # global camera poses
    glob_pos0 = np.load(f"scene_params/{scene_id}/glob_pos0.npy")
    glob_pos1 = np.load(f"scene_params/{scene_id}/glob_pos1.npy")
    # camera matrices
    K0 = np.load(f"scene_params/{scene_id}/K0.npy")
    K1 = np.load(f"scene_params/{scene_id}/K1.npy")
    P0 = np.load(f"scene_params/{scene_id}/P0.npy")
    P1 = np.load(f"scene_params/{scene_id}/P1.npy")
    RT0 = np.load(f"scene_params/{scene_id}/RT0.npy")
    RT1 = np.load(f"scene_params/{scene_id}/RT1.npy")
    # relative position and rotation
    # 3D vertices
    if scene_id in ["01", "02", "05"]:
        obj_name = "bust"
    elif scene_id in ["03", "04"]:
        obj_name = "x3d"
    elif scene_id == "05":
        obj_name = "turtle"
    elif scene_id == "06":
        obj_name = "castle"
    
    mesh = trimesh.load(f"assets/meshes/{obj_name}.obj")
    vertices = np.load(f"scene_params/{scene_id}/vertices.npy")
    mesh.vertices = vertices
    
    # out_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

    scene_params = {
        "glob_pos0": glob_pos0,
        "glob_pos1": glob_pos1,
        "K0": K0,
        "K1": K1,
        "P0": P0,
        "P1": P1,
        "RT0": RT0,
        "RT1": RT1
    }
    
    img0 = np.array(Image.open(f"assets/{scene_id}/img0.png"))
    img1 = np.array(Image.open(f"assets/{scene_id}/img1.png"))
    
    return scene_params, mesh, img0, img1

def load_scene_params2(scene_id):
    K0 = np.load(f"scene_params/{scene_id}/K0.npy")
    K1 = np.load(f"scene_params/{scene_id}/K1.npy")
    P0 = np.load(f"scene_params/{scene_id}/P0.npy")
    P1 = np.load(f"scene_params/{scene_id}/P1.npy")
    RT0 = np.load(f"scene_params/{scene_id}/RT0.npy")
    RT1 = np.load(f"scene_params/{scene_id}/RT1.npy")

    scene_params = {
        "K0": K0,
        "K1": K1,
        "P0": P0,
        "P1": P1,
        "RT0": RT0,
        "RT1": RT1
    }
    
    img0 = np.array(Image.open(f"assets/{scene_id}/img0.png"))
    img1 = np.array(Image.open(f"assets/{scene_id}/img1.png"))
    
    return scene_params, img0, img1

def draw_camera_frustum(pl, K, RT, image):
    height, width = image.shape[:2]
    R = RT[:, :3]
    t = RT[:, 3]
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = R.T
    cam_to_world[:3, 3] = -R.T @ t

    # Image corners in pixel coordinates
    corners_px = np.array([
        [0, 0],             # top-left
        [width, 0],         # top-right
        [width, height],    # bottom-right
        [0, height],        # bottom-left
    ])
    homog_corners_px = np.hstack([corners_px, np.ones((4, 1))])
    
    K_inv = np.linalg.inv(K)
    corners_cam = homog_corners_px @ K_inv.T
    # corners_cam = corners_cam / np.linalg.norm(corners_cam, axis=1, keepdims=True)
    focal_length = K[0, 0] / 1000
    corners_cam = corners_cam * focal_length
    corners_world = corners_cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]
    cam_center = cam_to_world[:3, 3]
    lines = np.zeros((8, 2, 3))
    lines[:4] = np.array([
        [cam_center, corners_world[0]],
        [cam_center, corners_world[1]],
        [cam_center, corners_world[2]],
        [cam_center, corners_world[3]],
    ])
    lines[4:8] = np.array([
        [corners_world[0], corners_world[1]],
        [corners_world[1], corners_world[2]],
        [corners_world[2], corners_world[3]],
        [corners_world[3], corners_world[0]],
    ])
    pl.add_lines(lines.reshape(-1, 3), color='grey', width=2)
    
    pl.add_lines(np.array([cam_center, cam_center + 0.1*R[0, :]]), color='red', width=2)
    pl.add_lines(np.array([cam_center, cam_center + 0.1*R[1, :]]), color='green', width=2)
    pl.add_lines(np.array([cam_center, cam_center + 0.1*R[2, :]]), color='blue', width=2)

   
def plot_epipolar_setup(mesh, img0, img1, scene_params):
    pl = pv.Plotter()
    pl.add_mesh(pv.wrap(mesh), color='white', point_size=5, render_points_as_spheres=True)
    c0 = scene_params["glob_pos0"]
    c1 = scene_params["glob_pos1"]
    pl.add_points(c0, color='black', point_size=10)
    pl.add_points(c1, color='black', point_size=10)
    K0 = scene_params["K0"]
    RT0 = scene_params["RT0"]
    K1 = scene_params["K1"]
    RT1 = scene_params["RT1"]

    draw_camera_frustum(pl, K0, RT0, img0)
    draw_camera_frustum(pl, K1, RT1, img1)
    # pl.add_lines(frustum1.reshape(-1, 3), color='grey', width=2)
    
    pl.camera_position = [0, 0.75, 0.15]
    pl.camera.focal_point = [0, 0, 0]
    pl.camera.up = [0, 0, 1]
    pl.camera.view_angle = 15
    return pl
    

def draw_dot(img, coords, color=(255, 0, 0, 255), radius=7):
    """
    Draw a dot on the image at the given coordinates.
    """
    i = Image.fromarray(img)
    draw = ImageDraw.Draw(i)
    x, y = coords
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return np.array(i)

def draw_line(img, line_params, color=(255, 0, 0, 255), width=5):
    """
    Draw a line on the image using the 3D line parametrization.
    """
    i = Image.fromarray(img)
    draw = ImageDraw.Draw(i)
    a, b, c = line_params
    if b != 0:
        x0, y0 = 0, int(-c / b)
        y1, x1 = img.shape[:2]
        y1 = int(-(a * x1 + c) / b)
    else:
        # Vertical line
        x0, y0 = int(-c / a), 0
        x1, y1 = int(-c / a), img.shape[0]
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    return np.array(i)


def superglue_correspondences(img0, img1, kth=0.003, mth=0.5):
    gray0 = cv.cvtColor(img0, cv.COLOR_RGBA2GRAY)
    gray1 = cv.cvtColor(img1, cv.COLOR_RGBA2GRAY)

    superglue_matcher = Matcher(
            {
                "superpoint": {
                    "input_shape": (-1, -1),
                    "keypoint_threshold": kth,
                },
                "superglue": {
                    "match_threshold": mth,
                },
                "use_gpu": False,
            }
        )

    pt0, pt1, _, _, matches = superglue_matcher.match(gray0, gray1)
    matches = np.array(matches)

    points0 = []
    points1 = []
    for m in matches:
        p0 = pt0[m.queryIdx].pt
        p1 = pt1[m.trainIdx].pt
        points0.append(p0)
        points1.append(p1)
        
    points0 = np.array(points0)
    points1 = np.array(points1)
    
    return points0, points1


def draw_matches(im0, img1, points0, points1):
    img0_copy = im0.copy()
    img1_copy = img1.copy()
    h0, w0 = im0.shape[:2]
    h1, w1 = img1.shape[:2]
    height = max(h0, h1)
    width = w0 + w1
    matched_img = np.zeros((height, width, 4), dtype=np.uint8)
    matched_img[:h0, :w0] = img0_copy
    matched_img[:h1, w0:w0+w1] = img1_copy
    
    for pt0, pt1 in zip(points0, points1):
        x0, y0 = int(pt0[0]), int(pt0[1])
        x1, y1 = int(pt1[0]), int(pt1[1])
        cv.circle(matched_img, (x0, y0), 5, (0, 255, 0, 255), -1)
        cv.circle(matched_img, (w0 + x1, y1), 5, (0, 255, 0, 255), -1)
        cv.line(matched_img, (x0, y0), (w0 + x1, y1), (255, 0, 0, 255), 1)
    
    return matched_img



def frustum_lines(orig, axes, depth=0.25, width=0.25, height=0.25):
    img_center = orig + depth*axes[2]
    top_left = img_center + axes[0]*width/2 - axes[1]*height/2
    top_right = img_center - axes[0]*width/2 - axes[1]*height/2
    bottom_left = img_center + axes[0]*width/2 + axes[1]*height/2
    bottom_right = img_center - axes[0]*width/2 + axes[1]*height/2
    # draw up triangle
    top = img_center - axes[1]*3/4*height
    lines = np.array([
        [orig, top_left],
        [orig, top_right],
        [orig, bottom_left],
        [orig, bottom_right],
        [top_left, top_right],
        [top_right, bottom_right],
        [bottom_right, bottom_left],
        [bottom_left, top_left],
        [top_left, top],
        [top_right, top],
    ]).reshape(-1, 3)
    return lines

def _draw_candidates(pl, cands):
    pl.add_lines(np.array([[0, 0, 0], [0.2, 0, 0]]), color="red", width=5)
    pl.add_lines(np.array([[0, 0, 0], [0, 0.2, 0]]), color="green", width=5)
    pl.add_lines(np.array([[0, 0, 0], [0, 0, 0.2]]), color="blue", width=5)
    pl.add_lines(frustum_lines(np.array([0, 0, 0]), np.eye(3)), color="grey", width=2)
    pl.add_point_labels(np.array([0,0,.25]), labels=[f"Camera 0"], point_size=10, font_size=20)
    
    # Loop through each candidate and draw its coordinate system
    for i, cand in enumerate(cands):
        x_axis = cand[0][:, 0]
        y_axis = cand[0][:, 1]
        z_axis = cand[0][:, 2]
        orig = cand[1]
        pl.add_lines(np.array([orig, orig + x_axis * 0.1]), color="red", width=2)
        pl.add_lines(np.array([orig, orig + y_axis * 0.1]), color="green", width=2)
        pl.add_lines(np.array([orig, orig + z_axis* 0.1]), color="blue", width=2)
        pl.add_point_labels([orig + z_axis * 0.25], labels=[f"Candidate {i+1}"], point_size=10, font_size=20)
        pl.add_lines(frustum_lines(orig, np.array([x_axis, y_axis, z_axis])), color="grey", width=2)
    
    return pl

def draw_candidates(candidates):
    pl = pv.Plotter()
    pl = _draw_candidates(pl, candidates)
    pl.show()
    


def draw_pair_with_reconstruction(K0, K1, RT1, img0, img1, points0, points1, triangulated_points, size=15):
    """
    Draw the camera pair with the triangulated points.
    """
    rgb_img0 = np.array(Image.fromarray(img0).convert("RGB"))
    rgb_img1 = np.array(Image.fromarray(img1).convert("RGB"))
    colors0 = rgb_img0[points0[:, 1].astype(int), points0[:, 0].astype(int)].astype(np.float32) / 255.0
    colors1 = rgb_img1[points1[:, 1].astype(int), points1[:, 0].astype(int)].astype(np.float32) / 255.0
    colors = colors0 * 0.5 + colors1 * 0.5
    pl = pv.Plotter()
    RT0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    draw_camera_frustum(pl, K0, RT0, img0)
    draw_camera_frustum(pl, K1, RT1, img1)
    pl.add_points(triangulated_points, scalars=colors, opacity=1.0, rgb=True, point_size=size, render_points_as_spheres=True)
    pl.show()
    
    
def test_normalize_points(points, normalized_points):
    """
    Plot original points and their normalized version side-by-side.
    
    Parameters:
        points (np.ndarray): 2D points to normalize and plot
    """
    
    # Create figure for comparison
    plt.figure(figsize=(12, 6))
    
    # Plot original points
    plt.subplot(1, 2, 1)
    plt.gca().invert_yaxis()
    plt.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.5)
    plt.title('Original Points')
    plt.grid(True)
    plt.axis('equal')
    
    # Plot normalized points
    plt.subplot(1, 2, 2)
    plt.scatter(normalized_points[:, 0], normalized_points[:, 1], c='red', alpha=0.5)
    plt.title('Normalized Points')
    plt.grid(True)
    plt.axis('equal')
    
    # Add statistics as text
    orig_center = np.mean(points, axis=0)
    orig_dist = np.mean(np.sqrt(np.sum((points - orig_center)**2, axis=1)))
    norm_center = np.mean(normalized_points, axis=0)
    norm_dist = np.mean(np.sqrt(np.sum((normalized_points - norm_center)**2, axis=1)))
    
    plt.figtext(0.5, 0.01, f"Original - Center: ({orig_center[0]:.2f}, {orig_center[1]:.2f}), Avg Distance: {orig_dist:.2f}\n"
                         f"Normalized - Center: ({norm_center[0]:.2f}, {norm_center[1]:.2f}), Avg Distance: {norm_dist:.2f}",
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    # Reverse the y-axis for the normalized points plot to match image coordinates
    plt.subplot(1, 2, 2)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()