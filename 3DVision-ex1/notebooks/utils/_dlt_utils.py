import numpy as np
import matplotlib.pyplot as plt

def load_given():
    points_2d = np.load("utils/points_2d_perspective.npy")
    points_3d = np.load("utils/points_3d.npy")
    calibration = np.load("utils/K.npy")
    points_2d *= np.array([640, 480])
    points_2d = points_2d.astype(np.int32)
    points_2d = points_2d.astype(np.float32) / np.array([640, 480])
    return points_3d, points_2d, calibration


def add_noise(points_2d, noise_level=5):
    if noise_level == 0:
        return points_2d
    noise = np.random.normal(0, noise_level, points_2d.shape)
    return points_2d + noise


def plot_2dpointclouds(points2d, labels=None):
    if labels is None:
        labels = ["PC" + str(i) for i in range(len(points2d))]    
    plt.figure(figsize=(8, 8))
    if type(points2d) is not list:
        points2d = [points2d]
    for i,cloud in enumerate(points2d):
        plt.scatter(cloud[:, 0], cloud[:, 1], label=labels[i], alpha=0.7, s=2)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_xaxis()
    plt.grid()
    plt.show()
    
def is_rot_mat(R):
    assert np.allclose(R @ R.T, np.eye(3)), "R is not orthogonal"
    assert np.isclose(np.linalg.det(R), 1.0), "det(R) is not 1"
    print("R is a valid rotation matrix")