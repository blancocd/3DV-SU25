from pathlib import Path
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from trimesh.exchange.obj import load_obj
from typing import Union


def load_mesh(
    path: Path, device: torch.device = torch.device("cpu")
) -> Union[Tensor, Tensor]:
    """Load a mesh."""
    with path.open("r") as f:
        mesh = load_obj(f)
    vertices = torch.tensor(mesh["geometry"][str(path)]["vertices"], dtype=torch.float32, device=device)
    faces = torch.tensor(mesh["geometry"][str(path)]["faces"], dtype=torch.int64, device=device)
    return vertices, faces


def get_bunny(
    bunny_path: Path = Path("data/bunny.obj"),
    device: torch.device = torch.device("cpu"),
) -> Union[Tensor, Tensor]:
    vertices, faces = load_mesh(bunny_path, device=device)

    # Center and rescale the bunny.
    maxima, _ = vertices.max(dim=0, keepdim=True)
    minima, _ = vertices.min(dim=0, keepdim=True)
    centroid = 0.5 * (maxima + minima)
    vertices -= centroid
    vertices /= (maxima - minima).max()

    return vertices, faces

def plot_point_cloud(
        vertices: Tensor,
        alpha: float = 0.5,
        max_points: int = 10_000,
        xlim: tuple[float, float] = (-1.0, 1.0),
        ylim: tuple[float, float] = (-1.0, 1.0),
        zlim: tuple[float, float] = (-1.0, 1.0),
    ):
    """Plot a point cloud."""
    vertices = vertices.cpu()

    batch, dim = vertices.shape

    if batch > max_points:
        vertices = np.random.default_rng().choice(vertices, max_points, replace=False)
    fig = plt.figure(figsize=(6, 6))
    if dim == 2:
        ax = fig.add_subplot(111)
    elif dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlabel("z")
        ax.set_zlim(zlim)
        ax.view_init(elev=120.0, azim=270)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.scatter(*vertices.T, alpha=alpha, marker=",", lw=0.5, s=1, color="black")
    plt.show()

if __name__ == "__main__":
    vertices, _ = get_bunny()
    plot_point_cloud(
        vertices,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        zlim=(-1.0, 1.0),
    )
