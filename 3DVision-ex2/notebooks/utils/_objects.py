from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import draw3d_arrow, get_plane_from_three_points, to_inhomogeneus, to_homogeneus
from ._matrices import get_plucker_matrix, get_projection_matrix


class GenericPoint:
    def __init__(self, X: np.ndarray, name: Optional[str] = None) -> None:
        self.values = X
        self.name = name

    def draw(
        self,
        f: float,
        px: float = 0.0,
        py: float = 0.0,
        C: Sequence[float] = (0.0, 0.0, 0.0),
        theta_x: float = 0.0,
        theta_y: float = 0.0,
        theta_z: float = 0.0,
        mx: float = 1.0,
        my: float = 1.0,
        s: float = 20.0,
        color: str = "tab:green",
        closed: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        if ax is None:
            ax = plt.gca()

        P = get_projection_matrix(
            f,
            px=px,
            py=py,
            mx=mx,
            my=my,
            theta_x=theta_x,
            theta_y=theta_y,
            theta_z=theta_z,
            C=C,
        )

        x = to_inhomogeneus(P @ to_homogeneus(self.values))
        ax.scatter(*x, s=s, color=color)
        if self.name is not None:
            ax.text(*x, self.name)

        return ax

    def draw3d(
        self,
        pi: np.ndarray,
        C: Sequence[float] = (0.0, 0.0, 0.0),
        s: float = 20.0,
        color: str = "tab:green",
        closed=True,
        ax: Optional[Axes3D] = None,
    ) -> Axes3D:
        if ax is None:
            ax = plt.gca() # projection="3d"
            # fig = plt.figure()  # newer version of plt
            # ax = fig.add_subplot(projection='3d')

        L = get_plucker_matrix(np.asarray(C), self.values)
        x = to_inhomogeneus(L @ pi)
        ax.scatter3D(*self.values, s=s, color=color)
        ax.scatter3D(*x, s=s, color=color)
        ax.plot(*np.c_[C, self.values], color="tab:gray", alpha=0.5, ls="--")
        if self.name is not None:
            ax.text(*self.values, self.name)
            ax.text(*x, self.name.lower())

        return ax


class Polygon:
    def __init__(self, xyz: np.ndarray) -> None:
        self.values = xyz

    def draw(
        self,
        f: float,
        px: float = 0.0,
        py: float = 0.0,
        C: Sequence[float] = (0.0, 0.0, 0.0),
        theta_x: float = 0.0,
        theta_y: float = 0.0,
        theta_z: float = 0.0,
        mx: float = 1.0,
        my: float = 1.0,
        s: float = 20.0,
        color: str = "tab:green",
        closed: bool = True,
        ax: Optional[plt.Axes] = None,
        notext: bool = False
    ) -> plt.Axes:
        if ax is None:
            ax = plt.gca()

        P = get_projection_matrix(
            f,
            px=px,
            py=py,
            mx=mx,
            my=my,
            theta_x=theta_x,
            theta_y=theta_y,
            theta_z=theta_z,
            C=C,
        )
        x_list = []
        for i, X in enumerate(self.values, 1):
            x = to_inhomogeneus(P @ to_homogeneus(X))
            ax.scatter(*x, s=s, color=color)
            if not notext:
                ax.text(*x, f"x{i}")
            x_list.append(x)

        if closed:
            x_list.append(x_list[0])

        ax.plot(*np.vstack(x_list).T, color=color)
        return ax
    
    def project(
        self,
        f: float,
        px: float = 0.0,
        py: float = 0.0,
        C: Sequence[float] = (0.0, 0.0, 0.0),
        theta_x: float = 0.0,
        theta_y: float = 0.0,
        theta_z: float = 0.0,
        mx: float = 1.0,
        my: float = 1.0,
        s: float = 20.0,
        closed: bool = True
    ):
        P = get_projection_matrix(
            f,
            px=px,
            py=py,
            mx=mx,
            my=my,
            theta_x=theta_x,
            theta_y=theta_y,
            theta_z=theta_z,
            C=C,
        )
        x_list = []
        for i, X in enumerate(self.values, 1):
            x = to_inhomogeneus(P @ to_homogeneus(X))
            x_list.append(x)

        if closed:
            x_list.append(x_list[0])

        return np.vstack(x_list)

    def draw3d(
        self,
        pi: np.ndarray,
        C: Sequence[float] = (0.0, 0.0, 0.0),
        s: float = 20.0,
        color: str = "tab:green",
        closed=True,
        ax: Optional[Axes3D] = None,
        notext: bool = False
    ) -> Axes3D:
        if ax is None:
            ax = plt.gca() # projection="3d"
            # fig = plt.figure()  # newer version of plt
            # ax = fig.add_subplot(projection='3d')
        xyz = self.values.copy()
        x_list = []
        for i, X in enumerate(xyz, 1):
            L = get_plucker_matrix(np.asarray(C), X)
            x = to_inhomogeneus(L @ pi)
            ax.scatter3D(*X, s=s, color=color)
            ax.scatter3D(*x, s=s, color=color)
            ax.plot(*np.c_[C, X], color="tab:gray", alpha=0.5, ls="--")
            if not notext:
                ax.text(*X, f"X{i}")
                ax.text(*x, f"x{i}")
            x_list.append(x)

        if closed:
            xyz = np.vstack([xyz, xyz[0, :]])
            x_list.append(x_list[0])

        ax.plot(*xyz.T, color=color)
        ax.plot(*np.vstack(x_list).T, color=color)
        return ax


class ReferenceFrame:
    def __init__(
        self,
        origin: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray,
        name: str,
    ) -> None:
        self.origin = origin
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.name = name

    def draw3d(
        self,
        head_length: float = 0.3,
        color: str = "tab:blue",
        ax: Optional[Axes3D] = None,
    ) -> Axes3D:
        if ax is None:
            ax = plt.gca() # projection="3d"
            # fig = plt.figure() # newer version of plt
            # ax = fig.add_subplot(projection='3d')
        ax.text(*self.origin + 0.5, f"({self.name})")
        ax = draw3d_arrow(
            ax=ax,
            arrow_location=self.origin,
            arrow_vector=self.dx,
            head_length=head_length,
            color=color,
            name="x",
        )
        ax = draw3d_arrow(
            ax=ax,
            arrow_location=self.origin,
            arrow_vector=self.dy,
            head_length=head_length,
            color=color,
            name="y",
        )
        ax = draw3d_arrow(
            ax=ax,
            arrow_location=self.origin,
            arrow_vector=self.dz,
            head_length=head_length,
            color=color,
            name="z",
        )
        return ax


class PrincipalAxis:
    def __init__(
        self, camera_center: np.ndarray, camera_dz: np.ndarray, f: float
    ) -> None:
        self.camera_center = camera_center
        self.camera_dz = camera_dz
        self.f = f
        self.p = camera_center + f * camera_dz

    def draw3d(
        self,
        head_length: float = 0.3,
        color: str = "tab:red",
        s: float = 20.0,
        ax: Optional[Axes3D] = None,
    ) -> Axes3D:
        if ax is None:
            ax = plt.gca()
            # fig = plt.figure()  # newer version of plt
            # ax = fig.add_subplot(projection='3d')

        draw3d_arrow(
            arrow_location=self.camera_center,
            arrow_vector=2.0 * self.f * self.camera_dz,
            head_length=head_length,
            color=color,
            name="Z",
            ax=ax,
        )
        ax.scatter3D(*self.p, s=s, color=color)
        ax.text(*self.p, "p")
        return ax


class Image:
    def __init__(self, heigth: int, width: int) -> None:
        self.heigth = heigth
        self.width = width

    def draw(self, color: str = "tab:gray", ax: Optional[plt.Axes] = None) -> plt.Axes:
        if ax is None:
            ax = plt.gca()

        ax.set_xticks(np.arange(0, self.width + 1))
        ax.set_yticks(np.arange(0, self.heigth + 1))
        ax.grid(color=color)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.heigth)
        ax.set_aspect("equal")
        return ax


class ImagePlane:
    def __init__(
        self,
        origin: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        heigth: int,
        width: int,
        mx: float = 1.0,
        my: float = 1.0,
    ) -> None:
        self.origin = origin
        self.dx = dx
        self.dy = dy
        self.heigth = heigth
        self.width = width
        self.mx = mx
        self.my = my
        self.pi = get_plane_from_three_points(origin, origin + dx, origin + dy)

    def draw3d(
        self, color: str = "tab:gray", alpha: float = 0.5, ax: Optional[Axes3D] = None
    ) -> Axes3D:
        if ax is None:
            # ax = plt.gca(projection="3d")
            ax = plt.gca()
            # fig = plt.figure()  # newer version of plt
            # ax = plt.add_subplot(projection='3d')

        xticks = np.arange(self.width + 1).reshape(-1, 1) * self.dx / self.mx
        yticks = np.arange(self.heigth + 1).reshape(-1, 1) * self.dy / self.my
        pts = (self.origin + xticks).reshape(-1, 1, 3) + yticks
        pts = pts.reshape(-1, 3)
        shape = len(xticks), len(yticks)
        X = pts[:, 0].reshape(shape)
        Y = pts[:, 1].reshape(shape)
        Z = pts[:, 2].reshape(shape)
        frame = np.c_[
            self.origin,
            self.origin + self.dx * self.width / self.mx,
            self.origin
            + self.dx * self.width / self.mx
            + self.dy * self.heigth / self.my,
            self.origin + self.dy * self.heigth / self.my,
            self.origin,
        ]
        ax.plot(*frame, color="black")
        ax.plot_wireframe(X, Y, Z, color=color, alpha=alpha)
        return ax
