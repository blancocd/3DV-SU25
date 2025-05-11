import numpy as np
from src.utils.given import load_pointcloud, visualize_multiple_projections


def look_at(eye, target, up): # (2 Points)
    """
    implement a function look_at(eye, target, up) that constructs the extrinsic 
    matrix of a camera, given its position and orientation in the world.

    Your task is to compute:
        R: a 3x3 rotation matrix that transforms points from world coordinates
        to camera coordinates. This defines the camera's orientation.
        t: a 3D translation vector such that when combined with R, they form
        the extrinsic camera matrix [R | t].

    The function should construct a right-handed coordinate system where:
        The z-axis points from the camera position toward the target.
        The x-axis is computed from the cross product of the up vector and the z-axis.
        The y-axis is chosen such that it forms a right-handed system with the x and z axes.
    
    Args:
        eye: Position of the camera in world coordinates
        target: Target point to look at in world coordinates
        up: Up vector in world coordinates
        
    Returns:
        R: Rotation matrix (world to camera)
        t: Translation vector for extrinsic matrix
    """
    R = np.eye(3)
    t = np.zeros((3,))

    ### Write your code here
    # raise NotImplementedError("look_at function not implemented")
    
    return R, t


def homogenize(points): # (1 Points)
    """
    implement a function homogenize(points) that converts a set of points from
    Euclidean coordinates to homogeneous coordinates.
    
    Your task is to convert each point to homogeneous coordinates by appending
    a 1 as an additional coordinate. This results in an N x (n+1) array, where
    each point is now in homogeneous form.

    Args:
        points: Nxn array of points
        
    Returns:
        points_homogeneous: Nx(n+1) array of homogeneous coordinates
    """
    points_homogeneous = np.ones((points.shape[0], points.shape[1] + 1)) # dummy

    ### Write your code here
    # raise NotImplementedError("homogenize function not implemented")
    
    return points_homogeneous


def dehomogenize(points): # (1 Points)
    """
    implement a function dehomogenize(points) that converts a set of points from homogeneous
    coordinates back to standard Euclidean coordinates.

    Your task is to perform the conversion by dividing the first n components of each point
    by the last component (the homogeneous scale factor). This will result in an N x n array
    of points in Euclidean space.
    
    Args:
        points: Nx(n+1) array of homogeneous coordinates
        
    Returns:
        points_2d: Nxn array of 2D points
    """
    points_2d = np.zeros((points.shape[0], points.shape[1] - 1)) # dummy

    ### Write your code here
    # raise NotImplementedError("dehomogenize function not implemented")
    
    return points_2d

def project_points_to_image(points_3d, K, R, t): # (6 Points)
    """
    implement a function project_points_to_image(points_3d, K, R, t) that projects 
    3D points in world coordinates onto a 2D image plane, using the camera's intrinsic and extrinsic parameters.

    Your task is to:
        - Convert the 3D points to homogeneous coordinates.
        - Apply the extrinsic transformation ([R | t]) to bring the points into the camera coordinate system.
        - Filter out points that lie behind the camera.
        - Apply the intrinsic matrix K to project the points onto the image plane.
        - Convert the projected points from homogeneous to 2D coordinates.
        - Check if the projected points fall within the image boundaries and update the visibility mask accordingly.
    
    Args:
        points_3d: Nx3 array of 3D points
        K: 3x3 intrinsic camera matrix
        R: 3x3 rotation matrix (camera extrinsic)
        t: 3x1 translation vector (camera extrinsic)
        
    Returns:
        points_2d: Nx2 array of 2D points
        mask: Boolean array indicating which points are in front of the camera
    """
    points_2d = np.zeros((points_3d.shape[0], 2)) # dummy
    mask = np.zeros((points_3d.shape[0],), dtype=bool) # dummy

    ### Write your code here
    # raise NotImplementedError("project_points_to_image function not implemented")

    return points_2d, mask


def orthographic_projection(points_3d, R, t, scale=1.0, image_width=640, image_height=480): # (6 Points)
    """
    In this exercise, you will implement a function orthographic_projection(points_3d, R, t, scale, image_width, image_height) 
    that projects 3D points in world coordinates onto a 2D image plane using orthographic projection.

    Different from the perspective projection, this function should use orthographic projection. This means that the depth
    information (z-coordinate) is not used for scaling the projected points.

    Key Difference from Perspective Projection:
        - In perspective projection, the 3D points are projected onto the image 
          plane using depth (z-coordinate), causing distant objects to appear smaller.
        - In orthographic projection, all points are scaled uniformly, and depth does
          not influence the size of the projected points. This means the z-coordinate
          is not used for scaling the points before projection.

    Your Task:
        - Convert the 3D points to homogeneous coordinates.
        - Apply the extrinsic transformation ([R | t]) to bring the points into the camera coordinate system.
        - Filter out points behind the camera.
        - Apply the orthographic projection.
        - Translate the points to be centered in the image.
        - Check if the projected points lie within the image bounds and update the visibility mask.

    Args:
        points_3d: Nx3 array of 3D points
        R: 3x3 rotation matrix (camera extrinsic)
        t: 3x1 translation vector (camera extrinsic)
        scale: Scaling factor for the projection
        image_width: Width of the image plane in pixels
        image_height: Height of the image plane in pixels
        
    Returns:
        points_2d: Nx2 array of 2D points
        mask: Boolean array indicating which points are visible in the image
    """
    points_2d = np.zeros((points_3d.shape[0], 2)) # dummy
    mask = np.zeros((points_3d.shape[0],), dtype=bool) # dummy

    ### Write your code here
    # raise NotImplementedError("orthographic_projection function not implemented")
    
    return points_2d, mask


def main():
    points_3d = load_pointcloud("data/bunny.obj")
    # rotate bunny to be upright
    r_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    points_3d = points_3d @ r_mat.T
    
    points_center = points_3d.mean(axis=0)
    
    # intrinsicss K
    image_width, image_height = 640.0, 480.0
    fx, fy = 525.0, 525.0
    cx, cy = image_width/2.0, image_height/2.0
    K = np.array([
        [fx / image_width, 0, cx / image_width],
        [0, fy / image_height, cy / image_height],
        [0, 0, 1]
    ])
    
    # Camera position in world coordinates
    C = np.random.rand(3) / 4
    print(C)
    
    R,t = look_at(C, points_center, np.array([0, 0, 1]))
    
    # Perspective projection
    points_2d_perspective, mask_perspective = project_points_to_image(
        points_3d, K, R, t
    )
    
    # Revert back to image ratio
    unnorm_p2d_perspective = points_2d_perspective.copy()
    unnorm_p2d_perspective *= np.array([image_width, image_height])

    # Orthographic projection
    points_2d_ortho, mask_ortho = orthographic_projection(
        points_3d, R, t, scale=2500, image_width=image_width, image_height=image_height
    )
    
    # Visualize the projections
    visualize_multiple_projections(
        points_3d, 
        [unnorm_p2d_perspective,points_2d_ortho],
        [mask_perspective, mask_ortho],
        ["Perspective", "Orthographic"],
        K, R, t, image_width, image_height
    )
    

if __name__ == "__main__":
    main()