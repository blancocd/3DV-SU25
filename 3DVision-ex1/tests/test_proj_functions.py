import numpy as np

from src.proj import (
    look_at,
    dehomogenize,
    homogenize,
    project_points_to_image,
    orthographic_projection,
)



def test_look_at():

    # Test case 1
    eye = np.ones(3)
    target = np.zeros(3)
    up = np.asarray([0.0, 1.0, 0.0])
    R, t = look_at(eye, target, up)
    
    R_sol = np.array([[-0.70710678,  0.,  0.70710678],
        [-0.40824829,  0.81649658, -0.40824829],
        [-0.57735027, -0.57735027, -0.57735027]])
    t_sol = np.array([0., 0., 1.73205081])

    assert np.allclose(R, R_sol)
    assert np.allclose(t, t_sol)

    # Test case 2
    eye = np.asarray([0.0, 0.0, 1.0])
    target = np.asarray([0.0, 0.0, 2.0])
    up = np.asarray([0.0, 1.0, 0.0])
    R, t = look_at(eye, target, up)

    R_sol = np.array([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
    t_sol = np.array([ 0.,  0., -1.])

    assert np.allclose(R, R_sol)
    assert np.allclose(t, t_sol)

    # Test case 3
    eye = np.asarray([3.6, 4.2, -5.9])
    target = np.asarray([1.0, 0.0, 1.0])
    up = np.asarray([0.0, 1.0, 0.0])
    R, t = look_at(eye, target, up)

    R_sol = np.array([[ 0.93577066, -0.,  0.35260923],
        [-0.17452055,  0.86892694,  0.46315068],
        [-0.30639166, -0.49494038,  0.81311633]])
    t_sol = np.array([-1.28837989, -0.28863014,  7.97914594])

    assert np.allclose(R, R_sol)
    assert np.allclose(t, t_sol)


def test_homogenize():
    points = np.array([[1, 2], [3, 4], [5, 6]])
    points_homogeneous = homogenize(points)
    expected_points_homogeneous = np.array([[1., 2., 1.],
                                    [3., 4., 1.],
                                    [5., 6., 1.]])
    assert np.allclose(points_homogeneous, expected_points_homogeneous)


def test_dehomogenize():
    points_homogeneous = np.array([[1., 2., 1.],
                                    [3., 4., 1.],
                                    [5., 6., 1.]])
    points_2d = dehomogenize(points_homogeneous)
    expected_points_2d = np.array([[1., 2.],
                                    [3., 4.],
                                    [5., 6.]])
    assert np.allclose(points_2d, expected_points_2d)


def test_project_points_to_image():
    points_3d = np.array([[1, 2, 3], [0.3, -1, 1], [-2, 2, 3]])
    
    image_width = 640
    image_height = 480
    image_width, image_height = 640.0, 480.0
    fx, fy = 525.0, 525.0
    cx, cy = image_width/2.0, image_height/2.0
    K = np.array([
        [fx / image_width, 0, cx / image_width],
        [0, fy / image_height, cy / image_height],
        [0, 0, 1]
    ])

    eye = np.asarray([4.0, 5.0, 1.0])
    target = np.asarray([0.0, 0.0, 2.0])
    up = np.asarray([0.0, 1.0, 0.0])
    R, t = look_at(eye, target, up)
    points_2d, mask = project_points_to_image(points_3d, K, R, t)
    
    points_2d *= np.array([image_width, image_height])

    expected_points_2d = np.array([[462.27595007, 323.42389172],
       [251.84727928, 160.41799803],
       [360.25368343, 485.34548898]])
    expected_mask = np.array([ True,  True, False])

    assert np.allclose(points_2d, expected_points_2d)
    assert np.all(mask == expected_mask)


def test_orthographic_projection():
    points_3d = np.array([[1, 2, 3], [0.3, -1, 1], [-20, 2, 3]])
    
    image_width = 640
    image_height = 480
    image_width, image_height = 640.0, 480.0

    eye = np.asarray([4.0, 0.0, 2.0])
    target = np.asarray([0.0, 0.0, 0.0])
    up = np.asarray([0.0, 1.0, 0.0])
    R, t = look_at(eye, target, up)

    scale = 200
    
    points_2d_ortho, mask_ortho = orthographic_projection(
        points_3d,
        R,
        t,
        scale=scale,
        image_width=image_width,
        image_height=image_height
    )

    expected_points_2d_ortho = np.array([[ 767.2135955 ,  640.],
                                [ 472.05262247,   40.],
                                [2645.5106966 ,  640.]])
    expected_mask_ortho = np.array([ False,  True, False])

    assert np.allclose(points_2d_ortho[:2], expected_points_2d_ortho[:2])
    assert np.all(mask_ortho == expected_mask_ortho)
