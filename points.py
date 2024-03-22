import numpy as np

def xyz(R, K, pixel_coords, depth):
    
    # Get the pixel coordinates
    u = pixel_coords[0]
    v = pixel_coords[1]

    # Get the intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Calculate the x, y, z coordinates
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    xyz = np.array([x[0], y[0], z[0], 1]).T
    xyz = np.dot(R, xyz)
    xyz = xyz[:3]

    return xyz
