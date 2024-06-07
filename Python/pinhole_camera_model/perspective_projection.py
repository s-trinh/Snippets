#!/usr/bin/env python3
import argparse
import numpy as np

np.set_printoptions(suppress=True)

def get_calibration_matrix_K(fov_H, width, height):
    f = (width / 2.0) * 1.0 / np.tan(fov_H / 2.0)
    K = np.eye(3)
    K[0,0] = f
    K[1,1] = f
    K[0,2] = width / 2.0
    K[1,2] = height / 2.0

    return K

def get_R_t_matrix(R, t):
    R_t = np.eye(4)
    R_t[0:3,0:3] = R
    R_t[0:3,3] = t.ravel()
    return R_t

def get_projection_matrix(R, t):
    c_T_o = get_R_t_matrix(R, t)

    proj = np.zeros((3,4))
    proj[:3,:3] = np.eye(3)
    P = proj @ c_T_o

    return P, c_T_o

def to_image_point(K, P):
    x = np.empty((2,1))
    x[0,0] = K[0,0] * P[0,0] + K[0,2]
    x[1,0] = K[1,1] * P[1,0] + K[1,2]

    return x

def to_meter_point(K, u, Z):
    X = np.empty((3,1))
    X[0,0] = (u[0,0] - K[0,2]) / K[0,0] * Z
    X[1,0] = (u[1,0] - K[1,2]) / K[1,1] * Z
    X[2,0] = Z

    return X

def main():
    parser = argparse.ArgumentParser(description='Perspective projection.')
    parser.add_argument("--fov", type=float, default=30, help='Camera fov in degree')
    parser.add_argument("--width", type=float, default=1024, help='Image width in pixel')
    parser.add_argument("--X", type=float, default=0.0005233, help='X coordinate in m')
    parser.add_argument("--Y", type=float, default=0, help='Y coordinate in m')
    parser.add_argument("--Z", type=float, default=1, help='Z coordinate in m')
    args = parser.parse_args()

    fov_H_deg = args.fov
    print(f"Camera fov: {fov_H_deg}°")
    width = args.width
    height = width
    print(f"Image width: {width} pixels")

    # Camera intrinsic parameters:
    K = get_calibration_matrix_K(np.deg2rad(fov_H_deg), width, height)
    print(f"Camera intrinsic parameters K:\n{K}")

    # 3D point
    X = args.X
    Y = args.Y
    Z = args.Z
    X_3d = np.array([[X], [Y], [Z], [1]])
    print(f"3D coordinate:\n{X_3d}")

    # Perspective projection
    P, c_T_o = get_projection_matrix(np.eye(3), np.zeros((3)))
    print(f"P:\n{P}\nc_T_o:\n{c_T_o}")

    # To image point
    x = to_image_point(K, P @ X_3d)
    print(f"3D point projected onto the image plane with top-left coordinate:\n{x}")

    # 3D point at the center
    X0 = 0
    Y0 = 0
    Z0 = 0
    X_3d_center = np.array([[X0], [Y0], [Z0], [1]])
    print(f"3D coordinate at the center:\n{X_3d_center}")

    # Perspective projection
    x_2d_center = to_image_point(K, P @ X_3d_center)
    print(f"3D point projected onto the image plane with top-left coordinate:\n{x_2d_center}")

    # Difference
    print(f"Difference:\n{x - x_2d_center}")

    # ------------------------------------------------------------------------------------------ #

    u0 = width/2
    print(f"\nu0: {u0}")
    one_pix = np.zeros((2,1))
    offset = 1
    one_pix[0,0] = u0 + offset
    one_pix[1,0] = height / 2
    print(f"one_pix:\n{one_pix}")

    print(f"K:\n{K}")
    print(f"X_3d:\n{X_3d}")
    one_pix_in_meter = to_meter_point(K, one_pix, X_3d[2,0])
    print(f"one_pix_in_meter:\n{one_pix_in_meter}")

if __name__ == '__main__':
    main()
