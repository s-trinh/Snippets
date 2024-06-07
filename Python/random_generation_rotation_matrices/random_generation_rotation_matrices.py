#!/usr/bin/env python3
# https://math.stackexchange.com/a/4832876
import matplotlib
import matplotlib.pyplot as plt
# needed to use a 3d projection with matplotlib <= 3.1
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

# https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/1602779#1602779
# http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
# https://github.com/alecjacobson/gptoolbox/blob/master/matrix/rand_rotation.m
def qr_full(num_samples=1):
    z = np.random.randn(num_samples, 3, 3)
    q, r = np.linalg.qr(z)
    sign = 2 * (np.diagonal(r, axis1=-2, axis2=-1) >= 0) - 1
    rot = q
    rot *= sign[..., None, :]
    rot[:, 0, :] *= np.linalg.det(rot)[..., None]
    return rot

# https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/1288873#1288873
# https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d/44701#44701
def randn_orthobasis(num_samples=1):
    z = np.random.randn(num_samples, 3, 3)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    z[:, 0] = np.cross(z[:, 1], z[:, 2], axis=-1)
    z[:, 0] = z[:, 0] / np.linalg.norm(z[:, 0], axis=-1, keepdims=True)
    z[:, 1] = np.cross(z[:, 2], z[:, 0], axis=-1)
    z[:, 1] = z[:, 1] / np.linalg.norm(z[:, 1], axis=-1, keepdims=True)
    return z

# https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/4394036#4394036
# https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d/44701#44701
def randn_axis(num_samples=1, corrected=True):
    u = np.random.uniform(0, 1, size=num_samples)
    z = np.random.randn(num_samples, 1, 3)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)

    if corrected:
        t = np.linspace(0, np.pi, 1024)
        cdf_psi = (t - np.sin(t)) / np.pi
        psi = np.interp(u, cdf_psi, t, left=0, right=np.pi)
    else:
        psi = 2 * np.pi * u

    # all_rot = rot3x3_from_axis_angle(z, psi)
    # print(f"all_rot={all_rot.shape}")

    return rot3x3_from_axis_angle(z, psi)

# https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/442423#442423
# https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d/44691#44691
def nbubis(num_samples=1, corrected=True):
    u1 = np.random.uniform(0, 1, size=num_samples)
    u2 = np.random.uniform(0, 1, size=num_samples)
    u3 = np.random.uniform(0, 1, size=num_samples)

    theta = np.arccos(2 * u1 - 1)
    phi = 2 * np.pi * u2
    axis_vector = [
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ]
    axis_vector = np.stack(axis_vector, axis=1).reshape(-1, 1, 3)

    if corrected:
        t = np.linspace(0, np.pi, 1024)
        cdf_psi = (t - np.sin(t)) / np.pi
        psi = np.interp(u3, cdf_psi, t, left=0, right=np.pi)
    else:
        psi = 2 * np.pi * u3

    return rot3x3_from_axis_angle(axis_vector, psi)

# https://math.stackexchange.com/questions/442418/random-generation-of-rotation-matrices/1602779#1602779
def qr_half(num_samples=1):
    z = np.random.randn(num_samples, 3, 3)
    q, r = np.linalg.qr(z)
    return q

# https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
def rot3x3_from_axis_angle(axis_vector, angle):
    angle = np.atleast_1d(angle)[..., None, None]
    K = np.cross(np.eye(3), axis_vector)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def plot_scatter(pointses, filename, kwargses):
    fig = plt.figure()
    # ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax = fig.add_subplot(projection="3d")
    for points, kwargs in zip(pointses, kwargses):
        ax.scatter(*np.asarray(points).T, marker=".", **kwargs)
    # ax.view_init(elev=45, azim=-45, roll=0)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    # ax.set_aspect("equal", adjustable="box")
    fig.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
    # plt.show()
    # exit()
    plt.close(fig)

def test_orientation_sampling(num_samples=1, no_rot=False):
    #   - "How to generate equidistributed points on the surface of a sphere", Markus Deserno
    #   - https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    #   - https://gist.github.com/dinob0t/9597525

    r = 1

    # cube_root = np.floor(np.cbrt(num_samples)).astype(int)
    cube_root = np.cbrt(num_samples).astype(int)
    if no_rot:
        nb_lon_lat = num_samples
        nb_ori = 1
    else:
        nb_lon_lat = cube_root*cube_root
        nb_ori = num_samples // nb_lon_lat
    print(f"cube_root={cube_root} ; nb_lon_lat={nb_lon_lat} ; nb_ori={nb_ori}")
    print(f"num_samples={num_samples} ; nb_lon_lat*nb_ori={nb_lon_lat*nb_ori}")

    all_rot = np.zeros((nb_lon_lat*nb_ori, 3, 3))

    a = 4.0 * np.pi*(r**2.0 / nb_lon_lat)
    d = np.sqrt(a)
    m_theta = int(round(np.pi / d))
    d_theta = np.pi / m_theta
    d_phi = a / d_theta
    pi_2 = np.pi/2

    m_upper_bound = m_theta

    cpt = 0
    for m in range(m_upper_bound):
        theta = np.pi * (m + 0.5) / m_theta
        m_phi = int(round(2.0 * np.pi * np.sin(theta) / d_phi))

        for n in range(m_phi):
            phi = 2.0 * np.pi * n / m_phi
            lon = phi
            lat = pi_2-theta

            for ori in range(nb_ori):
                if cpt < all_rot.shape[0]:
                    ENU_pose = np.array(
                        [
                            [-np.sin(lon), -np.sin(lat)*np.cos(lon), np.cos(lat)*np.cos(lon)],
                            [ np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)*np.sin(lon)],
                            [ 0,            np.cos(lat),             np.sin(lat)]
                        ]
                    )

                    # if ori > 0:
                    if not np.isclose(ori, 0): # use this?
                        ori_rad = ori * 2*np.pi / nb_ori

                        # ENU_pose(0,0) = -std::sin(lon); ENU_pose(0,1) = -std::sin(lat) * std::cos(lon); ENU_pose(0,2) = std::cos(lat) * std::cos(lon);
                        # ENU_pose(1,0) =  std::cos(lon); ENU_pose(1,1) = -std::sin(lat) * std::sin(lon); ENU_pose(1,2) = std::cos(lat) * std::sin(lon);
                        # ENU_pose(2,0) =   0;            ENU_pose(2,1) =  std::cos(lat);                 ENU_pose(2,2) = std::sin(lat);

                        z_axis = np.array([0, 0, 1])
                        up_pose = rot3x3_from_axis_angle(z_axis, ori_rad)

                        ENU_up_pose = ENU_pose @ up_pose
                    else:
                        ENU_up_pose = ENU_pose

                    all_rot[cpt,:,:] = ENU_up_pose
                    cpt += 1

    return all_rot

def get_rpy(alpha, beta, gamma):
    R_z = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha),  np.cos(alpha), 0],
            [0,             0,              1]
        ]
    )
    R_y = np.array(
        [
            [ np.cos(beta), 0, np.sin(beta)],
            [0,             1, 0],
            [-np.sin(beta), 0., np.cos(beta)]
        ]
    )
    R_x = np.array(
        [
            [1, 0,              0],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma),  np.cos(gamma)]
        ]
    )

    return R_z @ R_y @ R_x

def test_rpy_sampling(num_samples=1, corrected=True):
    nb_sampling = np.floor(np.cbrt(num_samples)).astype(int)
    # nb_sampling = np.cbrt(num_samples).astype(int)
    # Euler angles ranges?
    # http://motion.cs.illinois.edu/RoboticSystems/3DRotations.html#Singularities,-aka-gimbal-lock
    alpha_vec = np.linspace( 0,         2*np.pi, nb_sampling, endpoint=False)
    beta_vec =  np.linspace(-np.pi/2,   np.pi/2, nb_sampling, endpoint=True)
    gamma_vec = np.linspace( 0,         2*np.pi, nb_sampling, endpoint=False)

    all_rot = np.zeros( (len(alpha_vec)*len(beta_vec)*len(gamma_vec), 3, 3) )
    cpt = 0
    for alpha in alpha_vec:
        for beta in beta_vec:
            for gamma in gamma_vec:
                if cpt < all_rot.shape[0]:
                    all_rot[cpt, :, :] = get_rpy(alpha, beta, gamma)
                    cpt += 1

    return all_rot

def main():
    print(f"{matplotlib.__version__}")

    parser = argparse.ArgumentParser(description='Test rotation random sampling / custom rotation sampling.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num', type=int, default=1000,
                        help='Number of rotation points.')
    args = parser.parse_args()

    num_rot_pts = args.num
    print(f"Number of rotation points: {num_rot_pts}")

    METHODS = {
        "randn_orthobasis": randn_orthobasis,
        "randn_axis": randn_axis,
        "randn_axis_incorrect": lambda **kwargs: randn_axis(corrected=False, **kwargs),
        "nbubis": nbubis,
        "nbubis_incorrect": lambda **kwargs: nbubis(corrected=False, **kwargs),
        # "qr_half": qr_half,
        "qr_full": qr_full,
        "lon_lat_custom": test_orientation_sampling,
        "lon_lat_custom_no_rot": lambda **kwargs: test_orientation_sampling(no_rot=True, **kwargs),
        "euler_rpy_custom": test_rpy_sampling,
    }

    # x is the starting point; y contains various sample rotated points.
    # x = np.array([1.0, 0.0, 0.0])
    x = np.array([1 / 9, -4 / 9, 8 / 9])
    x /= np.linalg.norm(x)  # Normalize to unit vector, just in case.

    for name, func in METHODS.items():

        rot = func(num_samples=num_rot_pts // (2 if "_half" in name else 1))
        print(f"{name}: rot={len(rot)}")
        y = rot @ x
        plot_scatter(
            [y, [x]],
            f"rot3x3_{name}.png",
            [{"s": 1, "alpha": 0.5}, {"s": 64, "color": "#ff77cc"}],
        )

if __name__ == '__main__':
    main()