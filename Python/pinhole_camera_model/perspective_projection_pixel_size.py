#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
# https://stackoverflow.com/a/30890025
from matplotlib.ticker import FormatStrFormatter

# # https://stackoverflow.com/a/44435714
# np.set_printoptions(edgeitems=10,linewidth=180)

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
    # TODO:
    # P = np.einsum('ii, kii -> kii', K_, )
    # print("einsum.shape:", P.shape)

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
    parser = argparse.ArgumentParser(description='Plot some perspective camera functions.')
    parser.add_argument("--WAC", "--wac", action="store_true", help='Wide Angle Camera (30°)')
    parser.add_argument("--vision-range", type=int, choices=range(0, 4), help='Range: 0 <=> 50m to 15km ; ')
    # parser.add_argument("--tz-min", nargs='?', default=5000.0, help='Min tz')
    # parser.add_argument("--tz-max", nargs='?', default=40000.0, help='Max tz')
    args = parser.parse_args()

    if args.WAC:
        fov_H_deg = 30
    else:
        fov_H_deg = 5
    vision_range = args.vision_range
    print(f"Camera vision range: {vision_range}")
    fov_H = np.deg2rad(fov_H_deg)
    width = height = 1024.0
    sensor_size_1in_W = 12.8
    sensor_size_1in_H = 9.6
    u0 = width/2
    K = get_calibration_matrix_K(fov_H, width, height)
    print(f"fov_H: {np.rad2deg(fov_H)}° ; u0: {u0}")
    print("K:", K.shape)
    print("K:\n", K)

    R = np.zeros((3,3))
    R[0,1] = 1
    R[1,2] = -1
    R[2,0] = -1
    # TODO:
    # if args.WAC:
    #     if False:
    #         t_z_step = 1 / 10.0
    #         t_z = np.arange(50.0, 15000+t_z_step, t_z_step)
    #     else:
    #         t_z_step = 1 / 1000.0
    #         t_z = np.arange(0.1, 1+t_z_step, t_z_step)
    #     # t_z_step = 1 / 100.0
    #     # t_z = np.arange(50.0, 5000+t_z_step, t_z_step)
    #     # t_z_step = 1 / 1000.0
    #     # t_z = np.arange(0.1, 1+t_z_step, t_z_step)
    # else:
    #     t_z_step = 1
    #     t_z = np.arange(5000.0, 40000+t_z_step, t_z_step)

    # TODO:
    todo_elm_del = 12000
    if vision_range == 0:
        t_z_step = 1
        t_z = np.arange(5000.0, 40000+t_z_step, t_z_step)

    elif vision_range == 1:
        t_z_step = 1 / 10.0
        t_z = np.arange(500.0, 15000+t_z_step, t_z_step)

        todo_elm_del = 12000
    elif vision_range == 2:
        t_z_step = 1 / 100.0
        t_z = np.arange(50.0, 5000+t_z_step, t_z_step)

        todo_elm_del = 12000
    elif vision_range == 3:
        t_z_step = 1 / 1000.0
        t_z = np.arange(0.1, 1+t_z_step, t_z_step)
        # TODO:
        # t_z = np.arange(0.05, 1.2+t_z_step, t_z_step)

    t_z_min = t_z[0]
    print(f"min t_z: {t_z_min} ; max t_z: {t_z[-1]}")
    u_vec = np.empty((t_z.shape[0],1))
    x_vec = np.empty((t_z.shape[0],1))

    X0 = np.zeros((4,1))
    X0[3,0] = 1.0
    X1 = np.zeros((4,1))
    X1[1,0] = 1.0
    X1[3,0] = 1.0
    one_pix = np.zeros((2,1))
    # TODO:
    offset = 1
    one_pix[0,0] = u0+offset
    for i in range(t_z.shape[0]):
        t = np.zeros((3,1))
        t[2,0] = t_z[i]
        P, c_T_o = get_projection_matrix(R, t)

        x0 = P @ X0
        x0 /= x0[2]
        x1 = P @ X1
        x1 /= x1[2]

        u0 = to_image_point(K, x0)
        u1 = to_image_point(K, x1)
        # u_vec[i,0] = np.linalg.norm(u)
        u_vec[i,0] = np.linalg.norm(u1-u0) # u[0]

        one_pix_in_meter = to_meter_point(K, one_pix, t_z[i])
        x_vec[i,0] = np.linalg.norm(one_pix_in_meter[0])

    plt.figure(1)

    todo_idx_del = -1
    # TODO:
    # if False:
    # ax1 = plt.subplot(211)
    ax1 = plt.subplot(111)
    pix_size_mm = sensor_size_1in_W / width
    pix_size_cm = pix_size_mm / 10
    print(f"Pixel size: {pix_size_mm} mm")
    if args.WAC and vision_range == 3:
        u_vec = pix_size_cm * u_vec
    plt.plot(t_z, u_vec)

    if not args.WAC:
        # TODO:
        if vision_range == 2:
            ax1.set_yscale('log')
            plt.tick_params(axis='y', which='minor')
            ax1.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

        ax1.set_xlim(t_z[0], t_z[-1])
        ax1.set_ylim(u_vec[-1], u_vec[0])

        # TODO:
        # xticks = ax1.get_xticks()
        # todo_idx_del = np.where(xticks == todo_elm_del)[0][0]
        # print(f"xticks: {xticks}")
        # ax1.set_xticks(xticks)

        # yticks = np.append(ax1.get_yticks(), 1)
        # ax1.set_yticks(yticks)
    else:
        if vision_range != 3:
            ax1.set_yscale('log')
            plt.tick_params(axis='y', which='minor')
            ax1.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        left, right = ax1.get_xlim()
        # TODO:
        # ax1.set_xlim(left=50, right=15000)
        ax1.set_xlim(t_z[0], t_z[-1])
        xticks = ax1.get_xticks()
        # TODO:
        # xticks[0] = 50
        # xticks[-1] = 15000
        # ax1.set_xticks(xticks)

        bottom, top = ax1.set_ylim()
        if vision_range != 3:
            ax1.set_ylim(bottom=10**-1, top=10**2)
        print(f"u_vec[-1]: {u_vec[-1]} ; u_vec[0]: {u_vec[0]}")
        # yticks = ax1.get_yticks()
        # yticks[0] = u_vec[-1]
        # yticks[-1] = u_vec[0]
        # ax1.set_yticks(yticks)

        # TODO:
        # xt1 = ax1.get_xticks()
        # todo_idx_del = 0
        # xt1 = np.delete(xt1, todo_idx_del)
        # xt1 = np.append(xt1, 50.0)
        # print(f"xt1: {xt1}")
        # ax1.set_xticks(xt1)

    xmin, xmax, ymin, ymax = plt.axis()
    if not args.WAC:
        x_interp = np.interp((u0 + 1) - width/2.0, u_vec.ravel()[::-1], t_z.ravel()[::-1])
        print(f'x_interp: {x_interp}')
        plt.hlines((u0 + 1) - u0, xmin=xmin, xmax=x_interp, colors='green', linestyle='dashed')
        # plt.vlines(t_z.ravel()[np.round(x_interp*(1/t_z_step)).astype(int) - int(t_z_min)], ymin=ymin, ymax=(u0 + 1) - u0, colors='green', linestyle='dashed')
        plt.vlines(x_interp, ymin=ymin, ymax=(u0 + 1) - u0, colors='green', linestyle='dashed')
        # print('t_z.shape:', t_z.shape)
        if vision_range == 1:
            xt1 = np.append(ax1.get_xticks(), x_interp[0,0])
            # TODO:
            xt1 = np.delete(xt1, todo_idx_del)
            xt1 = np.insert(xt1, 2, 4000.0)
            xt1 = np.append(xt1, 500.0)
            xt1 = np.append(xt1, 15000.0)
            print(f"xt1: {xt1}")
            ax1.set_xticks(xt1)
        elif vision_range == 2:
            xt1 = ax1.get_xticks()
            todo_idx_del = 0
            xt1 = np.delete(xt1, todo_idx_del)
            xt1 = np.append(xt1, 50.0)
            xt1 = np.append(xt1, 2000.0)
            print(f"xt1: {xt1}")
            ax1.set_xticks(xt1)
    # TODO:
    else:
        if vision_range == 2:
            x_interp = np.interp((u0 + 1) - width/2.0, u_vec.ravel()[::-1], t_z.ravel()[::-1])
            plt.hlines((u0 + 1) - u0, xmin=xmin, xmax=x_interp, colors='green', linestyle='dashed')
            plt.vlines(x_interp, ymin=ymin, ymax=(u0 + 1) - u0, colors='green', linestyle='dashed')

            # xt1 = ax1.get_xticks()
            xt1 = np.append(ax1.get_xticks(), x_interp[0,0])
            todo_idx_del = 0
            xt1 = np.delete(xt1, todo_idx_del)
            todo_idx_del = 1
            xt1 = np.delete(xt1, todo_idx_del)
            xt1 = np.append(xt1, 50.0)
            # xt1 = np.append(xt1, 2000.0)
            print(f"xt1: {xt1}")
            ax1.set_xticks(xt1)
        elif vision_range == 3:
            if False:
                x_interp = np.interp((u0 + 1) - width/2.0, u_vec.ravel()[::-1], t_z.ravel()[::-1])
                print(f"TODO, x_interp: {x_interp}")
                plt.hlines((u0 + 1) - u0, xmin=xmin, xmax=x_interp, colors='green', linestyle='dashed')
                plt.vlines(x_interp, ymin=ymin, ymax=(u0 + 1) - u0, colors='green', linestyle='dashed')
            else:
                # ax1.set_yscale('log')
                t = np.zeros((3,1))
                t[2,0] = 1
                P, _ = get_projection_matrix(R, t)

                x0 = P @ X0
                x0 /= x0[2]
                x1 = P @ X1
                x1 /= x1[2]

                u0 = to_image_point(K, x0)
                u1 = to_image_point(K, x1)
                u_norm = np.linalg.norm(u1-u0)

                y_interp = pix_size_cm * u_norm
                print(f"TODO, u0: {u0} ; u1: {u1} ; u_norm: {u_norm}")
                print(f"TODO, y_interp: {y_interp}")
                plt.hlines(y_interp, xmin=xmin, xmax=xmax, colors='green', linestyle='dashed')
                plt.vlines(1.0, ymin=ymin, ymax=y_interp, colors='green', linestyle='dashed')

                yt1 = ax1.get_yticks()
                yt1 = np.append(yt1, y_interp)
                yt1 = np.append(yt1, u_vec[0])
                print(f"u_vec[0]: {u_vec[0]}")
                ax1.set_yticks(yt1)
                ax1.set_ylim(bottom=ymin)

            xt1 = ax1.get_xticks()
            todo_idx_del = 0
            xt1 = np.delete(xt1, todo_idx_del)
            xt1 = np.append(xt1, 0.1)
            print(f"xt1: {xt1}")
            ax1.set_xticks(xt1)

    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)
    if args.WAC and vision_range == 3:
        ax1.set_xlabel('Distance (m) between the camera and the object', fontsize=18)
        ax1.set_ylabel('Size (cm) of a 1 m object projected onto a 1" sensor', fontsize=18)
        # TODO:
        # ax1.set_title('Size (cm) of a 1 m object projected onto a 1" sensor wrt. the camera distance', fontsize=20)
        plt.title(f'Size (cm) of a 1 m object projected onto a 1" sensor wrt. the camera distance', fontsize=20)
    else:
        ax1.set_xlabel('Distance (m) between the camera and the object', fontsize=18)
        ax1.set_ylabel('Size (px) of a 1 m object', fontsize=18)
        # TODO:
        # ax1.set_title('Size (cm) of a 1 m object wrt. the camera distance', fontsize=20)
        plt.title(f'Size (cm) of a 1 m object wrt. the camera distance', fontsize=20)
    plt.suptitle(f'Camera with horizontal fov: {fov_H_deg}°', fontsize=30)
    # TODO:
    if not args.WAC:
        # https://stackoverflow.com/a/64830832
        x_ticks = ax1.xaxis.get_major_ticks()
        x_ticks[2].label1.set_visible(False)

    # TODO:
    plt.grid()
    plt.figure()
    # else:
    # ax2 = plt.subplot(212, sharex=ax1)
    ax2 = plt.subplot(111)
    # print(f"x_vec: {x_vec} ; size: {len(x_vec)}")

    if args.WAC and vision_range == 3:
        x_vec = x_vec * 1000.0

    plt.plot(t_z, x_vec)
    ax2.set_xlim(0)
    ax2.set_ylim(x_vec[0], x_vec[-1])
    xmin, xmax, ymin, ymax = plt.axis()
    print(f"xmin: {xmin} ; xmax: {xmax} ; ymin: {ymin} ; ymax: {ymax}")
    x_interp = np.interp((u0 + 1) - width/2.0, u_vec.ravel()[::-1], t_z.ravel()[::-1])
    print(f"x_interp: {x_interp}")
    if (vision_range != 2 and not (args.WAC and vision_range == 3)) or (vision_range == 2 and args.WAC):
        plt.hlines(1, xmin=xmin, xmax=x_interp, colors='green', linestyle='dashed')
        # plt.vlines(t_z.ravel()[np.round(x_interp*(1/t_z_step)).astype(int) - int(t_z_min)], ymin=ymin, ymax=1, colors='green', linestyle='dashed')
        plt.vlines(x_interp, ymin=ymin, ymax=1, colors='green', linestyle='dashed')
    elif args.WAC and vision_range == 3:
        t = np.zeros((3,1))
        t[2,0] = 1
        P, _ = get_projection_matrix(R, t)

        x0 = P @ X0
        x0 /= x0[2]
        x1 = P @ X1
        x1 /= x1[2]

        u0 = to_image_point(K, x0)
        u1 = to_image_point(K, x1)
        u_norm = np.linalg.norm(u1-u0)
        print(f"u_norm: {u_norm} ; u1: {u1}")

        one_pix_in_meter = to_meter_point(K, one_pix, t[2,0])
        x_vec_interp = np.linalg.norm(one_pix_in_meter[0]) * 1000

        print(f"x_vec_interp: {x_vec_interp} ; xmin: {xmin} ; t_z[0]: {t_z[0]}")
        plt.hlines(x_vec_interp, xmin=t_z[0], xmax=t[2,0], colors='green', linestyle='dashed')
        print(f"ymin: {ymin} ; ymax: {x_vec_interp}")
        plt.vlines(1, ymin=ymin, ymax=x_vec_interp, colors='green', linestyle='dashed')

        yt2 = ax2.get_yticks()
        yt2 = np.append(yt2, x_vec_interp)
        yt2 = np.append(yt2, x_vec[0])
        yt2 = np.append(yt2, x_vec[-1])
        # print(f"u_vec[0]: {u_vec[0]}")
        ax2.set_yticks(yt2)

        ax2.set_xlim(left=t_z[0], right=t_z[-1])
        xt2 = ax2.get_xticks()
        todo_idx_del = 0
        xt2 = np.delete(xt2, todo_idx_del)
        xt2 = np.append(xt2, t_z[0])
        print(f"xt2: {xt2}")
        ax2.set_xticks(xt2)
        ax2.set_yticks(yt2)

        ax2.set_ylim(bottom=x_vec[0], top=x_vec[-1])

    # TODO:
    # xt2 = np.append(ax2.get_xticks(), x_interp)
    xt2 = np.append(ax2.get_xticks(), x_interp[0,0])
    ax2.set_xticks(xt2)
    ax2.set_xlabel('Distance (m) between the camera and the object', fontsize=16)

    if vision_range == 1:
        xt2 = np.append(ax2.get_xticks(), x_interp[0,0])
        # TODO:
        todo_idx_del = 6
        xt2 = np.delete(xt2, todo_idx_del)
        xt2 = np.insert(xt2, 2, 4000.0)
        xt2 = np.append(xt2, 500.0)
        xt2 = np.append(xt2, 15000.0)
        print(f"xt2: {xt2}")
        ax2.set_xticks(xt2)
    elif vision_range == 2:
        xt2 = ax2.get_xticks()
        todo_idx_del = 0
        xt2 = np.delete(xt2, todo_idx_del)
        if args.WAC:
            todo_idx_del = 1
            xt2 = np.delete(xt2, todo_idx_del)
        xt2 = np.append(xt2, 50.0)
        # xt2 = np.append(xt2, 2000.0)
        print(f"xt2: {xt2}")
        ax2.set_xticks(xt2)

    ax2.xaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)
    if args.WAC and vision_range == 3:
        ax2.set_ylabel(f'Size (mm) of {offset} px', fontsize=16)
        ax2.set_title(f'Size (mm) of {offset} px wrt. the camera distance', fontsize=20)
    else:
        ax2.set_ylabel(f'Size (m) of {offset} px', fontsize=16)
        ax2.set_title(f'Size (m) of {offset} px wrt. the camera distance', fontsize=20)
    plt.suptitle(f'Camera with horizontal fov: {fov_H_deg}°', fontsize=30)
    # TODO:
    # if args.WAC:
    #     # https://stackoverflow.com/a/64830832
    #     x_ticks = ax2.xaxis.get_major_ticks()
    #     x_ticks[2].label1.set_visible(False)

    # TODO:
    # plt.suptitle(f'Camera with horizontal fov: {fov_H_deg}°', fontsize=30)
    # plt.title(f'Camera with horizontal fov: {fov_H_deg}°', fontsize=30)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
