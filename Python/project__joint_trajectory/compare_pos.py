#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


def compute_error_vector(pos_cmd_data, pos_mes_data):
    # error_vector = np.empty(pos_cmd_data.shape)

    # for i in range(pos_cmd_data.shape[0]):
    #     ts = pos_cmd_data[i,0]
    #     error_vector[i,0] = ts

    #     for j in range(1, pos_cmd_data.shape[1]):
    #         for cpt in range(i, pos_mes_data.shape[0]):
    #             ts2 = pos_mes_data[cpt,0]
    #             if np.isclose(ts, ts2):
    #                 error_vector[i,j] = pos_cmd_data[i,j] - pos_mes_data[cpt,j]
    #                 break

    error_vector = np.empty(pos_cmd_data.shape)

    start_idx = 0
    ts_start = pos_cmd_data[0,0]
    for i in range(pos_mes_data.shape[0]):
        ts2 = pos_mes_data[i,0]
        if np.isclose(ts_start, ts2):
            start_idx = i
            break

    for i in range(pos_cmd_data.shape[0]):
        ts = pos_cmd_data[i,0]
        error_vector[i,0] = ts

        for j in range(1, pos_cmd_data.shape[1]):
            error_vector[i,j] = pos_cmd_data[i,j] - pos_mes_data[i+start_idx,j]

    return error_vector


def main():
    parser = argparse.ArgumentParser(description='Plot joint positions / velocities / accelerations.')
    parser.add_argument("--file1", help='Path to the npy/npz file.')
    parser.add_argument("--file2", help='Path to the npy/npz file.')
    parser.add_argument("--ts-begin", default=0, type=int, help='Path to the npy/npz file.')
    parser.add_argument("--ts-end", default=-1, type=int, help='Path to the npy/npz file.')
    args = parser.parse_args()

    pos_cmd_data = np.loadtxt(args.file1)
    pos_mes_data = np.loadtxt(args.file2)

    print(f"pos_cmd shape={pos_cmd_data.shape}")
    print(f"pos_mes shape={pos_mes_data.shape}")

    fig, axs = plt.subplots(2,4)

    print(f"len(axs)={len(axs)}")
    print(f"shape(axs)={axs.shape}")
    print(f"len(shape(axs))={len(axs.shape)}")

    LABEL_SIZE = 12
    SUPTITLE_SIZE = 18
    LEGEND_SIZE = 24

    ts_begin = args.ts_begin
    ts_end = args.ts_end

    marker_style = ''

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            cpt = i*axs.shape[1] + j
            # flat --> https://matplotlib.org/stable/gallery/lines_bars_and_markers/markevery_demo.html#sphx-glr-gallery-lines-bars-and-markers-markevery-demo-py
            ax = axs.flat[cpt]

            ax.plot(pos_cmd_data[ts_begin:ts_end,0], pos_cmd_data[ts_begin:ts_end,cpt+1], label=f'cmd joint{cpt}', marker=marker_style)
            ax.plot(pos_mes_data[ts_begin:ts_end,0], pos_mes_data[ts_begin:ts_end,cpt+1], label=f'mes joint{cpt}', marker=marker_style)
            ax.legend()

            ax.set_xlabel('Time (s)', fontsize=LABEL_SIZE)
            if cpt == 0:
                ax.set_ylabel('m', fontsize=LABEL_SIZE)
            else:
                ax.set_ylabel('rad', fontsize=LABEL_SIZE)

    fig.suptitle('Joint positions', fontsize=LEGEND_SIZE)


    fig, axs = plt.subplots(2,4)

    # joint_errors = compute_error_vector(pos_cmd_data[ts_begin:ts_end,], pos_mes_data[ts_begin:ts_end,])
    joint_errors = compute_error_vector(pos_cmd_data, pos_mes_data)
    print(f"joint_errors: {joint_errors.shape}")

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            cpt = i*axs.shape[1] + j
            # flat --> https://matplotlib.org/stable/gallery/lines_bars_and_markers/markevery_demo.html#sphx-glr-gallery-lines-bars-and-markers-markevery-demo-py
            ax = axs.flat[cpt]
            ax.plot(joint_errors[:,0], joint_errors[:,cpt+1], label=f'joint{cpt}', marker=marker_style)
            ax.legend()

            ax.set_xlabel('Time (s)', fontsize=LABEL_SIZE)
            if cpt == 0:
                ax.set_ylabel('m', fontsize=LABEL_SIZE)
            else:
                ax.set_ylabel('rad', fontsize=LABEL_SIZE)

    fig.suptitle('Joint errors between command and measure', fontsize=LEGEND_SIZE)

    plt.show()

    # data_path = args.file
    # data = np.load(data_path)
    # positions_vec = data['positions_vec']
    # velocities_vec = data['velocities_vec']
    # accelerations_vec = data['accelerations_vec']
    # print(f"positions_vec shape: {positions_vec.shape}")
    # print(f"velocities_vec shape: {velocities_vec.shape}")
    # print(f"accelerations_vec shape: {accelerations_vec.shape}")

    # LABEL_SIZE = 12
    # SUPTITLE_SIZE = 18
    # LEGEND_SIZE = 14

    # marker_type = args.marker
    # left_xlim = args.left_xlim

    # time_from_start_vec = data['time_from_start_vec']
    # print(f"time_from_start_vec shape: {time_from_start_vec.shape}")

    # fig, axs = plt.subplots(3)
    # for i in range(1, positions_vec.shape[1]):
    #     axs[0].plot(time_from_start_vec[:], positions_vec[:,i], marker=marker_type, label=f'arm_joint_{i}')
    # axs[0].set_title('Positions', fontsize=SUPTITLE_SIZE)
    # axs[0].set_ylabel('rad', fontsize=LABEL_SIZE)

    # for i in range(1, velocities_vec.shape[1]):
    #     axs[1].plot(time_from_start_vec[:], velocities_vec[:,i], marker=marker_type, label=f'arm_joint_{i}')
    # axs[1].set_title('Velocities', fontsize=SUPTITLE_SIZE)
    # axs[1].set_ylabel('rad/s', fontsize=LABEL_SIZE)

    # for i in range(1, accelerations_vec.shape[1]):
    #     axs[2].plot(time_from_start_vec[:], accelerations_vec[:,i], marker=marker_type, label=f'arm_joint_{i}')
    # axs[2].set_title('Accelerations', fontsize=SUPTITLE_SIZE)
    # axs[2].set_xlabel('Time (s)', fontsize=LABEL_SIZE)
    # axs[2].set_ylabel('rad/s^2', fontsize=LABEL_SIZE)
    # axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=positions_vec.shape[1]-1, fontsize=LEGEND_SIZE)

    # for ax in axs:
    #     ax.grid()
    #     ax.set_xlim(left=left_xlim)
    #     # ax.legend(loc='upper center', ncol=7)
    #     # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.subplots_adjust(hspace=0.3) # See: https://stackoverflow.com/a/6541454
    # # fig.tight_layout() # See: https://stackoverflow.com/a/9827848


    # # For the translational joint (rail)
    # fig, axs = plt.subplots(3)

    # axs[0].plot(time_from_start_vec[:], positions_vec[:,0], marker=marker_type, label=f'rail_joint_1')
    # axs[0].grid()
    # axs[0].set_xlim(left=left_xlim)
    # axs[0].set_title('Positions', fontsize=SUPTITLE_SIZE)
    # axs[0].set_ylabel('m', fontsize=LABEL_SIZE)
    # axs[0].legend(fontsize=LEGEND_SIZE)

    # axs[1].plot(time_from_start_vec[:], velocities_vec[:,0], marker=marker_type, label=f'rail_joint_1')
    # axs[1].grid()
    # axs[1].set_xlim(left=left_xlim)
    # axs[1].set_title('Velocities', fontsize=SUPTITLE_SIZE)
    # axs[1].set_ylabel('m/s', fontsize=LABEL_SIZE)
    # axs[1].legend(fontsize=LEGEND_SIZE)

    # axs[2].plot(time_from_start_vec[:], accelerations_vec[:,0], marker=marker_type, label=f'rail_joint_1')
    # axs[2].grid()
    # axs[2].set_xlim(left=left_xlim)
    # axs[2].set_title('Accelerations', fontsize=SUPTITLE_SIZE)
    # axs[2].set_xlabel('Time (s)', fontsize=LABEL_SIZE)
    # axs[2].set_ylabel('m/s^2', fontsize=LABEL_SIZE)
    # axs[2].legend(fontsize=LEGEND_SIZE)

    # plt.show()

if __name__ == '__main__':
    main()
