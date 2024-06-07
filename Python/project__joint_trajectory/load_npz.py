#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


def main():
    parser = argparse.ArgumentParser(description='Plot joint positions / velocities / accelerations.')
    parser.add_argument("--file", "--data", help='Path to the npy/npz file.')
    parser.add_argument("--marker", default='+', help='Plot marker type.')
    parser.add_argument("--left-xlim", type=float, default=0, help='Plot left xlim.')
    parser.add_argument("--reverse", action='store_true')
    parser.add_argument("--no-xlim", action='store_true')
    parser.add_argument("--start-time", type=int, default=0, help='Start time index.')
    parser.add_argument("--end-time", type=int, default=-1, help='End time index.')
    args = parser.parse_args()

    data_path = args.file
    data = np.load(data_path)
    positions_vec = data['positions_vec']
    velocities_vec = data['velocities_vec']
    accelerations_vec = data['accelerations_vec']
    joints_size = positions_vec.shape[1]
    if 'joints_size' in data.keys():
        joints_size = data['joints_size'][0]
    joint_names = []
    joint_types = []
    only_revolute = True
    for idx in range(joints_size):
        name = 'joint_{:02d}_name'.format(idx)
        if name in data.keys():
            joint_names.append("".join([chr(item) for item in data[name]]))
        else:
            joint_names.append(name)

        type = 'joint_{:02d}_type'.format(idx)
        if type in data.keys():
            joint_types.append("".join([chr(item) for item in data[type]]))
            if joint_types[-1] != "Revolute":
                only_revolute = False
        else:
            joint_types.append("Revolute") # always assume revolute joint

    print(f"joints_size: {joints_size}")
    print(f"joint_names: {joint_names}")
    print(f"joint_types: {joint_types}")
    reverse = args.reverse
    no_xlim = args.no_xlim

    if reverse:
        positions_vec = positions_vec.transpose()
        velocities_vec = velocities_vec.transpose()
        accelerations_vec = accelerations_vec.transpose()

    print(f"positions_vec shape: {positions_vec.shape}")
    print(f"velocities_vec shape: {velocities_vec.shape}")
    print(f"accelerations_vec shape: {accelerations_vec.shape}")

    LABEL_SIZE = 12
    SUPTITLE_SIZE = 18
    LEGEND_SIZE = 14

    marker_type = args.marker
    left_xlim = args.left_xlim

    time_from_start_vec = data['time_from_start_vec']
    print(f"time_from_start_vec shape: {time_from_start_vec.shape}")

    start_time = args.start_time
    end_time = args.end_time
    print(f"Start time idx: {start_time} ; End time idx: {end_time}")
    print(f"Start time: {time_from_start_vec[start_time]} ; End time: {time_from_start_vec[end_time]}")

    fig, axs = plt.subplots(3)
    fig.canvas.set_window_title("1) " + data_path)
    nb_revolute_joints = 0
    for i in range(positions_vec.shape[1]):
        if joint_types[i] == "Revolute":
            nb_revolute_joints += 1
            axs[0].plot(time_from_start_vec[start_time:end_time], positions_vec[start_time:end_time,i], marker=marker_type, label=joint_names[i])
    axs[0].set_title('Positions', fontsize=SUPTITLE_SIZE)
    axs[0].set_ylabel('rad', fontsize=LABEL_SIZE)

    for i in range(velocities_vec.shape[1]):
        if joint_types[i] == "Revolute":
            axs[1].plot(time_from_start_vec[start_time:end_time], velocities_vec[start_time:end_time,i], marker=marker_type, label=joint_names[i])
    axs[1].set_title('Velocities', fontsize=SUPTITLE_SIZE)
    axs[1].set_ylabel('rad/s', fontsize=LABEL_SIZE)

    for i in range(accelerations_vec.shape[1]):
        if joint_types[i] == "Revolute":
            axs[2].plot(time_from_start_vec[start_time:end_time], accelerations_vec[start_time:end_time,i], marker=marker_type, label=joint_names[i])
    axs[2].set_title('Accelerations', fontsize=SUPTITLE_SIZE)
    axs[2].set_xlabel('Time (s)', fontsize=LABEL_SIZE)
    axs[2].set_ylabel('rad/s^2', fontsize=LABEL_SIZE)
    axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=nb_revolute_joints, fontsize=LEGEND_SIZE)

    for ax in axs:
        ax.grid()
        if not no_xlim:
            ax.set_xlim(left=left_xlim)
        # ax.legend(loc='upper center', ncol=7)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.subplots_adjust(hspace=0.3) # See: https://stackoverflow.com/a/6541454
    # fig.tight_layout() # See: https://stackoverflow.com/a/9827848


    # For prismatic joint (e.g. rail)
    if not only_revolute:
        fig, axs = plt.subplots(3)
        fig.canvas.set_window_title("2) " + data_path)

        nb_prismatic_joints = 0
        for i in range(positions_vec.shape[1]):
            if joint_types[i] != "Revolute":
                nb_prismatic_joints += 1
                axs[0].plot(time_from_start_vec[start_time:end_time], positions_vec[start_time:end_time,i], marker=marker_type, label=joint_names[i])
        axs[0].grid()
        if not no_xlim:
            axs[0].set_xlim(left=left_xlim)
        axs[0].set_title('Positions', fontsize=SUPTITLE_SIZE)
        axs[0].set_ylabel('m', fontsize=LABEL_SIZE)
        axs[0].legend(ncol=nb_prismatic_joints, fontsize=LEGEND_SIZE)

        for i in range(velocities_vec.shape[1]):
            if joint_types[i] == "Revolute":
                axs[1].plot(time_from_start_vec[start_time:end_time], velocities_vec[start_time:end_time,i], marker=marker_type, label=joint_names[i])
        axs[1].grid()
        if not no_xlim:
            axs[1].set_xlim(left=left_xlim)
        axs[1].set_title('Velocities', fontsize=SUPTITLE_SIZE)
        axs[1].set_ylabel('m/s', fontsize=LABEL_SIZE)
        axs[1].legend(ncol=nb_prismatic_joints, fontsize=LEGEND_SIZE)

        for i in range(accelerations_vec.shape[1]):
            if joint_types[i] == "Revolute":
                axs[2].plot(time_from_start_vec[start_time:end_time], accelerations_vec[start_time:end_time,i], marker=marker_type, label=joint_names[i])
        axs[2].grid()
        if not no_xlim:
            axs[2].set_xlim(left=left_xlim)
        axs[2].set_title('Accelerations', fontsize=SUPTITLE_SIZE)
        axs[2].set_xlabel('Time (s)', fontsize=LABEL_SIZE)
        axs[2].set_ylabel('m/s^2', fontsize=LABEL_SIZE)
        axs[2].legend(ncol=nb_prismatic_joints, fontsize=LEGEND_SIZE)

    plt.show()

if __name__ == '__main__':
    main()
