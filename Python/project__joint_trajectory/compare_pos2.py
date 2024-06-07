#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

np.set_printoptions(suppress=True)

LABEL_SIZE = 14
TITLE_SIZE = 18
SUPTITLE_SIZE = 24
LEGEND_SIZE = 14

def read_sensor_xml(path, sensor_positions, sensor_velocities, sensor_accelerations, sensor_time):
    print("Read xml data")
    treeSensor = ET.parse(path)
    rootSensor = treeSensor.getroot()
    i = 0
    foundAllSensors = False
    nbSensors = 0
    while i < len(rootSensor) and not foundAllSensors:
        child = rootSensor[i]
        print(child.tag, child.attrib)
        if child.tag == 'position_sensor':
            nbSensors += 1
        else:
            foundAllSensors = True
        i+=1
    print("nbSensors : ", nbSensors)
    for sensorId in range(nbSensors):
        sensor_positions.append([])
        sensor_velocities.append([])
        sensor_accelerations.append([])
    for point in rootSensor[nbSensors-1:]:
        if point.tag == 'point':
            for childId in range(len(point)):
                if childId < nbSensors and point[childId].tag=='position':
                    sensor_positions[childId].append(float(point[childId].attrib["value"]))
                elif childId < 2*nbSensors and point[childId].tag=='velocity':
                    sensor_velocities[childId-nbSensors].append(float(point[childId].attrib["value"]))
                elif childId < 3*nbSensors and point[childId].tag=='acceleration':
                    sensor_accelerations[childId-2*nbSensors].append(float(point[childId].attrib["value"]))
                elif point[childId].tag=='time_from_start':
                    sensor_time.append(float(point[childId].attrib["value"]))
    return nbSensors, rootSensor

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

def get_time_indices(time_vector, start_time_value, end_time_value):
    time_vec_1 = time_vector - start_time_value
    time_vec_2 = time_vector - end_time_value
    print(f"time_vec_1={time_vec_1.shape}")
    print(f"time_vec_2={time_vec_2.shape}")
    return np.argmin(np.abs(time_vec_1)), np.argmin(np.abs(time_vec_2))
    # return np.argmin(), np.argmin(time_vector - end_time_value)

def get_index_readjusting(cmd, meas, index_start, search_window=50):
    nb_elems = cmd.shape[0]
    print(f"cmd.shape={cmd.shape} ; meas.shape={meas.shape} ; meas_trunc.shape={meas[index_start:index_start+nb_elems].shape}")

    mse_vec = np.zeros(search_window)
    for i in range(search_window):
        error = np.linalg.norm(cmd - meas[index_start+i:index_start+nb_elems+i], axis=1)
        mse_vec[i] = np.mean(error)

    min_idx = np.argmin(mse_vec)
    # print(f"mse_vec.shape: {mse_vec.shape} ; mse_vec={mse_vec} ; min_idx={min_idx}")

    return min_idx


def main():
    parser = argparse.ArgumentParser(description='Plot joint positions / velocities / accelerations.')
    parser.add_argument("--npz", help='Path to the npy/npz file.')
    parser.add_argument("--xml", "--sensors", help='Path to the joint position, velocity and acceleration XML file.')
    parser.add_argument("--txt", "--FT", help='Path to the XML file containing F/T data.')
    parser.add_argument("--offset", type=int, help='Offset.')
    parser.add_argument("--nb-joints", default=6, type=int, help='Number of joints.')
    args = parser.parse_args()

    data_path = args.npz
    data = np.load(data_path)

    clock_now_seconds = data['clock_now_seconds']
    time_from_start_vec = data['time_from_start_vec']
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


    sensor_time_vec = []
    sensor_positions_list = []
    sensor_velocities_list = []
    sensor_accelerations_list = []

    nbSensors, rootSensor = read_sensor_xml(args.xml, sensor_positions_list, sensor_velocities_list, sensor_accelerations_list, sensor_time_vec)

    sensor_time = np.ascontiguousarray(sensor_time_vec)
    sensor_positions = np.ascontiguousarray(sensor_positions_list).T
    sensor_velocities = np.ascontiguousarray(sensor_velocities_list).T
    sensor_accelerations = np.ascontiguousarray(sensor_accelerations_list).T

    print(f"pos_cmd shape={positions_vec.shape}")
    print(f"sensor_time.shape={sensor_time.shape}")
    print(f"sensor_positions.shape={sensor_positions.shape}")
    print(f"sensor_velocities.shape={sensor_velocities.shape}")
    print(f"sensor_accelerations.shape={sensor_accelerations.shape}")
    print(f"nbSensors={nbSensors}")

    # TODO:
    nb_joints = args.nb_joints
    fig, axs = plt.subplots(nb_joints)

    print(f"start_time={clock_now_seconds + time_from_start_vec[0]} : end_time={clock_now_seconds + time_from_start_vec[-1]}")
    idx_start, idx_end = get_time_indices(sensor_time, clock_now_seconds + time_from_start_vec[0], clock_now_seconds + time_from_start_vec[-1])
    time_size = time_from_start_vec.shape[0]
    print(f"time_size={time_size} ; idx_start={idx_start} ; idx_end={idx_end}")

    min_index = get_index_readjusting(positions_vec, sensor_positions, idx_start)
    if args.offset is not None:
        min_index = args.offset
    print(f"min_index={min_index}")


    # Joint positions
    for idx, ax in enumerate(axs):
        ax.plot(clock_now_seconds + time_from_start_vec, positions_vec[:,idx], label="cmd")
        ax.plot(sensor_time[idx_start-min_index:idx_start+time_size-min_index],
                sensor_positions[idx_start:idx_start+time_size, idx], label="meas")

        ax.grid()
        ax.legend()
        ax.set_title("q_{:d}".format(idx+1), fontsize=TITLE_SIZE)

    fig.suptitle('Joint positions', fontsize=SUPTITLE_SIZE)


    # Joint velocities
    fig, axs = plt.subplots(nb_joints)
    for idx, ax in enumerate(axs):
        ax.plot(clock_now_seconds + time_from_start_vec, velocities_vec[:,idx], label="cmd")
        ax.plot(sensor_time[idx_start-min_index:idx_start+time_size-min_index],
                sensor_velocities[idx_start:idx_start+time_size, idx], label="meas")

        ax.grid()
        ax.legend()
        ax.set_title("qd_{:d}".format(idx+1), fontsize=TITLE_SIZE)

    fig.suptitle('Joint velocities', fontsize=SUPTITLE_SIZE)


    # Joint accelerations
    fig, axs = plt.subplots(nb_joints)
    for idx, ax in enumerate(axs):
        ax.plot(clock_now_seconds + time_from_start_vec, accelerations_vec[:,idx], label="cmd")
        ax.plot(sensor_time[idx_start-min_index:idx_start+time_size-min_index],
                sensor_accelerations[idx_start:idx_start+time_size, idx], label="meas")

        ax.grid()
        ax.legend()
        ax.set_title("qdd_{:d}".format(idx+1), fontsize=TITLE_SIZE)

    fig.suptitle('Joint accelerations', fontsize=SUPTITLE_SIZE)


    # Plot position, velocity and acceleration for each joint
    FT_data = None
    if args.txt is not None:
        FT_data_filepath = args.txt
        print(f"FT_data_filepath={FT_data_filepath}")
        FT_data = np.loadtxt(FT_data_filepath)
        print(f"FT_data={FT_data.shape}")

    for cpt_fig in range(nb_joints):
        fig, axs = plt.subplots(5)

        # Joint positions
        axs[0].plot(clock_now_seconds + time_from_start_vec, positions_vec[:,cpt_fig], label="cmd")
        axs[0].plot(sensor_time[idx_start-min_index:idx_start+time_size-min_index],
                sensor_positions[idx_start:idx_start+time_size, cpt_fig], label="meas")

        axs[0].grid()
        axs[0].legend()
        # axs[0].set_xlabel("Time (s)", fontsize=LABEL_SIZE)
        axs[0].set_ylabel("rad", fontsize=LABEL_SIZE)
        axs[0].set_title("Position", fontsize=TITLE_SIZE)

        # Joint velocities
        axs[1].plot(clock_now_seconds + time_from_start_vec, velocities_vec[:,cpt_fig], label="cmd")
        axs[1].plot(sensor_time[idx_start-min_index:idx_start+time_size-min_index],
                sensor_velocities[idx_start:idx_start+time_size, cpt_fig], label="meas")

        axs[1].grid()
        axs[1].legend()
        # axs[1].set_xlabel("Time (s)", fontsize=LABEL_SIZE)
        axs[1].set_ylabel("rad/s", fontsize=LABEL_SIZE)
        axs[1].set_title("Velocity", fontsize=TITLE_SIZE)

        # Joint accelerations
        axs[2].plot(clock_now_seconds + time_from_start_vec, accelerations_vec[:,cpt_fig], label="cmd")
        axs[2].plot(sensor_time[idx_start-min_index:idx_start+time_size-min_index],
                sensor_accelerations[idx_start:idx_start+time_size, cpt_fig], label="meas")

        axs[2].grid()
        axs[2].legend()
        # axs[2].set_xlabel("Time (s)", fontsize=LABEL_SIZE)
        axs[2].set_ylabel("rad/s^2", fontsize=LABEL_SIZE)
        axs[2].set_title("Acceleration", fontsize=TITLE_SIZE)

        if FT_data is not None:
            axs[3].plot(FT_data[idx_start-min_index:idx_start+time_size-min_index,0],
                    FT_data[idx_start:idx_start+time_size, 1], label="Tx")
            axs[3].plot(FT_data[idx_start-min_index:idx_start+time_size-min_index,0],
                    FT_data[idx_start:idx_start+time_size, 2], label="Ty")
            axs[3].plot(FT_data[idx_start-min_index:idx_start+time_size-min_index,0],
                    FT_data[idx_start:idx_start+time_size, 3], label="Tz")

            axs[3].grid()
            axs[3].legend()
            # axs[3].set_xlabel("Time (s)", fontsize=LABEL_SIZE)
            axs[3].set_ylabel("Nm", fontsize=LABEL_SIZE)
            axs[3].set_title("Torque", fontsize=TITLE_SIZE)


            axs[4].plot(FT_data[idx_start-min_index:idx_start+time_size-min_index,0],
                    FT_data[idx_start:idx_start+time_size, 4], label="Fx")
            axs[4].plot(FT_data[idx_start-min_index:idx_start+time_size-min_index,0],
                    FT_data[idx_start:idx_start+time_size, 5], label="Fy")
            axs[4].plot(FT_data[idx_start-min_index:idx_start+time_size-min_index,0],
                    FT_data[idx_start:idx_start+time_size, 6], label="Fz")

            axs[4].grid()
            axs[4].legend()
            axs[4].set_xlabel("Time (s)", fontsize=LABEL_SIZE+4)
            axs[4].set_ylabel("N", fontsize=LABEL_SIZE)
            axs[4].set_title("Force", fontsize=TITLE_SIZE)

        plt.subplots_adjust(hspace=0.5)
        fig.suptitle("q{:d}".format(cpt_fig+1), fontsize=SUPTITLE_SIZE)


    plt.show()

if __name__ == '__main__':
    main()
