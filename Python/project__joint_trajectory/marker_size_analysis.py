#!/usr/bin/env python
import argparse
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

LABEL_SIZE = 14
TITLE_SIZE = 18
SUPTITLE_SIZE = 24
LEGEND_SIZE = 14

def main():
    Z = 0.1 # m
    alpha = 60 # deg
    alpha_2 = np.deg2rad(alpha / 2)
    S = 0.1
    S_2 = S / 2
    max_delta_theta = 20 # deg
    max_trans_theta = 0.05 # deg
    nb_elems = 128
    orientation_error = np.linspace(0, np.deg2rad(max_delta_theta), nb_elems)
    position_error = np.linspace(0, max_trans_theta, nb_elems)

    print(f"orientation_error={orientation_error}")
    print(f"position_error={position_error}")

    xv, yv = np.meshgrid(position_error, orientation_error, indexing='ij')

    def z_func(pos_error, ori_error):
        return Z * np.tan(alpha_2 - ori_error) - S_2 - pos_error

    # constraints = np.zeros((nb_elems, nb_elems))

    # for i in range(nb_elems):
    #     for j in range(nb_elems):
    #         constraints[i,j] = Z * np.tan(alpha_2 - orientation_error[j]) - S_2 - position_error[i]

    # print(f"constraints={constraints}")

    Z = z_func(xv, yv)

    plot = plt.pcolormesh(position_error, orientation_error, Z, cmap='viridis', shading='flat')

    plt.colorbar(plot)
    plt.show()


if __name__ == '__main__':
    main()
