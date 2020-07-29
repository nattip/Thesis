# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: May 8, 2020
#
# Description: Signal Processing of COM
#   during stability testing in neuronormative
#   subjects for thesis research.
#
# Last updated: May 8, 2020

# import packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn import preprocessing
import skimage
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import analysis_lib as an

# constants
fs_com = 120
fs_cop = 1200
t_com = np.arange(0, 30, 1 / fs_com)
t_cop = np.arange(0, 30, 1 / fs_cop)
filepath = "/Users/natalietipton/Code/Data/SB01/SB01Trial_27.csv"


if __name__ == "__main__":

    # function call for reading CoM data from both 1 and 2 force plate files
    x_com, y_com, z_com = an.read_data_com(
        filepath, 36009, ["X", "Y", "Z"], 3600, [36010]
    )

    # x_cop, y_cop, x_cop_df = an.read_data_onefp(filepath, 3, ["Cx", "Cy"], 36000, [4])
    x_cop, y_cop = an.read_data_twofp(
        filepath, 3, ["Fz", "Cx", "Cy", "Fz.1", "Cx.1", "Cy.1"], 36000, [4]
    )

    n_com = len(x_com)
    t_corr_com = np.arange(-n_com / fs_com, n_com / fs_com - 1 / fs_com, 1 / fs_com)

    # standardize the signals
    x_com_standard = an.standardize(x_com)
    y_com_standard = an.standardize(y_com)
    z_com_standard = an.standardize(z_com)

    x_cop_standard = an.standardize(x_cop)
    y_cop_standard = an.standardize(y_cop)

    ################### CoM X-axis analysis ####################

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(t_com, x_com_standard, color="r")
    # plt.plot(t_cop, x_cop_standard)
    # plt.title("Standardized COP and COM data in AP direction for\nECFTanDF trial")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Distance From Origin (mm)")
    # plt.legend(["COM", "COP"])
    # plt.subplot(122)
    # plt.plot(t_com, y_com_standard, color="r")
    # plt.plot(t_cop, y_cop_standard)
    # plt.title("Standardized COP and COM data in ML direction for\nECFTanDF trial")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Distance From Origin (mm)")
    # plt.legend(["COM", "COP"])
    # plt.show()

    # plt.figure()
    # plt.subplot(121)
    # plt.plot(t_com, x_com)
    # plt.title("Raw COM data in AP direction for\nECFTanDF trial")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Distance From Origin (mm)")
    # plt.subplot(122)
    # plt.plot(t_com, y_com)
    # plt.title("Raw COM data in ML direction for\nECFTanDF trial")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Distance From Origin (mm)")
    # plt.show()

#######
# an.plot(t_com, x_com, "time (s)", "CoM", "Raw X signal", None, None)

# vel_x_com = an.deriv(t_com, x_com)
# acc_x_com = an.deriv(t_com[:-1], vel_x_com)

# print(
#     f"Average velocity of X CoM = {np.mean(sorted(vel_x_com, reverse=True)[:10])}"
# )
# print(
#     f"Average acceleration of X CoM = {np.mean(sorted(acc_x_com, reverse=True)[:10])}"
# )

# an.plot(t_com[:-1], vel_x_com, "time (s)", "Velocity", "CoM X Velocity", None, None)
# an.plot(
#     t_com[:-2],
#     acc_x_com,
#     "time (s)",
#     "Acceleration",
#     "CoM X Acceleration",
#     None,
#     None,
# )

# plt.figure()
# plt.plot(t_com, x_com_standard)
# plt.title("CoM in x axis")
# plt.show()

# auto_x_com = np.correlate(x_com_standard, x_com_standard, mode="full")
# cross_xy_com = np.correlate(x_com_standard, y_com_standard, mode="full")

# an.plot(
#     t_corr_com,
#     auto_x_com,
#     "time (s)",
#     "autocorrelation",
#     "Autocorrelation of x in CoM",
#     [0, 30],
#     None,
# )

# an.plot(
#     t_corr_com,
#     cross_xy_com,
#     "time (s)",
#     "cross correlation",
#     "X and Y CoM cross correlation",
#     None,
#     None,
# )

# n_auto_com = len(auto_x_com)
# freq_com = np.arange(0, fs_com, fs_com / n_auto_com)

# Sxx_com = np.fft.fft(auto_x_com)

# an.plot(
#     freq_com,
#     abs(Sxx_com),
#     "Frequency (Hz)",
#     "autopower",
#     "X of CoM Autopower",
#     [0, 1],
#     None,
# )

# Sxy_com = np.fft.fft(cross_xy_com)

# an.plot(
#     freq_com,
#     abs(Sxy_com),
#     "frequency (Hz)",
#     "cross power",
#     "Cross Power between X and Y CoM",
#     [0, 0.25],
#     None,
# )

# #################### CoM Y-axis analysis ####################

# an.plot(t_com, y_com, "time (s)", "CoM", "Raw Y signal", None, None)

# vel_y_com = an.deriv(t_com, y_com)
# acc_y_com = an.deriv(t_com[:-1], vel_y_com)

# print(
#     f"Average velocity of Y CoM = {np.mean(sorted(vel_y_com, reverse=True)[:10])}"
# )
# print(
#     f"Average acceleration of Y CoM = {np.mean(sorted(acc_y_com, reverse=True)[:10])}"
# )

# an.plot(t_com[:-1], vel_y_com, "time (s)", "Velocity", "CoM Y Velocity", None, None)
# an.plot(
#     t_com[:-2],
#     acc_y_com,
#     "time (s)",
#     "Acceleration",
#     "CoM Y Acceleration",
#     None,
#     None,
# )

# plt.figure()
# plt.plot(t_com, y_com_standard)
# plt.title("CoM in y axis")
# plt.show()

# auto_y_com = np.correlate(y_com_standard, y_com_standard, mode="full")

# an.plot(
#     t_corr_com,
#     auto_y_com,
#     "time (s)",
#     "autocorrelation",
#     "Autocorrelation of CoM Y",
#     [0, 30],
#     None,
# )

# Syy_com = np.fft.fft(auto_y_com)

# an.plot(
#     freq_com,
#     abs(Syy_com),
#     "frequency (Hz)",
#     "autopower",
#     "CoM Y Autopower",
#     [0, 1],
#     None,
# )

# # x_cop_df_resample = []
# # y_cop_df_resample = []
# # for i in range(0, n_com):
# #     x_cop_df_resample.append(x_cop_standard[i * 10])
# #     y_cop_df_resample.append(y_cop_standard[i * 10])

# # cross_x_cop_com = np.correlate(x_cop_df_resample, x_com_standard, mode="full")
# # cross_y_cop_com = np.correlate(y_cop_df_resample, y_com_standard, mode="full")

# # plot(
# #     t_corr_com,
# #     cross_x_cop_com,
# #     "time (s)",
# #     "cross correlation",
# #     "X CoM and X CoP cross correlation",
# #     [0, 30],
# #     None,
# # )

# # plot(
# #     t_corr_com,
# #     cross_y_cop_com,
# #     "time (s)",
# #     "cross correlation",
# #     "Y CoM and Y CoP cross correlation",
# #     [0, 30],
# #     None,
# # )

