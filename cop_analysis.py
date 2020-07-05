# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: February 22, 2020
#
# Description: Signal Processing of COP
#   during stability testing in neuronormative
#   subjects for thesis research.
#
# Last updated: May 13, 2020

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
from scipy import signal

# constants
fs_cop = 1200
t_cop = np.arange(0, 30, 1 / fs_cop)

filepath = "/Users/natalietipton/Code/center_of_pressure/data/SB01/SB01_Trial27_norm_spliced.csv"


#####################################################################################

if __name__ == "__main__":

    # function call for one force plate files
    # x_cop, y_cop, x_cop_df = an.read_data_onefp(filepath, 3, ["Cx", "Cy"], 36000, [4])

    # # function call for two force plate files
    # x_cop, y_cop = an.read_data_twofp(
    #     filepath, 3, ["Fz", "Cx", "Cy", "Fz.1", "Cx.1", "Cy.1"], 36000, [4]
    # )

    # x_cop = an.butter_lowpass_filter(x_cop, 6, fs_cop)
    # y_cop = an.butter_lowpass_filter(y_cop, 6, fs_cop)

    # get length of signals
    df = pd.read_csv(filepath, index_col=False)

    # convert data from dataframe into a numpy list
    df = df.to_numpy()
    values = np.delete(df, 0, 1)
    x_cop, y_cop = values.T

    n_cop = len(x_cop)

    # time signals for correlation
    t_corr_cop = np.arange(-n_cop / fs_cop, n_cop / fs_cop - 1 / fs_cop, 1 / fs_cop)

    # standardize the signals to between 0 - 1
    x_cop_standard = an.standardize(x_cop)
    y_cop_standard = an.standardize(y_cop)

    #################### CoP X-axis analysis ####################

    plt.figure()
    plt.subplot(121)
    plt.plot(t_cop, x_cop)
    plt.title("Raw COP data in AP direction for\nEOFTanDF trial")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.subplot(122)
    plt.plot(t_cop, y_cop)
    plt.title("Raw COP data in ML direction for\nEOFTanDF trial")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.show()
    #################
    # an.plot(t_cop, x_cop, "Time (s)", "COP", "Raw COP data in AP direction", None, None)

    vel_x_cop = an.deriv(t_cop, x_cop)
    vel_y_cop = an.deriv(t_cop, y_cop)
    acc_x_cop = an.deriv(t_cop[:-1], vel_x_cop)

    plt.figure()
    plt.subplot(121)
    plt.plot(t_cop[:-1], vel_x_cop)
    plt.title("COP Velocity in AP direction for\nbaseline ECFTanDF trial")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mm/s)")
    plt.subplot(122)
    plt.plot(t_cop[:-1], vel_y_cop)
    plt.title("COP Velocity in ML direction for\nbaseline ECFTanDF trial")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (mm/s)")
    plt.show()

# print(
#     f"Average velocity of X CoP = {np.mean(sorted(vel_x_cop, reverse=True)[:10])}"
# )
# print(
#     f"Average acceleration of X CoP = {np.mean(sorted(acc_x_cop, reverse=True)[:10])}"
# )

# an.plot(t_cop[:-1], vel_x_cop, "time (s)", "Velocity", "CoP X Velocity", None, None)
# an.plot(
#     t_cop[:-2],
#     acc_x_cop,
#     "time (s)",
#     "Acceleration",
#     "CoP X Acceleration",
#     None,
#     None,
# )

# # calculate auto and cross correlations
# auto_x_cop = np.correlate(x_cop_standard, x_cop_standard, mode="full")
# cross_xy_cop = np.correlate(x_cop_standard, y_cop_standard, mode="full")
# auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
# auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")

# an.plot(
#     t_corr_cop,
#     auto_x_cop,
#     "time (s)",
#     "autocorrelation",
#     "Autocorrelation of Cx",
#     [0, 30],
#     None,
# )

# an.plot(
#     t_corr_cop,
#     cross_xy_cop,
#     "time (s)",
#     "cross correlation",
#     "Cx and Cy cross correlation",
#     None,
#     None,
# )

# # calculate autopower
# n_auto_cop = len(auto_x_cop)
# freq_cop = np.arange(0, fs_cop, fs_cop / n_auto_cop)

# Sxx_cop = np.fft.fft(auto_x_cop)
# Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
# Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
# print(abs(np.max(Sxx_vel_cop)))
# print(f"Total Power of X CoP Velocity = {sum(abs(Sxx_vel_cop))}")

# plt.figure()
# plt.subplot(121)
# plt.plot(freq_cop[:-2], abs(Sxx_vel_cop))
# plt.title("PSD of COP Velocity in AP direction for\nECFTanDF trial")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.xlim([0, 10])
# plt.subplot(122)
# plt.plot(freq_cop[:-2], abs(Syy_vel_cop))
# plt.title("PSD of COP Velocity in ML direction for\nECFTanDF trial")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.xlim([0, 10])
# plt.show()

# an.plot(
#     freq_cop,
#     abs(Sxx_cop),
#     "Frequency (Hz)",
#     "autopower",
#     "Cx Autopower",
#     [0, 10],
#     None,
# )

# an.plot(
#     freq_cop[:-2],
#     abs(Sxx_vel_cop),
#     "Frequency (Hz)",
#     "autopower",
#     "Autopower of X CoP Velocity",
#     [0, 10],
#     None,
# )

# # calculate cross power
# Sxy_cop = np.fft.fft(cross_xy_cop)

# an.plot(
#     freq_cop,
#     abs(Sxy_cop),
#     "frequency (Hz)",
#     "cross power",
#     "Cross Power between Cx and Cy",
#     [0, 0.25],
#     None,
# )

# #################### CoP Y-axis analysis ####################

# an.plot(t_cop, y_cop, "time (s)", "CoP", "Raw Cy signal", None, None)

# vel_y_cop = an.deriv(t_cop, y_cop)
# acc_y_cop = an.deriv(t_cop[:-1], vel_y_cop)

# print(
#     f"Average velocity of Y CoP = {np.mean(sorted(vel_y_cop, reverse=True)[:10])}"
# )
# print(
#     f"Average acceleration of Y CoP = {np.mean(sorted(acc_y_cop, reverse=True)[:10])}"
# )

# an.plot(t_cop[:-1], vel_y_cop, "time (s)", "Velocity", "CoP Y Velocity", None, None)
# an.plot(
#     t_cop[:-2],
#     acc_y_cop,
#     "time (s)",
#     "Acceleration",
#     "CoP Y Acceleration",
#     None,
#     None,
# )

# auto_y_cop = np.correlate(y_cop_standard, y_cop_standard, mode="full")
# auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")

# an.plot(
#     t_corr_cop,
#     auto_y_cop,
#     "time (s)",
#     "autocorrelation",
#     "Autocorrelation of Cy",
#     [0, 30],
#     None,
# )

# Syy_cop = np.fft.fft(auto_y_cop)
# Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
# print(f"Total Power of Y CoP Velocity = {sum(abs(Syy_vel_cop))}")

# an.plot(
#     freq_cop,
#     abs(Syy_cop),
#     "frequency (Hz)",
#     "autopower",
#     "Cy Autopower",
#     [0, 10],
#     None,
# )

# an.plot(
#     freq_cop[:-2],
#     abs(Syy_vel_cop),
#     "frequency (Hz)",
#     "autopower",
#     "Autopower of CoP Y Velocity",
#     [0, 10],
#     None,
# )
###############
