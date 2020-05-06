# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: February 22, 2020
#
# Description: Signal Processing of COM and COP
#   during stability testing in neuronormative
#   subjects for thesis research.
#
# Last updated: May 4, 2020

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

# constants
fs_cop = 1200
t_cop = np.arange(0, 30, 1 / fs_cop)
fs_com = 120
t_com = np.arange(0, 30, 1 / fs_com)
filepath = "/Users/natalietipton/Code/Data/SB01/SB01Trial_19.csv"


# reads CoP data in from .csv files for data with feet together, one force plate
# returns: cx = CoP x position (AP), cy = CoP y position (ML)
# user specifies:
#   header = which row the data starts on (starting at 0, not 1 like the sheet)
#   usecols = names of the column headers that are to be included
#   nrows = number of data points to be read in
#   rows_skip = the number of any rows to not be included (starting at 0)
def read_data_onefp(filepath, row_start, cols, num_data, rows_skip):
    data = pd.read_csv(
        filepath,
        header=row_start,
        usecols=cols,
        nrows=num_data,
        dtype={"Cx": np.float64, "Cy": np.float64},
        skiprows=rows_skip,
    )

    # convert data frame into lists
    cx = data["Cx"].values.tolist()
    cy = data["Cy"].values.tolist()

    return cx, cy, data["Cx"]


# reads CoP data in from .csv files for feet tandem, two force plates
# combines CoP positions from both force plates into 1 resultant CoP
# returns: cx_combined = resultant CoP x position (AP),
#          cy_combined = resultant CoP y position (ML)
# user specifies:
#   header = which row the data starts on (starting at 0, not 1 like the sheet)
#   usecols = names of the column headers that are to be included
#   nrows = number of data points to be read in
#   rows_skip = the number of any rows to not be included (starting at 0)
def read_data_twofp(filepath, row_start, cols, num_data, rows_skip):
    data = pd.read_csv(
        filepath,
        header=row_start,
        usecols=["Fz", "Cx", "Cy", "Fz.1", "Cx.1", "Cy.1"],
        nrows=36000,
        dtype={
            "Fz": np.float64,
            "Cx": np.float64,
            "Cy": np.float64,
            "Fz.1": np.float64,
            "Cx.1": np.float64,
            "Cy.1": np.float64,
        },
        skiprows=[4],
    )

    # convert data frame into lists
    fz = data["Fz"].values.tolist()
    cx = data["Cx"].values.tolist()
    cy = data["Cy"].values.tolist()
    fz_1 = data["Fz.1"].values.tolist()
    cx_1 = data["Cx.1"].values.tolist()
    cy_1 = data["Cy.1"].values.tolist()

    cx_combined = []
    cy_combined = []

    # combine CoP from two force plates into one resultant CoP
    # combined(x,y) = COP1(x,y) + (F1(z) / F1(z) + F2(z)) + COP2(x,y) + (F2(z) / F1(z) + F2(z))
    for point in range(len(cx)):
        cx_combined.append(
            cx[point] * fz[point] / (fz[point] + fz_1[point])
            + cx_1[point] * fz_1[point] / (fz[point] + fz_1[point])
        )

        cy_combined.append(
            cy[point] * fz[point] / (fz[point] + fz_1[point])
            + cy_1[point] * fz_1[point] / (fz[point] + fz_1[point])
        )

    return cx_combined, cy_combined


# reads CoM data in from .csv files
# returns: x = CoM x position (AP), y = CoM y position (ML), z = CoM z position
# user specifies:
#   header = which row the data starts on (starting at 0, not 1 like the sheet)
#   usecols = names of the column headers that are to be included
#   nrows = number of data points to be read in
#   rows_skip = the number of any rows to not be included (starting at 0)
def read_data_com(filepath, row_start, cols, num_data, rows_skip):
    data = pd.read_csv(
        filepath,
        header=row_start,
        usecols=cols,
        nrows=num_data,
        dtype={"Cx": np.float64, "Cy": np.float64},
        skiprows=rows_skip,
    )

    # convert data frame into lists
    x = data["X"].values.tolist()
    y = data["Y"].values.tolist()
    z = data["Z"].values.tolist()

    return x, y, z


# standardizes the data given using the equation:
# standard[n] = (data[n] - mean[data]) / standard deviation[data]
def standardize(data):
    avg = np.mean(data)
    sd = np.std(data)

    standard = []

    for i in range(0, len(data)):
        standard.append((data[i] - avg) / sd)

    return standard


# creates a plot using matplotlib.pyplot
# user specifies:
#   x = data to be plotted on x axis
#   y = data to be plotted on y axis
#   xlabel = string for labeling x axis
#   ylabel = string for labeling y axis
#   title = string to titling plot
#   xlim = min and max limits for the x axis (use None for no lim)
#   ylim = min and max limits for the y axis (use None for no lim)
def plot(x, y, xlabel, ylabel, title, xlim, ylim):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def deriv(x, y):
    der = []
    der.append(0)

    for i in range(0, len(y) - 1):
        der.append((y[i + 1] - y[i]) / (x[i + 1] - x[i]))

    return der


#####################################################################################

if __name__ == "__main__":

    # function call for one force plate files
    x_cop, y_cop, x_cop_df = read_data_onefp(filepath, 3, ["Cx", "Cy"], 36000, [4])

    # function call for two force plate files
    # x_cop, y_cop = read_data_twofp(
    #     filepath, 3, ["Fz", "Cx", "Cy", "Fz.1", "Cx.1", "Cy.1"], 36000, [4]
    # )

    # function call for reading CoM data from both 1 and 2 force plate files
    x_com, y_com, z_com = read_data_com(filepath, 36009, ["X", "Y", "Z"], 3600, [36010])

    # get length of signals
    n_cop = len(x_cop)
    n_com = len(x_com)

    # time signals for correlation
    t_corr_cop = np.arange(-n_cop / fs_cop, n_cop / fs_cop - 1 / fs_cop, 1 / fs_cop)
    t_corr_com = np.arange(-n_com / fs_com, n_com / fs_com - 1 / fs_com, 1 / fs_com)

    # standardize the signals to between 0 - 1
    x_cop_standard = standardize(x_cop)
    y_cop_standard = standardize(y_cop)
    x_com_standard = standardize(x_com)
    y_com_standard = standardize(y_com)
    z_com_standard = standardize(z_com)

    #################### CoP X-axis analysis ####################

    plot(t_cop, x_cop, "time (s)", "CoP", "Raw Cx signal", None, None)

    vel_x_cop = []
    acc_x_cop = []

    vel_x_cop.append(0)
    acc_x_cop.append(0)

    vel_x_cop = deriv(t_cop, x_cop)
    acc_x_cop = deriv(t_cop, vel_x_cop)

    plot(t_cop, vel_x_cop, "time (s)", "Velocity", "CoP X Velocity", None, None)
    plot(t_cop, acc_x_cop, "time (s)", "Acceleration", "CoP X Acceleration", None, None)
    # calculate auto and cross correlations
    auto_x_cop = np.correlate(x_cop_standard, x_cop_standard, mode="full")
    cross_xy_cop = np.correlate(x_cop_standard, y_cop_standard, mode="full")

    plot(
        t_corr_cop,
        auto_x_cop,
        "time (s)",
        "autocorrelation",
        "Autocorrelation of Cx",
        [0, 30],
        None,
    )

    plot(
        t_corr_cop,
        cross_xy_cop,
        "time (s)",
        "cross correlation",
        "Cx and Cy cross correlation",
        None,
        None,
    )

    # calculate autopower
    n_auto_cop = len(auto_x_cop)
    freq_cop = np.arange(0, fs_cop, fs_cop / n_auto_cop)
    Sxx_cop = np.fft.fft(auto_x_cop)

    plot(
        freq_cop,
        abs(Sxx_cop),
        "Frequency (Hz)",
        "autopower",
        "Cx Autopower",
        [0, 1],
        None,
    )

    # calculate cross power
    Sxy_cop = np.fft.fft(cross_xy_cop)

    plot(
        freq_cop,
        abs(Sxy_cop),
        "frequency (Hz)",
        "cross power",
        "Cross Power between Cx and Cy",
        [0, 0.25],
        None,
    )

    # # increase magnitude of signal for more sensitivity in entropy
    # x_cop_multiplied = np.multiply(x_cop, 100)

    # # create overlapping windows
    # x_cop_array = np.asarray(x_cop_multiplied)
    # x_cop_overlap = skimage.util.view_as_windows(x_cop_array, 1800, step=900)
    # rows = x_cop_overlap.shape[0]

    # approx_entropy = []

    # # calculate moving approximate entropy
    # Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
    #     delayed(collect_approx_entropy)(ApEn(x_cop_overlap[row], 2, 10))
    #     for row in tqdm(range(0, rows))
    # )

    # # find average approximate entropy
    # avg_entr_x_cop = np.mean(approx_entropy)
    # print("Average entropy for X CoP =", avg_entr_x_cop)

    # entropy_windows = np.arange(0, rows)

    # plot(
    #     entropy_windows,
    #     approx_entropy,
    #     "Window",
    #     "Approximate Entropy",
    #     "Moving approximate entropy of Cx",
    #     None,
    #     [0, 0.4],
    # )

    #################### CoP Y-axis analysis ####################

    plot(t_cop, y_cop, "time (s)", "CoP", "Raw Cy signal", None, None)

    auto_y_cop = np.correlate(y_cop_standard, y_cop_standard, mode="full")

    plot(
        t_corr_cop,
        auto_y_cop,
        "time (s)",
        "autocorrelation",
        "Autocorrelation of Cy",
        [0, 30],
        None,
    )

    Syy_cop = np.fft.fft(auto_y_cop)

    plot(
        freq_cop,
        abs(Syy_cop),
        "frequency (Hz)",
        "autopower",
        "Cy Autopower",
        [0, 1],
        None,
    )

    # Entropy Stuff
    # y_cop_multiplied = np.multiply(y_cop, 100)
    # y_cop_array = np.asarray(y_cop_multiplied)
    # y_cop_overlap = skimage.util.view_as_windows(y_cop_array, 1800, step=900)

    # print(y_cop_overlap.shape)
    # rows = y_cop_overlap.shape[0]

    # approx_entropy = []

    # Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
    #     delayed(collect_approx_entropy)(ApEn(y_cop_overlap[row], 2, 10))
    #     for row in tqdm(range(0, rows))
    # )

    # avg_entr_y_cop = np.mean(approx_entropy)
    # print("Average entropy for Y CoP =", avg_entr_y_cop)

    # entropy_windows = np.arange(0, rows)

    # plot(
    #     entropy_windows,
    #     approx_entropy,
    #     "Window",
    #     "Approximate Entropy",
    #     "Moving approximate entropy of Cy",
    #     None,
    #     [0, 0.4],
    # )

    #################### CoM X-axis analysis ####################

    plot(t_com, x_com, "time (s)", "CoM", "Raw X signal", None, None)

    plt.figure()
    plt.plot(t_cop, x_cop_standard, t_com, x_com_standard)
    plt.title("CoP and CoM in x axis")
    plt.show()

    auto_x_com = np.correlate(x_com_standard, x_com_standard, mode="full")
    cross_xy_com = np.correlate(x_com_standard, y_com_standard, mode="full")

    plot(
        t_corr_com,
        auto_x_com,
        "time (s)",
        "autocorrelation",
        "Autocorrelation of x in CoM",
        [0, 30],
        None,
    )

    plot(
        t_corr_com,
        cross_xy_com,
        "time (s)",
        "cross correlation",
        "X and Y CoM cross correlation",
        None,
        None,
    )

    n_auto_com = len(auto_x_com)
    freq_com = np.arange(0, fs_com, fs_com / n_auto_com)

    Sxx_com = np.fft.fft(auto_x_com)

    plot(
        freq_com,
        abs(Sxx_com),
        "Frequency (Hz)",
        "autopower",
        "X of CoM Autopower",
        [0, 1],
        None,
    )

    Sxy_com = np.fft.fft(cross_xy_com)

    plot(
        freq_com,
        abs(Sxy_com),
        "frequency (Hz)",
        "cross power",
        "Cross Power between X and Y CoM",
        [0, 0.25],
        None,
    )

    # x_com_multiplied = np.multiply(x_com, 100)
    # x_com_array = np.asarray(x_com_multiplied)
    # x_com_overlap = skimage.util.view_as_windows(x_com_array, 180, step=90)

    # rows = x_com_overlap.shape[0]

    # approx_entropy = []

    # Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
    #     delayed(collect_approx_entropy)(ApEn(x_com_overlap[row], 2, 10))
    #     for row in tqdm(range(0, rows))
    # )

    # avg_entr_x_com = np.mean(approx_entropy)
    # print("Average entropy for X CoM =", avg_entr_x_com)

    # entropy_windows = np.arange(0, rows)

    # plot(
    #     entropy_windows,
    #     approx_entropy,
    #     "Window",
    #     "Approximate Entropy",
    #     "Moving approximate entropy of CoM X",
    #     None,
    #     [0, 0.4],
    # )

    #################### CoM Y-axis analysis ####################

    plot(t_com, y_com, "time (s)", "CoM", "Raw Y signal", None, None)

    plt.figure()
    plt.plot(t_cop, y_cop_standard, t_com, y_com_standard)
    plt.title("CoP and CoM in y axis")
    plt.show()

    auto_y_com = np.correlate(y_com_standard, y_com_standard, mode="full")

    plot(
        t_corr_com,
        auto_y_com,
        "time (s)",
        "autocorrelation",
        "Autocorrelation of CoM Y",
        [0, 30],
        None,
    )

    Syy_com = np.fft.fft(auto_y_com)

    plot(
        freq_com,
        abs(Syy_com),
        "frequency (Hz)",
        "autopower",
        "CoM Y Autopower",
        [0, 1],
        None,
    )

    # y_com_multiplied = np.multiply(y_com, 100)
    # y_com_array = np.asarray(y_com_multiplied)
    # y_com_overlap = skimage.util.view_as_windows(y_com_array, 180, step=90)

    # rows = y_com_overlap.shape[0]

    # approx_entropy = []

    # Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
    #     delayed(collect_approx_entropy)(ApEn(y_com_overlap[row], 2, 10))
    #     for row in tqdm(range(0, rows))
    # )

    # avg_entr_y_com = np.mean(approx_entropy)
    # print("Average entropy for Y CoM =", avg_entr_y_com)

    # entropy_windows = np.arange(0, rows)

    # plot(
    #     entropy_windows,
    #     approx_entropy,
    #     "Window",
    #     "Approximate Entropy",
    #     "Moving approximate entropy of CoM Y",
    #     None,
    #     [0, 0.4],
    # )

    x_cop_df_resample = []
    y_cop_df_resample = []
    for i in range(0, n_com):
        x_cop_df_resample.append(x_cop_standard[i * 10])
        y_cop_df_resample.append(y_cop_standard[i * 10])

    cross_x_cop_com = np.correlate(x_cop_df_resample, x_com_standard, mode="full")
    cross_y_cop_com = np.correlate(y_cop_df_resample, y_com_standard, mode="full")

    plot(
        t_corr_com,
        cross_x_cop_com,
        "time (s)",
        "cross correlation",
        "X CoM and X CoP cross correlation",
        [0, 30],
        None,
    )

    plot(
        t_corr_com,
        cross_y_cop_com,
        "time (s)",
        "cross correlation",
        "Y CoM and Y CoP cross correlation",
        [0, 30],
        None,
    )
