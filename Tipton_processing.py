# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: February 22, 2020g
#
# Description: Signal Processing of COM and COP
#   during stability testing in neuronormative
#   subjects for thesis research.
#
# Last updated: April 28, 2020

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

# reads data in from .csv files for data with feet together, one force plate
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

    return cx, cy


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


# reads data in from .csv files for feet tandem, two force plates
# combines CoP positions from both force plates into 1 resultant CoP
# returns: cx = CoP x position (AP), cy = CoP y position (ML)
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
    plt.show()


def ApEn(U, m, r) -> float:
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]

        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))


def collect_approx_entropy(x):
    approx_entropy.append(x)


#####################################################################################

if __name__ == "__main__":

    # x_cop, y_cop = read_data_onefp(filepath, 3, ["Cx", "Cy"], 36000, [4])
    x_com, y_com, z_com = read_data_com(filepath, 36009, ["X", "Y", "Z"], 3600, [36010])
    x_cop, y_cop = read_data_twofp(
        filepath, 3, ["Fz", "Cx", "Cy", "Fz.1", "Cx.1", "Cy.1"], 36000, [4]
    )

    n_cop = len(x_cop)
    n_com = len(x_com)

    t_corr_cop = np.arange(-n_cop / fs_cop, n_cop / fs_cop - 1 / fs_cop, 1 / fs_cop)
    t_corr_com = np.arange(-n_com / fs_com, n_com / fs_com - 1 / fs_com, 1 / fs_com)

    x_cop_standard = standardize(x_cop)
    y_cop_standard = standardize(y_cop)
    x_com_standard = standardize(x_com)
    y_com_standard = standardize(y_com)
    z_com_standard = standardize(z_com)

    #################### CoP X-axis analysis ####################

    plot(t_cop, x_cop, "time (s)", "CoP", "Raw Cx signal", None, None)

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

    n_auto_cop = len(auto_x_cop)
    freq_cop = np.arange(0, fs_cop, fs_cop / n_auto_cop)

    # create windows
    # rect_win = np.ones(n_auto)
    # ham_win = np.hamming(n_auto)

    # calculate PSD with both windows
    Sxx_cop = np.fft.fft(auto_x_cop)
    # Sxx_ham = np.fft.fft(np.multiply(auto, ham_win))
    plot(
        freq_cop,
        abs(Sxx_cop),
        "Frequency (Hz)",
        "autopower",
        "Cx Autopower",
        [0, 1],
        None,
    )

    Sxy_cop = np.fft.fft(cross_xy_cop)
    # Sxy_ham = np.fft.fft(np.multiply(cross, ham_win))

    plot(
        freq_cop,
        abs(Sxy_cop),
        "frequency (Hz)",
        "cross power",
        "Cross Power between Cx and Cy",
        [0, 0.25],
        None,
    )

    # Entropy Stuff
    # x_cop_array = np.asarray(x_cop)
    # x_cop_overlap = skimage.util.view_as_windows(x_cop_array, 1800, step=900)

    # rows = x_cop_overlap.shape[0]

    # approx_entropy = []

    # # for row in tqdm(range(0, rows)):
    # Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
    #     delayed(collect_approx_entropy)(ApEn(x_cop_overlap[row], 2, 10))
    #     for row in tqdm(range(0, rows))
    # )

    # entropy_windows = np.arange(0, rows)

    # plot(
    #     entropy_windows,
    #     approx_entropy,
    #     "Window",
    #     "Approximate Entropy",
    #     "Moving approximate entropy of Cx",
    #     None,
    #     None,
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

    # create windows
    # rect_win = np.ones(n_auto)
    # ham_win = np.hamming(n_auto)

    # calculate PSD with both windows
    Syy_cop = np.fft.fft(auto_y_cop)
    # Sxx_ham = np.fft.fft(np.multiply(auto, ham_win))
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
    # y_cop_array = np.asarray(y_cop)
    # y_cop_overlap = skimage.util.view_as_windows(y_cop_array, 1800, step=900)

    # print(y_cop_overlap.shape)
    # rows = y_cop_overlap.shape[0]

    # approx_entropy = []

    # # for row in tqdm(range(0, rows)):
    # Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
    #     delayed(collect_approx_entropy)(ApEn(y_cop_overlap[row], 2, 10))
    #     for row in tqdm(range(0, rows))
    # )

    # entropy_windows = np.arange(0, rows)

    # plot(
    #     entropy_windows,
    #     approx_entropy,
    #     "Window",
    #     "Approximate Entropy",
    #     "Moving approximate entropy of Cy",
    #     None,
    #     None,
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

    # create windows
    # rect_win = np.ones(n_auto)
    # ham_win = np.hamming(n_auto)

    # calculate PSD with both windows
    Sxx_com = np.fft.fft(auto_x_com)
    # Sxx_ham = np.fft.fft(np.multiply(auto, ham_win))
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
    # Sxy_ham = np.fft.fft(np.multiply(cross, ham_win))

    plot(
        freq_com,
        abs(Sxy_com),
        "frequency (Hz)",
        "cross power",
        "Cross Power between X and Y CoM",
        [0, 0.25],
        None,
    )

    x_com_array = np.asarray(x_com)
    x_com_overlap = skimage.util.view_as_windows(x_com_array, 180, step=90)

    rows = x_com_overlap.shape[0]

    approx_entropy = []

    # for row in tqdm(range(0, rows)):
    Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
        delayed(collect_approx_entropy)(ApEn(x_com_overlap[row], 2, 10))
        for row in tqdm(range(0, rows))
    )

    entropy_windows = np.arange(0, rows)

    plot(
        entropy_windows,
        approx_entropy,
        "Window",
        "Approximate Entropy",
        "Moving approximate entropy of CoM X",
        None,
        None,
    )

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

    # create windows
    # rect_win = np.ones(n_auto)
    # ham_win = np.hamming(n_auto)

    # calculate PSD with both windows
    Syy_com = np.fft.fft(auto_y_com)
    # Sxx_ham = np.fft.fft(np.multiply(auto, ham_win))
    plot(
        freq_com,
        abs(Syy_com),
        "frequency (Hz)",
        "autopower",
        "CoM Y Autopower",
        [0, 1],
        None,
    )

    y_com_array = np.asarray(y_com)
    y_com_overlap = skimage.util.view_as_windows(y_com_array, 180, step=90)

    rows = y_com_overlap.shape[0]

    approx_entropy = []

    # for row in tqdm(range(0, rows)):
    Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
        delayed(collect_approx_entropy)(ApEn(y_com_overlap[row], 2, 10))
        for row in tqdm(range(0, rows))
    )

    entropy_windows = np.arange(0, rows)

    plot(
        entropy_windows,
        approx_entropy,
        "Window",
        "Approximate Entropy",
        "Moving approximate entropy of CoM Y",
        None,
        None,
    )

    # WONT WORK BECAUSE COP AND COM ARE DIFFERENT SIZES
    # cross_x_cop_com = np.correlate(x_cop_standard, x_com_standard, mode="full")

    # plot(
    #     t_corr_com,
    #     cross_x_cop_com,
    #     "time (s)",
    #     "autocorrelation",
    #     "Autocorrelation of x in CoM",
    #     [0, 30],
    #     None,
    # )

    # cross_y_cop_com = np.correlate(y_cop_standard, y_com_standard, mode="full")

    # plot(
    #     t_corr_com,
    #     cross_y_cop_com,
    #     "time (s)",
    #     "autocorrelation",
    #     "Autocorrelation of x in CoM",
    #     [0, 30],
    #     None,
    # )
