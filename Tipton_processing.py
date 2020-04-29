# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: February 22, 2020
#
# Description: Signal Processing of COM and COP
#   during stability testing in neuronormative
#   subjects for thesis research.
#
# Last updated: April 14, 2020

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
fs = 1200
t_tot = np.arange(0, 30, 1 / fs)
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


#####################################################################################

if __name__ == "__main__":

    # cx, cy = read_data_onefp(filepath, 3, ["Cx", "Cy"], 36000, [4])
    cx, cy = read_data_twofp(
        filepath, 3, ["Fz", "Cx", "Cy", "Fz.1", "Cx.1", "Cy.1"], 36000, [4]
    )

    n_data = len(cx)

    t_corr = np.arange(-n_data / fs, n_data / fs - 1 / fs, 1 / fs)

    cx_standard = standardize(cx)
    cy_standard = standardize(cy)

    plot(t_tot, cx, "time (s)", "CoP", "Raw Cx signal", None, None)

    auto = np.correlate(cx_standard, cx_standard, mode="full")
    cross = np.correlate(cx_standard, cy_standard, mode="full")

    plot(
        t_corr,
        auto,
        "time (s)",
        "autocorrelation",
        "Autocorrelation of Cx",
        [0, 30],
        None,
    )

    plot(
        t_corr,
        cross,
        "time (s)",
        "cross correlation",
        "Cx and Cy cross correlation",
        None,
        None,
    )

    n_auto = len(auto)
    freq = np.arange(0, fs, fs / n_auto)

    # create windows
    # rect_win = np.ones(n_auto)
    # ham_win = np.hamming(n_auto)

    # calculate PSD with both windows
    Sxx = np.fft.fft(auto)
    # Sxx_ham = np.fft.fft(np.multiply(auto, ham_win))
    plot(freq, abs(Sxx), "frequency (Hz)", "autopower", "Cx Autopower", [0, 1], None)

    Sxy = np.fft.fft(cross)
    # Sxy_ham = np.fft.fft(np.multiply(cross, ham_win))

    plot(
        freq,
        abs(Sxy),
        "frequency (Hz)",
        "cross power",
        "Cross Power between Cx and Cy",
        [0, 0.25],
        None,
    )

    cx_array = np.asarray(cx)
    cx_overlap = skimage.util.view_as_windows(cx_array, 1800, step=900)

    print(cx_overlap.shape)
    rows = cx_overlap.shape[0]

    approx_entropy = []

    def collect_approx_entropy(x):
        approx_entropy.append(x)

    # for row in tqdm(range(0, rows)):
    Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
        delayed(collect_approx_entropy)(ApEn(cx_overlap[row], 2, 10))
        for row in tqdm(range(0, rows))
    )

    entropy_windows = np.arange(0, rows)

    plot(
        entropy_windows,
        approx_entropy,
        "Window",
        "Approximate Entropy",
        "Moving approximate entropy of Cx",
        None,
        None,
    )

    plot(t_tot, cy, "time (s)", "CoP", "Raw Cy signal", None, None)

    auto = np.correlate(cy_standard, cy_standard, mode="full")

    plot(
        t_corr,
        auto,
        "time (s)",
        "autocorrelation",
        "Autocorrelation of Cy",
        [0, 30],
        None,
    )

    n_auto = len(auto)
    freq = np.arange(0, fs, fs / n_auto)

    # create windows
    # rect_win = np.ones(n_auto)
    # ham_win = np.hamming(n_auto)

    # calculate PSD with both windows
    Syy = np.fft.fft(auto)
    # Sxx_ham = np.fft.fft(np.multiply(auto, ham_win))
    plot(freq, abs(Syy), "frequency (Hz)", "autopower", "Cy Autopower", [0, 1], None)

    cy_array = np.asarray(cy)
    cy_overlap = skimage.util.view_as_windows(cy_array, 1800, step=900)

    print(cy_overlap.shape)
    rows = cy_overlap.shape[0]

    approx_entropy = []

    # for row in tqdm(range(0, rows)):
    Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
        delayed(collect_approx_entropy)(ApEn(cy_overlap[row], 2, 10))
        for row in tqdm(range(0, rows))
    )

    entropy_windows = np.arange(0, rows)

    plot(
        entropy_windows,
        approx_entropy,
        "Window",
        "Approximate Entropy",
        "Moving approximate entropy of Cy",
        None,
        None,
    )
