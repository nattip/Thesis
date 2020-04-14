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

#   import packages
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
filepath = "/Users/natalietipton/Code/Data/SB01/SB01Trial_07.csv"

# reads data in from .csv files
# user specifies:
#   header = which row the data starts on (starting at 0, not 1 like the sheet)
#   usecols = names of the column headers that are to be included
#   nrows = number of data points to be read in
#   rows_skip = the number of any rows to not be included (starting at 0)
def read_data(filepath, row_start, cols, num_data, rows_skip):
    data = pd.read_csv(
        filepath,
        header=row_start,
        usecols=cols,
        nrows=num_data,
        dtype={"Cx": np.float64, "Cy": np.float64, "Cz": np.float64},
        skiprows=rows_skip,
    )

    # convert data frame into lists
    cx = data["Cx"].values.tolist()
    cy = data["Cy"].values.tolist()
    cz = data["Cz"].values.tolist()

    return cx, cy, cz


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
    # print(U, m, r)

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        # print(x, C)
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))


if __name__ == "__main__":

    cx, cy, cz = read_data(filepath, 3, ["Cx", "Cy", "Cz"], 36000, [4])
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
