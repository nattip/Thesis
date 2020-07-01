# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: May 6, 2020
#
# Description: Library of functions for use in
#   code relating to thesis research
#
# Last updated: May 13, 2020

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn import preprocessing
import skimage
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from scipy import signal

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


# calculates the point by point derivative between two lists
def deriv(x, y):
    der = []
    for i in range(0, len(y) - 1):
        der.append((y[i + 1] - y[i]) / (x[i + 1] - x[i]))

    return der


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


# collects approximate entroy into a list from each window
def collect_approx_entropy(ent_list, x):
    ent_list.append(x)


def approx_ent(x, win_size, overlap, mult, name) -> list:
    # increase magnitude of signal for more sensitivity in entropy
    x_multiplied = np.multiply(x, mult)

    # create overlapping windows
    x_array = np.asarray(x_multiplied)
    x_overlap = skimage.util.view_as_windows(x_array, win_size, step=overlap)
    rows = x_overlap.shape[0]

    approx_entropy = []

    # calculate moving approximate entropy
    Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
        delayed(collect_approx_entropy)(approx_entropy, (ApEn(x_overlap[row], 2, 10)))
        # delayed(collect_approx_entropy)(approx_entropy, (ApEn(x_overlap[row], 2, 10)))
        for row in tqdm(range(0, rows))
    )

    variance = []
    for row in range(0, rows):
        variance.append(np.var(x_overlap[row]))

    # find average approximate entropy
    # avg_entr = np.mean(approx_entropy)
    # print(f"Average entropy for {name} = {avg_entr}")

    # entropy_windows = np.arange(0, rows)

    # plot(
    #     entropy_windows,
    #     approx_entropy,
    #     "Window",
    #     "Approximate Entropy",
    #     f"Moving approximate entropy of {name}",
    #     None,
    #     [0, 0.4],
    # )

    # plot(
    #     entropy_windows,
    #     variance,
    #     "Window",
    #     "Variance",
    #     f"Moving Variance of {name}",
    #     None,
    #     None,
    # )

    return approx_entropy


def butter_lowpass(cutoff, fs, order=4):
    normal_cutoff = float(cutoff) / (fs / 2)
    b, a = signal.butter(order, normal_cutoff, btype="lowpass", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff_freq, fs, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
