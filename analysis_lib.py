# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: May 6, 2020
#
# Description: Library of functions for use in
#   code relating to thesis research
#
# Last updated: July 29, 2020

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm

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
# Inputs:
#   x = independent variable
#   y = dependent variable
def deriv(x, y):
    der = []
    for i in range(0, len(y) - 1):
        der.append((y[i + 1] - y[i]) / (x[i + 1] - x[i]))

    return der


# adapted from https://en.wikipedia.org/wiki/Approximate_entropy
# returns: the approximate entropy of a data set
# User inputs:
#   U = data set
#   m = length of compared runs
#   r = filter value
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
# Inputs:
#   ent_list = entrpoy value
#   x = list to append values to
def collect_approx_entropy(ent_list, x):
    ent_list.append(x)


# prepares data for moving approximate entropy calculation
# Inputs:
#   x = data
#   win_size = size of windows for moving calculation
#   overlap = amount of overlap in windows
#   mult = gain increase for data to exacerbate small differences
#   name = data descriptor for plot laleling
def approx_ent(x, win_size, overlap, mult, name) -> list:
    # increase magnitude of signal for more sensitivity in entropy
    x_multiplied = np.multiply(x, mult)

    # create overlapping windows
    x_array = np.asarray(x_multiplied)
    x_overlap = skimage.util.view_as_windows(x_array, win_size, step=overlap)
    rows = x_overlap.shape[0]

    approx_entropy = []

    # calculate moving approximate entropy in multiple threads to speed up process
    Parallel(n_jobs=multiprocessing.cpu_count(), require="sharedmem")(
        delayed(collect_approx_entropy)(approx_entropy, (ApEn(x_overlap[row], 2, 10)))
        for row in tqdm(range(0, rows))
    )

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


# function to obtain filter coefficients for butterworth, 4th order, 6 hz lowpass
# Inputs:
#   cutoff = cutoff frequency
#   fs = sampling frequency of data to be filtered
#   order = order of filter
# Outputs:
#   a,b = filter coefficients
def butter_lowpass(cutoff, fs, order=4):
    normal_cutoff = float(cutoff) / (fs / 2)
    b, a = signal.butter(order, normal_cutoff, btype="lowpass", analog=False)
    return b, a


# filters data using filter coefficients from butter_lowpass
# Inputs:
#   data = data to be filtered
#   cutoff_freq = lowpass cutoff frequency
#   fs = sampling frequency of data passed
#   order = order of filter
# Output:
#   y = filtered data
def butter_lowpass_filter(data, cutoff_freq, fs, order=4):
    b, a = butter_lowpass(cutoff_freq, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


# https://github.com/NathanMaton/prediction_intervals/blob/master/sklearn_prediction_interval_extension.ipynb
# returns a prediction interval for a linear regression prediction
# Inputs:
#   prediction = single prediction
#   y_test = test data that is being used for the regression
#   test_predictions = entire list of regression predictions
#   pi = confidence intervale
# Outputs:
#   lower = a single value of the lower bound
#   upper = a single value of the upper bound
def get_prediction_interval(prediction, y_test, test_predictions, pi=0.95):
    # get standard deviation of y_test
    sum_errs = np.sum((y_test - test_predictions) ** 2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)

    # get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = norm.ppf(ppf_lookup)
    interval = z_score * stdev

    # generate prediction interval lower and upper bound
    lower, upper = prediction - interval, prediction + interval

    return lower, upper


# calculates the regression and plots the curve with original
# signal and 95% prediction intervals
# Inputs:
#   X = x-axis data for regression
#   Y = y-axis data for regression
#   subplt = which subplot to plot for plotting both directions
#   dir = direction (AP/ML) of current data
# Outputs:
#   subplot of regression analysis
def poly(X, Y, subplt, dir):
    # determine polynomial peatures
    polynomial_features = PolynomialFeatures(degree=9)
    X_polynomial = polynomial_features.fit_transform(X)

    # create regression model and fit to the polynomial features
    model = LinearRegression()
    model.fit(X_polynomial, Y)

    # create a regression curve based on model
    y_polynomial_predictions = model.predict(X_polynomial)

    # calculte r squared value
    r2 = r2_score(Y, y_polynomial_predictions)

    # create upper and lower intervals
    X_upper = []
    X_lower = []

    # calculate upper and lower intervals
    for prediction in y_polynomial_predictions:
        lower, upper = get_prediction_interval(prediction, Y, y_polynomial_predictions)
        X_upper.append(upper)
        X_lower.append(lower)

    # create a label to show the r squared value on plot
    label = r"$R^{2}$ = " + str(round(r2, 3))

    # plot the original signal, regression curve, and prediction intervals
    plt.subplot(subplt)
    plt.plot(X, Y)
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, y_polynomial_predictions), key=sort_axis)
    x, y_polynomial_predictions = zip(*sorted_zip)
    plt.plot(x, y_polynomial_predictions, color="m")
    plt.plot(x, X_upper, color="r")
    plt.plot(x, X_lower, color="r")
    plt.xlabel("Time (s)")
    plt.ylabel("COP Approximation (mm from starting location)")
    plt.title(f"Regression model for all trials in the {dir} direction\n{label}")
    plt.legend(["Average Signal", "Model", "95% prediction interval"])
    plt.ylim([-10, 6])
