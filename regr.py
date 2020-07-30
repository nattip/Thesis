# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: May 6, 2020
#
# Description: Performs least squares polynomial
#   regression on data to obtain a baseline model
#
# Last updated: July 29, 2020

#####################################################################################

# import packages
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import analysis_lib as an

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm

# define root folder for data
ROOT = "/Users/natalietipton/Code/center_of_pressure/data"

# constants
fs_cop = 1200
t_cop = np.arange(0, 30, 1 / fs_cop)

# create dataframes for regression to
# regression_x_together = pd.DataFrame()
# regression_y_together = pd.DataFrame()

# create lists for all data to go into
all_x_data = []
all_y_data = []

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


#####################################################################################


if __name__ == "__main__":
    # determine all subdirectories in root directory
    folders = [x[0] for x in os.walk(ROOT)]
    # remove the root directory from folders
    folders.remove(ROOT)
    # find all file names in the subdirectories
    files = [
        (os.path.join(ROOT, folder), os.listdir(os.path.join(ROOT, folder)))
        for folder in folders
    ]

    # create a dictionary showing what filenames are within each folder
    dirs = {}
    for folder_name, file_list in files:
        # remove irrelevant file names
        if ".DS_Store" in file_list:
            file_list.remove(".DS_Store")
        dirs[folder_name] = file_list

    # loop through all files in all subdirectories
    for directory in dirs:
        # determine if it is subject 4 data that needs filtering
        if int(directory[-1]) == 4:
            sub4 = True
        else:
            sub4 = False

        for file in dirs[directory]:
            # determine trial number
            number = int(file[10:12])

            # change if statement to create groupings of different stability conditions
            # number < 7 = EOFT, 6 < number < 12 = ECFT, 11 < number < 17 = EOFTanDB
            # 16 < number < 22 = ECFTanDB, 21 < number < 27 = EOFTanDF,
            # 26 < number < 32 = ECFTanDF
            if number < 32:
                # read COP data
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                # convert to np array and obtain x and y axis data
                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                Xaxis, Yaxis = values.T

                # filter if subject 4 and trial less than 17
                if sub4 and number < 17:
                    Xaxis = an.butter_lowpass_filter(Xaxis, 6, fs_cop)
                    Yaxis = an.butter_lowpass_filter(Yaxis, 6, fs_cop)

                # turn data into delta values, showing deviation from starting position
                Xaxis = np.subtract(Xaxis, Xaxis[0])
                Yaxis = np.subtract(Yaxis, Yaxis[0])

                # save data from all files in condition range to a list
                all_x_data.append(Xaxis)
                all_y_data.append(Yaxis)

    # turn list into array
    all_x_data = np.array(all_x_data)
    all_y_data = np.array(all_y_data)

    # find average signal from all data in condition range
    avg_x_data = np.mean(all_x_data, axis=0)
    avg_y_data = np.mean(all_y_data, axis=0)

    avg_x_data = avg_x_data[:, np.newaxis]
    avg_y_data = avg_y_data[:, np.newaxis]
    t_cop = t_cop[:, np.newaxis]

    # calculate regression models for AP and ML directions and plot
    plt.figure()
    poly(t_cop, avg_x_data, 121, "AP")
    poly(t_cop, avg_y_data, 122, "ML")
    plt.show()
