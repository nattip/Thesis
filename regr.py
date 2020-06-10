# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: May 6, 2020
#
# Description: Performs least squares linear
#   regression on data to obtain a baseline model
#
# Last updated: May 13, 2020

import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import analysis_lib as an

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm


ROOT = "/Users/natalietipton/Code/center_of_pressure/data"
fs_cop = 1200
t_cop = np.arange(0, 30, 1 / fs_cop)

# t_cop = t_cop[:, np.newaxis]

regression_x_together = pd.DataFrame()
regression_y_together = pd.DataFrame()

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


def poly(X, Y):
    polynomial_features = PolynomialFeatures(degree=9)
    X_polynomial = polynomial_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_polynomial, Y)
    print(model.coef_)

    y_polynomial_predictions = model.predict(X_polynomial)

    rmse = np.sqrt(mean_squared_error(Y, y_polynomial_predictions))
    r2 = r2_score(Y, y_polynomial_predictions)
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

    X_upper = []
    X_lower = []

    for prediction in y_polynomial_predictions:
        lower, upper = get_prediction_interval(prediction, Y, y_polynomial_predictions)
        X_upper.append(upper)
        X_lower.append(lower)

    plt.plot(X, Y)
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, y_polynomial_predictions), key=sort_axis)
    x, y_polynomial_predictions = zip(*sorted_zip)
    plt.plot(x, y_polynomial_predictions, color="m")
    plt.plot(x, X_upper, color="r")
    plt.plot(x, X_lower, color="g")
    plt.show()

    # return y_polynomial_predictions


if __name__ == "__main__":
    # determing all subdirectories in root directory
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
        print(directory)
        if int(directory[-1]) == 4:
            sub4 = True
        else:
            sub4 = False

        for file in dirs[directory]:
            number = int(file[10:12])

            if number < 7:
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                Xaxis, Yaxis = values.T

                if sub4 and number < 17:
                    Xaxis = an.butter_lowpass_filter(Xaxis, 6, fs_cop)
                    Yaxis = an.butter_lowpass_filter(Yaxis, 6, fs_cop)

                # Xaxis = an.standardize(Xaxis)
                # Yaxis = an.standardize(Yaxis)

                all_x_data.append(Xaxis)
                all_y_data.append(Yaxis)

    all_x_data = np.array(all_x_data)
    all_y_data = np.array(all_y_data)

    avg_x_data = np.mean(all_x_data, axis=0)
    avg_y_data = np.mean(all_y_data, axis=0)

    an.plot(
        t_cop,
        avg_x_data,
        "time",
        "signal",
        "Average of all Feet Together X Axis",
        None,
        None,
    )
    an.plot(
        t_cop,
        avg_y_data,
        "time",
        "signal",
        "Average of all Feet Together Y Axis",
        None,
        None,
    )

    avg_x_data = avg_x_data[:, np.newaxis]
    avg_y_data = avg_y_data[:, np.newaxis]
    t_cop = t_cop[:, np.newaxis]

    poly(t_cop, avg_x_data)
    poly(t_cop, avg_y_data)
    #         # print(regression_x)

    #         regression_x = pd.DataFrame(regression_x.toarray())
    #         regression_y = pd.DataFrame(regression_y.toarray())

    #         regression_x_together = regression_x_together.DataFrame.append(
    #             regression_x
    #         )
    #         regression_y_together = regression_y_together.DataFrame.append(
    #             regression_y
    #         )

    # print(regression_x_together)
    ################################
    # def poly(X, Y):
    #     polynomial_features = PolynomialFeatures(degree=9)
    #     X_polynomial = polynomial_features.fit_transform(X)

    #     model = LinearRegression()
    #     model.fit(X_polynomial, Y)

    #     y_polynomial_predictions = model.predict(X_polynomial)

    #     rmse = np.sqrt(mean_squared_error(Y, y_polynomial_predictions))
    #     r2 = r2_score(Y, y_polynomial_predictions)
    #     print(f"RMSE: {rmse}")
    #     print(f"R2: {r2}")

    #     plt.plot(X, Y)
    #     sort_axis = operator.itemgetter(0)
    #     sorted_zip = sorted(zip(X, y_polynomial_predictions), key=sort_axis)
    #     x, y_polynomial_predictions = zip(*sorted_zip)
    #     plt.plot(x, y_polynomial_predictions, color="m")
    #     # plt.plot(x, X_upper, color="r")
    #     # plt.plot(x, X_lower, color="g")
    #     plt.show()
    # if __name__ == "__main__":

    # df = pd.read_csv(f"{ROOT}/SB02/SB02_Trial02_norm_spliced.csv", index_col=False)

    # df = df.to_numpy()
    # values = np.delete(df, 0, 1)
    # Xaxis, Yaxis = values.T
    # Xaxis = Xaxis[:, np.newaxis]
    # Yaxis = Yaxis[:, np.newaxis]
    # t_cop = t_cop[:, np.newaxis]

    # poly(t_cop, Xaxis)
    # poly(t_cop, Yaxis)
