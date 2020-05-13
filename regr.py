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


ROOT = "/Users/natalietipton/Code/center_of_pressure/data"
t_cop = np.arange(0, 30, 1 / 1200)
# t_cop = t_cop[:, np.newaxis]

regression_x_together = pd.DataFrame()
regression_y_together = pd.DataFrame()

all_x_data = []
all_y_data = []


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

    # bound = 1 - r2 / 2
    # X_upper = []
    # X_lower = []
    # for yy in y_polynomial_predictions:
    #     for x in yy:
    #         X_upper.append(x + x * bound)
    #         X_lower.append(x - x * bound)
    # X_upper = np.array(X_upper)
    # X_lower = np.array(X_lower)

    plt.plot(X, Y)
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, y_polynomial_predictions), key=sort_axis)
    x, y_polynomial_predictions = zip(*sorted_zip)
    plt.plot(x, y_polynomial_predictions, color="m")
    # plt.plot(x, X_upper, color="r")
    # plt.plot(x, X_lower, color="g")
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
        for file in dirs[directory]:
            number = int(file[10:12])

            if number > 26:
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                Xaxis, Yaxis = values.T

                Xaxis = an.standardize(Xaxis)
                Yaxis = an.standardize(Yaxis)
                # Xaxis = Xaxis[:, np.newaxis]
                # Yaxis = Yaxis[:, np.newaxis]

                all_x_data.append(Xaxis)
                all_y_data.append(Yaxis)

    all_x_data = np.array(all_x_data)
    all_y_data = np.array(all_y_data)

    avg_x_data = np.mean(all_x_data, axis=0)
    avg_y_data = np.mean(all_y_data, axis=0)
    # print(avg_x_data)
    an.plot(
        t_cop,
        avg_x_data,
        "time",
        "signal",
        "Average of all Feet Tandem, EC, DB, X Axis",
        None,
        None,
    )
    an.plot(
        t_cop,
        avg_y_data,
        "time",
        "signal",
        "Average of all Feet Tandem, EC, DB, Y Axis",
        None,
        None,
    )
    #         regression_x = poly(t_cop, Xaxis)
    #         regression_y = poly(t_cop, Yaxis)

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

#     df = pd.read_csv(f"{ROOT}/SB02/SB02_Trial02_norm_spliced.csv", index_col=False)

#     df = df.to_numpy()
#     values = np.delete(df, 0, 1)
#     Xaxis, Yaxis = values.T
#     Xaxis = Xaxis[:, np.newaxis]
#     Yaxis = Yaxis[:, np.newaxis]

#     poly(t_cop, Xaxis)
#     poly(t_cop, Yaxis)
