# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: May 6, 2020
#
# Description: Signal Processing of COM and COP
#   consisting of approximate entropy calcs
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
import analysis_lib as an
import os
import xlwt
from xlwt import Workbook

# import ray

# ray.init()

# constants
fs_cop = 1200
t_cop = np.arange(0, 30, 1 / fs_cop)
fs_com = 120
t_com = np.arange(0, 30, 1 / fs_com)
filepath = "/Users/natalietipton/Code/Data/SB01/SB01Trial_02.csv"
# ROOT = f"{os.environ.get('HOME')}/Code/center_of_pressure/data"

avg_ent_x_together = []
avg_ent_y_together = []
avg_ent_x_tandem = []
avg_ent_y_tandem = []

#####################################################################################

ROOT = "/Users/natalietipton/Code/center_of_pressure/data"

# constants
fs_cop = 1200
t_cop = np.arange(0, 30, 1 / fs_cop)
trial = -1
wb = Workbook()

# add_sheet is used to create sheet.
ent_results_x = wb.add_sheet("ent_results_x")
ent_results_y = wb.add_sheet("ent_results_y")


#####################################################################################

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

    ent_x_dict = {}
    ent_y_dict = {}

    # loop through all files in all subdirectories
    for directory in dirs:
        trial = trial + 1
        print(directory)
        if int(directory[-1]) == 4:
            sub4 = True
        else:
            sub4 = False

        # loop through each file in the directory
        for file in dirs[directory]:
            # Determine which trial number it is
            number = int(file[10:12])

            # read data
            df = pd.read_csv(os.path.join(directory, file), index_col=False)

            # convert data from dataframe into a numpy list
            df = df.to_numpy()
            values = np.delete(df, 0, 1)
            x_cop, y_cop = values.T

            # filter data if it is a subject 4 trial that was not pre-filtered
            if sub4 and number < 17:
                x_cop = an.butter_lowpass_filter(x_cop, 6, fs_cop)
                y_cop = an.butter_lowpass_filter(y_cop, 6, fs_cop)

            ap_ent_x = an.approx_ent(x_cop, 1800, 900, 100, "X CoP")
            ap_ent_y = an.approx_ent(y_cop, 1800, 900, 100, "Y CoP")

            ent_x_dict[f"{file}_x_ap_ent"] = ap_ent_x
            ent_y_dict[f"{file}_y_ap_ent"] = ap_ent_y

        ent_x_df = pd.DataFrame(ent_x_dict, columns=ent_x_dict.keys())
        ent_y_df = pd.DataFrame(ent_y_dict, columns=ent_y_dict.keys())

    ent_x_df.to_csv("x.csv")
    ent_y_df.to_csv("y.csv")


# def to_iter(obj):
#     while obj:
#         done, obj = ray.wait(obj)
#         yield ray.get(done[0])


# # @ray.remote
# def process_directory(directory):
#     for file in dirs[directory]:
#         number = int(file[10:12])
#         if number < 7:
#             df = pd.read_csv(os.path.join(directory, file), index_col=False)
#             df = df.to_numpy()
#             values = np.delete(df, 0, 1)
#             x_cop, y_cop = values.T

#             avg_ent_x_together.append(an.approx_ent(x_cop, 1800, 900, 100, "X CoP"))
#             avg_ent_y_together.append(an.approx_ent(y_cop, 1800, 900, 100, "Y CoP"))

#         elif number > 26:
#             df = pd.read_csv(os.path.join(directory, file), index_col=False)
#             df = df.to_numpy()
#             values = np.delete(df, 0, 1)
#             x_cop, y_cop = values.T

#             avg_ent_x_tandem.append(an.approx_ent(x_cop, 1800, 900, 100, "X CoP"))
#             avg_ent_y_tandem.append(an.approx_ent(y_cop, 1800, 900, 100, "Y CoP"))

#     delta_ent_x = np.mean(avg_ent_x_tandem) - np.mean(avg_ent_x_together)
#     delta_ent_y = np.mean(avg_ent_y_tandem) - np.mean(avg_ent_y_together)
#     mult_ent_x = np.mean(avg_ent_x_tandem) / np.mean(avg_ent_x_together)
#     mult_ent_y = np.mean(avg_ent_y_tandem) / np.mean(avg_ent_y_together)

#     print(
#         f"Entropy Delta X CoP = {format(round(delta_ent_x,2), 'e')}",
#         f"Entropy Delta Y CoP = {format(round(delta_ent_y,2), 'e')}",
#         f"Entropy Multiple X CoP = {format(round(mult_ent_x,2), 'e')}"
#         f"Entropy Multiple Y CoP = {format(round(mult_ent_y,2), 'e')}",
#     )

#     with open(f"{directory}_averages", "w+") as f:
#         f.writelines(
#             [
#                 f"Entropy Delta X CoP = {format(round(delta_ent_x,2), 'e')}",
#                 f"Entropy Delta Y CoP = {format(round(delta_ent_y,2), 'e')}",
#                 f"Entropy Multiple X CoP = {format(round(mult_ent_x,2), 'e')}"
#                 f"Entropy Multiple Y CoP = {format(round(mult_ent_y,2), 'e')}",
#             ]
#         )


# if __name__ == "__main__":

#     # # function call for one force plate files
#     x_cop, y_cop, x_cop_df = an.read_data_onefp(filepath, 3, ["Cx", "Cy"], 36000, [4])

#     # # function call for two force plate files
#     # # x_cop, y_cop = read_data_twofp(
#     # #     filepath, 3, ["Fz", "Cx", "Cy", "Fz.1", "Cx.1", "Cy.1"], 36000, [4]
#     # # )

#     # # function call for reading CoM data from both 1 and 2 force plate files
#     # x_com, y_com, z_com = an.read_data_com(
#     #     filepath, 36009, ["X", "Y", "Z"], 3600, [36010]
#     # )

#     # determing all subdirectories in root directory
#     # folders = [x[0] for x in os.walk(ROOT)]
#     # # remove the root directory from folders
#     # folders.remove(ROOT)
#     # # find all file names in the subdirectories
#     # files = [
#     #     (os.path.join(ROOT, folder), os.listdir(os.path.join(ROOT, folder)))
#     #     for folder in folders
#     # ]

#     # # create a dictionary showing what filenames are within each folder
#     # dirs = {}
#     # for folder_name, file_list in files:
#     #     # remove irrelevant file names
#     #     if ".DS_Store" in file_list:
#     #         file_list.remove(".DS_Store")
#     #     dirs[folder_name] = file_list

#     # # loop through all files in all subdirectories
#     # futures = [process_directory.remote(directory) for directory in dirs]
#     # # for directory in tqdm(dirs):
#     # #     process_directory(directory)
#     # for x in tqdm(to_iter(futures), total=len(futures)):
#     #     pass
#     ent_x = an.approx_ent(x_cop, 1800, 900, 100, "X CoP")
#     ent_y = an.approx_ent(y_cop, 1800, 900, 100, "Y CoP")

#     print(ent_x)
#     print(ent_y)
