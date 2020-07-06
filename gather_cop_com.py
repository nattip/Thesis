# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: July 5, 2020
#
# Description: Write 1 COP and COM signal
#   from each stability condition for each
#   subject to a .csv file
#
# Last updated: July 5, 2020

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
import analysis_lib as an
import os

# constants
fs_com = 120
fs_cop = 1200
t_com = np.arange(0, 30, 1 / fs_com)
t_cop = np.arange(0, 30, 1 / fs_cop)

cop_root = f"{os.environ.get('HOME')}/Code/center_of_pressure/data"
com_root = f"{os.environ.get('HOME')}/Code/Data"


#####################################################################################

if __name__ == "__main__":

    # determing all subdirectories in cop_root directory
    folders_cop = [x[0] for x in os.walk(cop_root)]
    # remove the root directory from folders
    folders_cop.remove(cop_root)
    # find all file names in the subdirectories
    files_cop = [
        (os.path.join(cop_root, folder), os.listdir(os.path.join(cop_root, folder)))
        for folder in folders_cop
    ]

    # create a dictionary showing what filenames are within each folder
    dirs_cop = {}
    for folder_name, file_list in files_cop:
        # remove irrelevant file names
        if ".DS_Store" in file_list:
            file_list.remove(".DS_Store")
        dirs_cop[folder_name] = file_list

    # determing all subdirectories in cop_root directory
    folders_com = [x[0] for x in os.walk(com_root)]
    # remove the root directory from folders
    folders_com.remove(com_root)
    # find all file names in the subdirectories
    files_com = [
        (os.path.join(com_root, folder), os.listdir(os.path.join(com_root, folder)))
        for folder in folders_com
    ]

    # create a dictionary showing what filenames are within each folder
    dirs_com = {}
    for folder_name, file_list in files_com:
        # remove irrelevant file names
        if ".DS_Store" in file_list:
            file_list.remove(".DS_Store")
        dirs_com[folder_name] = file_list

    cop_x_dict = {}
    cop_y_dict = {}

    # loop through all files in all subdirectories
    for directory in dirs_cop:
        print(directory)
        if int(directory[-1]) == 4:
            sub4 = True
        else:
            sub4 = False

        for file in dirs_cop[directory]:
            number = int(file[10:12])
            subject = int(file[2:4])

            if number not in [2, 7, 12, 17, 22, 31]:
                continue

            df = pd.read_csv(os.path.join(directory, file), index_col=False)

            # convert data from dataframe into a numpy list
            df = df.to_numpy()
            values = np.delete(df, 0, 1)
            x_cop, y_cop = values.T

            # filter data if it is a subject 4 trial that was not pre-filtered
            if sub4:
                x_cop = an.butter_lowpass_filter(x_cop, 6, fs_cop)
                y_cop = an.butter_lowpass_filter(y_cop, 6, fs_cop)

            x_cop_standard = an.standardize(x_cop)
            y_cop_standard = an.standardize(y_cop)

            # save ApEn results into a dictionary under the key of the file name
            cop_x_dict[f"SB0{subject}_Trial{number}"] = x_cop_standard
            cop_y_dict[f"SB0{subject}_Trial{number}"] = y_cop_standard

        # turn completed dictionaries into pandas dataframe
        cop_x_df = pd.DataFrame(cop_x_dict, columns=cop_x_dict.keys())
        cop_y_df = pd.DataFrame(cop_y_dict, columns=cop_y_dict.keys())

    # save both dataframes in .csv files
    cop_x_df.to_csv("x_cop.csv")
    cop_y_df.to_csv("y_cop.csv")

    com_x_dict = {}
    com_y_dict = {}

    for directory in dirs_com:
        print(directory)
        if int(directory[-1]) != 1:
            continue
        for file in dirs_com[directory]:
            number = int(file[10:12])
            subject = int(file[2:4])

            if number not in [2, 7, 12, 17, 22, 31]:
                continue

            print(os.path.join(directory, file))

            x_com, y_com, z_com = an.read_data_com(
                os.path.join(directory, file), 36009, ["X", "Y", "Z"], 3600, [36010]
            )

            # standardize the signals
            x_com_standard = an.standardize(x_com)
            y_com_standard = an.standardize(y_com)

            # save ApEn results into a dictionary under the key of the file name
            com_x_dict[f"SB0{subject}_Trial{number}"] = x_com_standard
            com_y_dict[f"SB0{subject}_Trial{number}"] = y_com_standard

        # turn completed dictionaries into pandas dataframe
        com_x_df = pd.DataFrame(com_x_dict, columns=com_x_dict.keys())
        com_y_df = pd.DataFrame(com_y_dict, columns=com_y_dict.keys())

    # save both dataframes in .csv files
    com_x_df.to_csv("x_1_com.csv")
    com_y_df.to_csv("y_1_com.csv")
