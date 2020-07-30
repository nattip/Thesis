# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: May 6, 2020
#
# Description: Compute approximate entropy
#   for all files in x and y and save them
#   in one .csv file for all x and another
#   for all y
#
# Last updated: July 29, 2020

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import analysis_lib as an
import os

#####################################################################################

# define root folder for data
ROOT = f"{os.environ.get('HOME')}/Code/center_of_pressure/data"

# constants
fs_cop = 1200
t_cop = np.arange(0, 30, 1 / fs_cop)

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

    # create lists to hold ApEn results for every file
    ent_x_dict = {}
    ent_y_dict = {}

    # loop through all files in all subdirectories
    for directory in dirs:
        print(directory)
        # determine if it is the directory for subjet 4 for filtering
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

            # convert data from dataframe into a numpy list and obtain x and y signals
            df = df.to_numpy()
            values = np.delete(df, 0, 1)
            x_cop, y_cop = values.T

            # filter data if it is a subject 4 trial that was not pre-filtered
            if sub4 and number < 17:
                x_cop = an.butter_lowpass_filter(x_cop, 6, fs_cop)
                y_cop = an.butter_lowpass_filter(y_cop, 6, fs_cop)

            # calculate moving ApEn for x and y COP signals
            ap_ent_x = an.approx_ent(x_cop, 1800, 900, 100, "X CoP")
            ap_ent_y = an.approx_ent(y_cop, 1800, 900, 100, "Y CoP")

            # save ApEn results into a dictionary under the key of the file name
            ent_x_dict[f"{file}_x_ap_ent"] = ap_ent_x
            ent_y_dict[f"{file}_y_ap_ent"] = ap_ent_y

        # turn completed dictionaries into pandas dataframe
        ent_x_df = pd.DataFrame(ent_x_dict, columns=ent_x_dict.keys())
        ent_y_df = pd.DataFrame(ent_y_dict, columns=ent_y_dict.keys())

    # save both dataframes in .csv files
    ent_x_df.to_csv("x.csv")
    ent_y_df.to_csv("y.csv")
