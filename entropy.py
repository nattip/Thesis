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
import analysis_libV2 as an
import os
import ray

ray.init()

# constants
fs_cop = 1200
t_cop = np.arange(0, 30, 1 / fs_cop)
fs_com = 120
t_com = np.arange(0, 30, 1 / fs_com)
# filepath = "/Users/natalietipton/Code/Data/SB01/SB01Trial_02.csv"
ROOT = f"{os.environ.get('HOME')}/Code/center_of_pressure/data"

avg_ent_x_together = []
avg_ent_y_together = []
avg_ent_x_tandem = []
avg_ent_y_tandem = []

#####################################################################################
def to_iter(obj):
    while obj:
        done, obj = ray.wait(obj)
        yield ray.get(done[0])


@ray.remote
def process_directory(directory):
    for file in dirs[directory]:
        number = int(file[10:12])
        if number < 7:
            df = pd.read_csv(os.path.join(directory, file), index_col=False)
            df = df.to_numpy()
            values = np.delete(df, 0, 1)
            x_cop, y_cop = values.T

            avg_ent_x_together.append(an.approx_ent(x_cop, 1800, 900, 100, "X CoP"))
            avg_ent_y_together.append(an.approx_ent(y_cop, 1800, 900, 100, "Y CoP"))

        elif number > 26:
            df = pd.read_csv(os.path.join(directory, file), index_col=False)
            df = df.to_numpy()
            values = np.delete(df, 0, 1)
            x_cop, y_cop = values.T

            avg_ent_x_tandem.append(an.approx_ent(x_cop, 1800, 900, 100, "X CoP"))
            avg_ent_y_tandem.append(an.approx_ent(y_cop, 1800, 900, 100, "Y CoP"))

    delta_ent_x = np.mean(avg_ent_x_tandem) - np.mean(avg_ent_x_together)
    delta_ent_y = np.mean(avg_ent_y_tandem) - np.mean(avg_ent_y_together)
    mult_ent_x = np.mean(avg_ent_x_tandem) / np.mean(avg_ent_x_together)
    mult_ent_y = np.mean(avg_ent_y_tandem) / np.mean(avg_ent_y_together)

    print(
        f"Entropy Delta X CoP = {format(round(delta_ent_x,2), 'e')}",
        f"Entropy Delta Y CoP = {format(round(delta_ent_y,2), 'e')}",
        f"Entropy Multiple X CoP = {format(round(mult_ent_x,2), 'e')}"
        f"Entropy Multiple Y CoP = {format(round(mult_ent_y,2), 'e')}",
    )

    with open(f"{directory}_averages", "w+") as f:
        f.writelines(
            [
                f"Entropy Delta X CoP = {format(round(delta_ent_x,2), 'e')}",
                f"Entropy Delta Y CoP = {format(round(delta_ent_y,2), 'e')}",
                f"Entropy Multiple X CoP = {format(round(mult_ent_x,2), 'e')}"
                f"Entropy Multiple Y CoP = {format(round(mult_ent_y,2), 'e')}",
            ]
        )


if __name__ == "__main__":

    # # function call for one force plate files
    # x_cop, y_cop, x_cop_df = an.read_data_onefp(filepath, 3, ["Cx", "Cy"], 36000, [4])

    # # function call for two force plate files
    # # x_cop, y_cop = read_data_twofp(
    # #     filepath, 3, ["Fz", "Cx", "Cy", "Fz.1", "Cx.1", "Cy.1"], 36000, [4]
    # # )

    # # function call for reading CoM data from both 1 and 2 force plate files
    # x_com, y_com, z_com = an.read_data_com(
    #     filepath, 36009, ["X", "Y", "Z"], 3600, [36010]
    # )

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
    futures = [process_directory.remote(directory) for directory in dirs]
    # for directory in tqdm(dirs):
    #     process_directory(directory)
    for x in tqdm(to_iter(futures), total=len(futures)):
        pass
    # an.approx_ent(x_com, 180, 90, 100, "X CoM")
    # an.approx_ent(y_com, 180, 90, 100, "Y CoM")
