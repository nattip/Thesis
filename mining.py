# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: April 28, 2020
#
# Description: Data Mining script to determine
#   how many files are missing CoM data
#
# Last updated: April 29, 2020

import pandas as pd
import os
import pickle

ROOT = "/Users/natalietipton/Code/Data"

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

    # loop through all files in all subdirectories and read in
    # first column of CoM data to determine how many missing cells
    for directory in dirs:
        for file in dirs[directory]:
            # skip the files known to have no CoM data, as they
            # will not read in correctly
            if (
                file == "SB05_Trial11.csv"
                or file == "SB05_Trial13.csv"
                or file == "SB05_Trial14.csv"
                or file == "SB05_Trial15.csv"
                or file == "SB05_Trial16.csv"
                or file == "SB05_Trial24.csv"
                or file == "SB05_Trial28.csv"
                or file == "SB05_Trial29.csv"
                or file == "SB05_Trial03.csv"
            ):
                continue

            # read first column of CoM data
            sheet = pd.read_csv(
                os.path.join(directory, file),
                engine="python",
                header=36008,
                usecols=[2],
                nrows=3600,
                skiprows=[36010],
            )

            # count empty cells in CoM data
            nulls = sheet.isna().sum().sum()
            # print number of empty cells if > 0
            if nulls > 0:
                print(file, "missing:", nulls)
