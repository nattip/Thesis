# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: April 28, 2020
#
# Description: Data Mining script to determine
#   how many files are missing CoM data
#
# Last updated: April 28, 2020

import pandas as pd
import os
import pickle

ROOT = "/Users/natalietipton/Code/Data"

if __name__ == "__main__":
    folders = [x[0] for x in os.walk(ROOT)]
    folders.remove(ROOT)
    files = [
        (os.path.join(ROOT, folder), os.listdir(os.path.join(ROOT, folder)))
        for folder in folders
    ]

    dirs = {}
    for folder_name, file_list in files:
        dirs[folder_name] = file_list

    for directory in dirs:
        if directory == "/Users/natalietipton/Code/Data/SB05":
            continue
        for file in dirs[directory]:
            print(os.path.join(directory, file))
            sheet = pd.read_csv(
                os.path.join(directory, file),
                engine="python",
                header=36008,
                usecols=[2],
                nrows=3600,
                skiprows=[36010],
            )

            if "X" not in sheet.columns:
                sheet.columns = ["X"]
                sheet.drop([0], inplace=True)
            print(sheet)
            counter = len(sheet[sheet.X == ""])
            print(counter)
