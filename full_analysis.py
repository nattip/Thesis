# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: May 9, 2020
#
# Description: Signal Processing of COP
#   during stability testing in neuronormative
#   subjects for thesis research.
#
# Last updated: May 13, 2020

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

ROOT = "/Users/natalietipton/Code/center_of_pressure/data"

# constants
fs_cop = 1200
t_cop = np.arange(0, 30, 1 / fs_cop)

# vel_x_avgs = []
# pow_x_tot = []
# vel_y_avgs = []
# pow_y_tot = []


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

    # loop through all files in all subdirectories
    for directory in dirs:
        print(directory)
        if int(directory[-1]) == 4:
            sub4 = True
        else:
            sub4 = False

        vel_x_avgs_together = []
        pow_x_tot_together = []
        vel_y_avgs_together = []
        pow_y_tot_together = []

        vel_x_avgs_tandem = []
        pow_x_tot_tandem = []
        vel_y_avgs_tandem = []
        pow_y_tot_tandem = []

        for file in dirs[directory]:
            number = int(file[10:12])

            # if number > 11:
            #     continue

            # if number > 6:
            #     continue
            # if number < 12:
            #     continue

            # if number < 27:
            #     continue

            if number < 12:
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                if sub4 and number < 17:
                    x_cop = an.butter_lowpass_filter(x_cop, 6, fs_cop)
                    y_cop = an.butter_lowpass_filter(y_cop, 6, fs_cop)

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_together.append(
                    np.mean(sorted(vel_x_cop, reverse=True)[:10])
                )

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_together.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_together.append(
                    np.mean(sorted(vel_y_cop, reverse=True)[:10])
                )

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_together.append(sum(abs(Syy_vel_cop)))

            elif number > 11:
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                if sub4 and number < 17:
                    x_cop = an.butter_lowpass_filter(x_cop, 6, fs_cop)
                    y_cop = an.butter_lowpass_filter(y_cop, 6, fs_cop)

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_tandem.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_tandem.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_tandem.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_tandem.append(sum(abs(Syy_vel_cop)))

            #         df = pd.read_csv(os.path.join(directory, file), index_col=False)

            #         df = df.to_numpy()
            #         values = np.delete(df, 0, 1)
            #         x_cop, y_cop = values.T

            #         vel_x_cop = an.deriv(t_cop, x_cop)
            #         vel_x_avgs.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

            #         auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
            #         Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
            #         pow_x_tot.append(sum(abs(Sxx_vel_cop)))

            #         vel_y_cop = an.deriv(t_cop, y_cop)
            #         vel_y_avgs.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

            #         auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
            #         Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
            #         pow_y_tot.append(sum(abs(Syy_vel_cop)))
        delta_vel_x = np.mean(vel_x_avgs_tandem) - np.mean(vel_x_avgs_together)
        delta_vel_y = np.mean(vel_y_avgs_tandem) - np.mean(vel_y_avgs_together)
        delta_pow_x = np.mean(pow_x_tot_tandem) - np.mean(pow_x_tot_together)
        delta_pow_y = np.mean(pow_y_tot_tandem) - np.mean(pow_y_tot_together)

        mult_vel_x = np.mean(vel_x_avgs_tandem) / np.mean(vel_x_avgs_together)
        mult_vel_y = np.mean(vel_y_avgs_tandem) / np.mean(vel_y_avgs_together)
        mult_pow_x = np.mean(pow_x_tot_tandem) / np.mean(pow_x_tot_together)
        mult_pow_y = np.mean(pow_y_tot_tandem) / np.mean(pow_y_tot_together)

        # print(f"Average Velocity Delta X CoP= {format(round(delta_vel_x,2), 'e')}")
        print(f"Multiple Velocity Delta X CoP = {format(round(mult_vel_x,2), 'e')}")
        # print(f"Average Velocity Delta Y CoP = {format(round(delta_vel_y,2), 'e')}")
        print(f"Multiple Velocity Delta Y CoP = {format(round(mult_vel_y,2), 'e')}")
        # print(f"Average Power Delta X CoP = {format(round(delta_pow_x,2), 'e')}")
        print(f"Multiple Power Delta X CoP = {format(round(mult_pow_x,2), 'e')}")
        # print(f"Average Power Delta Y CoP = {format(round(delta_pow_y,2), 'e')}")
        print(f"Multiple Power Delta Y CoP = {format(round(mult_pow_y,2), 'e')}")

        # print(f"Average Velocity X CoP = {format(round(np.mean(vel_x_avgs_together),2), 'e')}")
        # print(f"Average total power X CoP = {format(round(np.mean(pow_x_tot_together),2), 'e')}")
        # print(f"Average Velocity Y CoP = {format(round(np.mean(vel_y_avgs_together),2), 'e')}")
        # print(f"Average total power Y CoP = {format(round(np.mean(pow_y_tot_together),2), 'e')}")
