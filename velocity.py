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

        vel_x_avgs_eotog = []
        pow_x_tot_eotog = []
        vel_y_avgs_eotog = []
        pow_y_tot_eotog = []

        vel_x_avgs_ectog = []
        pow_x_tot_ectog = []
        vel_y_avgs_ectog = []
        pow_y_tot_ectog = []

        vel_x_avgs_eodftan = []
        pow_x_tot_eodftan = []
        vel_y_avgs_eodftan = []
        pow_y_tot_eodftan = []

        vel_x_avgs_ecdftan = []
        pow_x_tot_ecdftan = []
        vel_y_avgs_ecdftan = []
        pow_y_tot_ecdftan = []

        vel_x_avgs_eodbtan = []
        pow_x_tot_eodbtan = []
        vel_y_avgs_eodbtan = []
        pow_y_tot_eodbtan = []

        vel_x_avgs_ecdbtan = []
        pow_x_tot_ecdbtan = []
        vel_y_avgs_ecdbtan = []
        pow_y_tot_ecdbtan = []

        for file in dirs[directory]:
            number = int(file[10:12])

            if number < 7:
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                if sub4:
                    x_cop = an.butter_lowpass_filter(x_cop, 6, fs_cop)
                    y_cop = an.butter_lowpass_filter(y_cop, 6, fs_cop)

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_eotog.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_eotog.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_eotog.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_eotog.append(sum(abs(Syy_vel_cop)))

            elif 6 < number < 12:
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                if sub4:
                    x_cop = an.butter_lowpass_filter(x_cop, 6, fs_cop)
                    y_cop = an.butter_lowpass_filter(y_cop, 6, fs_cop)

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_ectog.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_ectog.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_ectog.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_ectog.append(sum(abs(Syy_vel_cop)))

            elif 11 < number < 17:
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                if sub4:
                    x_cop = an.butter_lowpass_filter(x_cop, 6, fs_cop)
                    y_cop = an.butter_lowpass_filter(y_cop, 6, fs_cop)

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_eodbtan.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_eodbtan.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_eodbtan.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_eodbtan.append(sum(abs(Syy_vel_cop)))

            elif 16 < number < 22:
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_ecdbtan.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_ecdbtan.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_ecdbtan.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_ecdbtan.append(sum(abs(Syy_vel_cop)))

            elif 21 < number < 27:
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_eodftan.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_eodftan.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_eodftan.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_eodftan.append(sum(abs(Syy_vel_cop)))

            elif 26 < number < 32:
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_ecdftan.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_ecdftan.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_ecdftan.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_ecdftan.append(sum(abs(Syy_vel_cop)))

        plt.style.use("ggplot")

        x = ["EOFT", "ECFT", "EODF", "EODB", "ECDF", "ECDB"]
        vels = [
            np.mean(vel_x_avgs_eotog),
            np.mean(vel_x_avgs_ectog),
            np.mean(vel_x_avgs_eodftan),
            np.mean(vel_x_avgs_eodbtan),
            np.mean(vel_x_avgs_ecdftan),
            np.mean(vel_x_avgs_ecdbtan),
        ]

        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, vels, color="green")
        plt.xlabel("Balance Condition")
        plt.ylabel("Average Velocity (mm/s)")
        plt.title(
            f"Average Velocity of Each Balance Condition for Subject {(directory[-1])}"
        )

        plt.xticks(x_pos, x)

        plt.show()
