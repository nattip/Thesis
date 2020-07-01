# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: June 10, 2020
#
# Description: Processing of the velocity
#   of COP signals.
#
# Last updated: June 29, 2020

# import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import analysis_lib as an
import os
import statistics
from scipy.stats import ttest_ind, ttest_ind_from_stats

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

    # vel_x_avgs_eotog = []
    # pow_x_tot_eotog = []
    # vel_y_avgs_eotog = []
    # pow_y_tot_eotog = []

    # vel_x_avgs_ectog = []
    # pow_x_tot_ectog = []
    # vel_y_avgs_ectog = []
    # pow_y_tot_ectog = []

    # vel_x_avgs_eodftan = []
    # pow_x_tot_eodftan = []
    # vel_y_avgs_eodftan = []
    # pow_y_tot_eodftan = []

    # vel_x_avgs_ecdftan = []
    # pow_x_tot_ecdftan = []
    # vel_y_avgs_ecdftan = []
    # pow_y_tot_ecdftan = []

    # vel_x_avgs_eodbtan = []
    # pow_x_tot_eodbtan = []
    # vel_y_avgs_eodbtan = []
    # pow_y_tot_eodbtan = []

    # vel_x_avgs_ecdbtan = []
    # pow_x_tot_ecdbtan = []
    # vel_y_avgs_ecdbtan = []
    # pow_y_tot_ecdbtan = []

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

        # create lists for vel and PSD of all trial groups
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

        # loop through each file in the directory
        for file in dirs[directory]:
            # Determine which trial number it is
            number = int(file[10:12])

            # if an EOFT trial
            if number < 7:
                # read data
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                # convert data from dataframe into a numpy list
                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                # filter data if it is a subject 4 trial that was not pre-filtered
                if sub4:
                    x_cop = an.butter_lowpass_filter(x_cop, 6, fs_cop)
                    y_cop = an.butter_lowpass_filter(y_cop, 6, fs_cop)

                # get the derivative of the x direction COP data
                vel_x_cop = an.deriv(t_cop, x_cop)

                # take the average of the first 10 data points from x velocity
                # (too time expensive to average all points, and this is accurate)
                vel_x_avgs_eotog.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                # calculate autocorrelation and then PSD of velocity of x COP
                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)

                # sum up all of the x cop frequency content
                pow_x_tot_eotog.append(sum(abs(Sxx_vel_cop)))

                # get the derivative of the y direction COP data
                vel_y_cop = an.deriv(t_cop, y_cop)

                # take the average of the first 10 data points from x velocity
                vel_y_avgs_eotog.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                # calculate autocorrelation and the PSD of velocity of y COP
                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)

                # sum up all of the y cop frequency content
                pow_y_tot_eotog.append(sum(abs(Syy_vel_cop)))

            # if an ECFT trial
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

            # if an EOFTanDB trial
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

            # if an ECFTanDB trial
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

            # if an EOFTanDF trial
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

            #  if an ECFTanDF trial
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

        # p_values_vel_x = []
        # p_values_vel_y = []
        # p_values_pow_x = []
        # p_values_pow_y = []

        # print(vel_x_avgs_eotog)

        # t_stat, p_val = ttest_ind(vel_x_avgs_eotog, vel_x_avgs_ectog, equal_var=False)
        # p_values_vel_x.append(p_val / 2)
        # t_stat, p_val = ttest_ind(vel_x_avgs_eotog, vel_x_avgs_eodftan, equal_var=False)
        # p_values_vel_x.append(p_val / 2)
        # t_stat, p_val = ttest_ind(vel_x_avgs_eotog, vel_x_avgs_eodbtan, equal_var=False)
        # p_values_vel_x.append(p_val / 2)
        # t_stat, p_val = ttest_ind(vel_x_avgs_eotog, vel_x_avgs_ecdftan, equal_var=False)
        # p_values_vel_x.append(p_val / 2)
        # t_stat, p_val = ttest_ind(vel_x_avgs_eotog, vel_x_avgs_ecdbtan, equal_var=False)
        # p_values_vel_x.append(p_val / 2)

        # t_stat, p_val = ttest_ind(pow_x_tot_eotog, pow_x_tot_ectog, equal_var=False)
        # p_values_pow_x.append(p_val / 2)
        # t_stat, p_val = ttest_ind(pow_x_tot_eotog, pow_x_tot_eodftan, equal_var=False)
        # p_values_pow_x.append(p_val / 2)
        # t_stat, p_val = ttest_ind(pow_x_tot_eotog, pow_x_tot_eodbtan, equal_var=False)
        # p_values_pow_x.append(p_val / 2)
        # t_stat, p_val = ttest_ind(pow_x_tot_eotog, pow_x_tot_ecdftan, equal_var=False)
        # p_values_pow_x.append(p_val / 2)
        # t_stat, p_val = ttest_ind(pow_x_tot_eotog, pow_x_tot_ecdbtan, equal_var=False)
        # p_values_pow_x.append(p_val / 2)

        # t_stat, p_val = ttest_ind(vel_y_avgs_eotog, vel_y_avgs_ectog, equal_var=False)
        # p_values_vel_y.append(p_val / 2)
        # t_stat, p_val = ttest_ind(vel_y_avgs_eotog, vel_y_avgs_eodftan, equal_var=False)
        # p_values_vel_y.append(p_val / 2)
        # t_stat, p_val = ttest_ind(vel_y_avgs_eotog, vel_y_avgs_eodbtan, equal_var=False)
        # p_values_vel_y.append(p_val / 2)
        # t_stat, p_val = ttest_ind(vel_y_avgs_eotog, vel_y_avgs_ecdftan, equal_var=False)
        # p_values_vel_y.append(p_val / 2)
        # t_stat, p_val = ttest_ind(vel_y_avgs_eotog, vel_y_avgs_ecdbtan, equal_var=False)
        # p_values_vel_y.append(p_val / 2)

        # t_stat, p_val = ttest_ind(pow_y_tot_eotog, pow_y_tot_ectog, equal_var=False)
        # p_values_pow_y.append(p_val / 2)
        # t_stat, p_val = ttest_ind(pow_y_tot_eotog, pow_y_tot_eodftan, equal_var=False)
        # p_values_pow_y.append(p_val / 2)
        # t_stat, p_val = ttest_ind(pow_y_tot_eotog, pow_y_tot_eodbtan, equal_var=False)
        # p_values_pow_y.append(p_val / 2)
        # t_stat, p_val = ttest_ind(pow_y_tot_eotog, pow_y_tot_ecdftan, equal_var=False)
        # p_values_pow_y.append(p_val / 2)
        # t_stat, p_val = ttest_ind(pow_y_tot_eotog, pow_y_tot_ecdbtan, equal_var=False)
        # p_values_pow_y.append(p_val / 2)

        # plot 4 subplots of the average velocity and frequency content
        # in the X and Y directions for each subject and all conditions
        plt.figure()

        # style of plots
        plt.style.use("ggplot")
        plt.subplot(141)

        # x axis labels
        x = ["EOFT", "ECFT", "EODF", "EODB", "ECDF", "ECDB"]

        # y axis data
        vels = [
            np.mean(vel_x_avgs_eotog),
            np.mean(vel_x_avgs_ectog),
            np.mean(vel_x_avgs_eodftan),
            np.mean(vel_x_avgs_eodbtan),
            np.mean(vel_x_avgs_ecdftan),
            np.mean(vel_x_avgs_ecdbtan),
        ]

        # creates an x-axis position for each stability condition
        x_pos = [i for i, _ in enumerate(x)]

        # create bars plot with different colors for each
        plt.bar(x_pos, vels, color="rgbkm")

        # label x axis
        plt.xlabel(
            "Balance Condition", fontdict={"fontsize": 9},
        )

        # label y axis
        plt.ylabel(
            "Average Velocity (mm/s)", fontdict={"fontsize": 9},
        )

        # title plot
        plt.title(
            f"Average Velocity of Each Balance Condition\nin X for Subject {(directory[-1])}",
            fontdict={"fontsize": 9},
        )

        # create x-axis ticks
        plt.xticks(x_pos, x)

        # plt.show()
        # plt.subplot(4,1,2)
        # plt.style.use("ggplot")

        # x = ["ECFT", "EODF", "EODB", "ECDF", "ECDB"]

        # x_pos = [i for i, _ in enumerate(x)]

        # plt.bar(x_pos, p_values_vel_x, color="blue")
        # plt.xlabel("Balance Condition")
        # plt.ylabel("p-value")
        # plt.title(f"P-values for X velocity for subject {(directory[-1])}")

        # plt.xticks(x_pos, x)
        # plt.ylim(0, 0.06)

        # plt.show()
        plt.subplot(142)
        plt.style.use("ggplot")

        x = ["EOFT", "ECFT", "EODF", "EODB", "ECDF", "ECDB"]
        vels = [
            np.mean(vel_y_avgs_eotog),
            np.mean(vel_y_avgs_ectog),
            np.mean(vel_y_avgs_eodftan),
            np.mean(vel_y_avgs_eodbtan),
            np.mean(vel_y_avgs_ecdftan),
            np.mean(vel_y_avgs_ecdbtan),
        ]

        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, vels, color="rgbkm")

        plt.xlabel(
            "Balance Condition", fontdict={"fontsize": 9},
        )
        plt.ylabel(
            "Average Velocity (mm/s)", fontdict={"fontsize": 9},
        )
        plt.title(
            f"Average Velocity of Each Balance Condition\nin Y for Subject {(directory[-1])}",
            fontdict={"fontsize": 9},
        )

        plt.xticks(x_pos, x)

        plt.subplot(143)
        plt.style.use("ggplot")

        x = ["EOFT", "ECFT", "EODF", "EODB", "ECDF", "ECDB"]
        vels = [
            np.mean(pow_x_tot_eotog),
            np.mean(pow_x_tot_ectog),
            np.mean(pow_x_tot_eodftan),
            np.mean(pow_x_tot_eodbtan),
            np.mean(pow_x_tot_ecdftan),
            np.mean(pow_x_tot_ecdbtan),
        ]

        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, vels, color="rgbkm")

        plt.xlabel(
            "Balance Condition", fontdict={"fontsize": 9},
        )
        plt.ylabel(
            "Average Velocity (mm/s)", fontdict={"fontsize": 9},
        )
        plt.title(
            f"Average Total Power of Each Balance Condition\nin X for Subject {(directory[-1])}",
            fontdict={"fontsize": 9},
        )

        plt.xticks(x_pos, x)

        # plt.show()
        # plt.subplot(4,1,1)
        # x = ["ECFT", "EODF", "EODB", "ECDF", "ECDB"]

        # x_pos = [i for i, _ in enumerate(x)]

        # plt.bar(x_pos, p_values_pow_x, color="blue")
        # plt.xlabel("Balance Condition")
        # plt.ylabel("p-value")
        # plt.title(f"P-values for X Power for subject {(directory[-1])}")

        # plt.xticks(x_pos, x)
        # plt.ylim(0, 0.06)

        # plt.show()

        # plt.show()

        # x = ["ECFT", "EODF", "EODB", "ECDF", "ECDB"]

        # x_pos = [i for i, _ in enumerate(x)]

        # plt.bar(x_pos, p_values_vel_y, color="blue")
        # plt.xlabel("Balance Condition")
        # plt.ylabel("p-value")
        # plt.title(f"P-values for Y velocity for subject {(directory[-1])}")

        # plt.xticks(x_pos, x)
        # plt.ylim(0, 0.06)

        # plt.show()
        plt.subplot(144)
        plt.style.use("ggplot")

        x = ["EOFT", "ECFT", "EODF", "EODB", "ECDF", "ECDB"]
        vels = [
            np.mean(pow_y_tot_eotog),
            np.mean(pow_y_tot_ectog),
            np.mean(pow_y_tot_eodftan),
            np.mean(pow_y_tot_eodbtan),
            np.mean(pow_y_tot_ecdftan),
            np.mean(pow_y_tot_ecdbtan),
        ]

        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, vels, color="rgbkm")
        plt.xlabel(
            "Balance Condition", fontdict={"fontsize": 9},
        )
        plt.ylabel(
            "Average Velocity (mm/s)", fontdict={"fontsize": 9},
        )
        plt.title(
            f"Average Total Power of Each Balance Condition\nin Y for Subject {(directory[-1])}",
            fontdict={"fontsize": 9},
        )

        plt.xticks(x_pos, x)

        # show all four subplots
        plt.show()

        # x = ["ECFT", "EODF", "EODB", "ECDF", "ECDB"]

        # x_pos = [i for i, _ in enumerate(x)]

        # plt.bar(x_pos, p_values_pow_y, color="blue")
        # plt.xlabel("Balance Condition")
        # plt.ylabel("p-value")
        # plt.title(f"P-values for Y power for subject {(directory[-1])}")

        # plt.xticks(x_pos, x)
        # plt.ylim(0, 0.06)

        # plt.show()

        #########################

