# velocity.py
#
# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: June 10, 2020
#
# Description: Processing of the velocity
#   of COP signals.
#
# Last updated: July 29, 2020

#####################################################################################

# import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import analysis_lib as an
import os
import statistics

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

    # loop through all files in all subdirectories
    for directory in dirs:
        print(directory)

        # create lists to store vel and psd for x and y data
        x_trial_vel = []
        y_trial_vel = []
        x_trial_pow = []
        y_trial_pow = []
        trial_cond = []

        # determine if directory is for subject 4
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
            # Determine which trial number and subject the file is
            number = int(file[10:12])
            subject = int(file[2:4])

            # change subject number to fit plot formatting
            if subject == 4:
                subject = 3
            elif subject == 5:
                subject = 4
            elif subject == 6:
                subject = 5
            elif subject == 8:
                subject = 6

            # if an EOFT trial
            if number < 7:
                trial_cond.append("EOFT")
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
                x_trial_vel.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                # calculate autocorrelation and then PSD of velocity of x COP
                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)

                # sum up all of the x cop frequency content
                pow_x_tot_eotog.append(sum(abs(Sxx_vel_cop)))
                x_trial_pow.append(sum(abs(Sxx_vel_cop)))

                # get the derivative of the y direction COP data
                vel_y_cop = an.deriv(t_cop, y_cop)

                # take the average of the first 10 data points from x velocity
                vel_y_avgs_eotog.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))
                y_trial_vel.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                # calculate autocorrelation and the PSD of velocity of y COP
                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)

                # sum up all of the y cop frequency content
                pow_y_tot_eotog.append(sum(abs(Syy_vel_cop)))
                y_trial_pow.append(sum(abs(Syy_vel_cop)))

            # if an ECFT trial: complete same steps as above
            elif 6 < number < 12:
                trial_cond.append("ECFT")
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                if sub4:
                    x_cop = an.butter_lowpass_filter(x_cop, 6, fs_cop)
                    y_cop = an.butter_lowpass_filter(y_cop, 6, fs_cop)

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_ectog.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))
                x_trial_vel.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_ectog.append(sum(abs(Sxx_vel_cop)))
                x_trial_pow.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_ectog.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))
                y_trial_vel.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_ectog.append(sum(abs(Syy_vel_cop)))
                y_trial_pow.append(sum(abs(Syy_vel_cop)))

            # if an EOFTanDB trial
            elif 11 < number < 17:
                trial_cond.append("EOFTanDB")
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                if sub4:
                    x_cop = an.butter_lowpass_filter(x_cop, 6, fs_cop)
                    y_cop = an.butter_lowpass_filter(y_cop, 6, fs_cop)

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_eodbtan.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))
                x_trial_vel.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_eodbtan.append(sum(abs(Sxx_vel_cop)))
                x_trial_pow.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_eodbtan.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))
                y_trial_vel.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_eodbtan.append(sum(abs(Syy_vel_cop)))
                y_trial_pow.append(sum(abs(Syy_vel_cop)))

            # if an ECFTanDB trial
            elif 16 < number < 22:
                trial_cond.append("ECFTanDB")
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_ecdbtan.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))
                x_trial_vel.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_ecdbtan.append(sum(abs(Sxx_vel_cop)))
                x_trial_pow.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_ecdbtan.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))
                y_trial_vel.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_ecdbtan.append(sum(abs(Syy_vel_cop)))
                y_trial_pow.append(sum(abs(Syy_vel_cop)))

            # if an EOFTanDF trial
            elif 21 < number < 27:
                trial_cond.append("EOFTanDF")
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_eodftan.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))
                x_trial_vel.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_eodftan.append(sum(abs(Sxx_vel_cop)))
                x_trial_pow.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_eodftan.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))
                y_trial_vel.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_eodftan.append(sum(abs(Syy_vel_cop)))
                y_trial_pow.append(sum(abs(Syy_vel_cop)))

            #  if an ECFTanDF trial
            elif 26 < number < 32:
                trial_cond.append("ECFTanDF")
                df = pd.read_csv(os.path.join(directory, file), index_col=False)

                df = df.to_numpy()
                values = np.delete(df, 0, 1)
                x_cop, y_cop = values.T

                vel_x_cop = an.deriv(t_cop, x_cop)
                vel_x_avgs_ecdftan.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))
                x_trial_vel.append(np.mean(sorted(vel_x_cop, reverse=True)[:10]))

                auto_vel_x_cop = np.correlate(vel_x_cop, vel_x_cop, mode="full")
                Sxx_vel_cop = np.fft.fft(auto_vel_x_cop)
                pow_x_tot_ecdftan.append(sum(abs(Sxx_vel_cop)))
                x_trial_pow.append(sum(abs(Sxx_vel_cop)))

                vel_y_cop = an.deriv(t_cop, y_cop)
                vel_y_avgs_ecdftan.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))
                y_trial_vel.append(np.mean(sorted(vel_y_cop, reverse=True)[:10]))

                auto_vel_y_cop = np.correlate(vel_y_cop, vel_y_cop, mode="full")
                Syy_vel_cop = np.fft.fft(auto_vel_y_cop)
                pow_y_tot_ecdftan.append(sum(abs(Syy_vel_cop)))
                y_trial_pow.append(sum(abs(Syy_vel_cop)))

        # save velocity data from all conditions in both directions to a .csv
        zipped = list(
            zip(x_trial_vel, y_trial_vel, x_trial_pow, y_trial_pow, trial_cond)
        )
        df_for_stats = pd.DataFrame(
            zipped, columns=["xVelocity", "yVelocity", "xPower", "yPower", "condition"]
        )
        df_for_stats.to_csv(f"sb{directory[-1]}_velocity_by_condition.csv")

        # plot 4 subplots of the average velocity and frequency content
        # in the X and Y directions for each subject and all conditions
        plt.figure()

        # style of plots
        plt.style.use("ggplot")
        plt.subplot(141)

        # x axis labels
        x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

        # y axis data
        vels = [
            np.mean(vel_x_avgs_eotog),
            np.mean(vel_x_avgs_ectog),
            np.mean(vel_x_avgs_eodbtan),
            np.mean(vel_x_avgs_eodftan),
            np.mean(vel_x_avgs_ecdbtan),
            np.mean(vel_x_avgs_ecdftan),
        ]

        # standard error bars
        error = [
            np.std(vel_x_avgs_eotog) / np.sqrt(len(vel_x_avgs_eotog)),
            np.std(vel_x_avgs_ectog) / np.sqrt(len(vel_x_avgs_ectog)),
            np.std(vel_x_avgs_eodbtan) / np.sqrt(len(vel_x_avgs_eodbtan)),
            np.std(vel_x_avgs_eodftan) / np.sqrt(len(vel_x_avgs_eodftan)),
            np.std(vel_x_avgs_ecdbtan) / np.sqrt(len(vel_x_avgs_ecdbtan)),
            np.std(vel_x_avgs_ecdftan) / np.sqrt(len(vel_x_avgs_ecdftan)),
        ]

        # creates an x-axis position for each stability condition
        x_pos = [i for i, _ in enumerate(x)]

        # create bars plot with different colors for each
        plt.bar(
            x_pos, vels, yerr=error, capsize=3, color="grbymc",
        )

        # label x axis
        plt.xlabel(
            "Balance Condition", fontdict={"fontsize": 11},
        )

        # label y axis
        plt.ylabel(
            "Average Velocity (mm/s)", fontdict={"fontsize": 11},
        )
        # title plot
        plt.title(
            f"Average Velocity of Each Balance Condition\nin AP for Subject {subject}",
            fontdict={"fontsize": 11},
        )

        # create x-axis ticks
        plt.xticks(x_pos, x)

        plt.subplot(142)
        plt.style.use("ggplot")

        x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]
        vels = [
            np.mean(vel_y_avgs_eotog),
            np.mean(vel_y_avgs_ectog),
            np.mean(vel_y_avgs_eodbtan),
            np.mean(vel_y_avgs_eodftan),
            np.mean(vel_y_avgs_ecdbtan),
            np.mean(vel_y_avgs_ecdftan),
        ]

        # standard error bars
        error = [
            np.std(vel_y_avgs_eotog) / np.sqrt(len(vel_y_avgs_eotog)),
            np.std(vel_y_avgs_ectog) / np.sqrt(len(vel_y_avgs_ectog)),
            np.std(vel_y_avgs_eodbtan) / np.sqrt(len(vel_y_avgs_eodbtan)),
            np.std(vel_y_avgs_eodftan) / np.sqrt(len(vel_y_avgs_eodftan)),
            np.std(vel_y_avgs_ecdbtan) / np.sqrt(len(vel_y_avgs_ecdbtan)),
            np.std(vel_y_avgs_ecdftan) / np.sqrt(len(vel_y_avgs_ecdftan)),
        ]

        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(
            x_pos, vels, yerr=error, capsize=3, color="grbymc",
        )

        plt.xlabel(
            "Balance Condition", fontdict={"fontsize": 11},
        )
        plt.ylabel(
            "Average Velocity (mm/s)", fontdict={"fontsize": 11},
        )
        plt.title(
            f"Average Velocity of Each Balance Condition\nin ML for Subject {subject}",
            fontdict={"fontsize": 11},
        )

        plt.xticks(x_pos, x)

        plt.subplot(143)
        plt.style.use("ggplot")

        x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]
        vels = [
            np.mean(pow_x_tot_eotog),
            np.mean(pow_x_tot_ectog),
            np.mean(pow_x_tot_eodbtan),
            np.mean(pow_x_tot_eodftan),
            np.mean(pow_x_tot_ecdbtan),
            np.mean(pow_x_tot_ecdftan),
        ]

        # standard error bars
        error = [
            np.std(pow_x_tot_eotog) / np.sqrt(len(pow_x_tot_eotog)),
            np.std(pow_x_tot_ectog) / np.sqrt(len(pow_x_tot_ectog)),
            np.std(pow_x_tot_eodbtan) / np.sqrt(len(pow_x_tot_eodbtan)),
            np.std(pow_x_tot_eodftan) / np.sqrt(len(pow_x_tot_eodftan)),
            np.std(pow_x_tot_ecdbtan) / np.sqrt(len(pow_x_tot_ecdbtan)),
            np.std(pow_x_tot_ecdftan) / np.sqrt(len(pow_x_tot_ecdftan)),
        ]

        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, vels, yerr=error, capsize=3, color="grbymc")

        plt.xlabel(
            "Balance Condition", fontdict={"fontsize": 11},
        )
        plt.ylabel(
            "Magnitude", fontdict={"fontsize": 11},
        )
        plt.title(
            f"Average Total Power of Each Balance Condition\nin AP for Subject {subject}",
            fontdict={"fontsize": 11},
        )

        plt.xticks(x_pos, x)

        plt.subplot(144)
        plt.style.use("ggplot")

        x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]
        vels = [
            np.mean(pow_y_tot_eotog),
            np.mean(pow_y_tot_ectog),
            np.mean(pow_y_tot_eodbtan),
            np.mean(pow_y_tot_eodftan),
            np.mean(pow_y_tot_ecdbtan),
            np.mean(pow_y_tot_ecdftan),
        ]

        # standard error bars
        error = [
            np.std(pow_y_tot_eotog) / np.sqrt(len(pow_y_tot_eotog)),
            np.std(pow_y_tot_ectog) / np.sqrt(len(pow_y_tot_ectog)),
            np.std(pow_y_tot_eodbtan) / np.sqrt(len(pow_y_tot_eodbtan)),
            np.std(pow_y_tot_eodftan) / np.sqrt(len(pow_y_tot_eodftan)),
            np.std(pow_y_tot_ecdbtan) / np.sqrt(len(pow_y_tot_ecdbtan)),
            np.std(pow_y_tot_ecdftan) / np.sqrt(len(pow_y_tot_ecdftan)),
        ]

        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, vels, yerr=error, capsize=3, color="grbymc")
        plt.xlabel(
            "Balance Condition", fontdict={"fontsize": 11},
        )
        plt.ylabel(
            "Magnitude", fontdict={"fontsize": 11},
        )
        plt.title(
            f"Average Total Power of Each Balance Condition\nin ML for Subject {subject}",
            fontdict={"fontsize": 11},
        )

        plt.xticks(x_pos, x)

        # show all four subplots
        plt.show()

        #########################

        # plot 4 subplots of the average velocity and frequency content
        # in the X and Y directions for each subject and all conditions
        plt.figure()

        # style of plots
        plt.style.use("ggplot")
        plt.subplot(141)

        # x axis labels
        x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]

        # y axis data
        vels = [
            np.mean(vel_x_avgs_ectog) - np.mean(vel_x_avgs_eotog),
            np.mean(vel_x_avgs_eodbtan) - np.mean(vel_x_avgs_eotog),
            np.mean(vel_x_avgs_eodftan) - np.mean(vel_x_avgs_eotog),
            np.mean(vel_x_avgs_ecdbtan) - np.mean(vel_x_avgs_eotog),
            np.mean(vel_x_avgs_ecdftan) - np.mean(vel_x_avgs_eotog),
        ]

        # standard error bars
        error = [
            np.std(vel_x_avgs_ectog) / np.sqrt(len(vel_x_avgs_ectog)),
            np.std(vel_x_avgs_eodbtan) / np.sqrt(len(vel_x_avgs_eodbtan)),
            np.std(vel_x_avgs_eodftan) / np.sqrt(len(vel_x_avgs_eodftan)),
            np.std(vel_x_avgs_ecdbtan) / np.sqrt(len(vel_x_avgs_ecdbtan)),
            np.std(vel_x_avgs_ecdftan) / np.sqrt(len(vel_x_avgs_ecdftan)),
        ]

        # creates an x-axis position for each stability condition
        x_pos = [i for i, _ in enumerate(x)]

        # create bars plot with different colors for each
        plt.bar(
            x_pos, vels, yerr=error, capsize=3, color="rbymc",
        )

        # label x axis
        plt.xlabel(
            "Balance Condition", fontdict={"fontsize": 11},
        )

        # label y axis
        plt.ylabel(
            "Velocity Difference (mm/s)", fontdict={"fontsize": 11},
        )
        # title plot
        plt.title(
            f"Difference in velocity from EOFT condition\nin AP for Subject {subject}",
            fontdict={"fontsize": 11},
        )

        # create x-axis ticks
        plt.xticks(x_pos, x)

        plt.subplot(142)
        plt.style.use("ggplot")

        x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]
        vels = [
            np.mean(vel_y_avgs_ectog) - np.mean(vel_y_avgs_eotog),
            np.mean(vel_y_avgs_eodbtan) - np.mean(vel_y_avgs_eotog),
            np.mean(vel_y_avgs_eodftan) - np.mean(vel_y_avgs_eotog),
            np.mean(vel_y_avgs_ecdbtan) - np.mean(vel_y_avgs_eotog),
            np.mean(vel_y_avgs_ecdftan) - np.mean(vel_y_avgs_eotog),
        ]

        # standard error bars
        error = [
            np.std(vel_y_avgs_ectog) / np.sqrt(len(vel_y_avgs_ectog)),
            np.std(vel_y_avgs_eodbtan) / np.sqrt(len(vel_y_avgs_eodbtan)),
            np.std(vel_y_avgs_eodftan) / np.sqrt(len(vel_y_avgs_eodftan)),
            np.std(vel_y_avgs_ecdbtan) / np.sqrt(len(vel_y_avgs_ecdbtan)),
            np.std(vel_y_avgs_ecdftan) / np.sqrt(len(vel_y_avgs_ecdftan)),
        ]

        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(
            x_pos, vels, yerr=error, capsize=3, color="rbymc",
        )

        plt.xlabel(
            "Balance Condition", fontdict={"fontsize": 11},
        )
        plt.ylabel(
            "Velocity Difference (mm/s)", fontdict={"fontsize": 11},
        )
        plt.title(
            f"Difference in velocity form EOFT condition\nin ML for Subject {subject}",
            fontdict={"fontsize": 11},
        )

        plt.xticks(x_pos, x)

        plt.subplot(143)
        plt.style.use("ggplot")

        x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]
        vels = [
            np.mean(pow_x_tot_ectog) - np.mean(pow_x_tot_eotog),
            np.mean(pow_x_tot_eodbtan) - np.mean(pow_x_tot_eotog),
            np.mean(pow_x_tot_eodftan) - np.mean(pow_x_tot_eotog),
            np.mean(pow_x_tot_ecdbtan) - np.mean(pow_x_tot_eotog),
            np.mean(pow_x_tot_ecdftan) - np.mean(pow_x_tot_eotog),
        ]

        # standard error bars
        error = [
            np.std(pow_x_tot_ectog) / np.sqrt(len(pow_x_tot_ectog)),
            np.std(pow_x_tot_eodbtan) / np.sqrt(len(pow_x_tot_eodbtan)),
            np.std(pow_x_tot_eodftan) / np.sqrt(len(pow_x_tot_eodftan)),
            np.std(pow_x_tot_ecdbtan) / np.sqrt(len(pow_x_tot_ecdbtan)),
            np.std(pow_x_tot_ecdftan) / np.sqrt(len(pow_x_tot_ecdftan)),
        ]

        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, vels, yerr=error, capsize=3, color="rbymc")

        plt.xlabel(
            "Balance Condition", fontdict={"fontsize": 11},
        )
        plt.ylabel(
            "Power Difference", fontdict={"fontsize": 11},
        )
        plt.title(
            f"Difference in total power from EOFT condition\nin AP for Subject {subject}",
            fontdict={"fontsize": 11},
        )

        plt.xticks(x_pos, x)

        plt.subplot(144)
        plt.style.use("ggplot")

        x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]
        vels = [
            np.mean(pow_y_tot_ectog) - np.mean(pow_y_tot_eotog),
            np.mean(pow_y_tot_eodbtan) - np.mean(pow_y_tot_eotog),
            np.mean(pow_y_tot_eodftan) - np.mean(pow_y_tot_eotog),
            np.mean(pow_y_tot_ecdbtan) - np.mean(pow_y_tot_eotog),
            np.mean(pow_y_tot_ecdftan) - np.mean(pow_y_tot_eotog),
        ]

        # standard error bars
        error = [
            np.std(pow_y_tot_ectog) / np.sqrt(len(pow_y_tot_ectog)),
            np.std(pow_y_tot_eodbtan) / np.sqrt(len(pow_y_tot_eodbtan)),
            np.std(pow_y_tot_eodftan) / np.sqrt(len(pow_y_tot_eodftan)),
            np.std(pow_y_tot_ecdbtan) / np.sqrt(len(pow_y_tot_ecdbtan)),
            np.std(pow_y_tot_ecdftan) / np.sqrt(len(pow_y_tot_ecdftan)),
        ]

        x_pos = [i for i, _ in enumerate(x)]

        plt.bar(x_pos, vels, yerr=error, capsize=3, color="rbymc")
        plt.xlabel(
            "Balance Condition", fontdict={"fontsize": 11},
        )
        plt.ylabel(
            "Power Difference", fontdict={"fontsize": 11},
        )
        plt.title(
            f"Difference in total power from EOFT condition\nin ML for Subject {subject}",
            fontdict={"fontsize": 11},
        )

        plt.xticks(x_pos, x)

        # show all four subplots
        plt.show()

        #########################

