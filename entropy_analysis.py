# entropy_analysis.py
#
# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: July 3, 2020
#
# Description: Perform analysis on files
#   containing Approximate entropy results
#   for all files in x and y directions
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

# constants
fs_cop = 1200
t_cop = np.arange(0, 30, 1 / fs_cop)

# create dictionaries to hold the trial entropy
# for each subject in both directions
sb01_x = {}
sb02_x = {}
sb04_x = {}
sb05_x = {}
sb06_x = {}
sb08_x = {}

sb01_y = {}
sb02_y = {}
sb04_y = {}
sb05_y = {}
sb06_y = {}
sb08_y = {}

#####################################################################################

if __name__ == "__main__":

    # read files with x and y ApEn data
    x_ent = pd.read_csv("x.csv", index_col=False)
    y_ent = pd.read_csv("y.csv", index_col=False)

    # cycle through each column in x file
    for col in x_ent.columns:
        # determine the subject and trial number for column
        number = int(col[10:12])
        subject = int(col[2:4])

        # turn column of df into a list
        sig = x_ent[col].to_list()

        # create dictionary key for trial number and save data to it
        # for each subject
        if subject == 1:
            sb01_x[number] = sig
        elif subject == 2:
            sb02_x[number] = sig
        elif subject == 4:
            sb04_x[number] = sig
        elif subject == 5:
            sb05_x[number] = sig
        elif subject == 6:
            sb06_x[number] = sig
        elif subject == 8:
            sb08_x[number] = sig

    # repeat above for y data
    for col in y_ent.columns:
        number = int(col[10:12])
        subject = int(col[2:4])

        sig = y_ent[col].to_list()

        if subject == 1:
            sb01_y[number] = sig
        elif subject == 2:
            sb02_y[number] = sig
        elif subject == 4:
            sb04_y[number] = sig
        elif subject == 5:
            sb05_y[number] = sig
        elif subject == 6:
            sb06_y[number] = sig
        elif subject == 8:
            sb08_y[number] = sig

    ####################### Subject 1 #########################

    # create lists to save different stability conditions to
    x_eoft_ent = []
    x_ecft_ent = []
    x_eoftandb_ent = []
    x_ecftandb_ent = []
    x_eoftandf_ent = []
    x_ecftandf_ent = []

    y_eoft_ent = []
    y_ecft_ent = []
    y_eoftandb_ent = []
    y_ecftandb_ent = []
    y_eoftandf_ent = []
    y_ecftandf_ent = []

    x_trial_ent = []
    y_trial_ent = []
    trial_cond = []

    # loop through all trials for subject 1 in x
    for trial in sb01_x.keys():
        # save ApEn to proper stability condition list
        if trial < 7:
            trial_cond.append("EOFT")
            x_eoft_ent.append(np.mean(sb01_x[trial]))
            y_eoft_ent.append(np.mean(sb01_y[trial]))
            x_trial_ent.append(np.mean(sb01_x[trial]))
            y_trial_ent.append(np.mean(sb01_y[trial]))
        elif 6 < trial < 12:
            trial_cond.append("ECFT")
            x_ecft_ent.append(np.mean(sb01_x[trial]))
            y_ecft_ent.append(np.mean(sb01_y[trial]))
            x_trial_ent.append(np.mean(sb01_x[trial]))
            y_trial_ent.append(np.mean(sb01_y[trial]))
        elif 11 < trial < 17:
            trial_cond.append("EOFTanDB")
            x_eoftandb_ent.append(np.mean(sb01_x[trial]))
            y_eoftandb_ent.append(np.mean(sb01_y[trial]))
            x_trial_ent.append(np.mean(sb01_x[trial]))
            y_trial_ent.append(np.mean(sb01_y[trial]))
        elif 16 < trial < 22:
            trial_cond.append("ECFTanDB")
            x_ecftandb_ent.append(np.mean(sb01_x[trial]))
            y_ecftandb_ent.append(np.mean(sb01_y[trial]))
            x_trial_ent.append(np.mean(sb01_x[trial]))
            y_trial_ent.append(np.mean(sb01_y[trial]))
        elif 21 < trial < 27:
            trial_cond.append("EOFTanDF")
            x_eoftandf_ent.append(np.mean(sb01_x[trial]))
            y_eoftandf_ent.append(np.mean(sb01_y[trial]))
            x_trial_ent.append(np.mean(sb01_x[trial]))
            y_trial_ent.append(np.mean(sb01_y[trial]))
        elif 26 < trial < 32:
            trial_cond.append("ECFTanDF")
            x_ecftandf_ent.append(np.mean(sb01_x[trial]))
            y_ecftandf_ent.append(np.mean(sb01_y[trial]))
            x_trial_ent.append(np.mean(sb01_x[trial]))
            y_trial_ent.append(np.mean(sb01_y[trial]))

    # save x and y entropies by trial and condition to a .csv file
    zipped = list(zip(x_trial_ent, y_trial_ent, trial_cond))
    df_for_stats = pd.DataFrame(zipped, columns=["xEntropy", "yEntropy", "condition"])
    df_for_stats.to_csv(f"sb1_entropy_by_condition.csv")

    # create windows to plot moving entropy against
    windows = np.arange(0, 39, 1)

    # plot entropy
    plt.figure()
    plt.subplot(121)
    plt.plot(windows, sb01_x[27])
    plt.title("Approximate Entropy in AP direction for\nECFTanDF trial")
    plt.xlabel("Window")
    plt.ylabel("Magnitude")
    plt.ylim([0, 0.4])
    plt.subplot(122)
    plt.plot(windows, sb01_y[27])
    plt.title("Approximate Entropy in ML direction for\nECFTanDF trial")
    plt.xlabel("Window")
    plt.ylabel("Magnitude")
    plt.ylim([0, 0.4])
    plt.show()

    # create new figure to plot 2 subplots
    plt.figure()

    # style of plots
    plt.style.use("ggplot")
    plt.subplot(121)

    # x axis labels
    x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(x_eoft_ent),
        np.mean(x_ecft_ent),
        np.mean(x_eoftandb_ent),
        np.mean(x_eoftandf_ent),
        np.mean(x_ecftandb_ent),
        np.mean(x_ecftandf_ent),
    ]

    # standard error bars
    error = [
        np.std(x_eoft_ent) / np.sqrt(len(x_eoft_ent)),
        np.std(x_ecft_ent) / np.sqrt(len(x_ecft_ent)),
        np.std(x_eoftandb_ent) / np.sqrt(len(x_eoftandb_ent)),
        np.std(x_eoftandf_ent) / np.sqrt(len(x_eoftandf_ent)),
        np.std(x_ecftandb_ent) / np.sqrt(len(x_ecftandb_ent)),
        np.std(x_ecftandf_ent) / np.sqrt(len(x_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="grbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "Average Approximate Entropy", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Average ApEn of Each Balance Condition\nin AP for Subject 1",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.style.use("ggplot")
    plt.subplot(122)

    # x axis labels
    x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(y_eoft_ent),
        np.mean(y_ecft_ent),
        np.mean(y_eoftandb_ent),
        np.mean(y_eoftandf_ent),
        np.mean(y_ecftandb_ent),
        np.mean(y_ecftandf_ent),
    ]

    # standard error bars
    error = [
        np.std(y_eoft_ent) / np.sqrt(len(y_eoft_ent)),
        np.std(y_ecft_ent) / np.sqrt(len(y_ecft_ent)),
        np.std(y_eoftandb_ent) / np.sqrt(len(y_eoftandb_ent)),
        np.std(y_eoftandf_ent) / np.sqrt(len(y_eoftandf_ent)),
        np.std(y_ecftandb_ent) / np.sqrt(len(y_ecftandb_ent)),
        np.std(y_ecftandf_ent) / np.sqrt(len(y_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="grbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "Average Approximate Entropy", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Average ApEn of Each Balance Condition\nin ML for Subject 1",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.show()

    # create new figure to plot 2 subplots
    plt.figure()

    # style of plots
    plt.style.use("ggplot")
    plt.subplot(121)

    # x axis labels
    x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(x_ecft_ent) - np.mean(x_eoft_ent),
        np.mean(x_eoftandb_ent) - np.mean(x_eoft_ent),
        np.mean(x_eoftandf_ent) - np.mean(x_eoft_ent),
        np.mean(x_ecftandb_ent) - np.mean(x_eoft_ent),
        np.mean(x_ecftandf_ent) - np.mean(x_eoft_ent),
    ]

    # standard error bars
    error = [
        np.std(x_ecft_ent) / np.sqrt(len(x_ecft_ent)),
        np.std(x_eoftandb_ent) / np.sqrt(len(x_eoftandb_ent)),
        np.std(x_eoftandf_ent) / np.sqrt(len(x_eoftandf_ent)),
        np.std(x_ecftandb_ent) / np.sqrt(len(x_ecftandb_ent)),
        np.std(x_ecftandf_ent) / np.sqrt(len(x_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="rbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "ApEn Difference", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Difference in ApEn from EOFT condition\nin AP for Subject 1",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.style.use("ggplot")
    plt.subplot(122)

    # x axis labels
    x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(y_ecft_ent) - np.mean(y_eoft_ent),
        np.mean(y_eoftandb_ent) - np.mean(y_eoft_ent),
        np.mean(y_eoftandf_ent) - np.mean(y_eoft_ent),
        np.mean(y_ecftandb_ent) - np.mean(y_eoft_ent),
        np.mean(y_ecftandf_ent) - np.mean(y_eoft_ent),
    ]

    # standard error bars
    error = [
        np.std(y_ecft_ent) / np.sqrt(len(y_ecft_ent)),
        np.std(y_eoftandb_ent) / np.sqrt(len(y_eoftandb_ent)),
        np.std(y_eoftandf_ent) / np.sqrt(len(y_eoftandf_ent)),
        np.std(y_ecftandb_ent) / np.sqrt(len(y_ecftandb_ent)),
        np.std(y_ecftandf_ent) / np.sqrt(len(y_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="rbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "ApEn Difference", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Difference in ApEn between EOFT condition\nin ML for Subject 1",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.show()

    ####################### Subject 2 #########################

    x_eoft_ent = []
    x_ecft_ent = []
    x_eoftandb_ent = []
    x_ecftandb_ent = []
    x_eoftandf_ent = []
    x_ecftandf_ent = []

    y_eoft_ent = []
    y_ecft_ent = []
    y_eoftandb_ent = []
    y_ecftandb_ent = []
    y_eoftandf_ent = []
    y_ecftandf_ent = []

    x_trial_ent = []
    y_trial_ent = []
    trial_cond = []

    for trial in sb02_x.keys():
        if trial < 7:
            trial_cond.append("EOFT")
            x_eoft_ent.append(np.mean(sb02_x[trial]))
            y_eoft_ent.append(np.mean(sb02_y[trial]))
            x_trial_ent.append(np.mean(sb02_x[trial]))
            y_trial_ent.append(np.mean(sb02_y[trial]))
        elif 6 < trial < 12:
            trial_cond.append("ECFT")
            x_ecft_ent.append(np.mean(sb02_x[trial]))
            y_ecft_ent.append(np.mean(sb02_y[trial]))
            x_trial_ent.append(np.mean(sb02_x[trial]))
            y_trial_ent.append(np.mean(sb02_y[trial]))
        elif 11 < trial < 17:
            trial_cond.append("EOFTanDB")
            x_eoftandb_ent.append(np.mean(sb02_x[trial]))
            y_eoftandb_ent.append(np.mean(sb02_y[trial]))
            x_trial_ent.append(np.mean(sb02_x[trial]))
            y_trial_ent.append(np.mean(sb02_y[trial]))
        elif 16 < trial < 22:
            trial_cond.append("ECFTanDB")
            x_ecftandb_ent.append(np.mean(sb02_x[trial]))
            y_ecftandb_ent.append(np.mean(sb02_y[trial]))
            x_trial_ent.append(np.mean(sb02_x[trial]))
            y_trial_ent.append(np.mean(sb02_y[trial]))
        elif 21 < trial < 27:
            trial_cond.append("EOFTanDF")
            x_eoftandf_ent.append(np.mean(sb02_x[trial]))
            y_eoftandf_ent.append(np.mean(sb02_y[trial]))
            x_trial_ent.append(np.mean(sb02_x[trial]))
            y_trial_ent.append(np.mean(sb02_y[trial]))
        elif 26 < trial < 32:
            trial_cond.append("ECFTanDF")
            x_ecftandf_ent.append(np.mean(sb02_x[trial]))
            y_ecftandf_ent.append(np.mean(sb02_y[trial]))
            x_trial_ent.append(np.mean(sb02_x[trial]))
            y_trial_ent.append(np.mean(sb02_y[trial]))

    zipped = list(zip(x_trial_ent, y_trial_ent, trial_cond))
    df_for_stats = pd.DataFrame(zipped, columns=["xEntropy", "yEntropy", "condition"])
    df_for_stats.to_csv(f"sb2_entropy_by_condition.csv")

    plt.figure()

    # style of plots
    plt.style.use("ggplot")
    plt.subplot(121)

    # x axis labels
    x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(x_eoft_ent),
        np.mean(x_ecft_ent),
        np.mean(x_eoftandb_ent),
        np.mean(x_eoftandf_ent),
        np.mean(x_ecftandb_ent),
        np.mean(x_ecftandf_ent),
    ]

    # standard error bars
    error = [
        np.std(x_eoft_ent) / np.sqrt(len(x_eoft_ent)),
        np.std(x_ecft_ent) / np.sqrt(len(x_ecft_ent)),
        np.std(x_eoftandb_ent) / np.sqrt(len(x_eoftandb_ent)),
        np.std(x_eoftandf_ent) / np.sqrt(len(x_eoftandf_ent)),
        np.std(x_ecftandb_ent) / np.sqrt(len(x_ecftandb_ent)),
        np.std(x_ecftandf_ent) / np.sqrt(len(x_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="grbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "Average Approximate Entropy", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Average ApEn of Each Balance Condition\nin AP for Subject 2",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.style.use("ggplot")
    plt.subplot(122)

    # x axis labels
    x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(y_eoft_ent),
        np.mean(y_ecft_ent),
        np.mean(y_eoftandb_ent),
        np.mean(y_eoftandf_ent),
        np.mean(y_ecftandb_ent),
        np.mean(y_ecftandf_ent),
    ]

    # standard error bars
    error = [
        np.std(y_eoft_ent) / np.sqrt(len(y_eoft_ent)),
        np.std(y_ecft_ent) / np.sqrt(len(y_ecft_ent)),
        np.std(y_eoftandb_ent) / np.sqrt(len(y_eoftandb_ent)),
        np.std(y_eoftandf_ent) / np.sqrt(len(y_eoftandf_ent)),
        np.std(y_ecftandb_ent) / np.sqrt(len(y_ecftandb_ent)),
        np.std(y_ecftandf_ent) / np.sqrt(len(y_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="grbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "Average Approximate Entropy", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Average ApEn of Each Balance Condition\nin ML for Subject 2",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.show()

    # create new figure to plot 2 subplots
    plt.figure()

    # style of plots
    plt.style.use("ggplot")
    plt.subplot(121)

    # x axis labels
    x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(x_ecft_ent) - np.mean(x_eoft_ent),
        np.mean(x_eoftandb_ent) - np.mean(x_eoft_ent),
        np.mean(x_eoftandf_ent) - np.mean(x_eoft_ent),
        np.mean(x_ecftandb_ent) - np.mean(x_eoft_ent),
        np.mean(x_ecftandf_ent) - np.mean(x_eoft_ent),
    ]

    # standard error bars
    error = [
        np.std(x_ecft_ent) / np.sqrt(len(x_ecft_ent)),
        np.std(x_eoftandb_ent) / np.sqrt(len(x_eoftandb_ent)),
        np.std(x_eoftandf_ent) / np.sqrt(len(x_eoftandf_ent)),
        np.std(x_ecftandb_ent) / np.sqrt(len(x_ecftandb_ent)),
        np.std(x_ecftandf_ent) / np.sqrt(len(x_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="rbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "ApEn Difference", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Difference in ApEn from EOFT condition\nin AP for Subject 2",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.style.use("ggplot")
    plt.subplot(122)

    # x axis labels
    x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(y_ecft_ent) - np.mean(y_eoft_ent),
        np.mean(y_eoftandb_ent) - np.mean(y_eoft_ent),
        np.mean(y_eoftandf_ent) - np.mean(y_eoft_ent),
        np.mean(y_ecftandb_ent) - np.mean(y_eoft_ent),
        np.mean(y_ecftandf_ent) - np.mean(y_eoft_ent),
    ]

    # standard error bars
    error = [
        np.std(y_ecft_ent) / np.sqrt(len(y_ecft_ent)),
        np.std(y_eoftandb_ent) / np.sqrt(len(y_eoftandb_ent)),
        np.std(y_eoftandf_ent) / np.sqrt(len(y_eoftandf_ent)),
        np.std(y_ecftandb_ent) / np.sqrt(len(y_ecftandb_ent)),
        np.std(y_ecftandf_ent) / np.sqrt(len(y_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="rbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "ApEn Difference", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Difference in ApEn between EOFT condition\nin ML for Subject 2",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.show()

    ####################### Subject 4 #########################

    x_eoft_ent = []
    x_ecft_ent = []
    x_eoftandb_ent = []
    x_ecftandb_ent = []
    x_eoftandf_ent = []
    x_ecftandf_ent = []

    y_eoft_ent = []
    y_ecft_ent = []
    y_eoftandb_ent = []
    y_ecftandb_ent = []
    y_eoftandf_ent = []
    y_ecftandf_ent = []

    x_trial_ent = []
    y_trial_ent = []
    trial_cond = []

    for trial in sb04_x.keys():
        if trial < 7:
            trial_cond.append("EOFT")
            x_eoft_ent.append(np.mean(sb04_x[trial]))
            y_eoft_ent.append(np.mean(sb04_y[trial]))
            x_trial_ent.append(np.mean(sb04_x[trial]))
            y_trial_ent.append(np.mean(sb04_y[trial]))
        elif 6 < trial < 12:
            trial_cond.append("ECFT")
            x_ecft_ent.append(np.mean(sb04_x[trial]))
            y_ecft_ent.append(np.mean(sb04_y[trial]))
            x_trial_ent.append(np.mean(sb04_x[trial]))
            y_trial_ent.append(np.mean(sb04_y[trial]))
        elif 11 < trial < 17:
            trial_cond.append("EOFTanDB")
            x_eoftandb_ent.append(np.mean(sb04_x[trial]))
            y_eoftandb_ent.append(np.mean(sb04_y[trial]))
            x_trial_ent.append(np.mean(sb04_x[trial]))
            y_trial_ent.append(np.mean(sb04_y[trial]))
        elif 16 < trial < 22:
            trial_cond.append("ECFTanDB")
            x_ecftandb_ent.append(np.mean(sb04_x[trial]))
            y_ecftandb_ent.append(np.mean(sb04_y[trial]))
            x_trial_ent.append(np.mean(sb04_x[trial]))
            y_trial_ent.append(np.mean(sb04_y[trial]))
        elif 21 < trial < 27:
            trial_cond.append("EOFTanDF")
            x_eoftandf_ent.append(np.mean(sb04_x[trial]))
            y_eoftandf_ent.append(np.mean(sb04_y[trial]))
            x_trial_ent.append(np.mean(sb04_x[trial]))
            y_trial_ent.append(np.mean(sb04_y[trial]))
        elif 26 < trial < 32:
            trial_cond.append("ECFTanDF")
            x_ecftandf_ent.append(np.mean(sb04_x[trial]))
            y_ecftandf_ent.append(np.mean(sb04_y[trial]))
            x_trial_ent.append(np.mean(sb04_x[trial]))
            y_trial_ent.append(np.mean(sb04_y[trial]))

    zipped = list(zip(x_trial_ent, y_trial_ent, trial_cond))
    df_for_stats = pd.DataFrame(zipped, columns=["xEntropy", "yEntropy", "condition"])
    df_for_stats.to_csv(f"sb4_entropy_by_condition.csv")

    plt.figure()

    # style of plots
    plt.style.use("ggplot")
    plt.subplot(121)

    # x axis labels
    x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(x_eoft_ent),
        np.mean(x_ecft_ent),
        np.mean(x_eoftandb_ent),
        np.mean(x_eoftandf_ent),
        np.mean(x_ecftandb_ent),
        np.mean(x_ecftandf_ent),
    ]

    # standard error bars
    error = [
        np.std(x_eoft_ent) / np.sqrt(len(x_eoft_ent)),
        np.std(x_ecft_ent) / np.sqrt(len(x_ecft_ent)),
        np.std(x_eoftandb_ent) / np.sqrt(len(x_eoftandb_ent)),
        np.std(x_eoftandf_ent) / np.sqrt(len(x_eoftandf_ent)),
        np.std(x_ecftandb_ent) / np.sqrt(len(x_ecftandb_ent)),
        np.std(x_ecftandf_ent) / np.sqrt(len(x_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="grbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "Average Approximate Entropy", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Average ApEn of Each Balance Condition\nin AP for Subject 3",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.style.use("ggplot")
    plt.subplot(122)

    # x axis labels
    x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(y_eoft_ent),
        np.mean(y_ecft_ent),
        np.mean(y_eoftandb_ent),
        np.mean(y_eoftandf_ent),
        np.mean(y_ecftandb_ent),
        np.mean(y_ecftandf_ent),
    ]

    # standard error bars
    error = [
        np.std(y_eoft_ent) / np.sqrt(len(y_eoft_ent)),
        np.std(y_ecft_ent) / np.sqrt(len(y_ecft_ent)),
        np.std(y_eoftandb_ent) / np.sqrt(len(y_eoftandb_ent)),
        np.std(y_eoftandf_ent) / np.sqrt(len(y_eoftandf_ent)),
        np.std(y_ecftandb_ent) / np.sqrt(len(y_ecftandb_ent)),
        np.std(y_ecftandf_ent) / np.sqrt(len(y_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="grbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "Average Approximate Entropy", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Average ApEn of Each Balance Condition\nin ML for Subject 3",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.show()

    # create new figure to plot 2 subplots
    plt.figure()

    # style of plots
    plt.style.use("ggplot")
    plt.subplot(121)

    # x axis labels
    x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(x_ecft_ent) - np.mean(x_eoft_ent),
        np.mean(x_eoftandb_ent) - np.mean(x_eoft_ent),
        np.mean(x_eoftandf_ent) - np.mean(x_eoft_ent),
        np.mean(x_ecftandb_ent) - np.mean(x_eoft_ent),
        np.mean(x_ecftandf_ent) - np.mean(x_eoft_ent),
    ]

    # standard error bars
    error = [
        np.std(x_ecft_ent) / np.sqrt(len(x_ecft_ent)),
        np.std(x_eoftandb_ent) / np.sqrt(len(x_eoftandb_ent)),
        np.std(x_eoftandf_ent) / np.sqrt(len(x_eoftandf_ent)),
        np.std(x_ecftandb_ent) / np.sqrt(len(x_ecftandb_ent)),
        np.std(x_ecftandf_ent) / np.sqrt(len(x_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="rbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "ApEn Difference", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Difference in ApEn from EOFT condition\nin AP for Subject 3",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.style.use("ggplot")
    plt.subplot(122)

    # x axis labels
    x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(y_ecft_ent) - np.mean(y_eoft_ent),
        np.mean(y_eoftandb_ent) - np.mean(y_eoft_ent),
        np.mean(y_eoftandf_ent) - np.mean(y_eoft_ent),
        np.mean(y_ecftandb_ent) - np.mean(y_eoft_ent),
        np.mean(y_ecftandf_ent) - np.mean(y_eoft_ent),
    ]

    # standard error bars
    error = [
        np.std(y_ecft_ent) / np.sqrt(len(y_ecft_ent)),
        np.std(y_eoftandb_ent) / np.sqrt(len(y_eoftandb_ent)),
        np.std(y_eoftandf_ent) / np.sqrt(len(y_eoftandf_ent)),
        np.std(y_ecftandb_ent) / np.sqrt(len(y_ecftandb_ent)),
        np.std(y_ecftandf_ent) / np.sqrt(len(y_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="rbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "ApEn Difference", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Difference in ApEn between EOFT condition\nin ML for Subject 3",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.show()

    ####################### Subject 5 #########################

    x_eoft_ent = []
    x_ecft_ent = []
    x_eoftandb_ent = []
    x_ecftandb_ent = []
    x_eoftandf_ent = []
    x_ecftandf_ent = []

    y_eoft_ent = []
    y_ecft_ent = []
    y_eoftandb_ent = []
    y_ecftandb_ent = []
    y_eoftandf_ent = []
    y_ecftandf_ent = []

    x_trial_ent = []
    y_trial_ent = []
    trial_cond = []

    for trial in sb05_x.keys():
        if trial < 7:
            trial_cond.append("EOFT")
            x_eoft_ent.append(np.mean(sb05_x[trial]))
            y_eoft_ent.append(np.mean(sb05_y[trial]))
            x_trial_ent.append(np.mean(sb05_x[trial]))
            y_trial_ent.append(np.mean(sb05_y[trial]))
        elif 6 < trial < 12:
            trial_cond.append("ECFT")
            x_ecft_ent.append(np.mean(sb05_x[trial]))
            y_ecft_ent.append(np.mean(sb05_y[trial]))
            x_trial_ent.append(np.mean(sb05_x[trial]))
            y_trial_ent.append(np.mean(sb05_y[trial]))
        elif 11 < trial < 17:
            trial_cond.append("EOFTanDB")
            x_eoftandb_ent.append(np.mean(sb05_x[trial]))
            y_eoftandb_ent.append(np.mean(sb05_y[trial]))
            x_trial_ent.append(np.mean(sb05_x[trial]))
            y_trial_ent.append(np.mean(sb05_y[trial]))
        elif 16 < trial < 22:
            trial_cond.append("ECFTanDB")
            x_ecftandb_ent.append(np.mean(sb05_x[trial]))
            y_ecftandb_ent.append(np.mean(sb05_y[trial]))
            x_trial_ent.append(np.mean(sb05_x[trial]))
            y_trial_ent.append(np.mean(sb05_y[trial]))
        elif 21 < trial < 27:
            trial_cond.append("EOFTanDF")
            x_eoftandf_ent.append(np.mean(sb05_x[trial]))
            y_eoftandf_ent.append(np.mean(sb05_y[trial]))
            x_trial_ent.append(np.mean(sb05_x[trial]))
            y_trial_ent.append(np.mean(sb05_y[trial]))
        elif 26 < trial < 32:
            trial_cond.append("ECFTanDF")
            x_ecftandf_ent.append(np.mean(sb05_x[trial]))
            y_ecftandf_ent.append(np.mean(sb05_y[trial]))
            x_trial_ent.append(np.mean(sb05_x[trial]))
            y_trial_ent.append(np.mean(sb05_y[trial]))

    zipped = list(zip(x_trial_ent, y_trial_ent, trial_cond))
    df_for_stats = pd.DataFrame(zipped, columns=["xEntropy", "yEntropy", "condition"])
    df_for_stats.to_csv(f"sb5_entropy_by_condition.csv")

    plt.figure()

    # style of plots
    plt.style.use("ggplot")
    plt.subplot(121)

    # x axis labels
    x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(x_eoft_ent),
        np.mean(x_ecft_ent),
        np.mean(x_eoftandb_ent),
        np.mean(x_eoftandf_ent),
        np.mean(x_ecftandb_ent),
        np.mean(x_ecftandf_ent),
    ]

    # standard error bars
    error = [
        np.std(x_eoft_ent) / np.sqrt(len(x_eoft_ent)),
        np.std(x_ecft_ent) / np.sqrt(len(x_ecft_ent)),
        np.std(x_eoftandb_ent) / np.sqrt(len(x_eoftandb_ent)),
        np.std(x_eoftandf_ent) / np.sqrt(len(x_eoftandf_ent)),
        np.std(x_ecftandb_ent) / np.sqrt(len(x_ecftandb_ent)),
        np.std(x_ecftandf_ent) / np.sqrt(len(x_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="grbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "Average Approximate Entropy", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Average ApEn of Each Balance Condition\nin AP for Subject 4",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.style.use("ggplot")
    plt.subplot(122)

    # x axis labels
    x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(y_eoft_ent),
        np.mean(y_ecft_ent),
        np.mean(y_eoftandb_ent),
        np.mean(y_eoftandf_ent),
        np.mean(y_ecftandb_ent),
        np.mean(y_ecftandf_ent),
    ]

    # standard error bars
    error = [
        np.std(y_eoft_ent) / np.sqrt(len(y_eoft_ent)),
        np.std(y_ecft_ent) / np.sqrt(len(y_ecft_ent)),
        np.std(y_eoftandb_ent) / np.sqrt(len(y_eoftandb_ent)),
        np.std(y_eoftandf_ent) / np.sqrt(len(y_eoftandf_ent)),
        np.std(y_ecftandb_ent) / np.sqrt(len(y_ecftandb_ent)),
        np.std(y_ecftandf_ent) / np.sqrt(len(y_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="grbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "Average Approximate Entropy", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Average ApEn of Each Balance Condition\nin ML for Subject 4",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.show()

    # create new figure to plot 2 subplots
    plt.figure()

    # style of plots
    plt.style.use("ggplot")

    # x axis labels
    x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(x_ecft_ent) - np.mean(x_eoft_ent),
        np.mean(x_eoftandb_ent) - np.mean(x_eoft_ent),
        np.mean(x_eoftandf_ent) - np.mean(x_eoft_ent),
        np.mean(x_ecftandb_ent) - np.mean(x_eoft_ent),
        np.mean(x_ecftandf_ent) - np.mean(x_eoft_ent),
    ]

    # standard error bars
    error = [
        np.std(x_ecft_ent) / np.sqrt(len(x_ecft_ent)),
        np.std(x_eoftandb_ent) / np.sqrt(len(x_eoftandb_ent)),
        np.std(x_eoftandf_ent) / np.sqrt(len(x_eoftandf_ent)),
        np.std(x_ecftandb_ent) / np.sqrt(len(x_ecftandb_ent)),
        np.std(x_ecftandf_ent) / np.sqrt(len(x_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="rbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "ApEn Difference", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Difference in ApEn from EOFT condition\nin AP for Subject 4",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)
    plt.show()

    ####################### Subject 6 #########################

    x_eoft_ent = []
    x_ecft_ent = []
    x_eoftandb_ent = []
    x_ecftandb_ent = []
    x_eoftandf_ent = []
    x_ecftandf_ent = []

    y_eoft_ent = []
    y_ecft_ent = []
    y_eoftandb_ent = []
    y_ecftandb_ent = []
    y_eoftandf_ent = []
    y_ecftandf_ent = []

    x_trial_ent = []
    y_trial_ent = []
    trial_cond = []

    for trial in sb06_x.keys():
        if trial < 7:
            trial_cond.append("EOFT")
            x_eoft_ent.append(np.mean(sb06_x[trial]))
            y_eoft_ent.append(np.mean(sb06_y[trial]))
            x_trial_ent.append(np.mean(sb06_x[trial]))
            y_trial_ent.append(np.mean(sb06_y[trial]))
        elif 6 < trial < 12:
            trial_cond.append("ECFT")
            x_ecft_ent.append(np.mean(sb06_x[trial]))
            y_ecft_ent.append(np.mean(sb06_y[trial]))
            x_trial_ent.append(np.mean(sb06_x[trial]))
            y_trial_ent.append(np.mean(sb06_y[trial]))
        elif 11 < trial < 17:
            trial_cond.append("EOFTanDB")
            x_eoftandb_ent.append(np.mean(sb06_x[trial]))
            y_eoftandb_ent.append(np.mean(sb06_y[trial]))
            x_trial_ent.append(np.mean(sb06_x[trial]))
            y_trial_ent.append(np.mean(sb06_y[trial]))
        elif 16 < trial < 22:
            trial_cond.append("ECFTanDB")
            x_ecftandb_ent.append(np.mean(sb06_x[trial]))
            y_ecftandb_ent.append(np.mean(sb06_y[trial]))
            x_trial_ent.append(np.mean(sb06_x[trial]))
            y_trial_ent.append(np.mean(sb06_y[trial]))
        elif 21 < trial < 27:
            trial_cond.append("EOFTanDF")
            x_eoftandf_ent.append(np.mean(sb06_x[trial]))
            y_eoftandf_ent.append(np.mean(sb06_y[trial]))
            x_trial_ent.append(np.mean(sb06_x[trial]))
            y_trial_ent.append(np.mean(sb06_y[trial]))
        elif 26 < trial < 32:
            trial_cond.append("ECFTanDF")
            x_ecftandf_ent.append(np.mean(sb06_x[trial]))
            y_ecftandf_ent.append(np.mean(sb06_y[trial]))
            x_trial_ent.append(np.mean(sb06_x[trial]))
            y_trial_ent.append(np.mean(sb06_y[trial]))

    zipped = list(zip(x_trial_ent, y_trial_ent, trial_cond))
    df_for_stats = pd.DataFrame(zipped, columns=["xEntropy", "yEntropy", "condition"])
    df_for_stats.to_csv(f"sb6_entropy_by_condition.csv")

    plt.figure()

    # style of plots
    plt.style.use("ggplot")
    plt.subplot(121)

    # x axis labels
    x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(x_eoft_ent),
        np.mean(x_ecft_ent),
        np.mean(x_eoftandb_ent),
        np.mean(x_eoftandf_ent),
        np.mean(x_ecftandb_ent),
        np.mean(x_ecftandf_ent),
    ]

    # standard error bars
    error = [
        np.std(x_eoft_ent) / np.sqrt(len(x_eoft_ent)),
        np.std(x_ecft_ent) / np.sqrt(len(x_ecft_ent)),
        np.std(x_eoftandb_ent) / np.sqrt(len(x_eoftandb_ent)),
        np.std(x_eoftandf_ent) / np.sqrt(len(x_eoftandf_ent)),
        np.std(x_ecftandb_ent) / np.sqrt(len(x_ecftandb_ent)),
        np.std(x_ecftandf_ent) / np.sqrt(len(x_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="grbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "Average Approximate Entropy", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Average ApEn of Each Balance Condition\nin AP for Subject 5",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.style.use("ggplot")
    plt.subplot(122)

    # x axis labels
    x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(y_eoft_ent),
        np.mean(y_ecft_ent),
        np.mean(y_eoftandb_ent),
        np.mean(y_eoftandf_ent),
        np.mean(y_ecftandb_ent),
        np.mean(y_ecftandf_ent),
    ]

    # standard error bars
    error = [
        np.std(y_eoft_ent) / np.sqrt(len(y_eoft_ent)),
        np.std(y_ecft_ent) / np.sqrt(len(y_ecft_ent)),
        np.std(y_eoftandb_ent) / np.sqrt(len(y_eoftandb_ent)),
        np.std(y_eoftandf_ent) / np.sqrt(len(y_eoftandf_ent)),
        np.std(y_ecftandb_ent) / np.sqrt(len(y_ecftandb_ent)),
        np.std(y_ecftandf_ent) / np.sqrt(len(y_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="grbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "Average Approximate Entropy", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Average ApEn of Each Balance Condition\nin ML for Subject 5",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.show()

    # create new figure to plot 2 subplots
    plt.figure()

    # style of plots
    plt.style.use("ggplot")
    plt.subplot(121)

    # x axis labels
    x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(x_ecft_ent) - np.mean(x_eoft_ent),
        np.mean(x_eoftandb_ent) - np.mean(x_eoft_ent),
        np.mean(x_eoftandf_ent) - np.mean(x_eoft_ent),
        np.mean(x_ecftandb_ent) - np.mean(x_eoft_ent),
        np.mean(x_ecftandf_ent) - np.mean(x_eoft_ent),
    ]

    # standard error bars
    error = [
        np.std(x_ecft_ent) / np.sqrt(len(x_ecft_ent)),
        np.std(x_eoftandb_ent) / np.sqrt(len(x_eoftandb_ent)),
        np.std(x_eoftandf_ent) / np.sqrt(len(x_eoftandf_ent)),
        np.std(x_ecftandb_ent) / np.sqrt(len(x_ecftandb_ent)),
        np.std(x_ecftandf_ent) / np.sqrt(len(x_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="rbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "ApEn Difference", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Difference in ApEn from EOFT condition\nin AP for Subject 5",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.style.use("ggplot")
    plt.subplot(122)

    # x axis labels
    x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(y_ecft_ent) - np.mean(y_eoft_ent),
        np.mean(y_eoftandb_ent) - np.mean(y_eoft_ent),
        np.mean(y_eoftandf_ent) - np.mean(y_eoft_ent),
        np.mean(y_ecftandb_ent) - np.mean(y_eoft_ent),
        np.mean(y_ecftandf_ent) - np.mean(y_eoft_ent),
    ]

    # standard error bars
    error = [
        np.std(y_ecft_ent) / np.sqrt(len(y_ecft_ent)),
        np.std(y_eoftandb_ent) / np.sqrt(len(y_eoftandb_ent)),
        np.std(y_eoftandf_ent) / np.sqrt(len(y_eoftandf_ent)),
        np.std(y_ecftandb_ent) / np.sqrt(len(y_ecftandb_ent)),
        np.std(y_ecftandf_ent) / np.sqrt(len(y_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="rbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "ApEn Difference", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Difference in ApEn between EOFT condition\nin ML for Subject 5",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.show()

    ####################### Subject 8 #########################

    x_eoft_ent = []
    x_ecft_ent = []
    x_eoftandb_ent = []
    x_ecftandb_ent = []
    x_eoftandf_ent = []
    x_ecftandf_ent = []

    y_eoft_ent = []
    y_ecft_ent = []
    y_eoftandb_ent = []
    y_ecftandb_ent = []
    y_eoftandf_ent = []
    y_ecftandf_ent = []

    x_trial_ent = []
    y_trial_ent = []
    trial_cond = []

    for trial in sb08_x.keys():
        if trial < 7:
            trial_cond.append("EOFT")
            x_eoft_ent.append(np.mean(sb08_x[trial]))
            y_eoft_ent.append(np.mean(sb08_y[trial]))
            x_trial_ent.append(np.mean(sb08_x[trial]))
            y_trial_ent.append(np.mean(sb08_y[trial]))
        elif 6 < trial < 12:
            trial_cond.append("ECFT")
            x_ecft_ent.append(np.mean(sb08_x[trial]))
            y_ecft_ent.append(np.mean(sb08_y[trial]))
            x_trial_ent.append(np.mean(sb08_x[trial]))
            y_trial_ent.append(np.mean(sb08_y[trial]))
        elif 11 < trial < 17:
            trial_cond.append("EOFTanDB")
            x_eoftandb_ent.append(np.mean(sb08_x[trial]))
            y_eoftandb_ent.append(np.mean(sb08_y[trial]))
            x_trial_ent.append(np.mean(sb08_x[trial]))
            y_trial_ent.append(np.mean(sb08_y[trial]))
        elif 16 < trial < 22:
            trial_cond.append("ECFTanDB")
            x_ecftandb_ent.append(np.mean(sb08_x[trial]))
            y_ecftandb_ent.append(np.mean(sb08_y[trial]))
            x_trial_ent.append(np.mean(sb08_x[trial]))
            y_trial_ent.append(np.mean(sb08_y[trial]))
        elif 21 < trial < 27:
            trial_cond.append("EOFTanDF")
            x_eoftandf_ent.append(np.mean(sb08_x[trial]))
            y_eoftandf_ent.append(np.mean(sb08_y[trial]))
            x_trial_ent.append(np.mean(sb08_x[trial]))
            y_trial_ent.append(np.mean(sb08_y[trial]))
        elif 26 < trial < 32:
            trial_cond.append("ECFTanDF")
            x_ecftandf_ent.append(np.mean(sb08_x[trial]))
            y_ecftandf_ent.append(np.mean(sb08_y[trial]))
            x_trial_ent.append(np.mean(sb08_x[trial]))
            y_trial_ent.append(np.mean(sb08_y[trial]))

    zipped = list(zip(x_trial_ent, y_trial_ent, trial_cond))
    df_for_stats = pd.DataFrame(zipped, columns=["xEntropy", "yEntropy", "condition"])
    df_for_stats.to_csv(f"sb8_entropy_by_condition.csv")

    plt.figure()

    # style of plots
    plt.style.use("ggplot")
    plt.subplot(121)

    # x axis labels
    x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(x_eoft_ent),
        np.mean(x_ecft_ent),
        np.mean(x_eoftandb_ent),
        np.mean(x_eoftandf_ent),
        np.mean(x_ecftandb_ent),
        np.mean(x_ecftandf_ent),
    ]

    # standard error bars
    error = [
        np.std(x_eoft_ent) / np.sqrt(len(x_eoft_ent)),
        np.std(x_ecft_ent) / np.sqrt(len(x_ecft_ent)),
        np.std(x_eoftandb_ent) / np.sqrt(len(x_eoftandb_ent)),
        np.std(x_eoftandf_ent) / np.sqrt(len(x_eoftandf_ent)),
        np.std(x_ecftandb_ent) / np.sqrt(len(x_ecftandb_ent)),
        np.std(x_ecftandf_ent) / np.sqrt(len(x_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="grbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "Average Approximate Entropy", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Average ApEn of Each Balance Condition\nin AP for Subject 6",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.style.use("ggplot")
    plt.subplot(122)

    # x axis labels
    x = ["EOFT", "ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(y_eoft_ent),
        np.mean(y_ecft_ent),
        np.mean(y_eoftandb_ent),
        np.mean(y_eoftandf_ent),
        np.mean(y_ecftandb_ent),
        np.mean(y_ecftandf_ent),
    ]

    # standard error bars
    error = [
        np.std(y_eoft_ent) / np.sqrt(len(y_eoft_ent)),
        np.std(y_ecft_ent) / np.sqrt(len(y_ecft_ent)),
        np.std(y_eoftandb_ent) / np.sqrt(len(y_eoftandb_ent)),
        np.std(y_eoftandf_ent) / np.sqrt(len(y_eoftandf_ent)),
        np.std(y_ecftandb_ent) / np.sqrt(len(y_ecftandb_ent)),
        np.std(y_ecftandf_ent) / np.sqrt(len(y_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="grbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "Average Approximate Entropy", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Average ApEn of Each Balance Condition\nin ML for Subject 6",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.show()

    # create new figure to plot 2 subplots
    plt.figure()

    # style of plots
    plt.style.use("ggplot")
    plt.subplot(121)

    # x axis labels
    x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(x_ecft_ent) - np.mean(x_eoft_ent),
        np.mean(x_eoftandb_ent) - np.mean(x_eoft_ent),
        np.mean(x_eoftandf_ent) - np.mean(x_eoft_ent),
        np.mean(x_ecftandb_ent) - np.mean(x_eoft_ent),
        np.mean(x_ecftandf_ent) - np.mean(x_eoft_ent),
    ]

    # standard error bars
    error = [
        np.std(x_ecft_ent) / np.sqrt(len(x_ecft_ent)),
        np.std(x_eoftandb_ent) / np.sqrt(len(x_eoftandb_ent)),
        np.std(x_eoftandf_ent) / np.sqrt(len(x_eoftandf_ent)),
        np.std(x_ecftandb_ent) / np.sqrt(len(x_ecftandb_ent)),
        np.std(x_ecftandf_ent) / np.sqrt(len(x_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="rbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "ApEn Difference", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Difference in ApEn from EOFT condition\nin AP for Subject 6",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.style.use("ggplot")
    plt.subplot(122)

    # x axis labels
    x = ["ECFT", "EODB", "EODF", "ECDB", "ECDF"]

    # y axis data
    ents = [
        np.mean(y_ecft_ent) - np.mean(y_eoft_ent),
        np.mean(y_eoftandb_ent) - np.mean(y_eoft_ent),
        np.mean(y_eoftandf_ent) - np.mean(y_eoft_ent),
        np.mean(y_ecftandb_ent) - np.mean(y_eoft_ent),
        np.mean(y_ecftandf_ent) - np.mean(y_eoft_ent),
    ]

    # standard error bars
    error = [
        np.std(y_ecft_ent) / np.sqrt(len(y_ecft_ent)),
        np.std(y_eoftandb_ent) / np.sqrt(len(y_eoftandb_ent)),
        np.std(y_eoftandf_ent) / np.sqrt(len(y_eoftandf_ent)),
        np.std(y_ecftandb_ent) / np.sqrt(len(y_ecftandb_ent)),
        np.std(y_ecftandf_ent) / np.sqrt(len(y_ecftandf_ent)),
    ]

    # creates an x-axis position for each stability condition
    x_pos = [i for i, _ in enumerate(x)]
    # create bars plot with different colors for each
    plt.bar(x_pos, ents, yerr=error, capsize=3, color="rbymc")
    # label x axis
    plt.xlabel(
        "Balance Condition", fontdict={"fontsize": 11},
    )
    # label y axis
    plt.ylabel(
        "ApEn Difference", fontdict={"fontsize": 11},
    )
    # title plot
    plt.title(
        f"Difference in ApEn between EOFT condition\nin ML for Subject 6",
        fontdict={"fontsize": 11},
    )
    # # create x-axis ticks
    plt.xticks(x_pos, x)

    plt.show()
