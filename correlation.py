# correlation.py
#
# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: July 5, 2020
#
# Description: Signal Processing of COM
#   and COP to obtain auto and cross
#   correlation for comparison.
#
# Last updated: July 29, 2020

# import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import analysis_lib as an
from scipy.stats import pearsonr

# constants
fs_com = 120
fs_cop = 1200
t_com = np.arange(0, 30, 1 / fs_com)
t_cop = np.arange(0, 30, 1 / fs_cop)

#####################################################################################

if __name__ == "__main__":

    # read in COP data
    x_cop = pd.read_csv("x_cop.csv", index_col=False)
    y_cop = pd.read_csv("y_cop.csv", index_col=False)

    # read in COM data
    x_com = pd.read_csv("x_com.csv", index_col=False)
    y_com = pd.read_csv("y_com.csv", index_col=False)

    # create dictionaries and lists for x-axis data
    corr_dict = {}
    eoft = []
    ecft = []
    eoftandb = []
    eoftandf = []
    ecftandb = []
    ecftandf = []

    # cycle through each column in x file
    for col in x_cop.columns:
        # determine the subject and trial number for column

        # determine subject and trial number of current data
        number = int(col[10:12])
        subject = int(col[2:4])

        # turn column of df into a list
        cop_sig = x_cop[col].to_list()
        com_sig = x_com[col].to_list()

        # create time vector for correlation signal
        n_com = len(com_sig)
        t_corr = np.arange(-n_com / fs_com, n_com / fs_com - 1 / fs_com, 1 / fs_com)

        # resample COP data to match fs of COM
        cop_sig_resample = []
        for i in range(0, n_com):
            cop_sig_resample.append(cop_sig[i * 10])

        # calculate pearson's correlation coefficient between COP and COM
        corr, _ = pearsonr(cop_sig_resample, com_sig)

        # calculate autocorrelation of COP
        auto_corr = np.correlate(cop_sig_resample, cop_sig_resample, mode="full")

        # calculate cross correlation between COP and COM
        cross_corr = np.correlate(cop_sig_resample, com_sig, mode="full")

        # split up pearson's coefficients by condition
        if number < 7:
            eoft.append(corr)
        elif 6 < number < 12:
            ecft.append(corr)
        elif 11 < number < 17:
            eoftandb.append(corr)
        elif 16 < number < 22:
            ecftandb.append(corr)
        elif 21 < number < 27:
            eoftandf.append(corr)
        elif 26 < number < 32:
            ecftandf.append(corr)

        # set values to keys of each condition in dictionary for pearsons
        corr_dict["EOFT"] = eoft
        corr_dict["ECFT"] = ecft
        corr_dict["EOFTanDB"] = eoftandb
        corr_dict["ECFTanDB"] = ecftandb
        corr_dict["EOFTanDF"] = eoftandf
        corr_dict["ECFTanDF"] = ecftandf

        # plot auto and cross correlations of one EOFT trial
        if subject == 1 and number == 2:
            plt.figure()
            plt.subplot(121)
            plt.plot(t_corr, auto_corr, color="r")
            plt.plot(t_corr, cross_corr)
            plt.title(
                "Autocorrelation of COP and Cross-Correlation of COP/COM\nin AP direction for EOFT trial"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Magnitude")
            plt.legend(["Auto", "Cross"])

    # save pearson's coefficients in dataframe format to a .csv file
    corr_df = pd.DataFrame(corr_dict, columns=corr_dict.keys())
    corr_df.to_csv("x_correlations.csv")

    # repeat above for the y-axis data
    corr_dict = {}
    eoft = []
    ecft = []
    eoftandb = []
    eoftandf = []
    ecftandb = []
    ecftandf = []

    for col in y_cop.columns:

        number = int(col[10:12])
        subject = int(col[2:4])

        cop_sig = y_cop[col].to_list()
        com_sig = y_com[col].to_list()

        n_com = len(com_sig)
        t_corr = np.arange(-n_com / fs_com, n_com / fs_com - 1 / fs_com, 1 / fs_com)

        cop_sig_resample = []
        for i in range(0, n_com):
            cop_sig_resample.append(cop_sig[i * 10])

        corr, _ = pearsonr(cop_sig_resample, com_sig)
        auto_corr = np.correlate(cop_sig_resample, cop_sig_resample, mode="full")
        cross_corr = np.correlate(cop_sig_resample, com_sig, mode="full")

        if number < 7:
            eoft.append(corr)
        elif 6 < number < 12:
            ecft.append(corr)
        elif 11 < number < 17:
            eoftandb.append(corr)
        elif 16 < number < 22:
            ecftandb.append(corr)
        elif 21 < number < 27:
            eoftandf.append(corr)
        elif 26 < number < 32:
            ecftandf.append(corr)

        corr_dict["EOFT"] = eoft
        corr_dict["ECFT"] = ecft
        corr_dict["EOFTanDB"] = eoftandb
        corr_dict["ECFTanDB"] = ecftandb
        corr_dict["EOFTanDF"] = eoftandf
        corr_dict["ECFTanDF"] = ecftandf

        if subject == 1 and number == 2:
            plt.subplot(122)
            plt.plot(t_corr, auto_corr, color="r")
            plt.plot(t_corr, cross_corr)
            plt.title(
                "Autocorrelation of COP and Cross-Correlation of COP/COM\nin ML direction for EOFT trial"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Magnitude")
            plt.legend(["Auto", "Cross"])
            plt.show()

    corr_df = pd.DataFrame(corr_dict, columns=corr_dict.keys())
    corr_df.to_csv("y_correlations.csv")

