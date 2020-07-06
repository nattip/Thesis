# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: July 5, 2020
#
# Description: Signal Processing of COM
#   and COP to obtain auto and cross
#   correlation for comparison.
#
# Last updated: July 5, 2020

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
from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.stats import pearsonr

# constants
fs_com = 120
fs_cop = 1200
t_com = np.arange(0, 30, 1 / fs_com)
t_cop = np.arange(0, 30, 1 / fs_cop)

corr_dict = {}
eoft = []
ecft = []
eoftandb = []
eoftandf = []
ecftandb = []
ecftandf = []
#####################################################################################

if __name__ == "__main__":

    x_cop = pd.read_csv("x_cop.csv", index_col=False)
    y_cop = pd.read_csv("y_cop.csv", index_col=False)

    x_com = pd.read_csv("x_com.csv", index_col=False)
    y_com = pd.read_csv("y_com.csv", index_col=False)

    # cycle through each column in x file
    for col in x_cop.columns:
        # determine the subject and trial number for column

        number = int(col[10:12])
        subject = int(col[2:4])

        # turn column of df into a list
        cop_sig = x_cop[col].to_list()
        com_sig = x_com[col].to_list()

        n_com = len(com_sig)
        t_corr = np.arange(-n_com / fs_com, n_com / fs_com - 1 / fs_com, 1 / fs_com)

        cop_sig_resample = []
        for i in range(0, n_com):
            cop_sig_resample.append(cop_sig[i * 10])

        corr, _ = pearsonr(cop_sig_resample, com_sig)

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

    corr_df = pd.DataFrame(corr_dict, columns=corr_dict.keys())
    corr_df.to_csv("correlations.csv")

    # auto_corr = np.correlate(cop_sig_resample, cop_sig_resample, mode="full")
    # cross_corr = np.correlate(cop_sig_resample, com_sig, mode="full")

    # auto_corr = an.standardize(auto_corr)
    # cross_corr = an.standardize(cross_corr)

    # diff = np.subtract(auto_corr, cross_corr)

    # # t_stat, p_val = ttest_ind(auto_corr, cross_corr)
    # # print(p_val)

    # an.plot(
    #     t_corr,
    #     diff,
    #     "time (s)",
    #     "diff",
    #     "X CoM and X CoP cross correlation",
    #     [0, 30],
    #     None,
    # )

    # plt.figure()
    # plt.plot(t_corr, auto_corr)
    # plt.plot(t_corr, cross_corr)
    # plt.xlim([0, 30])
    # plt.show()

    # an.plot(
    #     t_corr, auto_corr, "time (s)", "diff", "auto", [0, 30], None,
    # )

    # an.plot(
    #     t_corr, cross_corr, "time (s)", "diff", "cross", [0, 30], None,
    # )

