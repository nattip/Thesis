# Written by: Natalie Tipton
# Advisor: Dr. Samhita Rhodes
#
# Created on: February 22, 2020
#
# Description: Signal Processing of COM and COP
#   during stability testing in neuronormative
#   subjects.
#
# Last updated: February 22, 2020

#   import packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# constants
fs = 1200
t_tot = 30

# read in data
data = pd.read_csv(
    r"/Users/natalietipton/Code/Thesis_Data/SB01/SB01Trial_02.csv",  # file path
    header=3,  # data starts in row 3
    usecols=["Cx", "Cy", "Cz"],  # columns we want
    nrows=36000,  # read 30 seconds of data points
    dtype={
        "Cx": np.float64,
        "Cy": np.float64,
        "Cz": np.float64,
    },  # specify data type of columns
    skiprows=[4],  # skip row between column titles and data points
)

# convert data frame into lists
cx = data["Cx"].values.tolist()
cy = data["Cy"].values.tolist()
cz = data["Cz"].values.tolist()

# plot x, y plot of COP
plt.figure()
plt.scatter(cx, cy)
plt.title("X, Y position scatterplot")
plt.xlabel("X position (mm)")
plt.ylabel("Y position (mm)")
plt.show()

# plot x, y, z plot of COP
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(cx, cy, cz)
plt.title("X, Y, Z position scatterplot")
ax.set_xlabel("X position (mm)")
ax.set_ylabel("Y position (mm)")
ax.set_zlabel("Z position (mm)")
plt.show()

# plot x data in time and frequency domain
np.random.seed(19680801)
dt = 1 / fs
t = np.arange(0, t_tot, dt)
nse = np.random.randn(len(t))
r = np.exp(-t / 0.05)

cnse = np.convolve(nse, r) * dt
cnse = cnse[: len(t)]
s = cx

plt.figure()
plt.subplot(211)
plt.title("Cx")
plt.xlabel("Time (s)")
plt.ylabel("X Position (mm)")
plt.plot(t, s)
plt.subplot(212)
plt.psd(s, 512, 1 / dt)
plt.show()

# plot y data in time and frequency domain
s = cy

plt.figure()
plt.subplot(211)
plt.title("Cy")
plt.xlabel("Time (s)")
plt.ylabel("Y Position (mm)")
plt.plot(t, s)
plt.subplot(212)
plt.psd(s, 512, 1 / dt)
plt.show()

# plot z data in time and frequency domain
s = cz

plt.figure()
plt.subplot(211)
plt.title("Cz")
plt.xlabel("Time (s)")
plt.ylabel("Z Position (mm)")
plt.plot(t, s)
plt.subplot(212)
plt.psd(s, 512, 1 / dt)
plt.show()

