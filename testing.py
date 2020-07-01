import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import analysis_lib as an

# ######## auto/cross correlation #########

# # time = np.arange(0, 10, 0.1)
# # time_corr


# def ApEn(U, m, r) -> float:
#     """Approximate_entropy."""

#     def _maxdist(x_i, x_j):
#         return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

#     def _phi(m):
#         x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
#         C = [
#             len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
#             for x_i in x
#         ]
#         return (N - m + 1.0) ** (-1) * sum(np.log(C))

#     N = len(U)

#     return abs(_phi(m + 1) - _phi(m))


fx = 5
fy = 2
fs = 100
t = np.arange(0, 3, 1 / fs)
y = np.sin(2 * np.pi * fy * t) + np.cos(2 * np.pi * fx * t)
x = np.cos(2 * np.pi * fx * t)

N = len(y)
t_corr = np.arange(-N / fs, N / fs - 1 / fs, 1 / fs)

plt.figure()
plt.plot(t, y)
plt.show()

auto = np.correlate(y, y, mode="full")
cross = np.correlate(y, x, mode="full")

# plt.figure()
# plt.plot(t_corr, auto)
# plt.xlabel("time")
# plt.ylabel("autocorrelation")
# plt.show()

# plt.figure()
# plt.plot(t_corr, cross)
# plt.xlabel("time")
# plt.ylabel("cross correlation")
# plt.show()


# ############### auto power ####################

N = len(auto)
# # freq = np.arange(0, (N - 1) * (fs / N), fs / N)
freq = np.arange(0, fs, fs / N)


# create windows
rect_win = np.ones(N)
ham_win = np.hamming(N)

# calculate PSD with both windows
Sxx_rect = np.fft.fft(np.multiply(auto, rect_win))
Sxx_ham = np.fft.fft(np.multiply(auto, ham_win))

plt.figure()
plt.plot(freq, abs(Sxx_rect))
plt.xlabel("frequency (Hz)")
plt.ylabel("autopower")
plt.title("PSD of 2 Hz sine wave plus 5 Hz cosine wave")
plt.xlim(0, 10)
plt.show()

# ############### cross power ####################

# Sxy_rect = np.fft.fft(np.multiply(cross, rect_win))
# Sxy_ham = np.fft.fft(np.multiply(cross, ham_win))

# plt.figure()
# plt.plot(freq, abs(Sxy_rect))
# plt.xlabel("frequency")
# plt.ylabel("crosspower")
# plt.xlim(0, 10)
# plt.show()


# # Usage example
# U = np.array([85, 80, 89] * 17)
# print(ApEn(U, 2, 3))
# # 1.0996541105257052e-05

# randU = np.random.choice([85, 80, 89], size=17 * 3)
# print(ApEn(randU, 2, 3))
# # 0.8626664154888908

############### combining ####################
# reads data in from .csv files
# user specifies:
#   header = which row the data starts on (starting at 0, not 1 like the sheet)
#   usecols = names of the column headers that are to be included
#   nrows = number of data points to be read in
#   rows_skip = the number of any rows to not be included (starting at 0)

# data = pd.read_csv(
#     "/Users/natalietipton/Code/Data/SB01/SB01Trial_19.csv",
#     header=3,
#     usecols=["Fz", "Cx", "Cy", "Fz.1", "Cx.1", "Cy.1",],
#     nrows=36000,
#     dtype={
#         "Fz": np.float64,
#         "Cx": np.float64,
#         "Cy": np.float64,
#         "Fz.1": np.float64,
#         "Cx.1": np.float64,
#         "Cy.1": np.float64,
#     },
#     skiprows=[4],
# )

# # convert data frame into lists
# fz = data["Fz"].values.tolist()
# cx = data["Cx"].values.tolist()
# cy = data["Cy"].values.tolist()
# fz_1 = data["Fz.1"].values.tolist()
# cx_1 = data["Cx.1"].values.tolist()
# cy_1 = data["Cy.1"].values.tolist()


# print(cx_combined[0])
# print(cy_combined[0])

######### ap en examples ########

# fx = 5
# fs = 100
# t = np.arange(0, 3, 1 / fs)
# sine = np.sin(2 * np.pi * fx * t)
# noise = np.random.random(size=len(t))

# ent_sin = an.ApEn(sine, 2, 1)
# ent_noise = an.ApEn(noise, 2, 1)

# an.plot(
#     t, sine, "time (s)", "Magnitude", f"Sine Wave, Entropy = {ent_sin}", None, None,
# )

# an.plot(
#     t,
#     noise,
#     "time (s)",
#     "Magnitude",
#     f"White Noise, Entropy = {ent_noise}",
#     None,
#     None,
# )

