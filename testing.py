import matplotlib.pyplot as plt
import numpy as np

######## auto/cross correlation #########

# time = np.arange(0, 10, 0.1)
# time_corr


def ApEn(U, m, r) -> float:
    """Approximate_entropy."""

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))


fx = 5
fy = 2
fs = 100
t = np.arange(0, 3, 1 / fs)
y = np.sin(2 * np.pi * fy * t)
x = np.cos(2 * np.pi * fx * t)

N = len(y)
t_corr = np.arange(-N / fs, N / fs - 1 / fs, 1 / fs)

plt.figure()
plt.plot(t, y)
plt.show()

auto = np.correlate(y, y, mode="full")
cross = np.correlate(y, x, mode="full")

plt.figure()
plt.plot(t_corr, auto)
plt.xlabel("time")
plt.ylabel("autocorrelation")
plt.show()

plt.figure()
plt.plot(t_corr, cross)
plt.xlabel("time")
plt.ylabel("cross correlation")
plt.show()


############### auto power ####################

N = len(auto)
# freq = np.arange(0, (N - 1) * (fs / N), fs / N)
freq = np.arange(0, fs, fs / N)


# create windows
rect_win = np.ones(N)
ham_win = np.hamming(N)

# calculate PSD with both windows
Sxx_rect = np.fft.fft(np.multiply(auto, rect_win))
Sxx_ham = np.fft.fft(np.multiply(auto, ham_win))

plt.figure()
plt.plot(freq, abs(Sxx_rect))
plt.xlabel("frequency")
plt.ylabel("autopower")
plt.xlim(0, 10)
plt.show()

############### cross power ####################

Sxy_rect = np.fft.fft(np.multiply(cross, rect_win))
Sxy_ham = np.fft.fft(np.multiply(cross, ham_win))

plt.figure()
plt.plot(freq, abs(Sxy_rect))
plt.xlabel("frequency")
plt.ylabel("crosspower")
plt.xlim(0, 10)
plt.show()


# Usage example
U = np.array([85, 80, 89] * 17)
print(ApEn(U, 2, 3))
# 1.0996541105257052e-05

randU = np.random.choice([85, 80, 89], size=17 * 3)
print(ApEn(randU, 2, 3))
# 0.8626664154888908
