import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.fft import rfft, rfftfreq


def moving_average(x, w):
    m_avg = np.convolve(x, np.ones(w), 'valid') / w
    # duplicate the last element
    # in order to fit the original length
    m_avg_extended = np.append(m_avg, np.ones(w-1)*m_avg[-1])
    return m_avg_extended


file = os.path.join(os.getcwd(), r"results\vor.avi\result.csv")
df = np.loadtxt(file, delimiter=',')
df = df[df[:, 0].argsort()]

fps = 60
w = 30
c_time = df[:, 1]
c_normArea_L = moving_average(df[:, 8], w)

_, ax = plt.subplots()
plt.plot(c_time, c_normArea_L)
plt.show()

nSamples = len(c_time)
sample_interval = 1/fps
x = rfftfreq(nSamples, sample_interval)
y = abs(
    rfft(moving_average(c_normArea_L, w)))
_, ax = plt.subplots()
plt.plot(x, y)
plt.show()
print(len(x), len(y))
