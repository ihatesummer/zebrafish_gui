import numpy as np
from os.path import exists
from os import remove
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, rfft, rfftfreq
data_frame = np.loadtxt(
    r"C:\fish_gui\results\B13WT_1_c1m_30hz_js_5rpm_70mm-1.avi\result.csv",
    delimiter=',')
# Sort by frame number
data_frame = data_frame[
    data_frame[:, 0].argsort()]
# Load each column
c_frame_no = data_frame[:, 0]
c_time = data_frame[:, 1]
c_bDetected = data_frame[:, 2]
c_angle_B = data_frame[:, 3]
c_angle_L = data_frame[:, 4]
c_angle_wrtB_L = data_frame[:, 5]
c_angVel_L = data_frame[:, 6]
c_angVel_wrtB_L = data_frame[:, 7]
c_angle_R = data_frame[:, 8]
c_angle_wrtB_R = data_frame[:, 9]
c_angVel_R = data_frame[:, 10]
c_angVel_wrtB_R = data_frame[:, 11]

nSamples = int(max(c_frame_no))
print("nSamples:", nSamples)
sample_interval = c_time[1]
print("sample interval:", sample_interval)
freq = rfftfreq(nSamples, sample_interval)
print("freq:", freq)

wave = c_angle_wrtB_L
norm_wave = (wave / len(wave))
yf = rfft(norm_wave)

plt.plot(freq, np.abs(yf), 'brown')
plt.show()