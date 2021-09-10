import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths


def moving_average(x, w):
    m_avg = np.convolve(x, np.ones(w), 'valid') / w
    # duplicate the last element
    # in order to fit the original length
    m_avg_extended = np.append(m_avg, np.ones(w-1)*m_avg[-1])
    return m_avg_extended


def get_angVel(angle_list, deltaTime):
    angVel_list = angle_list.copy()
    for i in range(1, len(angVel_list)):
        angVel_list[i] = \
            (angle_list[i] - angle_list[i-1]) / deltaTime
    angVel_list[0] = angVel_list[1]
    return angVel_list


file = os.path.join(os.getcwd(), r"results\B13WT_1_c1m_30hz_js_5rpm_70mm-1.avi\result.csv")
df = np.loadtxt(file, delimiter=',')
df = df[df[:, 0].argsort()]

c_frame_no = df[:, 0]
c_time = df[:, 1]
c_bDetected = df[:, 2]
c_angle_wrtB_L = df[:, 5]

fps = 30
w = 15
t = c_time[:1000]
_, ax = plt.subplots()
ax.set_xlim(xmin=0, xmax=17)
angles = moving_average(c_angle_wrtB_L[:1000], w)
angVels = get_angVel(angles, 1/30)
plt.plot(t, angVels, '-o', markersize=3)
peaks, details = find_peaks(-angVels, prominence=15)
print(t[peaks])
print(details['prominences'])

plt.plot(t[peaks], angVels[peaks], "x")
plt.show()

idx_bNonPeak = np.ones(1000, dtype=bool)  # Initialize as all true
wl = 0.8
wr = 0.5
for i in range (-int(wl*fps), int(wr*fps)):
    idx_bNonPeak[peaks-i] = False
_, ax = plt.subplots()
ax.set_xlim(xmin=0, xmax=17)
print(idx_bNonPeak)
plt.plot(t[idx_bNonPeak], angVels[idx_bNonPeak])
plt.show()
plt.savefig(f"window={w}_angvel_direct.png")