import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch

def moving_average(x, w):
    m_avg = np.convolve(x, np.ones(w), 'valid') / w
    # duplicate the last element
    # in order to fit the original length
    m_avg_extended = np.append(m_avg, np.ones(w-1)*m_avg[-1])
    return m_avg_extended

def get_angVel(angle_list, deltaTime):
    angVel_list = angle_list.copy()
    angVel_list[0] = 0
    for i in range(1, len(angVel_list)):
        angVel_list[i] = \
            (angle_list[i] - angle_list[i-1]) / \
                deltaTime
    return angVel_list

file = os.path.join(os.getcwd(), r"results\B13WT_1_c1m_30hz_js_5rpm_70mm-1.avi\result.csv")
df = np.loadtxt(file, delimiter=',')
df = df[df[:, 0].argsort()]

fps = 30
w = 15
c_time = df[:, 1]
c_angle_L = moving_average(df[:, 5], w)
c_angle_R = moving_average(df[:, 7], w)
c_angVel_L = get_angVel(c_angle_L, 1/fps)
c_angVel_R = get_angVel(c_angle_R, 1/fps)

# t = np.linspace(0,50,fps*50)
# cos = np.cos(t)
# _, ax = plt.subplots()
# plt.plot(t, cos)
# plt.show()

# _, ax = plt.subplots()
# plt.semilogy(rfftfreq(len(t), 1/fps), rfft(cos))
# plt.show()

# _, ax = plt.subplots()
# f, Pxx = welch(cos, fps)
# plt.semilogy(t, cos)
# plt.show()

_, ax = plt.subplots()
plt.plot(c_time, c_angVel_L)
plt.plot(c_time, c_angVel_R)
plt.title("time domain, angular velocity [deg/sec]")
plt.show()

nSamples = len(c_time)
sample_interval = 1/fps
x = rfftfreq(nSamples, sample_interval)
y1 = abs(rfft(c_angVel_L))
y2 = abs(rfft(c_angVel_R))
_, ax = plt.subplots()
plt.semilogy(x, y1)
plt.semilogy(x, y2)
plt.title("rfft")
plt.show()

f, Pxx = welch(c_angVel_L, fps)
f2, Pxx2 = welch(c_angVel_R, fps)
_, ax = plt.subplots()
plt.semilogy(f, Pxx)
plt.semilogy(f2, Pxx2)
plt.title("Welch")
plt.show()
