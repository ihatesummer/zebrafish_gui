import numpy as np
import os
import matplotlib.pyplot as plt


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


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
c_angVel_wrtB_L = df[:, 7]

for w in [1, 5, 15, 100]:
    _, ax = plt.subplots()
    ax.set_xlim(xmin=0, xmax=17)
    angles = moving_average(c_angle_wrtB_L[:500], w)
    plt.plot(c_time[:500-w+1], angles)
    plt.savefig(f"window={w}_angle.png")

    _, ax = plt.subplots()
    ax.set_xlim(xmin=0, xmax=17)
    angVels = get_angVel(angles, 1/30)
    plt.plot(c_time[:500-w+1], angVels)
    plt.savefig(f"window={w}_angvel.png")

    _, ax = plt.subplots()
    ax.set_xlim(xmin=0, xmax=17)
    angVels = moving_average(c_angVel_wrtB_L[:500], w)
    plt.plot(c_time[:500-w+1], angVels)
    plt.savefig(f"window={w}_angvel_direct.png")