import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.signal import find_peaks, welch, butter, lfilter
np.set_printoptions(precision=2)


def main():
    data_file = r"results\test_vor.m4v\result.csv"
    data_frame = np.loadtxt(data_file, delimiter=',')

    # Sort by frame number
    data_frame = data_frame[data_frame[:, 0].argsort()]
    # Load each column
    c_frame_no = data_frame[:, 0]
    c_time = data_frame[:, 1]
    c_bDetected = data_frame[:, 2]
    c_angle_B = interpolate(data_frame[:, 3], c_frame_no, c_bDetected)
    c_angle_L = interpolate(data_frame[:, 4], c_frame_no, c_bDetected)
    c_angle_wrtB_L = interpolate(data_frame[:, 5], c_frame_no, c_bDetected)
    c_angle_R = interpolate(data_frame[:, 6], c_frame_no, c_bDetected)
    c_angle_wrtB_R = interpolate(data_frame[:, 7], c_frame_no, c_bDetected)
    c_area_norm_L = interpolate(data_frame[:, 8], c_frame_no, c_bDetected)
    c_area_norm_R = interpolate(data_frame[:, 9], c_frame_no, c_bDetected)
    c_ax_ratio_L = interpolate(data_frame[:, 10], c_frame_no, c_bDetected)
    c_ax_ratio_R = interpolate(data_frame[:, 11], c_frame_no, c_bDetected)
    fps = 1 / (c_time[1] - c_time[0])


    norm_area_L = moving_average(c_area_norm_L, 15)
    norm_area_R = moving_average(c_area_norm_R, 15)

    idx_start = int(10*fps)
    idx_end = int(40*fps)
    norm_area_L = norm_area_L[idx_start:idx_end]
    norm_area_R = norm_area_R[idx_start:idx_end]
    c_time = c_time[idx_start:idx_end]

    duration = c_time[-1] - c_time[0]
    
    f_L, fft_L = welch(norm_area_L, fs=fps)
    f_R, fft_R = welch(norm_area_R, fs=fps)
    normalizer_L = integrate.cumtrapz(fft_L, f_L)[-1]
    normalizer_R = integrate.cumtrapz(fft_R, f_R)[-1]
    print(normalizer_L)
    print(normalizer_R)

    order = 6
    cutoff = 1
    b, a = butter(order, cutoff, fs=fps, btype='low', analog=False)
    norm_area_L_LPF = lfilter(b, a, norm_area_L)
    norm_area_R_LPF = lfilter(b, a, norm_area_R)

    f_L, fft_L_LPF = welch(norm_area_L_LPF, fs=fps)
    f_R, fft_R_LPF = welch(norm_area_R_LPF, fs=fps)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout()
    axes[0, 0].plot(c_time, norm_area_L, "black", label="original", linewidth=1)
    axes[1, 0].plot(c_time, norm_area_R, "black", label="original", linewidth=1)
    axes[0, 0].plot(c_time, norm_area_L_LPF, "red", alpha=0.5, label="LPF", linewidth=2)
    axes[1, 0].plot(c_time, norm_area_R_LPF, "red", alpha=0.5, label="LPF", linewidth=2)
    axes[0, 0].set_xlim(10, 40)
    axes[1, 0].set_xlim(10, 40)
    axes[0, 0].grid()
    axes[1, 0].grid()
    axes[0, 0].legend()
    axes[1, 0].legend()
    axes[0, 0].set_title("Time domain (L)")
    axes[1, 0].set_title("Time domain (R)")

    axes[0, 1].plot(f_L, fft_L, "black", label="original", linewidth=1)
    axes[1, 1].plot(f_R, fft_R, "black", label="original", linewidth=1)
    axes[0, 1].plot(f_L, fft_L_LPF, "red", alpha=0.5, label="LPF", linewidth=2)
    axes[1, 1].plot(f_R, fft_R_LPF, "red", alpha=0.5, label="LPF", linewidth=2)
    axes[0, 1].set_xlim(0, 3)
    axes[1, 1].set_xlim(0, 3)
    axes[0, 1].grid()
    axes[1, 1].grid()
    axes[0, 1].minorticks_on()
    axes[1, 1].minorticks_on()
    axes[0, 1].legend()
    axes[1, 1].legend()
    axes[0, 1].set_title("Frequency domain (L)")
    axes[1, 1].set_title("Frequency domain (R)")

    plt.show()


def moving_average(x, window_size):
    w = window_size
    moving_avg = np.convolve(x, np.ones(w), 'valid') / w
    # duplicate the last element
    # in order to fit the original length
    m_avg_extended = np.append(moving_avg,
                               np.ones(w-1)*moving_avg[-1])
    return m_avg_extended


def get_first_available(y_part, bDetected_part):
    if bDetected_part[0] == True:
        return y_part[0]
    else:
        # Recursion until available
        return get_first_available(
            y_part[1:], bDetected_part[1:])

def get_last_available(y_part, bDetected_part):
    if bDetected_part[-1] == True:
        return y_part[-1]
    else:
        # Recursion until available
        return get_last_available(
            y_part[:-1], bDetected_part[:-1])


def interpolate(y, c_frame_no, c_bDetected):
    idx_last = int(max(c_frame_no))
    if c_bDetected[0] == False:
        y[0] = get_first_available(y[1:], c_bDetected[1:])
    if c_bDetected[idx_last] == False:
        y[idx_last] = get_last_available(y[:-1], c_bDetected[:-1])
    for i in range (1, idx_last-1):
        if c_bDetected[i] == False:
            if len(y[:i]) == 1:
                prev = y[:i]
            else:
                prev = get_last_available(y[:i], c_bDetected[:i])
            if len(y[i+1:]) == 1:
                next = y[i+1:]
            else:
                next = get_first_available(y[i+1:], c_bDetected[i+1:])
            y[i] = (prev + next) / 2
        else:
            pass
    return y


def get_angVel(angle_list, fps):
    deltaTime = 1/fps
    angVel_list = angle_list.copy()
    for i in range(1, len(angVel_list)):
        angVel_list[i] = \
            (angle_list[i] - angle_list[i-1]) / deltaTime
    angVel_list[0] = angVel_list[1]
    return angVel_list


def get_lowPeak_idx(y, fps):
    prominence = 10
    peaks, _ = find_peaks(-y, prominence, distance=fps)
    return peaks


def get_slowPhase_maxima(angVel, low_peaks):
    max_indices = []
    start_frame = 0
    for i in range(len(low_peaks)):
        target_window = angVel[start_frame: low_peaks[i]]
        max_indices.append(np.argmax(target_window) + start_frame)
        start_frame = low_peaks[i] + 1
    # one remaining window after the last peak
    target_window = angVel[start_frame:]
    max_indices.append(np.argmax(target_window) + start_frame)
    return max_indices


if __name__ == '__main__':
    main()
