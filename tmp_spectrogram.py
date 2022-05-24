import numpy as np
import matplotlib.pyplot as plt

def main():
    data_file = r"results\2021-12-20_CON 1-1.avi\result.csv"
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

    angle = moving_average(c_angle_wrtB_L, 15)
    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout()
    axes[0].plot(c_time, angle)
    axes[0].set_xlim(0, max(c_time))
    axes[1].specgram(angle, Fs=fps, cmap='inferno')
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
            y_part[1:], bDetected_part[1:]
        )

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


if __name__ == '__main__':
    main()
