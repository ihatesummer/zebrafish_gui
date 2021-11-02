import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch, square

def compare_domains(t, y, FPS):
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    axes[0].plot(t, y)
    axes[1].semilogy(rfftfreq(len(t), 1/FPS), rfft(y))
    f, Pxx = welch(y, FPS)
    axes[2].semilogy(f, Pxx)
    axes[0].set_title("time domain")
    axes[1].set_title("RFFT")
    axes[2].set_title("Welch")
    plt.show()


def generate_neg_impulse_train(t, delta_interval):
    neg_impulse_train = np.zeros(shape=np.shape(t))
    for i in range(0, len(t)):
        time = i/FPS
        if time%delta_interval == 0:
            neg_impulse_train[i] = -1
    return neg_impulse_train

def generate_rect(t, period, duty_cycle):
    rect = np.zeros(shape=np.shape(t))
    bRise = True
    next_stop = period*duty_cycle
    for i in range(0, len(t)):
        time = i/FPS
        if time > next_stop:
            bRise = not bRise
            duty_cycle = 1 - duty_cycle
            next_stop += period*duty_cycle
        if bRise:
            rect[i] = 1
        else:
            rect[i] = 0
    return rect

FPS = 30
start_time = 0
end_time = 30
t = np.linspace(start_time, end_time, end_time*FPS)

# y = np.cos(t)
# compare_domains(t, y, FPS)

# neg_impulse_train = generate_neg_impulse_train(t, 2)
# compare_domains(t, neg_impulse_train, FPS)

rect = generate_rect(t, 2, 0.5)
compare_domains(t, rect, FPS)
