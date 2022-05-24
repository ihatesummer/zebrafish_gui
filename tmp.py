import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(0, 2*np.pi, 1000)
sig = np.sin(2*np.pi*t**2)
dt = t[1] - t[0]
fps = int(1/dt)

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.tight_layout()
axes[0].plot(t, sig)
axes[0].set_xlim(0, 2*np.pi)
axes[1].specgram(sig, Fs=fps, cmap='inferno')
plt.show()
