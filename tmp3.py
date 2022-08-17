import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

t = np.linspace(0, np.pi, 1000)
y = np.sin(t)
y_int = integrate.cumtrapz(y, t)[-1]
print(y_int)

plt.plot(t, y)
plt.xlim(0, np.pi)
plt.ylim(0, 1)
plt.show()