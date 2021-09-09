import numpy as np

w = 5
a = np.linspace(0, 4, 5)
print(a)
b = np.append(a, np.ones(w-1)*a[-1])
print(b)