import numpy as np

a = np.linspace(0, 4, 5)
print(a)
idx = a > 2
print(idx)
print(not idx)
b = a[a > 2]
print(b)