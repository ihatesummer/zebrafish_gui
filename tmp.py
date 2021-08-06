import numpy as np
from os.path import exists
from os import remove
from datetime import datetime
import matplotlib.pyplot as plt

grid = [None, None]

grid[0] = np.linspace(0,1,6)
# grid[1] = np.linspace(0.2, 0.6, 5)

xgrid, ygrid = grid
print(grid)
print(type(ygrid) == np.ndarray)
print(all(type(g) != np.ndarray for g in grid))

# _, ax = plt.subplots()
# ax.set_xlim(xmin=0, xmax=1)
# ax.set_ylim(ymin=0, ymax=1)
# ax.set_xticks(grid[0])
# # ax.set_yticks(grid[1])
# # ax.xaxis.grid(True)
# ax.yaxis.grid(True)
# plt.xlabel("Select...")
# plt.ylabel("Select...")
# plt.show()
