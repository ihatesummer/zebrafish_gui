# %%
import matplotlib.pyplot as plt
from numpy import loadtxt, shape, set_printoptions
from os.path import join
from os import getcwd

set_printoptions(precision=1)

vid_name = input("video name: ")
DEST_PATH = join(getcwd(), "image", vid_name+"_processed")
data_frame = loadtxt(join(DEST_PATH, "result.csv"), delimiter=',')
data_frame = data_frame[data_frame[:, 0].argsort()]  # Sort by frame number
frame_no = data_frame[:, 0]
angle_eyeL = data_frame[:, -2]
angle_eyeR = data_frame[:, -1]
angle_body = data_frame[:, -3]

fig, ax = plt.subplots()
plt.plot(frame_no, angle_eyeL - angle_body,
         label="left eye w.r.t body", linewidth=0.5)
plt.plot(frame_no, angle_eyeR - angle_body,
         label="right eye w.r.t body", linewidth=0.5)
ax.set_xlim(xmin=0, xmax=max(frame_no))
ax.set_ylim(ymin=-20, ymax=50)
ax.grid()
plt.xlabel("frame")
plt.ylabel("angle (degree)")
plt.legend(loc="upper right")
plt.savefig(join(DEST_PATH, "result.png"))
