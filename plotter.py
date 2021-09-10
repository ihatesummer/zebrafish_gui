# %%
import matplotlib.pyplot as plt
from numpy import max, min, ndarray
from scipy.signal import find_peaks


def generate_blank(pathname):
    _, ax = plt.subplots()
    ax.set_xlim(xmin=0, xmax=1)
    ax.set_ylim(ymin=-1, ymax=1)
    ax.grid()
    plt.xlabel("Select...")
    plt.ylabel("Select...")
    plt.savefig(pathname)


def peak_preview(output, x, x_label, x_range,
                 y_list, y_label, y_range,
                 lr_selected,
                 custom_grid,
                 custom_label,
                 custom_eye_label,
                 custom_colors,
                 graph_title,
                 peak_prominence):
    frame_duration = x[1] - x[0]
    fps = 1/frame_duration
    _, ax = plt.subplots()
    if len(y_list) == 2:
        for i, y_arr in enumerate(y_list):
            ax.plot(x, y_arr,
                    color=custom_colors[i],
                    label=custom_eye_label[i],
                    linewidth=1)
            peaks, _ = find_peaks(-y_arr, peak_prominence, distance=fps)
            ax.legend(loc="upper right")
            ax.plot(x[peaks], y_arr[peaks], "x", markersize=10, color=custom_colors[i])
            print(f"Peak occurrences: {x[peaks]} second")
    elif len(y_list) == 1:
        y = y_list[0]
        if lr_selected[0] == "left":
            my_color = custom_colors[0]
        if lr_selected[0] == "right":
            my_color = custom_colors[1]
        ax.plot(x, y,
                color=my_color,
                linewidth=1)
        peaks, _ = find_peaks(-y, peak_prominence, distance=fps)
        ax.legend(loc="upper right")
        ax.plot(x[peaks], y[peaks], "x", markersize=10, color=my_color)
        print(f"Peak occurrences: {x[peaks]} second")
        ax.plot(x, y_list[0], my_color, linewidth=1)
    else:
        print("ERROR: Wrong input for y.")
        pass

    if x_range == [0, 0]:
        ax.set_xlim(xmin=min(x), xmax=max(x))
    else:
        ax.set_xlim(xmin=x_range[0], xmax=x_range[1])

    if y_range == [0, 0]:
        ax.set_ylim(ymin=min(y_list), ymax=max(y_list))
    else:
        ax.set_ylim(ymin=y_range[0], ymax=y_range[1])

    xgrid, ygrid = custom_grid
    if all(type(g) != ndarray for g in custom_grid):
        ax.grid()
    else:
        if type(xgrid) == ndarray:
            ax.set_xticks(xgrid)
            ax.xaxis.grid(True)
        if type(ygrid) == ndarray:
            ax.set_yticks(ygrid)
            ax.yaxis.grid(True)

    if custom_label[0] != "":
        ax.set_xlabel(custom_label[0])
    else:
        ax.set_xlabel(x_label)
    if custom_label[1] != "":
        ax.set_ylabel(custom_label[1])
    else:
        ax.set_ylabel(y_label)

    if graph_title != "":
        ax.set_title(graph_title)
    plt.savefig(output)  # png
    plt.savefig(output[:-4]+".pdf")  # pdf
    plt.close('all')

def main(output, x, x_label, x_range,
         y_list, y_label, y_range,
         lr_selected,
         custom_grid,
         custom_label,
         custom_eye_label,
         custom_colors,
         graph_title, xAxis):
    _, ax = plt.subplots()

    if len(y_list) == 2:
        if xAxis == "freq":
            for i, y_arr in enumerate(y_list):
                ax.semilogy(x, y_arr,
                        color=custom_colors[i],
                        label=custom_eye_label[i],
                        linewidth=1)
                ax.legend(loc="upper right")
        else:
            for i, y_arr in enumerate(y_list):
                ax.plot(x, y_arr,
                        color=custom_colors[i],
                        label=custom_eye_label[i],
                        linewidth=1)
                ax.legend(loc="upper right")
    elif len(y_list) == 1:
        if lr_selected[0] == "left":
            my_color = custom_colors[0]
        if lr_selected[0] == "right":
            my_color = custom_colors[1]
        if xAxis == "freq":
            ax.semilogy(x, y_list[0], my_color, linewidth=1)
        else:
            ax.plot(x, y_list[0], my_color, linewidth=1)

    else:
        print("ERROR: Wrong input for y.")
        pass

    if x_range == [0, 0]:
        ax.set_xlim(xmin=min(x), xmax=max(x))
    else:
        ax.set_xlim(xmin=x_range[0], xmax=x_range[1])

    if y_range == [0, 0]:
        ax.set_ylim(ymin=min(y_list), ymax=max(y_list))
    else:
        ax.set_ylim(ymin=y_range[0], ymax=y_range[1])

    xgrid, ygrid = custom_grid
    if all(type(g) != ndarray for g in custom_grid):
        ax.grid()
    else:
        if type(xgrid) == ndarray:
            ax.set_xticks(xgrid)
            ax.xaxis.grid(True)
        if type(ygrid) == ndarray:
            ax.set_yticks(ygrid)
            ax.yaxis.grid(True)

    if custom_label[0] != "":
        ax.set_xlabel(custom_label[0])
    else:
        ax.set_xlabel(x_label)
    if custom_label[1] != "":
        ax.set_ylabel(custom_label[1])
    else:
        ax.set_ylabel(y_label)

    if graph_title != "":
        ax.set_title(graph_title)
    plt.savefig(output)  # png
    plt.savefig(output[:-4]+".pdf")  # pdf
    plt.close('all')

if __name__ == "__main__":
    print("WARNING: this is not the main module.")
