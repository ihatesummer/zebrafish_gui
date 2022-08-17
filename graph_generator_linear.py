import matplotlib.pyplot as plt
from numpy import argmax, max, min, ndarray, mean
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


def show_spv(output, time, x_label, x_range,
             y_list, y_label, y_range,
             lr_selected,
             custom_grid,
             custom_label,
             custom_eye_label,
             custom_colors,
             graph_title, idxs):
    _, ax = plt.subplots()

    if len(y_list) == 2:
        for i, y_arr in enumerate(y_list):
            idx = idxs[i]
            x = time[idx]
            y = y_arr[idx]
            ax.plot(x, y,
                    color=custom_colors[i],
                    label=custom_eye_label[i],
                    linewidth=1)
            ax.legend(loc="upper right")
    elif len(y_list) == 1:
        if lr_selected[0] == "left":
            my_color = custom_colors[0]
        if lr_selected[0] == "right":
            my_color = custom_colors[1]
        idx = idxs[0]
        x = time[idx]
        y = y_list[0]
        y = y[idx]
        ax.plot(x, y, my_color, linewidth=1)

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

def show_bpm(output, bpms, eye_selection):
    if len(eye_selection) == 2 and eye_selection[0] == 'right':
        eye_selection = ['left', 'right']
    _, ax = plt.subplots()
    pos_x = 0.25
    pos_y = 0.5
    for i, bpm in enumerate(bpms):
        bpm_string = f"Beats ({eye_selection[i]}): {bpm:.4f}" 
        ax.text(pos_x, pos_y-i/10, bpm_string, fontsize=15)
    plt.savefig(output)  # png
    plt.savefig(output[:-4]+".pdf")  # pdf
    plt.close('all')


def show_sacc_freq(output,
                   time, x_label, x_range,
                   y_list, y_label, y_range,
                   lr_selected,
                   custom_grid,
                   custom_label,
                   custom_eye_label,
                   custom_colors,
                   graph_title, idxs):

    _, ax = plt.subplots()
    if len(y_list) == 2:
        markers = ['o', 's']
        for i, y in enumerate(y_list):
            idx = idxs[i]
            x = time[idx]
            ax.plot(x, y,
                    color=custom_colors[i],
                    label=custom_eye_label[i],
                    marker = markers[i],
                    linestyle='None',
                    alpha = 0.8)
            ax.legend(loc="upper right")
    elif len(y_list) == 1:
        if lr_selected[0] == "left":
            my_color = custom_colors[0]
        if lr_selected[0] == "right":
            my_color = custom_colors[1]
        idx = idxs[0]
        x = time[idx]
        y = y_list[0]
        ax.plot(x, y, my_color,
                marker='o',
                linestyle='None')
    else:
        print("ERROR: Wrong input for y.")
        pass

    if x_range == [0, 0]:
        ax.set_xlim(0, xmax=max(x)+1)
    else:
        ax.set_xlim(xmin=x_range[0], xmax=x_range[1])

    if y_range == [0, 0]:
        pass
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


def print_maximum(x, y, x_range):
    if x_range != [0, 0]:
        frame_duration = x[1] - x[0]
        fps = 1/frame_duration
        idx_low = int(x_range[0] * fps)
        idx_high = int(x_range[1] * fps)
        x = x[idx_low:idx_high]
        y = y[idx_low:idx_high]
    xmax = x[argmax(y)]
    ymax = y.max()
    print(f"Maximum: {ymax:e}, at {xmax:.3f}")


def main(output, x, x_label, x_range,
         y_list, y_label, y_range,
         lr_selected,
         custom_grid,
         custom_label,
         custom_eye_label,
         custom_colors,
         graph_title, xAxis,
         flip_left, flip_right):
    _, ax = plt.subplots()

    if len(y_list) == 2:
        if xAxis == "freq":
            for i, y_arr in enumerate(y_list):
                ax.plot(x, y_arr,
                        color=custom_colors[i],
                        label=custom_eye_label[i],
                        linewidth=2)
                ax.legend(loc="upper right")
                print_maximum(x, y_arr, x_range)
        else:
            for i, y_arr in enumerate(y_list):
                if i==0 and flip_left:
                    avg = mean(y_arr)
                    y_arr *= -1
                    new_avg = mean(y_arr)
                    y_arr += (avg-new_avg)
                if i==1 and flip_right:
                    avg = mean(y_arr)
                    y_arr *= -1
                    new_avg = mean(y_arr)
                    y_arr += (avg-new_avg)
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
            ax.plot(x, y_list[0],
                    my_color,
                    linewidth=2)
            ax.legend(loc="upper right")
            print_maximum(x, y_list[0], x_range)
        else:
            if (lr_selected[0] == "left" and flip_left) or (lr_selected[0] == "right" and flip_right):
                avg = mean(y_list[0])
                y_list[0] *= -1
                new_avg = mean(y_list[0])
                y_list[0] += (avg-new_avg)
            ax.plot(x, y_list[0],
                    my_color, linewidth=1)

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


def main_separate(output, x, x_label, x_range,
                  y_list, y_label, y_range,
                  lr_selected,
                  custom_grid,
                  custom_label,
                  custom_eye_label,
                  custom_colors,
                  graph_title, xAxis,
                  flip_left, flip_right):
    _, axes = plt.subplots(nrows=2, ncols=1,
                           tight_layout=True)

    if len(y_list) == 2:
        if xAxis == "freq":
            for i, y_arr in enumerate(y_list):
                axes[i].plot(x, y_arr,
                        color=custom_colors[i],
                        label=custom_eye_label[i],
                        linewidth=2)
                print_maximum(x, y_arr, x_range)
        else:
            for i, y_arr in enumerate(y_list):
                axes[i].plot(x, y_arr,
                             color=custom_colors[i],
                             label=custom_eye_label[i],
                             linewidth=1)   
    else:
        print("ERROR: Wrong input for y.")
        pass

    if x_range == [0, 0]:
        axes[0].set_xlim(xmin=min(x), xmax=max(x))
        axes[1].set_xlim(xmin=min(x), xmax=max(x))
    else:
        axes[0].set_xlim(xmin=x_range[0], xmax=x_range[1])
        axes[1].set_xlim(xmin=x_range[0], xmax=x_range[1])

    if y_range == [0, 0]:
        axes[0].set_ylim(ymin=min(y_list), ymax=max(y_list))
        axes[1].set_ylim(ymin=min(y_list), ymax=max(y_list))
    else:
        axes[0].set_ylim(ymin=y_range[0], ymax=y_range[1])
        axes[1].set_ylim(ymin=y_range[0], ymax=y_range[1])

    xgrid, ygrid = custom_grid
    if all(type(g) != ndarray for g in custom_grid):
        axes[0].grid()
        axes[1].grid()
    else:
        if type(xgrid) == ndarray:
            axes[0].set_xticks(xgrid)
            axes[1].set_xticks(xgrid)
            axes[0].xaxis.grid(True)
            axes[1].xaxis.grid(True)
        if type(ygrid) == ndarray:
            axes[0].set_yticks(ygrid)
            axes[1].set_yticks(ygrid)
            axes[0].yaxis.grid(True)
            axes[1].yaxis.grid(True)

    if custom_label[0] != "":
        axes[0].set_xlabel(custom_label[0])
        axes[1].set_xlabel(custom_label[0])
    else:
        axes[0].set_xlabel(x_label)
        axes[1].set_xlabel(x_label)
    if custom_label[1] != "":
        axes[0].set_ylabel(custom_label[1])
        axes[1].set_ylabel(custom_label[1])
    else:
        axes[0].set_ylabel(y_label)
        axes[1].set_ylabel(y_label)

    if graph_title != "":
        axes[0].set_title(graph_title + " (L)")
        axes[1].set_title(graph_title + " (R)")
    plt.savefig(output[:-4]+"_separate.png")  # png
    plt.savefig(output[:-4]+"_separate.pdf")  # pdf
    plt.close('all')


if __name__ == "__main__":
    print("WARNING: this is not the main module.")
