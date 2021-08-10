# %%
import matplotlib.pyplot as plt
from numpy import max, min, ndarray
from os.path import join


def generate_blank(pathname):
    _, ax = plt.subplots()
    ax.set_xlim(xmin=0, xmax=1)
    ax.set_ylim(ymin=-1, ymax=1)
    ax.grid()
    plt.xlabel("Select...")
    plt.ylabel("Select...")
    plt.savefig(pathname)

def main(output, x, x_label, x_range,
         y_list, y_label, y_range,
         custom_grid, custom_label,
         graph_title, bDetected):
    _, ax = plt.subplots()
    if len(y_list) == 2:
        labels = ["Left", "Right"]
        colors = ["r", "g"]
        for i, y_arr in enumerate(y_list):
            ax.plot(x, y_arr,
                    color=colors[i],
                    label=labels[i],
                    linewidth=0.5)
            ax.legend()
    elif len(y_list) == 1:
        ax.plot(x, y_list[0], 'black', linewidth=1)

    if x_range == [0, 0]:
        ax.set_xlim(xmin=0, xmax=max(x))
    else:
        ax.set_xlim(xmin=x_range[0], xmax=x_range[1])

    if y_range == [0, 0]:
        ax.set_ylim(ymin=min(y_list)-10, ymax=max(y_list)+10)
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
