import numpy as np
import matplotlib.pyplot as plt

style_dict = {
    # FONT
    'font.size': 14,
    'font.family': 'sans-serif',
    'font.stretch': 'semi-expanded',
    'font.style': 'normal',
    'font.variant': 'normal',
    'font.weight': 'normal',
    'font.sans-serif': ['TeX Gyre Heros'],

    # # AXES
    'axes.linewidth': 1,        # spine thickness
    'axes.xmargin': 0.05,       # spine distance from grid
    'axes.ymargin': 0.05,
    'axes.titlepad': 10.0,
    'axes.labelpad': 5,

    # FIGURE
    'figure.dpi': 100,
    'figure.frameon': False,
    'figure.figsize': [8, 6],

    # LINES
    'lines.linestyle': '-',
    # 'lines.color': '#000000',
    "lines.linewidth": 2,
    "lines.markersize": 6,

    # LEGEND
    # 'legend.framealpha': 1,
    'legend.frameon': False,
    # 'legend.handleheight': 2.5,
    # 'legend.handlelength': 2.5,
    'legend.loc': 'upper right',
    # 'legend.labelspacing': 0.8,
    # 'legend.borderpad': 0.5,
    # 'legend.columnspacing': 2.0,

    # "legend.numpoints": 1,
    # "legend.labelspacing": 0.3,
    # "legend.handlelength": 2,
    # "legend.borderpad": 1.0,

    # SAVEFIG
    'savefig.bbox': 'tight',
    'savefig.dpi': 'figure',
    'savefig.edgecolor': 'white',
    'savefig.format': 'png',
    "savefig.transparent": False,

    # X-TICK
    'xtick.alignment': 'center',
    'xtick.direction': 'in',
    'xtick.bottom': True,
    'xtick.top': True,
    'xtick.labelbottom': True,
    'xtick.labeltop': False,
    'xtick.major.width': 1.0,
    'xtick.minor.visible': True,
    'xtick.minor.width': 0.5,
    "xtick.major.size": 5,
    "xtick.minor.size": 3,

    # Y-TICK
    'ytick.alignment': 'center_baseline',
    'ytick.direction': 'in',
    'ytick.labelleft': True,
    'ytick.labelright': False,
    'ytick.left': True,
    'ytick.right': True,
    "ytick.major.size": 10,
    "ytick.minor.size": 5,

    'ytick.minor.visible': True,
    'ytick.minor.width': 0.5,
}

plt.rcParams.update(style_dict)



def position_legend(fig, ax):
    """
    Extend y-axis range such that legend can be placed w/o overlap with plot.

    Args
    ----
        fig: *matplotlib figure*
            Figure object which the legend should be placed on.
        ax: *matplotlib axis*
            Axis object which will hold the legend.

    """
    # Split legend into columns such that there are never more than 3 rows
    ncol = int(np.floor(len(ax.get_legend_handles_labels()[1])/3 + 0.9))
    # Place the legend in the upper right to be able to get the position to
    # adjust the axis range
    leg = ax.legend(loc="upper right", ncol=ncol)
    # Draw the plot to make it possible to extract actual positions.
    plt.draw()
    # Get legend positions in pixels of the figures
    a = leg.get_frame().get_bbox().bounds
    ylow, yup = ax.get_ylim()
    # Transform the figure pixels to data coordinates
    q = ax.transData.inverted().transform((0, a[1]))
    # Claculate the new upper limit of the y-axis depending on whether the plot
    # is on a logarithmic scale or not
    if ax.get_yscale() == "log":
        newYup = (10 ** ((np.log10(q[1]) * np.log10(ylow) + np.log10(yup) ** 2
                  - 2 * np.log10(ylow) * np.log10(yup)) / (np.log10(q[1])
                  - np.log10(ylow))))
    else:
        newYup = (q[1] * ylow + yup ** 2 - 2 * ylow * yup) / (q[1] - ylow)

    # Set the new axis limits
    ax.set_ylim([ylow, newYup])



# def get_bin_center(bin_edges):
#     """
#     Get bin centers for given bin edges.

#     Args
#     ----
#         binEdges: *array of floats*
#             Edges of the bins for which the centers should be calculated

#     Returns
#     -------
#         *array of floats*
#             Array with the center of each bin.

#     """
#     return (
#         (bin_edges[:-1] + bin_edges[1:]) / 2.,
#         (bin_edges[1:] - bin_edges[:-1]) / 2.
#     )

# def plotErrorBarPlot(binCenters, binLength, values, outName,
#                      legend=False, xlabel="", ylabel="", text=""):
#     """
#     Plot multiple errorbars by passing a dictionary of the values.

#     Args
#     ----
#         binCenters: *array*
#             x-coordinates for all errorbars
#         binLength: *array*
#             x-error for all errorbars
#         values: *dict*
#             Dictionary containing the values to be plotted. The key of each
#             element will be used in the legend of the plot. The values is a
#             list of two arrays, the first one giving the y-values,
#             the second one the y-error.
#         outName: *str*
#             Name of the file in which the plot will be saved.
#         legend: *bool*
#             Default=False; Whether to draw the legend on the plot
#         xlabel: *str*
#             Default=""; Label to use for the x-axis.
#         ylabel: *str*
#             Default=""; Label to use for the y-axis.
#         text: *str*
#             Additional text to be placed on the plot.

#     """
#     fig, ax = plt.subplots()
