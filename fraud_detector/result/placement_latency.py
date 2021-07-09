import getopt
import os
import sys
from math import ceil

import matplotlib
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter, LogLocator
from numpy import double
from numpy.ma import arange

# mpl.use('Agg')

import matplotlib.pyplot as plt
import pylab
from matplotlib.font_manager import FontProperties

OPT_FONT_NAME = 'Helvetica'
TICK_FONT_SIZE = 24
LABEL_FONT_SIZE = 28
LEGEND_FONT_SIZE = 30
LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE)
LEGEND_FP = FontProperties(style='normal', size=LEGEND_FONT_SIZE)
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)

MARKERS = (['o', 's', 'v', "^", "h", "v", ">", "x", "d", "<", "|", "", "+", "_"])
# you may want to change the color map for different figures
COLOR_MAP = ('#B03A2E', '#2874A6', '#239B56', '#7D3C98', '#F1C40F', '#F5CBA7', '#82E0AA', '#AEB6BF', '#AA4499')
# you may want to change the patterns for different figures
PATTERNS = (["", "////", "\\\\", "//", "o", "", "||", "-", "//", "\\", "o", "O", "////", ".", "|||", "o", "---", "+", "\\\\", "*"])
LABEL_WEIGHT = 'bold'
LINE_COLORS = COLOR_MAP
LINE_WIDTH = 3.0
MARKER_SIZE = 10.0
MARKER_FREQUENCY = 1000

mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
mpl.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
mpl.rcParams['font.family'] = OPT_FONT_NAME
matplotlib.rcParams['pdf.fonttype'] = 42

FIGURE_FOLDER = './'

# there are some embedding problems if directly exporting the pdf figure using matplotlib.
# so we generate the eps format first and convert it to pdf.
def ConvertEpsToPdf(dir_filename):
    os.system("epstopdf --outfile " + dir_filename + ".pdf " + dir_filename + ".eps")
    os.system("rm -rf " + dir_filename + ".eps")

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# example for reading csv file
def ReadFile(latency_file="/home/hya/end2end.out"):
    x_axis = []
    y_axis = []

    # col = []
    # coly = []
    # temp_dict = {}
    # start_ts = 0
    # f = open("/home/myc/samza-hello-samza/test-megaphone-40960")
    # read = f.readlines()
    # for r in read:
    #     if r.find("endToEnd latency: ") != -1:
    #         if start_ts == 0:
    #             start_ts = int(int(r.split("ts: ")[1][:13]) / 1000)
    #         ts = int(int(r.split("ts: ")[1][:13]) / 1000) - start_ts
    #         latency = int(r.split("endToEnd latency: ")[1])
    #         if ts not in temp_dict:
    #             temp_dict[ts] = []
    #         temp_dict[ts].append(latency)
    #
    # for ts in temp_dict:
    #     coly.append(sum(temp_dict[ts]) / len(temp_dict[ts]))
    #     col.append(ts)
    # # x_axis.append([x -10 for x in col][10:])
    # x_axis.append(col[10:120])
    # y_axis.append(coly[10:120])
    #
    # col = []
    # coly = []
    # temp_dict = {}
    # start_ts = 0
    # f = open("/home/myc/samza-hello-samza/test_trisk_comparison")
    # read = f.readlines()
    # for r in read:
    #     if r.find("endToEnd latency: ") != -1:
    #         if start_ts == 0:
    #             start_ts = int(int(r.split("ts: ")[1][:13]) / 1000)
    #         ts = int(int(r.split("ts: ")[1][:13]) / 1000) - start_ts
    #         latency = int(r.split("endToEnd latency: ")[1])
    #         if ts not in temp_dict:
    #             temp_dict[ts] = []
    #         temp_dict[ts].append(latency)
    #
    # for ts in temp_dict:
    #     coly.append(sum(temp_dict[ts]) / len(temp_dict[ts]))
    #     col.append(ts)
    # # x_axis.append([x -10 for x in col][10:])
    # x_axis.append(col[10:120])
    # y_axis.append(coly[10:120])

    col = []
    coly = []
    temp_dict = {}
    start_ts = 0
    # f = open("/home/myc/workspace/flink-related/flink-1.11/build-target/trisk-remap-10000-200-10-5000-1000-40960-2-1-stable/flink-myc-taskexecutor-0-myc-amd.out")
    f = open(latency_file)
    read = f.readlines()
    for r in read:
        if r.find("endToEnd latency: ") != -1:
            if RepresentsInt(r.split("endToEnd latency: ")[1]) and len(r.split("ts: ")) == 2 :
                if start_ts == 0:
                    start_ts = int(int(r.split("ts: ")[1][:13]) / 1000)
                ts = int(int(r.split("ts: ")[1][:13]) / 1000) - start_ts
                latency = int(r.split("endToEnd latency: ")[1])
                if ts not in temp_dict:
                    temp_dict[ts] = []
                temp_dict[ts].append(latency)

    sorted_temp_dict = sorted(temp_dict)
    start_ts = sorted_temp_dict[0]

    for ts in sorted_temp_dict:
        coly.append(sum(temp_dict[ts]) / len(temp_dict[ts]))
        col.append(ts - start_ts)
    # x_axis.append([float(x+3) for x in col][0:100])
    # y_axis.append(coly[0:100])
    # x_axis.append(col[0:100])
    # y_axis.append(coly[0:100])
    x_axis.append(col)
    y_axis.append(coly)
    print(col)
    print(coly)


    # col = []
    # coly = []
    # temp_dict = {}
    # for i in range(0,10):
    #     ts = 0
    #     f = open("/data/trisk/Splitter FlatMap-{}.output".format(i))
    #     read = f.readlines()
    #     for r in read:
    #         if r.find("endToEndLantecy: ") != -1:
    #             latency = int(r.split("endToEndLantecy: ")[1][:2])
    #             # col.append(ts)
    #             # coly.append(latency)
    #             if ts not in temp_dict:
    #                 temp_dict[ts] = []
    #             temp_dict[ts].append(latency)
    #             ts += 1
    #
    # for ts in temp_dict:
    #     coly.append(sum(temp_dict[ts]) / len(temp_dict[ts]))
    #     col.append(ts)
    # x_axis.append(col)
    # y_axis.append(coly)

    return x_axis, y_axis


def read_cpu_file(path="cpu.out"):
    reads = open(path).readlines()
    cpu_metric = []
    time = []
    i = 0
    # padding the ealy usage to 0 which means we don't know the usage, this is because cpu recorder
    # can not start as quickly as cluster.
    n_records = len(reads) - 1
    for r in reads[1:]:
        sp = r.split(",")
        cpu_metric.append(float(sp[0]))
        time.append(570 - n_records + i)
        i += 1
    return [cpu_metric], [time]

# draw a line chart
def DrawFigure(xvalues, yvalues, legend_labels, x_label, y_label, title, allow_legend, ylim=(0, 1000)):
    # you may change the figure size on your own.
    fig = plt.figure(figsize=(9, 4))
    figure = fig.add_subplot(111)

    FIGURE_LABEL = legend_labels

    x_values = xvalues
    y_values = yvalues
    lines = [None] * (len(FIGURE_LABEL))
    for i in range(len(y_values)):
        lines[i], = figure.plot(x_values[i], y_values[i], color=LINE_COLORS[i], \
                               linewidth=LINE_WIDTH, marker=MARKERS[i], \
                               markersize=MARKER_SIZE, label=FIGURE_LABEL[i],
                                markeredgewidth=1, markeredgecolor='k',
                                markevery=2)
    plt.axvline(x=50, color='tab:gray', label='axvline - full height')
    # sometimes you may not want to draw legends.
    if allow_legend == True:
        plt.legend(lines,
                   FIGURE_LABEL,
                   prop=LEGEND_FP,
                   loc='upper center',
                   ncol=4,
                   #                     mode='expand',
                   bbox_to_anchor=(0.5, 1.2), shadow=False,
                   columnspacing=0.1,
                   frameon=True, borderaxespad=0.0, handlelength=1.5,
                   handletextpad=0.1,
                   labelspacing=0.1)
    plt.grid(axis='y', color='gray')
    # plt.yscale('log')
    plt.xlabel(x_label, fontproperties=LABEL_FP)
    plt.ylabel(y_label, fontproperties=LABEL_FP)
    plt.ylim(ylim[0], ylim[1])
    # plt.ylim(100, 400)
    # plt.xlim(100, 400)
    plt.title(title)
    plt.show()

    # plt.savefig(FIGURE_FOLDER + "/" + filename + ".pdf", bbox_inches='tight')

if __name__ == "__main__":
    x_axis, y_axis = ReadFile(latency_file="end2end.rep1")
    # legend_labels = ["Megaphone", "Trisk", "Flink"]
    legend_labels = ["Flink"]
    legend = False
    DrawFigure(x_axis, y_axis, legend_labels, "time(s)", "latency(ms)", "placement_latency", legend, ylim=(100, 500))

    cpu, t = read_cpu_file(path="flamingo.cpu")
    DrawFigure(t, cpu, legend_labels, "time(s)", "cpu util", "flamingo cpu usage", legend)
    cpu, t = read_cpu_file(path="eagle.cpu")
    DrawFigure(t, cpu, legend_labels, "time(s)", "cpu util", "eagle cpu usage", legend)