import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.ticker import MaxNLocator


def plotDistance(data, pos, y_label, label, color, axes):
    #  Plot the trajectories
    # Now data are plotted for every seconds, we may want to subsample them for every minute
    # print(data.shape)
    # data = data[0::6]
    # print(data.shape)
    # data = data[0::60]
    # print(data.shape)
    x = np.arange(0, data.shape[0], 1, dtype=float) / 6.0
    axs[pos].plot(x, data[:, 0], label=label, color=color)
    axs[pos].fill_between(x, data[:, 0]-data[:, 1], data[:, 0] +
                          data[:, 1], alpha=0.05, edgecolor=color, facecolor=color)
    axs[pos].set_ylabel(y_label, fontsize=14)
    axs[pos].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[pos].set_xlim(min(x), max(x))
    axs[pos].set_ylim(-1, 35)
    if (pos == 1):
        axs[pos].set_ylim(-1, 15)


def calculateMetricsAndPlot(data, ylabel, label_list, color_list, axs, axs_pos):
    # Calculate metrics (MSE)
    if (axs_pos == 0):
        print("=== TOPOLOGICAL DISTANCE ===")
    else:
        print("=== EUCLIDEAN DISTANCE ===")
    for i in range(len(data)):
        print("----" + label_list[i] + "----")
        mean_error = round(np.nanmean(data[i][:, 0]), 2)
        variance = round(np.nanstd(data[i][:, 0]), 2)
        print("     Mean error: " + str(mean_error) + "(" + str(variance) + ")")

    for i in range(len(data)):
        # plotDistance(data[i][:,0,:], pos=0, y_label="X-error[m]",          label=label_list[i], color=color_list[i], axes=axs)
        # plotDistance(data[i][:,1,:], pos=1, y_label="Y-error[m]",          label=label_list[i], color=color_list[i], axes=axs)
        plotDistance(data[i], pos=axs_pos, y_label=ylabel,
                     label=label_list[i], color=color_list[i], axes=axs)


def normalizeLenData(data):
    min_len = np.min([x.shape[0] for x in data])
    # print(min_len)
    data = [x[:min_len] for x in data]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, nargs="+", default=["03-02-2021-17-35-01"],
                        help="Experiment name")
    parser.add_argument("--root", type=str, default=os.environ['DATA_DIR'],
                        help="Folder path")
    args = parser.parse_args()

    # folder to save results
    out_folder = os.path.join(args.root, "out_" + "_".join(args.experiments))
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print("I will save the plots to {}".format(out_folder))

    # We have a given number of methods to compare such as
    # 1) lidar only
    # 2) RFID only
    # 3) lidar_RFID
    # and for all of them we need gt and pf data
    # gps_connected_result    = np.load(args.root + "result_gps_connected.npy")
    # gps_unconnected_result    = np.load(args.root + "result_gps_unconnected.npy")
    # lidar_result    = np.load(args.root + "result_lidar.npy")
    # rfid_result     = np.load(args.root + "result_rfid.npy")

    # For debug
    # rfid_result=np.random.normal(rfid_result,5.0)
    # combined_result=np.random.normal(combined_result,5.0)
    # Set up plot
    rows = 2
    cols = 1
    parameters = {'axes.labelsize': 8,
                  'ytick.labelsize': 8,
                  'xtick.labelsize': 8,
                  'legend.fontsize': 8}
    plt.rcParams.update(parameters)
    fig = plt.figure(figsize=(12, 8))
    axs = fig.subplots(rows, cols, sharex=True, sharey=False)
    # fig.suptitle("Tags localization error")
    plt.xlabel("Minutes", fontsize=14)
    label_list = ["TPF", "BayesFilter", ]
    color_list = ["b", "r", "#6666ff", "#ff6666"]
    for tagi in range(3):
        topo_combined_result = np.load(args.root + "topo_result{}.npy".format(tagi))
        topo_estimated_node_result = np.load(
            args.root + "bayes_topo_result{}.npy".format(tagi))

        combined_result = np.load(args.root + "metric_result{}.npy".format(tagi))
        estimated_node_result = np.load(
            args.root + "bayes_metric_result{}.npy".format(tagi))



        # TODO: Check that they sizes are the sames
        # data = [gps_connected_result, gps_unconnected_result, lidar_result, rfid_result, combined_result]
        topo_data = [topo_combined_result, topo_estimated_node_result]
        metric_data = [combined_result, estimated_node_result]
        # label_list=["GPS-connected", "GPS-unconnected", "Lidar", "RFID", "Combined"]


        topo_data = normalizeLenData(topo_data)
        metric_data = normalizeLenData(metric_data)

        calculateMetricsAndPlot(
            topo_data, "Topological Error[nodes]", label_list, color_list, axs, 0)
        calculateMetricsAndPlot(
            metric_data, "Euclidan Error[m]", label_list, color_list, axs, 1)

    # plt.show()
    fig.tight_layout()
    # , bbox_to_anchor=(0.5, 0.95))
    axs[0].legend(ncol=2, loc='upper right', fontsize=12)

    fig.savefig(fname=os.path.join(out_folder, "distance.png"), dpi=300)
