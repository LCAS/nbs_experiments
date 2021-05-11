import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.ticker import MaxNLocator


def plotDistance(data, pos, y_label, label, color, style, axes):
    #  Plot the trajectories
    # Now data are plotted for every seconds, we may want to subsample them for every minute
    # print(data.shape)
    # data = data[0::6]
    # print(data.shape)
    # data = data[0::60]
    # print(data.shape)
    x = np.arange(0, data.shape[0], 1, dtype=float) / 6.0
    if label is not None and not label.startswith("picker"):
        axs[pos].plot(x[::3], data[::3, 0], label=label, color=color)
        axs[pos].fill_between(x[::3], data[::3, 0]-data[::3, 1], data[::3, 0] +
                            data[::3, 1], alpha=0.05, edgecolor=color, facecolor=color)
    else:
        axs[pos].plot(x[::2], data[::2, 0], label=label, color=color, linestyle=style)
    axs[pos].set_ylabel(y_label, fontsize=14)
    axs[pos].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[pos].set_xlim(min(x), max(x))
    axs[pos].set_ylim(0, 12)
    axs[pos].tick_params(axis='both', which='major', labelsize=14)
    axs[pos].tick_params(axis='both', which='minor', labelsize=14)
    if (pos == 1):
        axs[pos].set_ylim(0, 35)
    


def calculateMetricsAndPlot(data, ylabel, label_list, color_list, style, axs, axs_pos):
    # Calculate metrics (MSE)
    if (axs_pos == 1):
        print("=== TOPOLOGICAL DISTANCE ===")
    else:
        print("=== EUCLIDEAN DISTANCE ===")
    for i in range(len(data)):
        print("----" + str(label_list[i]) + "----")
        mean_error = round(np.nanmean(data[i][:, 0]), 2)
        variance = round(np.nanstd(data[i][:, 0]), 2)
        print("     Mean error: " + str(mean_error) + "(" + str(variance) + ")")

    for i in range(len(data)):
        # plotDistance(data[i][:,0,:], pos=0, y_label="X-error[m]",          label=label_list[i], color=color_list[i], axes=axs)
        # plotDistance(data[i][:,1,:], pos=1, y_label="Y-error[m]",          label=label_list[i], color=color_list[i], axes=axs)
        plotDistance(data[i], pos=axs_pos, y_label=ylabel,
                     label=label_list[i], color=color_list[i], style=style, axes=axs)


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
    label_list = ["RFID+LIDAR+GPS(ours)", "NoMonitor", "CostantSpeed", "Dondrup et al.[24]"]
    pickers_label_list = [
        ["picker1", "picker1"],
        ["picker2", "picker2"],
        ["picker3", "picker3"]
    ]
    color_list = ["g", "b", "c", "r"]
    pickers_color_list = ["#6666ff66", "#ff666666"]
    pickers_style = ["--", "-.", ":"]
    tot_topo_combined_result = []
    tot_topo_estimated_node_result = []
    tot_combined_result = []
    tot_estimated_node_result = []
    # for tagi in range(3):
            
    #     topo_combined_result = np.load(args.root + "topo_result{}.npy".format(tagi))
    #     topo_estimated_node_result = np.load(
    #         args.root + "bayes_topo_result{}.npy".format(tagi))

    #     combined_result = np.load(args.root + "metric_result{}.npy".format(tagi))
    #     estimated_node_result = np.load(
    #         args.root + "bayes_metric_result{}.npy".format(tagi))

    #     # if len(tot_topo_combined_result) == 0:
    #     #     tot_topo_combined_result = topo_combined_result
    #     #     tot_topo_estimated_node_result = topo_estimated_node_result
    #     #     tot_combined_result = combined_result
    #     #     tot_estimated_node_result = estimated_node_result
    #     # else:
    #     #     tot_topo_combined_result = np.hstack((tot_topo_combined_result, topo_combined_result))
    #     #     tot_topo_estimated_node_result = np.hstack((tot_topo_estimated_node_result, topo_estimated_node_result))
    #     #     tot_combined_result = np.hstack((tot_combined_result, combined_result))
    #     #     tot_estimated_node_result = np.hstack((tot_estimated_node_result, estimated_node_result))

    #     # TODO: Check that they sizes are the sames
    #     # data = [gps_connected_result, gps_unconnected_result, lidar_result, rfid_result, combined_result]
    #     topo_data = [topo_combined_result, topo_estimated_node_result]
    #     metric_data = [combined_result, estimated_node_result]
    #     # label_list=["GPS-connected", "GPS-unconnected", "Lidar", "RFID", "Combined"]


    #     topo_data = normalizeLenData(topo_data)
    #     metric_data = normalizeLenData(metric_data)

    #     # calculateMetricsAndPlot(
    #     #     topo_data, "Topological Error[nodes]", pickers_label_list[tagi], pickers_color_list, pickers_style[tagi], axs, 0)
    #     # calculateMetricsAndPlot(
    #     #     metric_data, "Euclidan Error[m]", pickers_label_list[tagi], pickers_color_list, pickers_style[tagi], axs, 1)
        
    tot_topo_combined_result = np.load(
        args.root + "topo_result{}.npy".format("_combined"))
    tot_topo_nothreshold_result = np.load(
        args.root + "topo_result{}.npy".format("_nothreshold"))
    tot_topo_costantspeed_result = np.load(
        args.root + "topo_result{}.npy".format("_costantspeed"))
    tot_topo_bayes_result = np.load(
        args.root + "topo_result{}.npy".format("_bayes"))

    tot_combined_result = np.load(
        args.root + "metric_result{}.npy".format("_combined"))
    tot_nothreshold_result = np.load(
        args.root + "metric_result{}.npy".format("_nothreshold"))
    tot_costantspeed_result = np.load(
        args.root + "metric_result{}.npy".format("_costantspeed"))
    tot_bayes_result = np.load(
        args.root + "metric_result{}.npy".format("_bayes"))
    # TODO: Check that they sizes are the sames
    # data = [gps_connected_result, gps_unconnected_result, lidar_result, rfid_result, combined_result]
    topo_data = [tot_topo_combined_result, tot_topo_nothreshold_result, tot_topo_costantspeed_result, tot_topo_bayes_result]
    metric_data = [tot_combined_result, tot_nothreshold_result, tot_costantspeed_result, tot_bayes_result]
    # label_list=["GPS-connected", "GPS-unconnected", "Lidar", "RFID", "Combined"]

    topo_data = normalizeLenData(topo_data)
    metric_data = normalizeLenData(metric_data)

    calculateMetricsAndPlot(
        metric_data, "Euclidan Error[m]", label_list, color_list, "-", axs, 0)
    calculateMetricsAndPlot(
        topo_data, "Topological Error[nodes]", label_list, color_list, "-", axs, 1)


    # plt.show()
    fig.tight_layout()
    # , bbox_to_anchor=(0.5, 0.95))
    axs[0].legend(ncol=2, loc='upper right', fontsize=16)

    fig.savefig(fname=os.path.join(out_folder, "distance.pdf"), dpi=300)
