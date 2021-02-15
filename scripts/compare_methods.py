import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.ticker import MaxNLocator

def plotDistance(data, pos, y_label, label, color, axes):
    #  Plot the trajectories
    x = np.arange(0, data.shape[0] , 1)
    axs[pos].plot(x, data[:,-2], label=label, color=color)
    axs[pos].fill_between(x, data[:, -2]-data[:, -1], data[:, -2]+data[:, -1], alpha=0.1, edgecolor=color, facecolor=color)
    axs[pos].set_ylabel(y_label, fontsize=14)
    axs[pos].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[pos].set_xlim(min(x), max(x))
    axs[pos].set_ylim(-3, 8)


if __name__== "__main__":
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
    lidar_result    = np.load(args.root + "result_lidar.npy")
    rfid_result     = np.load(args.root + "result_rfid.npy")
    combined_result = np.load(args.root + "result_combined.npy")

    # For debug
    # rfid_result=np.random.normal(rfid_result,5.0)
    # combined_result=np.random.normal(combined_result,5.0)
    
    # TODO: Check that they sizes are the sames
    data = [lidar_result, rfid_result, combined_result]
    label_list=["Lidar", "RFID", "Combined"]
    color_list=["r", "b", "g"]
    min_len = np.min([x.shape[0] for x in data])
    # print(min_len)
    data = [x[:min_len] for x in data]

    # Calculate metrics (MSE)
    for i in range(len(data)):
        print("----" + label_list[i] + "----")
        mean_error = round(np.mean(data[i][:, 2, :]),2)
        variance = round(np.std(data[i][:, 2, :]),2)
        print("     Mean error: " + str(mean_error) + "(" + str(variance) + ")")

    # Set up plot
    rows = 3
    cols = 1
    parameters = {'axes.labelsize': 8,
                    'ytick.labelsize': 8, 
                    'xtick.labelsize': 8,
                    'legend.fontsize': 8}
    plt.rcParams.update(parameters)
    fig = plt.figure(figsize=(12,8))
    axs = fig.subplots(rows, cols, sharex=True, sharey=False)
    # fig.suptitle("Tags localization error")
    plt.xlabel("NBS iterations", fontsize=14)
    
    for i in range(len(data)):
        plotDistance(data[i][:,0,:], pos=0, y_label="X-error[m]",          label=label_list[i], color=color_list[i], axes=axs)
        plotDistance(data[i][:,1,:], pos=1, y_label="Y-error[m]",          label=label_list[i], color=color_list[i], axes=axs)
        plotDistance(data[i][:,2,:], pos=2, y_label="Euclidean error[m]",  label=label_list[i], color=color_list[i], axes=axs)


    fig.tight_layout()
    axs[0].legend(ncol=3,loc='upper right', fontsize=12) #, bbox_to_anchor=(0.5, 0.95))

    fig.savefig(fname=os.path.join(out_folder, "distance.png"), dpi=300)