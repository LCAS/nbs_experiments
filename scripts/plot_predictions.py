import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import argparse


def prepareData(data):
    # Normalise length to keep the shortest trajectory
    min_len = np.min([x.shape[0] for x in data])
    final = np.zeros(shape=(min_len, len(data) + 2, 2) )
    for i in range(len(data)):
        while data[i].shape[0] > min_len:
            data[i] = data[i][:min_len, ...]
        data[i] = data[i][:, 0, :]
        final[:,i, 0] = data[i][:, 0]  # (:,:,0) if for the x
        final[:,i, 1] = data[i][:, 1]  # (:,:,1) if for the y
        # Calculate average and std dev 
    final[:,-2, :] = np.average(final[:,0:-2, :], axis=1)
    final[:,-1, :] = np.std(final[:,0:-2, :], axis=1)
    # final = (timestamp, <run_1, run_2, run_3,..., avg, std>, <x, y>)
    return final

def plotTrajectory(data, traj, label, color, marker, axes):
    #  Plot the trajectories
    axes.plot(data[:, traj, 0], data[:, traj, 1], label=label, color=color,  marker=marker)
    # plt.fill_between(data[:,-2, 0], data[:, -2, 1]-data[:, -1, 1], data[:, -2, 1]+data[:, -1, 1], alpha=0.2, edgecolor=color, facecolor=color)
    num_items = np.arange(0, len(data))
    for i in range(len(data)):
        axes.text(data[i, traj, 0], data[i, traj, 1], str(num_items[i]), color=color, fontsize=12)
    axes.set_xlim(min(data[:, traj, 0])-2, max(data[:, traj, 0])+2)
    axes.set_ylim(min(data[:, traj, 1])-2, max(data[:, traj, 1])+2)
    #set aspect ratio to 1
    # ratio = 1.0
    # x_left, x_right = axes.get_xlim()
    # y_low, y_high = axes.get_ylim()
    # axes.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    axes.set_aspect(1)

    
def computeDistance(pf_list, gt_list):
    distance = np.zeros(shape=(pf_list[0].shape[0], pf_list[0].shape[1]+1, len(pf_list) ))
    for run in range(len(gt_list)):
        euclidean_distance = gt_list[run] - pf_list[run]
        tmp = np.zeros((euclidean_distance.shape[0],euclidean_distance.shape[1]+1))
        tmp[:,:-1] = euclidean_distance
        euclidean_distance = tmp
        euclidean_distance[:, -1] = np.sqrt(pow(euclidean_distance[:, 0], 2) + pow(euclidean_distance[:,1],2))
        # distance_list.append(euclidean_distance)
        distance[:,:,run] = euclidean_distance
    result = np.zeros(shape=(len(distance), 3, 2))
    # distance = (timestamp, <x, y, euclidean>, run)
    # result = (timestamp, <x, y, euclidean>, <avg, std>)
    result[:, :, 0] = np.average(np.absolute(distance), axis=2)
    result[:, :, 1] = np.std(np.absolute(distance), axis=2)
    return result

def plotDistance(data, pos, y_label, label, color, axes):
    #  Plot the trajectories
    x = np.arange(0, data.shape[0], 1)
    axs[pos].plot(x, data[:,-2], label=label, color=color)
    axs[pos].fill_between(x, data[:, -2]-data[:, -1], data[:, -2]+data[:, -1], alpha=0.2, edgecolor=color, facecolor=color)
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
    parser.add_argument("--tags", type=str, nargs="+", default=['0'],
                        help="tag IDS to consider")

    args = parser.parse_args()

    # folder to save results
    out_folder = os.path.join(args.root, "out_" + "_".join(args.experiments))
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print("I will save the plots to {}".format(out_folder))

    run = len(args.experiments)
    gt_list = []
    gps_list = []
    pf_list = []
    # tag = "1"

    for i in range(run):
        for t in args.tags:
            tag_id = int(t)+1
            tmp_gt = np.genfromtxt(open(os.path.join(args.root, args.experiments[i], "gt_tag_pose_" + t + ".csv")), delimiter=",", skip_header=1)
            tmp_gps = np.genfromtxt(open(os.path.join(args.root, args.experiments[i], "gps_tag_pose_" + t + ".csv")), delimiter=",", skip_header=1)
            tmp_pf = np.genfromtxt(open(os.path.join(args.root, args.experiments[i], "pf_tag_pose_" + t + ".csv")), delimiter=",", skip_header=1)
            tmp_gt = np.expand_dims(tmp_gt, axis=1)  # Add one column needed later for stacking
            tmp_gps = np.expand_dims(tmp_gps, axis=1)  # Add one column needed later for stacking
            tmp_pf = np.expand_dims(tmp_pf, axis=1)  # Add one column needed later for stacking
            gt_list.append(tmp_gt)
            gps_list.append(tmp_gps)
            pf_list.append(tmp_pf)
            print("Experiment {}_tag{}: size {}".format(args.experiments[i], t, tmp_gps.shape))

    gt = prepareData(gt_list)
    gps = prepareData(gps_list)
    pf = prepareData(pf_list)


    # Plot only one trajectory out of the batch considered
    random_traj_index = np.random.randint(low=0, high=run)
    fig, ax = plt.subplots()
    plotTrajectory(data=gps, traj=random_traj_index, label="gps", color="g", marker="x", axes=ax)
    plotTrajectory(data=gt, traj=random_traj_index, label="gt", color="r", marker="o", axes=ax)
    plotTrajectory(data=pf, traj=random_traj_index, label="pf", color="b", marker="*", axes=ax)

    plt.legend()
    plt.title("Waypoint prediction", fontsize=14)
    plt.xlabel("X-distance[m]")
    plt.ylabel("Y-distance[m]")
    plt.savefig(fname=os.path.join(out_folder, "waypoints_prediction.png"), dpi=300)
    # plt.show()

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

    result = computeDistance(pf_list, gt_list)
    # We want to save on disk the average and std value just computed 
    # for then comparing it with different methods
    np.save(out_folder + "/result", result)


    plotDistance(result[:,0,:], pos=0, y_label="X-error[m]", label="X", color="r", axes=axs)
    plotDistance(result[:,1,:], pos=1, y_label="Y-error[m]", label="Y", color="b", axes=axs)
    plotDistance(result[:,2,:], pos=2, y_label="Euclidean error[m]", label="Euclidean", color="g", axes=axs)


    fig.tight_layout()
    # fig.legend(ncol=3,loc='upper center', bbox_to_anchor=(0.5, 0.95))

    fig.savefig(fname=os.path.join(out_folder, "distance.png"), dpi=300)
