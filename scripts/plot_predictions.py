import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


def prepareData(data):
    # Normalise lenght
    max_len_index = np.max([x.shape[0] for x in data])
    final = np.zeros(shape=(max_len_index, len(data) + 2, 2) )
    for i in range(len(data)):
        while data[i].shape[0] < max_len_index:
            data[i] = np.vstack([data[i], data[i][-1,:]])
        data[i] = data[i][:, 0, :]
        final[:,i, 0] = data[i][:, 0]  # (:,:,0) if for the x
        final[:,i, 1] = data[i][:, 1]  # (:,:,1) if for the y
        # Calculate average and std dev 
    final[:,-2, :] = np.average(final[:,0:-2, :], axis=1)
    final[:,-1, :] = np.std(final[:,0:-2, :], axis=1)
    # final = (timestamp, <run_1, run_2, run_3,..., avg, std>, <x, y>)
    return final

def plotTrajectory(data, traj, label, color, marker):
    #  Plot the trajectories
    plt.plot(data[:, traj, 0], data[:, traj, 1], label=label, color=color,  marker=marker)
    # plt.fill_between(data[:,-2, 0], data[:, -2, 1]-data[:, -1, 1], data[:, -2, 1]+data[:, -1, 1], alpha=0.2, edgecolor=color, facecolor=color)
    num_items = np.arange(0, len(data))
    for i in range(len(data)):
        plt.text(data[i,-2, 0], data[i,-2, 1], str(num_items[i]), color=color, fontsize=12)
    
    
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
    x = np.arange(1, data.shape[0] + 1, 1)
    axs[pos].plot(x, data[:,-2], label=label, color=color)
    axs[pos].fill_between(x, data[:, -2]-data[:, -1], data[:, -2]+data[:, -1], alpha=0.2, edgecolor=color, facecolor=color)
    axs[pos].set_ylabel(y_label)


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

    run = len(args.experiments)
    gt_list = []
    gps_list = []
    pf_list = []
    tag = "1"

    for i in range(run):
        tmp_gt = np.genfromtxt(open(os.path.join(args.root, args.experiments[i], "gt_tag_pose_" + tag + ".csv")), delimiter=",", skip_header=1)
        tmp_gps = np.genfromtxt(open(os.path.join(args.root, args.experiments[i], "gps_tag_pose_" + tag + ".csv")), delimiter=",", skip_header=1)
        tmp_pf = np.genfromtxt(open(os.path.join(args.root, args.experiments[i], "pf_tag_pose_" + tag + ".csv")), delimiter=",", skip_header=1)
        tmp_gt = np.expand_dims(tmp_gt, axis=1)  # Add one column needed later for stacking
        tmp_gps = np.expand_dims(tmp_gps, axis=1)  # Add one column needed later for stacking
        tmp_pf = np.expand_dims(tmp_pf, axis=1)  # Add one column needed later for stacking
        gt_list.append(tmp_gt)
        gps_list.append(tmp_gps)
        pf_list.append(tmp_pf)

    gt = prepareData(gt_list)
    gps = prepareData(gps_list)
    pf = prepareData(pf_list)

    # Plot only one trajectory out of the batch considered
    random_traj_index = np.random.randint(low=0, high=run)
    plotTrajectory(data=gps, traj=random_traj_index, label="gps", color="g", marker="x" )
    plotTrajectory(data=gt, traj=random_traj_index, label="gt", color="r", marker="o" )
    plotTrajectory(data=pf, traj=random_traj_index, label="pf", color="b", marker="*" )

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
    fig = plt.figure(figsize=(6,8))
    axs = fig.subplots(rows, cols, sharex=True, sharey=False)
    fig.suptitle("Tags localization error")
    plt.xlabel("NBS iterations")
    result = computeDistance(pf_list, gt_list)


    plotDistance(result[:,0,:], pos=0, y_label="Displacement[m]", label="X", color="r", axes=axs)
    plotDistance(result[:,1,:], pos=1, y_label="Displacement[m]", label="Y", color="b", axes=axs)
    plotDistance(result[:,2,:], pos=2, y_label="Displacement[m]", label="Euclidean", color="g", axes=axs)


    # # fig.tight_layout()
    fig.legend(ncol=3,loc='upper center', bbox_to_anchor=(0.5, 0.95))

    fig.savefig(fname=os.path.join(out_folder, "distance.png"), dpi=300)
