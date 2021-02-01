import numpy as np
import matplotlib.pyplot as plt

root = "/home/pulver/Desktop/MCDM/mcdm/29-01-202116-37-00/"
tag = "1"

gt  = np.genfromtxt(root + "gt_tag_pose_" + tag + ".csv", delimiter=",", skip_header=1)
gps = np.genfromtxt(root + "gps_tag_pose_" + tag + ".csv", delimiter=",", skip_header=1)
pf  = np.genfromtxt(root + "pf_tag_pose_" + tag + ".csv", delimiter=",", skip_header=1)

assert gt.shape[0] == pf.shape[0]
# NOTE: GPS signal can have more entry because at different rate
num_items = np.arange(0, gt.shape[0])
# print(gt[:,0])
data_to_plot = 12
start_index = 0

fig = plt.figure()

# for rows in gt.shape[0]:
plt.plot(gt[start_index:start_index+data_to_plot,0], gt[start_index:start_index+data_to_plot,1], color="r", marker="o", label="gt")
plt.plot(gps[start_index:start_index+data_to_plot,0], gps[start_index:start_index+data_to_plot,1], color="g", marker="x", label="gps", alpha=0.2)
plt.plot(pf[start_index:start_index+data_to_plot,0], pf[start_index:start_index+data_to_plot,1], color="b", marker="*", label="pf")

for i in range(data_to_plot):
    plt.text(gt[start_index+i,0], gt[start_index+i,1], str(num_items[i]), color="red", fontsize=12)
    plt.text(pf[start_index+i,0], pf[start_index+i,1], str(num_items[i]), color="blue", fontsize=12)
    plt.text(gps[start_index+i,0], gps[start_index+i,1], str(num_items[i]), color="green", fontsize=12)

plt.legend()
plt.title("Waypoint prediction", fontsize=14)
plt.savefig(fname=root + "waypoints_prediction.png", dpi=300)

# Clear figure and plot distance pf-gt
plt.clf()

# distance = np.sqrt((x1 - x2)^2 + (y1 - y2)^2)
euclidean_distance = gt - pf
tmp = np.zeros((euclidean_distance.shape[0],euclidean_distance.shape[1]+1))
tmp[:,:-1] = euclidean_distance
euclidean_distance = tmp
euclidean_distance[:, -1] = np.sqrt(pow(euclidean_distance[:, 0], 2) + pow(euclidean_distance[:,1],2))
# euclidean_distance[:, 1] = np.sqrt(euclidean_distance[:, -1])
timestamps = np.arange(start=1, stop=len(euclidean_distance)+1)

fig, axs = plt.subplots(3, 1, sharex=True)
fig.suptitle("Distance between prediction and ground truth", fontsize=14)
axs[0].plot(timestamps, euclidean_distance[start_index:start_index+data_to_plot,0], color="r", marker="o", label="X-displacement[m]")
axs[1].plot(timestamps, euclidean_distance[start_index:start_index+data_to_plot,1], color="b", marker="o", label="Y-displacement[m]")
axs[2].plot(timestamps, euclidean_distance[start_index:start_index+data_to_plot,2], color="g", marker="o", label="Euclidean Distance [m]")

axs[0].yaxis.set_label_position("right")
axs[1].yaxis.set_label_position("right")
axs[2].yaxis.set_label_position("right")
axs[0].set_ylabel("X-displacement[m]", fontsize=8)
axs[1].set_ylabel("Y-displacement[m]", fontsize=8)
axs[2].set_ylabel("Euclidean distance[m]", fontsize=8)
axs[2].set_xlabel("NBS iterations")

# fig.tight_layout()
# fig.legend(ncol=3,loc='upper center', bbox_to_anchor=(0.5, 0.95))

fig.savefig(fname=root + "distance.png", dpi=300)