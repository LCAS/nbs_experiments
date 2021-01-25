import numpy as np
import matplotlib.pyplot as plt

root = "/home/pulver/Desktop/topoNBS/"
tag = "1"

gt  = np.genfromtxt(root + "gt_tag_pose_" + tag + ".csv", delimiter=",", skip_header=1)
gps = np.genfromtxt(root + "gps_tag_pose_tag_" + tag + ".csv", delimiter=",", skip_header=1)
pf  = np.genfromtxt(root + "pf_tag_pose_" + tag + ".csv", delimiter=",", skip_header=1)

assert gt.shape[0] == pf.shape[0]
# NOTE: GPS signal can have more entry because at different rate
num_items = np.arange(0, gt.shape[0])
# print(gt[:,0])
data_to_plot = 20
start_index = 12

# for rows in gt.shape[0]:
plt.plot(gt[start_index:start_index+data_to_plot,0], gt[start_index:start_index+data_to_plot,1], color="r", marker="o", label="gt")
# plt.plot(gps[:int(gps.shape[0]/2),0], gps[:int(gps.shape[0]/2),1], color="g", marker="x", label="gps", alpha=0.2)
plt.plot(pf[start_index:start_index+data_to_plot,0], pf[start_index:start_index+data_to_plot,1], color="b", marker="*", label="pf")

for i in range(data_to_plot):
    plt.text(gt[start_index+i,0], gt[start_index+i,1], str(num_items[i]), color="red", fontsize=12)
    plt.text(pf[start_index+i,0], pf[start_index+i,1], str(num_items[i]), color="blue", fontsize=12)

plt.legend()
plt.title("Noisy GPS", fontsize=20)
plt.show()