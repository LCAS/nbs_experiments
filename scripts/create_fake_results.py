import numpy as np
import os

root = "/home/pulver/Desktop/MCDM/mcdm/"
experiment = "29-01-202117-38-22/"

fake_results_num = 2

gps = np.genfromtxt(open(root + experiment + "gps_tag_pose_1.csv"), delimiter=",", skip_header=1)
gt = np.genfromtxt(open(root + experiment + "gt_tag_pose_1.csv"), delimiter=",", skip_header=1)
pf = np.genfromtxt(open(root + experiment + "pf_tag_pose_1.csv"), delimiter=",", skip_header=1)

for i in range(1, fake_results_num + 1):
    # First, create a non existing folder
    path = os.path.join(root, "tmp_" + str(i))
    if not os.path.exists(path):
        os.makedirs(path)
    tmp_gps = np.random.normal(gps, 0.2)
    tmp_gt = np.random.normal(gt, 0.2)
    tmp_pf = np.random.normal(pf, 0.2)
    np.savetxt(path + "/gps_tag_pose_1.csv", tmp_gps, delimiter=',', header="gps_x, gps_y")
    np.savetxt(path + "/gt_tag_pose_1.csv", tmp_gt, delimiter=',', header="gt_x, gt_y")
    np.savetxt(path + "/pf_tag_pose_1.csv", tmp_pf, delimiter=',', header="pf_x, pf_y")
