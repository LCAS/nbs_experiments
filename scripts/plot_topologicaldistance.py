import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import argparse
import rosbag
import yaml
import rospy
import csv
from nav_msgs.msg import Odometry

tdistances = None

def find_distance(n1, n2):
    if tdistances is not None:
        key = n1 + n2
        if key in tdistances:
            return tdistances[key]

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bags", type=str, nargs="+", default=[],
                        help="Experiment name")
    parser.add_argument("--root", type=str, default=os.environ['DATA_DIR'],
                        help="Folder path")
    # parser.add_argument("--tags", type=str, nargs="+", default=['0'],
    #                     help="tag IDS to consider")

    args = parser.parse_args()

    out_folder = args.root #os.path.join(args.root, "out_" + "_".join(args.experiments))
    # if not os.path.exists(out_folder):
    #     os.makedirs(out_folder)
    # print("I will save the plots to {}".format(out_folder))
    bag_names = []
    if len(args.bags) > 0:
        bag_names = args.bags
    else:
        for f in os.listdir(out_folder):
            if f.endswith(".bag"):
                bag_names.append(f)
    
    bags = [
        rosbag.Bag(os.path.join(args.root, bname)) for bname in bag_names
    ]

    with open("../maps/riseholme_poly_act_rfid_sim.yaml") as f:
        tmap = yaml.safe_load(f)
        # print(tmap[0])
        node_names = [node['node']['name'] for node in tmap]
        node_positions = [[node['node']['pose']['position']
                        ['x'], node['node']['pose']['position']['y']] for node in tmap]
        # print(nodes)


    with open("../maps/riseholme_map.csv") as f:
        tdistances = csv.reader(f, delimiter=",")
        tdistances = {dist[0]: dist[1] for dist in tdistances} 
        # print(tdistances)

    all_bags_times = []
    all_bags_distances = []
    all_bags_gtnodes = []
    all_bags_enodes = []
    all_bags_nbssteps = []
    step_size = rospy.Duration(secs=10) #second
    for i, bag in enumerate(bags):
        print(bag_names[i])
        times = []
        distances = []
        gtnodes = []
        enodes = []
        nbssteps = []
        last_time = None
        last_gtnode = None
        last_enode = None
        nbs_step = False
        for j, (topic, msg, ts) in enumerate(bag.read_messages(topics=['/tag_1/estimated_node', '/poses/1', '/thorvald/rfid_grid_map_node/rfid_belief_maps'])):
            # update time
            if last_time is None:
                last_time = ts
            elif (last_time + step_size) >= ts:
                if not (last_gtnode is None or last_enode is None):
                    distance = find_distance(last_enode, last_gtnode)
                    if distance is None:
                        print("Distance not found {} {}".format(
                            last_enode, last_gtnode))
                        distances.append(np.nan)
                    else:
                        distances.append(int(distance))
                    times.append(j*step_size.secs)
                    gtnodes.append(last_gtnode)
                    enodes.append(last_enode)
                    nbssteps.append(nbs_step)
                    nbs_step = False
                last_time = ts
            if topic == '/poses/1':
                tpose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
                closest = np.argmin(np.sqrt(np.sum((np.array(node_positions) - tpose) ** 2, axis=1)))
                last_gtnode = node_names[closest]
            elif topic == '/tag_1/estimated_node':
                last_enode = msg.data
            elif topic == '/thorvald/rfid_grid_map_node/rfid_belief_maps':
                nbs_step = True

        if len(distances) > 0:
            avg_distance = np.average(distances)
            print("\taverage_distance {}".format(avg_distance))

            all_bags_distances.append(np.array(distances))
            all_bags_times.append(np.array(times))
            all_bags_gtnodes.append(np.array(gtnodes))
            all_bags_enodes.append(np.array(enodes))
            all_bags_nbssteps.append(np.array(nbssteps))
            # print(distances.shape)

    
    ### all with same size (smallest)
    min_size = min([t.shape[0] for t in all_bags_times])
    for i in range(len(all_bags_times)):
        all_bags_distances[i] = all_bags_distances[i][:min_size]
        all_bags_times[i] = all_bags_times[i][:min_size]
        all_bags_gtnodes[i] = all_bags_gtnodes[i][:min_size]
        all_bags_enodes[i] = all_bags_enodes[i][:min_size]
        all_bags_nbssteps[i] = all_bags_nbssteps[i][:min_size]

    averages = np.average(all_bags_distances, axis=0)
    stds = np.std(all_bags_distances, axis=0)

    print(averages.shape)
    

    fig = plt.figure(figsize=(12, 8))
    plt.plot(all_bags_times[0], averages, label='label', color='r')
    plt.fill_between(all_bags_times[0], averages-stds, averages +
                          stds, alpha=0.2, edgecolor='r', facecolor='r')
    # plt.set_ylabel(y_label, fontsize=14)
    # plt.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # plt.set_xlim(min(all_bags_times[0]), max(all_bags_times[0]))
    # axs[pos].set_ylim(-3, 8)
    # plotDistance(result[:, 0, :], pos=0, y_label="X-error[m]",
                #  label="X", color="r") 
    # plt.legend()
    # plt.title("Topological distance estimation vs ground truth", fontsize=14)
    # plt.xlabel("X-distance[m]")
    # plt.ylabel("Nodes")
    plt.show()
    plt.savefig(fname=os.path.join(out_folder, bag_names[i].replace(".bag", ".png")), dpi=300)
    # plotTrajectory(data=gt, traj=random_traj_index,
    #             label="gt", color="r", marker="o")
    # plotTrajectory(data=pf, traj=random_traj_index,
    #             label="pf", color="b", marker="*")
    
    print("\t\tTOTAL AVERAGE: {}".format(np.average(all_bags_distances)))

