import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import argparse
import rosbag
import yaml
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

    all_bags_distances = [] 
    for i, bag in enumerate(bags):
        print(bag_names[i])
        distances = []
        last_gtnode = None
        for topic, msg, ts in bag.read_messages(topics=['/tag_1/estimated_node', '/poses/1']):
            if topic == '/poses/1':
                tpose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
                closest = np.argmin(np.sqrt(np.sum((np.array(node_positions) - tpose) ** 2, axis=1)))
                # print(closest, node_names[closest],
                #       node_positions[closest], tpose)
                last_gtnode = node_names[closest]
            if topic == '/tag_1/estimated_node':
                if last_gtnode is None:
                    continue
                last_enode = msg.data
                distance = find_distance(last_enode, last_gtnode)
                if distance is None:
                    print("Distance not found {} {}".format(last_enode, last_gtnode))
                else:
                    distances.append(int(distance))
        if len(distances) > 0:
            avg_distance = np.average(distances)
            print("\taverage_distance {}".format(avg_distance))

            all_bags_distances.append(avg_distance)
    print("\t\tTOTAL AVERAGE: {}".format(np.average(all_bags_distances)))


