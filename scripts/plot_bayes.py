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

def find_eucliden_distance(n1, n2):
    if tdistances is not None:
        # print(n1)
        # print(n2)
        return np.sqrt(pow(n1[0]- n2[0], 2) + pow(n1[1] - n2[1],2))
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

    # stores the bayes tracker uuids corresponding to the pickers
    tag_uuids = [
        "",
        "",
        ""
    ]

    all_bags_times = []
    all_bags_distances = []
    all_bags_topo_distances = []
    all_bags_bayesdistances = []
    all_bags_topo_bayesdistances = []
    all_bags_gtnodes = []
    all_bags_enodes = []
    all_bags_bayesnodes = []
    all_bags_nbssteps = []
    step_size = rospy.Duration(secs=10) #seconds
    for i, bag in enumerate(bags):
        print(i, bag_names[i])
        ###trajectory vectors
        times = [[], [], []]
        distances = [[], [], []]
        topo_distances = [[], [], []]
        bayesdistances = [[], [], []]
        topo_bayesdistances = [[], [], []]
        gtnodes = [[], [], []]
        enodes = [[], [], []]
        bayesnodes = [[], [], []]
        nbssteps = [[], [], []]
        ###last read values
        last_time = None
        last_gtnode = [None, None, None]
        last_enode = [None, None, None]
        last_bayesnode = [None, None, None]
        nbs_step = False
        last_enode_pose = [None, None, None]
        last_bayesnode_pose = [None, None, None]
        tpose = [None, None, None]
        for j, (topic, msg, ts) in enumerate(bag.read_messages(topics=['/tag_1/estimated_node', '/poses/1', '/tag_2/estimated_node', '/poses/2', '/tag_3/estimated_node', '/poses/3',  '/thorvald/rfid_grid_map_node/rfid_belief_maps', '/people_tracker_filter/positions_throttle'])):
            # update time
            if last_time is None:
                last_time = ts
            elif ts.to_sec() >= (last_time.to_sec() + step_size.secs):
                for tagi in range(3):
                    if not (last_gtnode[tagi] is None or last_enode[tagi] is None):
                        # get distance of PF estimate and GT
                        topo_distance = find_distance(last_enode[tagi], last_gtnode[tagi])
                        distance = find_eucliden_distance(last_enode_pose[tagi], tpose[tagi])
                        if distance is None:
                            print("Distance not found {} {}".format(
                                last_enode, last_gtnode))
                            distances[tagi].append(np.nan)
                            topo_distances[tagi].append(np.nan)
                        else:
                            distances[tagi].append(int(distance))
                            topo_distances[tagi].append(int(topo_distance))
                        # get distance of bayes tracker and GT
                        if not (last_bayesnode[tagi] is None or last_bayesnode_pose[tagi] is None):
                            topo_distance = find_distance(last_bayesnode[tagi], last_gtnode[tagi])
                            distance = find_eucliden_distance(last_bayesnode_pose[tagi], tpose[tagi])
                            if distance is None:
                                print("Distance not found {} {}".format(
                                    last_enode, last_gtnode))
                                bayesdistances[tagi].append(np.nan)
                                topo_bayesdistances[tagi].append(np.nan)
                            else:
                                bayesdistances[tagi].append(int(distance))
                                topo_bayesdistances[tagi].append(int(topo_distance))
                        else:
                            bayesdistances[tagi].append(np.nan)
                            topo_bayesdistances[tagi].append(np.nan)


                        # update general stuff
                        gtnodes[tagi].append(last_gtnode[tagi])
                        enodes[tagi].append(last_enode[tagi])
                        bayesnodes[tagi].append(last_bayesnode[tagi])
                        nbssteps[tagi].append(nbs_step)
                        times[tagi].append(j*step_size.secs)
                    else:
                        distances[tagi].append(np.nan)
                        topo_distances[tagi].append(np.nan)
                        bayesdistances[tagi].append(np.nan)
                        topo_bayesdistances[tagi].append(np.nan)
                        gtnodes[tagi].append(None)
                        enodes[tagi].append(None)
                        bayesnodes[tagi].append(None)
                        nbssteps[tagi].append(nbs_step)
                        times[tagi].append(j*step_size.secs)
                nbs_step = False
                last_time = rospy.Time(secs=ts.secs, nsecs=ts.nsecs)
            if topic == '/poses/1':
                tpose[0] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
                closest = np.argmin(np.sqrt(np.sum((np.array(node_positions) - tpose[0]) ** 2, axis=1)))
                last_gtnode[0] = node_names[closest]
            elif topic == '/tag_1/estimated_node':
                last_enode[0] = msg.data
                index = node_names.index(last_enode[0])
                last_enode_pose[0] = node_positions[index]
            elif topic == '/poses/2':
                tpose[1] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
                closest = np.argmin(np.sqrt(np.sum((np.array(node_positions) - tpose[1]) ** 2, axis=1)))
                last_gtnode[1] = node_names[closest]
            elif topic == '/tag_2/estimated_node':
                last_enode[1] = msg.data
                index = node_names.index(last_enode[1])
                last_enode_pose[1] = node_positions[index]
            elif topic == '/poses/3':
                tpose[2] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
                closest = np.argmin(np.sqrt(np.sum((np.array(node_positions) - tpose[2]) ** 2, axis=1)))
                last_gtnode[2] = node_names[closest]
            elif topic == '/tag_3/estimated_node':
                last_enode[2] = msg.data
                index = node_names.index(last_enode[2])
                last_enode_pose[2] = node_positions[index]
            elif topic == '/thorvald/rfid_grid_map_node/rfid_belief_maps':
                nbs_step = True
            elif topic == '/people_tracker_filter/positions_throttle':
                for tag_idx in range(3):
                    # check if still being tracked
                    if tag_uuids[tag_idx] in msg.uuids:
                        ti = msg.uuids.index(tag_uuids[tag_idx])
                        # find closest topological node
                        _tpose = np.array([msg.poses[ti].position.x, msg.poses[ti].position.y])
                        closest = np.argmin(np.sqrt(np.sum((np.array(node_positions) - _tpose) ** 2, axis=1)))
                        last_bayesnode[tag_idx] = node_names[closest]
                        last_bayesnode_pose[tag_idx] = node_positions[closest]
                    elif len(msg.uuids) > 0:
                        # find tracked closest to gt which is not already tracked for another tag
                        tracks_positions = np.array([[pose.position.x, pose.position.y] for pose in msg.poses])
                        ord_idx = np.argsort(np.sqrt(np.sum((tracks_positions - tpose[tag_idx]) ** 2, axis=1)))
                        theone = None
                        for candidateidx in ord_idx:
                            if msg.uuids[candidateidx] not in tag_uuids:
                                theone = np.argmin(
                                    np.sqrt(np.sum((np.array(node_positions) - tracks_positions[candidateidx]) ** 2, axis=1)))
                                break 
                        if theone is None:
                            last_bayesnode[tag_idx] = None
                            last_bayesnode_pose[tag_idx] = None       
                        else:        
                            last_bayesnode[tag_idx] = node_names[theone]
                            last_bayesnode_pose[tag_idx] = node_positions[closest]
                    else:
                        last_bayesnode[tag_idx] = None
                        last_bayesnode_pose[tag_idx] = None


        if len(distances) > 0:
            # avg_distance = np.average(distances)
            # avg_topo_distance = np.average(topo_distances)
            print("\tsize {}".format(np.array(distances[0]).shape))
            # print("\taverage_distance {}".format(avg_distance))
            # print("\taverage_topo_distance {}".format(avg_topo_distance))

            # NOTE for now put all the data for each tag together
            for tagi in range(3):
                all_bags_times.append(np.array(times[tagi]))
                all_bags_distances.append(np.array(distances[tagi]))
                all_bags_bayesdistances.append(np.array(bayesdistances[tagi]))
                all_bags_topo_distances.append(np.array(topo_distances[tagi]))
                all_bags_topo_bayesdistances.append(np.array(topo_bayesdistances[tagi]))
                all_bags_gtnodes.append(np.array(gtnodes[tagi]))
                all_bags_enodes.append(np.array(enodes[tagi]))
                all_bags_bayesnodes.append(np.array(bayesnodes[tagi]))
                all_bags_nbssteps.append(np.array(nbssteps[tagi]))
            # print(distances.shape)

    
    ### all with same size (smallest)
    min_size = min([t.shape[0] for t in all_bags_times])
    for i in range(len(all_bags_times)):
        all_bags_distances[i] = all_bags_distances[i][:min_size]
        all_bags_topo_distances[i] = all_bags_topo_distances[i][:min_size]
        all_bags_bayesdistances[i] = all_bags_bayesdistances[i][:min_size]
        all_bags_topo_bayesdistances[i] = all_bags_topo_bayesdistances[i][:min_size]
        all_bags_times[i] = all_bags_times[i][:min_size]
        all_bags_gtnodes[i] = all_bags_gtnodes[i][:min_size]
        all_bags_enodes[i] = all_bags_enodes[i][:min_size]
        all_bags_bayesnodes[i] = all_bags_bayesnodes[i][:min_size]
        all_bags_nbssteps[i] = all_bags_nbssteps[i][:min_size]
        

    # fig = plt.figure(figsize=(12, 8))
    # for i in range(0, len(all_bags_bayesdistances), 3):
    #     if (i != 0):
    #         plt.plot(all_bags_bayesdistances[i], label=str(i/3))
    # plt.fill_between(all_bags_times[0], averages-stds, averages +
    #                       stds, alpha=0.2, edgecolor='r', facecolor='r')
    # plt.legend()
    # plt.show()
    # exit(0)
    # plt.savefig(fname=os.path.join(out_folder, bag_names[i].replace(".bag", ".png")), dpi=300)
    
    print("TPF:")
    print("\t\tTOTAL AVERAGE: {}".format(np.nanmean(all_bags_distances)))
    print("\t\tTOTAL STD: {}".format(np.nanstd(all_bags_distances)))
    print("\t\tTOTAL TOPO AVERAGE: {}".format(
        np.nanmean(all_bags_topo_distances)))
    print("\t\tTOTAL TOPO STD: {}".format(
        np.nanstd(all_bags_topo_distances)))
    print("Bayes filter:")
    print("\t\tTOTAL AVERAGE: {}".format(np.nanmean(all_bags_bayesdistances)))
    print("\t\tTOTAL STD: {}".format(np.nanstd(all_bags_bayesdistances)))
    print("\t\tTOTAL TOPO AVERAGE: {}".format(
        np.nanmean(all_bags_topo_bayesdistances)))
    print("\t\tTOTAL TOPO STD: {}".format(
        np.nanstd(all_bags_topo_bayesdistances)))

    # for tagi in range(3):
    #     averages = np.nanmean(all_bags_distances[tagi::3], axis=0)
    #     stds = np.nanstd(all_bags_distances[tagi::3], axis=0)

    #     topo_averages = np.nanmean(all_bags_topo_distances[tagi::3], axis=0)
    #     topo_stds = np.nanstd(all_bags_topo_distances[tagi::3], axis=0)

    #     bayesaverages = np.nanmean(all_bags_bayesdistances[tagi::3], axis=0)
    #     bayesstds = np.nanstd(all_bags_bayesdistances[tagi::3], axis=0)
    #     bayestopo_averages = np.nanmean(all_bags_topo_bayesdistances[tagi::3], axis=0)
    #     bayestopo_stds = np.nanstd(all_bags_topo_bayesdistances[tagi::3], axis=0)

    #     averages = np.expand_dims(averages, axis=1)
    #     stds = np.expand_dims(stds, axis=1)
    #     result = np.concatenate((averages, stds), axis=1)
    #     np.save(out_folder + "/metric_result{}".format(tagi), result)

    #     topo_averages = np.expand_dims(topo_averages, axis=1)
    #     topo_stds = np.expand_dims(topo_stds, axis=1)
    #     result = np.concatenate((topo_averages, topo_stds), axis=1)
    #     np.save(out_folder + "/topo_result{}".format(tagi), result)

    #     bayesaverages = np.expand_dims(bayesaverages, axis=1)
    #     bayesstds = np.expand_dims(bayesstds, axis=1)
    #     result = np.concatenate((bayesaverages, bayesstds), axis=1)
    #     np.save(out_folder + "/bayes_metric_result{}".format(tagi), result)

    #     bayestopo_averages = np.expand_dims(bayestopo_averages, axis=1)
    #     bayestopo_stds = np.expand_dims(bayestopo_stds, axis=1)
    #     result = np.concatenate((bayestopo_averages, bayestopo_stds), axis=1)
    #     np.save(out_folder + "/bayes_topo_result{}".format(tagi), result)

    averages = np.nanmean(all_bags_distances, axis=0)
    stds = np.nanstd(all_bags_distances, axis=0)

    topo_averages = np.nanmean(all_bags_topo_distances, axis=0)
    topo_stds = np.nanstd(all_bags_topo_distances, axis=0)

    bayesaverages = np.nanmean(all_bags_bayesdistances, axis=0)
    bayesstds = np.nanstd(all_bags_bayesdistances, axis=0)

    # plt.plot(bayesaverages, label="avg")
    # plt.legend()
    # plt.show()
    # exit(0)
    bayestopo_averages = np.nanmean(
        all_bags_topo_bayesdistances, axis=0)
    bayestopo_stds = np.nanstd(
        all_bags_topo_bayesdistances, axis=0)

    averages = np.expand_dims(averages, axis=1)
    stds = np.expand_dims(stds, axis=1)
    result = np.concatenate((averages, stds), axis=1)
    np.save(out_folder + "/metric_result{}".format("tot"), result)

    topo_averages = np.expand_dims(topo_averages, axis=1)
    topo_stds = np.expand_dims(topo_stds, axis=1)
    result = np.concatenate((topo_averages, topo_stds), axis=1)
    np.save(out_folder + "/topo_result{}".format("tot"), result)

    bayesaverages = np.expand_dims(bayesaverages, axis=1)
    bayesstds = np.expand_dims(bayesstds, axis=1)
    result = np.concatenate((bayesaverages, bayesstds), axis=1)
    np.save(out_folder + "/bayes_metric_result{}".format("tot"), result)

    bayestopo_averages = np.expand_dims(bayestopo_averages, axis=1)
    bayestopo_stds = np.expand_dims(bayestopo_stds, axis=1)
    result = np.concatenate((bayestopo_averages, bayestopo_stds), axis=1)
    np.save(out_folder + "/bayes_topo_result{}".format("tot"), result)

