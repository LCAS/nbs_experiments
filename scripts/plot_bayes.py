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

    offset = 0.2
    start_tunnel_x = 5.53 + offset
    end_tunnel_x = start_tunnel_x + 27.0 - (2*offset)

    all_bags_times = []
    all_bags_distances = []
    all_bags_topo_distances = []
    all_bags_bayesdistances = []
    all_bags_topo_bayesdistances = []
    all_bags_noisydistances = []
    all_bags_topo_noisydistances = []
    all_bags_gtnodes = []
    all_bags_enodes = []
    all_bags_bayesnodes = []
    all_bags_nbssteps = []
    all_bags_endtunnel = []
    step_size = rospy.Duration(secs=10) #seconds
    for i, bag in enumerate(bags):
        print(bag_names[i])
        # stores the bayes tracker uuids corresponding to the pickers
        tag_uuids = [
            "",
            "",
            ""
        ]
        ###trajectory vectors
        times = [[], [], []]
        distances = [[], [], []]
        topo_distances = [[], [], []]
        bayesdistances = [[], [], []]
        topo_bayesdistances = [[], [], []]
        noisydistances = [[], [], []]
        topo_noisydistances = [[], [], []]
        gtnodes = [[], [], []]
        enodes = [[], [], []]
        bayesnodes = [[], [], []]
        nbssteps = [[], [], []]
        endtunnel = [[], [], []]
        ###last read values
        last_time = None
        last_gtnode = [None, None, None]
        last_noisynode = [None, None, None]
        last_enode = [None, None, None]
        last_bayesnode = [None, None, None]
        nbs_step = False
        last_noisynode_pose = [None, None, None]
        last_enode_pose = [None, None, None]
        last_bayesnode_pose = [None, None, None]
        tpose = [None, None, None]
        noisypose = [None, None, None]
        last_endtunnel = [0, 0, 0]
        ###
        # plt.figure()
        ###
        for j, (topic, msg, ts) in enumerate(bag.read_messages(topics=['/tag_1/estimated_node', '/poses/1', '/tag_1/pose_obs', '/tag_2/estimated_node', '/poses/2', '/tag_2/pose_obs', '/tag_3/estimated_node', '/poses/3',  '/tag_3/pose_obs', '/thorvald/rfid_grid_map_node/rfid_belief_maps', '/people_tracker_filter/positions_throttle'])):
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
                        # get distance of noisy gps and GT
                        if not (last_noisynode[tagi] is None or last_noisynode_pose[tagi] is None):
                            topo_distance = find_distance(last_noisynode[tagi], last_gtnode[tagi])
                            distance = find_eucliden_distance(last_noisynode_pose[tagi], tpose[tagi])
                            if distance is None:
                                print("Distance not found {} {}".format(
                                    last_enode, last_gtnode))
                                noisydistances[tagi].append(np.nan)
                                topo_noisydistances[tagi].append(np.nan)
                            else:
                                noisydistances[tagi].append(int(distance))
                                topo_noisydistances[tagi].append(int(topo_distance))
                        else:
                            noisydistances[tagi].append(np.nan)
                            topo_noisydistances[tagi].append(np.nan)


                        # update general stuff
                        gtnodes[tagi].append(last_gtnode[tagi])
                        enodes[tagi].append(last_enode[tagi])
                        bayesnodes[tagi].append(last_bayesnode[tagi])
                        nbssteps[tagi].append(nbs_step)
                        times[tagi].append(j*step_size.secs)
                        endtunnel[tagi].append(last_endtunnel[tagi])
                    else:
                        distances[tagi].append(np.nan)
                        topo_distances[tagi].append(np.nan)
                        bayesdistances[tagi].append(np.nan)
                        topo_bayesdistances[tagi].append(np.nan)
                        noisydistances[tagi].append(np.nan)
                        topo_noisydistances[tagi].append(np.nan)
                        gtnodes[tagi].append(None)
                        enodes[tagi].append(None)
                        bayesnodes[tagi].append(None)
                        nbssteps[tagi].append(nbs_step)
                        times[tagi].append(j*step_size.secs)
                        endtunnel[tagi].append(last_endtunnel[tagi])
                nbs_step = False
                last_time = rospy.Time(secs=ts.secs, nsecs=ts.nsecs)
            if topic == '/poses/1':
                tpose[0] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
                closest = np.argmin(np.sqrt(np.sum((np.array(node_positions) - tpose[0]) ** 2, axis=1)))
                last_gtnode[0] = node_names[closest]
                if tpose[0][0] < start_tunnel_x:
                    last_endtunnel[0] = 1
                elif tpose[0][0] > end_tunnel_x:
                    last_endtunnel[0] = -1
                else:
                    last_endtunnel[0] = 0
            elif topic == '/tag_1/estimated_node':
                last_enode[0] = msg.data
                index = node_names.index(last_enode[0])
                last_enode_pose[0] = node_positions[index]
            elif topic == '/tag_1/pose_obs':
                noisypose[0] = np.array(
                    [msg.pose.pose.pose.position.x, msg.pose.pose.pose.position.y])
                closest = np.argmin(
                    np.sqrt(np.sum((np.array(node_positions) - noisypose[0]) ** 2, axis=1)))
                last_noisynode[0] = node_names[closest]
                last_noisynode_pose[0] = node_positions[closest]
            elif topic == '/poses/2':
                tpose[1] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
                closest = np.argmin(np.sqrt(np.sum((np.array(node_positions) - tpose[1]) ** 2, axis=1)))
                last_gtnode[1] = node_names[closest]
                if tpose[1][0] < start_tunnel_x:
                    last_endtunnel[1] = 1
                elif tpose[1][0] > end_tunnel_x:
                    last_endtunnel[1] = -1
                else:
                    last_endtunnel[1] = 0
            elif topic == '/tag_2/estimated_node':
                last_enode[1] = msg.data
                index = node_names.index(last_enode[1])
                last_enode_pose[1] = node_positions[index]
            elif topic == '/tag_2/pose_obs':
                noisypose[1] = np.array(
                    [msg.pose.pose.pose.position.x, msg.pose.pose.pose.position.y])
                closest = np.argmin(
                    np.sqrt(np.sum((np.array(node_positions) - noisypose[1]) ** 2, axis=1)))
                last_noisynode[1] = node_names[closest]
                last_noisynode_pose[1] = node_positions[closest]
            elif topic == '/poses/3':
                tpose[2] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
                closest = np.argmin(np.sqrt(np.sum((np.array(node_positions) - tpose[2]) ** 2, axis=1)))
                last_gtnode[2] = node_names[closest]
                if tpose[2][0] < start_tunnel_x:
                    last_endtunnel[2] = 1
                elif tpose[2][0] > end_tunnel_x:
                    last_endtunnel[2] = -1
                else:
                    last_endtunnel[2] = 0
            elif topic == '/tag_3/estimated_node':
                last_enode[2] = msg.data
                index = node_names.index(last_enode[2])
                last_enode_pose[2] = node_positions[index]
            elif topic == '/tag_3/pose_obs':
                noisypose[2] = np.array(
                    [msg.pose.pose.pose.position.x, msg.pose.pose.pose.position.y])
                closest = np.argmin(
                    np.sqrt(np.sum((np.array(node_positions) - noisypose[2]) ** 2, axis=1)))
                last_noisynode[2] = node_names[closest]
                last_noisynode_pose[2] = node_positions[closest]
            elif topic == '/thorvald/rfid_grid_map_node/rfid_belief_maps':
                nbs_step = True
            elif topic == '/people_tracker_filter/positions_throttle':
                if any([el is None for el in noisypose]):
                    continue

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
                        tag_uuids = [""] * 3
                        last_bayesnode = [None, None, None]
                        last_bayesnode_pose = [None, None, None]

                        #stack all the distances
                        tracks_positions = np.array([[pose.position.x, pose.position.y] for pose in msg.poses])
                        distance_matrix = np.vstack((
                            np.sqrt(
                                np.sum((tracks_positions - noisypose[0]) ** 2, axis=1)),
                            np.sqrt(
                                np.sum((tracks_positions - noisypose[1]) ** 2, axis=1)),
                            np.sqrt(
                                np.sum((tracks_positions - noisypose[2]) ** 2, axis=1))
                        ))
                        # print(distance_matrix)
                        det_to_assign = min(len(tracks_positions), len(tag_uuids))
                        while det_to_assign > 0:
                            indices = np.unravel_index(
                                distance_matrix.argmin(), distance_matrix.shape)
                            # print(indices)
                            _tagidx = indices[0]
                            _candidateidx = indices[1]

                            # find the closest node
                            theone = np.argmin(
                                np.sqrt(np.sum((np.array(node_positions) - tracks_positions[_candidateidx]) ** 2, axis=1)))
                            
                            # assign uuid and node
                            tag_uuids[_tagidx] = msg.uuids[_candidateidx]
                            last_bayesnode[_tagidx] = node_names[theone]
                            last_bayesnode_pose[_tagidx] = node_positions[theone]
                            
                            # delete row for the picker already assigned 
                            # distance_matrix = np.delete(
                            #     distance_matrix, indices[0], 0)
                            distance_matrix[indices[0], :] = np.inf
                            # ... and column for the tracked detection
                            # distance_matrix = np.delete(
                            #     distance_matrix, indices[1], 1)
                            distance_matrix[:, indices[1]] = np.inf
                            det_to_assign -= 1
                            # print(distance_matrix)    
                        break

                    else:
                        tag_uuids[tag_idx] = ""
                        last_bayesnode[tag_idx] = None
                        last_bayesnode_pose[tag_idx] = None


        # ### DEBUG scatter the gt trajectory and the bayes trajectory
        # plt.figure()
        # plt.scatter(np.arange(len(bayesdistances[0])), np.array(bayesdistances[0]))
        # # print(np.array(node_positions).shape)
        # # print(np.array(bayesnodes[0]).shape, np.array(enodes[0]).shape)
        # # for i, (bnode, enode) in enumerate(zip(bayesnodes[0], enodes[0])):
        # #     bposes = np.array(node_positions)[np.in1d(node_names, bnode)]
        # #     eposes = np.array(node_positions)[np.in1d(node_names, enode)]
        # #     # print(bposes, eposes)
        # #     try:
        # #         diff2d = np.abs(eposes - bposes)[0]
        # #     except:
        # #         diff2d = [np.nan, np.nan]
        # #     # print(diff2d)
        # #     # plt.plot(bposes[:, 0], bposes[:, 1], 'r')
        # #     # plt.plot(eposes[:, 0], eposes[:, 1], 'g')
        # #     plt.plot([i], [diff2d[0]], 'go')
        # #     plt.plot([i], [diff2d[1]], 'bo')
        # plt.show()
        # # bayesnodes[0]
        # ###

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
                all_bags_noisydistances.append(np.array(noisydistances[tagi]))
                all_bags_topo_distances.append(np.array(topo_distances[tagi]))
                all_bags_topo_bayesdistances.append(np.array(topo_bayesdistances[tagi]))
                all_bags_topo_noisydistances.append(np.array(topo_noisydistances[tagi]))
                all_bags_gtnodes.append(np.array(gtnodes[tagi]))
                all_bags_enodes.append(np.array(enodes[tagi]))
                all_bags_bayesnodes.append(np.array(bayesnodes[tagi]))
                all_bags_nbssteps.append(np.array(nbssteps[tagi]))
                all_bags_endtunnel.append(np.array(endtunnel[tagi]))
            # print(distances.shape)

    
    ### all with same size (smallest)
    min_size = min([t.shape[0] for t in all_bags_times])
    for i in range(len(all_bags_times)):
        all_bags_distances[i] = all_bags_distances[i][:min_size]
        all_bags_topo_distances[i] = all_bags_topo_distances[i][:min_size]
        all_bags_bayesdistances[i] = all_bags_bayesdistances[i][:min_size]
        all_bags_topo_bayesdistances[i] = all_bags_topo_bayesdistances[i][:min_size]
        all_bags_noisydistances[i] = all_bags_noisydistances[i][:min_size]
        all_bags_topo_noisydistances[i] = all_bags_topo_noisydistances[i][:min_size]
        all_bags_times[i] = all_bags_times[i][:min_size]
        all_bags_gtnodes[i] = all_bags_gtnodes[i][:min_size]
        all_bags_enodes[i] = all_bags_enodes[i][:min_size]
        all_bags_bayesnodes[i] = all_bags_bayesnodes[i][:min_size]
        all_bags_nbssteps[i] = all_bags_nbssteps[i][:min_size]
        all_bags_endtunnel[i] = all_bags_endtunnel[i][:min_size]
        

    # fig = plt.figure(figsize=(12, 8))
    # plt.plot(all_bags_times[0], averages, label='label', color='r')
    # plt.fill_between(all_bags_times[0], averages-stds, averages +
    #                       stds, alpha=0.2, edgecolor='r', facecolor='r')
    # plt.show()
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
    print("Noisy GPS:")
    print("\t\tTOTAL AVERAGE: {}".format(np.nanmean(all_bags_noisydistances)))
    print("\t\tTOTAL STD: {}".format(np.nanstd(all_bags_noisydistances)))
    print("\t\tTOTAL TOPO AVERAGE: {}".format(
        np.nanmean(all_bags_topo_noisydistances)))
    print("\t\tTOTAL TOPO STD: {}".format(
        np.nanstd(all_bags_topo_noisydistances)))

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
    bayestopo_averages = np.nanmean(
        all_bags_topo_bayesdistances, axis=0)
    bayestopo_stds = np.nanstd(
        all_bags_topo_bayesdistances, axis=0)

    noisyaverages = np.nanmean(all_bags_noisydistances, axis=0)
    noisystds = np.nanstd(all_bags_noisydistances, axis=0)
    noisytopo_averages = np.nanmean(
        all_bags_topo_noisydistances, axis=0)
    noisytopo_stds = np.nanstd(
        all_bags_topo_noisydistances, axis=0)

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

    noisyaverages = np.expand_dims(noisyaverages, axis=1)
    noisystds = np.expand_dims(noisystds, axis=1)
    result = np.concatenate((noisyaverages, noisystds), axis=1)
    np.save(out_folder + "/noisygps_metric_result{}".format("tot"), result)

    noisytopo_averages = np.expand_dims(noisytopo_averages, axis=1)
    noisytopo_stds = np.expand_dims(noisytopo_stds, axis=1)
    result = np.concatenate((noisytopo_averages, noisytopo_stds), axis=1)
    np.save(out_folder + "/noisygps_topo_result{}".format("tot"), result)


    np.save(out_folder + "/tunnelswitch", all_bags_endtunnel)