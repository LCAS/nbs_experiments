#!/usr/bin/env python

'''
Offline rosbag parser!
Bag contains: tf topic with robot tf and rfid tags tf
              rfid readings
              laser readings 

'''


import rosbag
from tf_bag import BagTfTransformer

import tf
import pandas as pd
import rospy
from rfid_node.msg import TagReading


def getRelativeXYYaw(bag_tf, orig_frame, dest_frame, t):
    translation, quaternion = bag_tf.lookupTransform(orig_frame, dest_frame, t)
    (rel_x, rel_y, rel_yaw) = getXYYaw(translation, quaternion)
    return (rel_x, rel_y, rel_yaw)


def getXYYaw(translat, rotat):
    rel_x = translat[0]
    rel_y = translat[1]
    (rel_rol, rel_pitch, rel_yaw) = tf.transformations.euler_from_quaternion(rotat)
    return (rel_x, rel_y, rel_yaw)


# Main function.
if __name__ == '__main__':
    #folder = '/home/manolofc/catkin_ws/src/RFID/pr_model/tests/linda/'
    #folder = '/home/manolofc/ownCloud/RFID-bags/'
    #saveFile = folder + '20dB-Linda-FAB-LAB-V3.csv'
    #bagFile = folder + '2000-Linda-FAB-LAB.bag'

    folder = '/home/manolofc/Desktop/success_INB3ENG/'
    saveFile = folder + 'INB3ENG_0.2_0.6_0.2_date__2019-09-09-18-50-52.csv'
    bagFile = folder + 'INB3ENG_0.2_0.6_0.2_date__2019-09-09-18-50-52.bag'
    tagSet=set()
    detectecTagSet=set()
    allTagSet=set()

    # tags surveyed in bag....
    tagSet.add('390100010000000000000002')
    tagSet.add('390100010000000000000004')
    tagSet.add('390100010000000000000005')
    tagSet.add('390100010000000000000007')
    tagCoveragePercent = 0.0
    isFirstStat = True
    rob_x = 0.0
    rob_y = 0.0
    rob_yaw = 0.0
    rfid_reading_topic = '/lastTag'
    mcdm_stats_topic = '/mcdm_stats'
    robot_frame = 'base_footprint'
    map_frame = "map"

    labels = ['Time', 'robot_x_m', 'robot_y_m', 'robot_yaw_rad','tagCoveragePercent', 'coveragePercent', 'numConfiguration', 'backTracking']
    dataEntries = []

    print("Procesing rosbag file: " + bagFile)
    bag = rosbag.Bag(bagFile)
    print("Creating bag transformer (may take a while)")
    bag_transformer = BagTfTransformer(bag)

    lastPrint = rospy.Time(0)
    printInc = rospy.Duration(30)

    print("Iterating over bag file...")
    # main loop
    for topic, msg, t in bag.read_messages():
        if ((t-lastPrint) > printInc):
            print("T: " + str(t))
            lastPrint = t

        if topic == rfid_reading_topic:
            tid = str(msg.ID)
            allTagSet.add(tid)
            if tid in tagSet:
                detectecTagSet.add(tid)

            tagCoveragePercent = 100.0 * float(len(detectecTagSet)) / float(len(tagSet))

        if topic == mcdm_stats_topic:
            
            raw_stats = msg.data.split(',')
            if isFirstStat:
                isFirstStat = False
                print("MDCM giving data about: "+msg.data)
            else:
                (rob_x, rob_y, rob_yaw) = getRelativeXYYaw(bag_transformer, map_frame, robot_frame, t)
                coveragePercent = float(raw_stats[0])
                numConfiguration = float(raw_stats[1])
                backTracking = float(raw_stats[2])

                # entry order and elements MUST MATCH LABELS!!!!!
                entry = (str(t), rob_x, rob_y, rob_yaw, tagCoveragePercent, coveragePercent, numConfiguration, backTracking)
                dataEntries.append(entry)

    # save and close
    bag.close()

    df = pd.DataFrame.from_records(dataEntries, columns=labels)
    print("detected tags: "+str(detectecTagSet))
    print("All tags: "+str(allTagSet))
    print("Saving data to csv")
    df.to_csv(saveFile, index=False)
    print("Done")
