#!/bin/bash

# set the following values properly
workspace_folder='/home/manolofc/workspace/tiago-dev/'
tiago_ip="10.5.42.42" # Tiago WIFI IP can be found by pinging tiago-29c
my_ip="10.5.42.168" # your WIFI IP

# you shouldn't need to change the following lines ...
export ROS_MASTER_URI="http://"$tiago_ip":11311"
export ROS_IP=$my_ip
source $workspace_folder"devel/setup.bash"

echo -e "Don't forget to launch browser at http://"$tiago_ip":8080/ and stop in Statup tab the modules: \n enrichme, \n localizer,\n map_configuration_server,\n map_server,\n move_base,\n navigation_sm"
