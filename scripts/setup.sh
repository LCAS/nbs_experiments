#!/usr/bin/env sh
export TMULE=1

this_file=$(readlink -f "$BASH_SOURCE")
this_dir=$(dirname $this_file)
ws_root=$(readlink -f "$this_dir/../../..")
export DISPLAY=:0
export ROBOT_NAME=linda
export LINDA_WS="$ws_root"

if [ -r "$LINDA_WS" ]; then echo "YES"; source "$LINDA_WS/devel/setup.bash"; else source /opt/ros/kinetic/setup.bash; fi
echo $LINDA_WS

# VARs likely to change or depend on the scenario.

#export ROS_IP="10.82.0.70"
export ROS_IP="127.0.0.1"
export ROS_MASTER_URI="http://"$ROS_IP":11311"

#TESTCODE=INB3123
#LOC_MAP_YAML="$(rospack find nbs_experiments)/maps/inb3123/map_real_sci/cropped.yaml"
#NAV_MAP_YAML="$(rospack find nbs_experiments)/maps/inb3123/map_real_sci_mcdm/cropped.yaml"
#STARTING_POSE_PX="-5.585"
#STARTING_POSE_PY="-8.798"
#STARTING_POSE_OZ="-0.999"
#STARTING_POSE_OW="0.012"

# ........................................................................................
TESTCODE=INB3ENG
LOC_MAP_YAML="$(rospack find nbs_experiments)/maps/inb3eng/scitos_map_real/map.yaml"
NAV_MAP_YAML="$(rospack find nbs_experiments)/maps/inb3eng/scitos_map_real_mcdm/map.yaml"

PARAM_1=0.2
PARAM_2=0.6
PARAM_3=0.2
STARTING_POSE_PX="-33.6"
STARTING_POSE_PY="8.3"
STARTING_POSE_OZ="0.682"
STARTING_POSE_OW="0.731"
