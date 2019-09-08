#!/usr/bin/env sh
export TMULE=1

this_file=$(readlink -f "$BASH_SOURCE")
this_dir=$(dirname $this_file)
ws_root=$(readlink -f "$this_dir/../../..")
export DISPLAY=:0
export ROBOT_NAME=linda
export LINDA_WS="$ws_root"
#export ROS_IP="10.82.0.70"
export ROS_IP="127.0.0.1"
export ROS_MASTER_URI="http://"$ROS_IP":11311"


if [ -r "$LINDA_WS" ]; then echo "YES"; source "$LINDA_WS/devel/setup.bash"; else source /opt/ros/kinetic/setup.bash; fi

echo $LINDA_WS
