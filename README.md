# mcdm_experiments
This repo contains INB3123 maps and gazebo worlds for experiments. It's main use is to get an environment for mcdm experiments with Tiago.


# Simulation
Main launcher is `INB3123_experiment.launch` which contains almost everything.

# Setting up LCAS software.
If you still haven't set up this awesome software, you won't be able to go on. Please proceed with:

     # This is just to cache you admin password for the next steps
     sudo ls
     # curl is required for the next step
     sudo apt-get update && sudo apt-get install curl
     # And this should install everything required
     curl https://raw.githubusercontent.com/LCAS/rosdistro/master/lcas-rosdistro-setup.sh | bash -

Or whatever new instructions Marc has kindly compiled for us [here](https://github.com/LCAS/rosdistro/wiki#using-the-l-cas-repository-if-you-just-want-to-use-our-software)

# Installing Tiago in Kinetic with Gazebo8 compatibility


Mostly we will use same instructions than in PAL tutorials (see [here](http://wiki.ros.org/Robots/TIAGo/Tutorials/Installation/TiagoSimulation)), but we will use our own rosinstall file from here. Follow these instructions:

    mkdir ~/workspace/tiago
    cd ~/workspace/tiago
    curl https://raw.githubusercontent.com/MFernandezCarmona/mcdm_experiments/master/tiago_lcas.rosinstall

    rosinstall src /opt/ros/kinetic tiago_lcas.rosinstall

    # not sure about the init here ...
    #sudo rosdep init
    rosdep update

    rosdep install --from-paths src --ignore-src --rosdistro kinetic --skip-keys="opencv2 opencv2-nonfree pal_laser_filters speed_limit  sensor_to_cloud hokuyo_node libdw-dev python-graphitesend-pip python-statsd pal_filters pal_vo_server pal_usb_utils pal_pcl pal_pcl_points_throttle_and_filter pal_karto pal_local_joint_control camera_calibration_files pal_startup_msgs pal-orbbec-openni2 dummy_actuators_manager pal_local_planner gravity_compensation_controller current_limit_controller dynamic_footprint dynamixel_cpp tf_lookup"

It will produce some errors regarding gazebo7, but it's fine. After that we can compile:

    source /opt/ros/kinetic/setup.bash
    catkin build -DCATKIN_ENABLE_TESTING=0

Amtec fails sometimes. Just retry compiling it with:

    catkin build amtec

If all compiles, you should be able to install
