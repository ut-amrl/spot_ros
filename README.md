# Spot ROS Driver

![CP Spot](cp_spot.jpg)

## Prerequisite
```
pip3 install bosdyn-client bosdyn-mission bosdyn-api bosdyn-core
```


## Documentation

Check-out the usage and user documentation [HERE](http://www.clearpathrobotics.com/assets/guides/melodic/spot-ros/)


# Building Quick-Start

NOTE: please follow the link above for the complete documentation. You will need to configure the networking on both
your computer and the base Spot platform before you can run the driver.

## Install Dependencies

```bash
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-melodic-ros-base

sudo apt update
sudo apt install -y python3-pip
pip3 install cython
pip3 install bosdyn-client bosdyn-mission bosdyn-api bosdyn-core
pip3 install empy
```



## Building for Melodic

Please note that the Spot SDK uses Python3, which is not officially supported by ROS Melodic.  If you encounter an error
of this form:

```bash
Traceback (most recent call last):
  File "/home/administrator/catkin_ws/src/spot_ros/spot_driver/scripts/spot_ros", line 3, in <module>
    from spot_driver.spot_ros import SpotROS
  File "/home/administrator/catkin_ws/src/spot_ros/spot_driver/src/spot_driver/spot_ros.py", line 19, in <module>
    import tf2_ros
  File "/opt/ros/melodic/lib/python2.7/dist-packages/tf2_ros/__init__.py", line 38, in <module>
    from tf2_py import *
  File "/opt/ros/melodic/lib/python2.7/dist-packages/tf2_py/__init__.py", line 38, in <module>
    from ._tf2 import *
ImportError: dynamic module does not define module export function (PyInit__tf2)
```

when launching the driver, please follow these steps:

1.  `rm -rf devel/ build/ install/` -- this will remove any old build artifacts from your workspace

2. `git clone https://github.com/ros/geometry2 --branch 0.6.5` into your `src` folder

3. rebuild your workspace with

```bash
catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

4. re-run `source devel/setup.bash`

5. start the driver with `roslaunch spot_driver driver.launch`

# For those who are sick of docker and just want to use the native system
1. create a ROS workspace with following structure, and then clone this repo under `src/`
```tree
Your ROS workspace name
├── src
│   ├── spot_ros
```
2. checkout branch `ros_no_docker`
```shell
cd {workspace_name}/src/spot_ros
git checkout ros_no_docker
```
3. go to website: https://robostack.github.io/GettingStarted.html and follow the steps to create your appropriate conda environment. Below are the code snippets
```shell
conda install mamba -c conda-forge
mamba create -n ros_env
mamba activate ros_env

# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-staging
# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults
```

4. install ROS environment
```shell
# Install ros-noetic into the environment (ROS1)
mamba install ros-noetic-desktop
mamba install ros-noetic-joy	
mamba install ros-noetic-interactive-marker-twist-server
mamba install ros-noetic-teleop-twist-joy
mamba install ros-noetic-twist-mux
```

5. go back to the root of the your workspace and then `catkin_make`
```shell
catkin_make
```

6. source your devel
```shell
source devel/setup.bash
```

7. to test if this works, fill in the correct username and password under `launch/driver.launch` and finally run `roslaunch spot_ros driver.launch`

You should be able to see the spot showing a green light and standing up. If you still encounter problem, don't hesitate to contact Zichao on Slack.