# Using Generative Adversarial Network in Robotic Vision 

![Combine SSD and GAN](https://github.com/championway/gan_rv/blob/master/image/ssd_gan.gif)

# Development Environment
- Ubuntu 16.04
- [ROS kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)
- [Point Cloud Library](http://pointclouds.org/)
- [Realsense ROS](https://github.com/intel-ros/realsense)
- Gazebo 7.14


## Build
```
$ cd
$ git clone https://github.com/championway/gan_rv
$ cd ~/gan_rv/catkin_ws
$ source /opt/ros/kinetic/setup.bash
$ catkin_make
```
Note:
Do the following everytime as you open new terminals

```
$ cd ~/gan_rv
$ source environment.sh
```

## Pixelwise classification combining SSD and GAN

- SSD model: https://drive.google.com/open?id=1TIXA0t9OMXW6_ojoSd9jfSIlbSbagyWA
- GAN model: https://drive.google.com/open?id=1eCMgsKO7VXYg77_mAtGMSBT9Z4sod4b1
- Rosbag file: https://drive.google.com/open?id=1onTrb6HU35V7RusSw_bV8Nzmt7PLiR99

```
RGB image --> mask
$ rosrun classify gan_crop.py

RGB image + Depth image + Mask --> Point Cloud
$ roslaunch pcl_exercise mask_to_point.launch

Run rosbag recorded in tunnel(Real data)
$ rosbag play artifact_part.bag
```