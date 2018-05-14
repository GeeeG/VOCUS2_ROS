#include <ros/ros.h>
#include "VOCUS_ROS.h"

int main(int argc, char** argv)
{
	ros::init(argc, argv, "VOCUS_ROS");

	VOCUS_ROS vocus_ros;
	ros::spin();
	return 0;
}