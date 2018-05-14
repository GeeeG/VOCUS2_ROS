#ifndef VOCUS_ROS_H
#define VOCUS_ROS_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include "VOCUS2.h"
#include <mutex>
#include <dynamic_reconfigure/server.h>
#include <vocus2_ros/vocus2_rosConfig.h>

#include <image_geometry/pinhole_camera_model.h>

class VOCUS_ROS
{

public:


    // default constructor
  VOCUS_ROS();
  ~VOCUS_ROS();

  void restoreDefaultConfiguration();

    // this sets VOCUS's own configuration object's settings to the one given in the ROS configuration 
    // coming from the dynamic_reconfigure module
  void setVOCUSConfigFromROSConfig(VOCUS2_Cfg& vocus_cfg, const vocus2_ros::vocus2_rosConfig &config);

    // callback that is called when a new image is published to the topic
  void imageCb(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& info_msg);

    // callback that is called after the configuration in dynamic_reconfigure has been changed
  void callback(vocus2_ros::vocus2_rosConfig &config, uint32_t level);

private:
  // the VOCUS2 object that will 
  // process all the images
	VOCUS2 _vocus;

  ros::NodeHandle _nh;
  image_transport::ImageTransport _it;
  image_transport::CameraSubscriber _cam_sub;
  image_transport::Publisher _image_pub;
  image_transport::Publisher _image_sal_pub;
  ros::Publisher _poi_pub;

  image_geometry::PinholeCameraModel _cam;

  // dynamic reconfigure server and callbacktype
  dynamic_reconfigure::Server<vocus2_ros::vocus2_rosConfig> _server;
  dynamic_reconfigure::Server<vocus2_ros::vocus2_rosConfig>::CallbackType _f;

  // VOCUS configuration
  VOCUS2_Cfg _cfg;
  // ROS configuration
  vocus2_ros::vocus2_rosConfig _config;
  std::mutex cfg_mutex;
  boost::recursive_mutex config_mutex;

  // parameters that are not included in VOCUS's configuration but should be
  // configurable from the dynamic_reconfigure module
  int TOPDOWN_SEARCH = -1;
  int TOPDOWN_LEARN = -1;
  float MSR_THRESH = 0.75; // most salient region
  double CENTER_BIAS = 0.000005;
  bool COMPUTE_CBIAS = false;
  int NUM_FOCI = 1;

  // helper bools to enable button-like behavior of check boxes
  bool HAS_LEARNED = false;
  bool RESTORE_DEFAULT = false;

  const std::string OPENCV_WINDOW = "Image window";


};

  #endif // VOCUS_ROS_H
