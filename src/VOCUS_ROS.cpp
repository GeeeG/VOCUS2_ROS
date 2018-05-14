#include "VOCUS_ROS.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PointStamped.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <iostream>
#include <sstream>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <sys/stat.h>

#include "ImageFunctions.h"
#include "HelperFunctions.h"

using namespace cv;


VOCUS_ROS::VOCUS_ROS() : _it(_nh)
{
	_f = boost::bind(&VOCUS_ROS::callback, this, _1, _2);
	_server.setCallback(_f);
	_cam_sub = _it.subscribeCamera("image_in", 1, &VOCUS_ROS::imageCb, this);
	//_image_sub = _it.subscribe("/usb_cam/image_raw", 1,
	//	&VOCUS_ROS::imageCb, this);
	//_image_pub = _it.advertise("/image_converter/output_video", 1);
	_image_pub = _it.advertise("image_out", 1);
	_image_sal_pub = _it.advertise("saliency_image_out", 1);
        _poi_pub = _nh.advertise<geometry_msgs::PointStamped>("saliency_poi", 1);
}

VOCUS_ROS::~VOCUS_ROS()
{}


void VOCUS_ROS::restoreDefaultConfiguration()
{
	exit(0);
	boost::recursive_mutex::scoped_lock lock(config_mutex); 
	vocus2_ros::vocus2_rosConfig config;
	_server.getConfigDefault(config);
	_server.updateConfig(config);
	COMPUTE_CBIAS = config.center_bias;
	CENTER_BIAS = config.center_bias_value;
	NUM_FOCI = config.num_foci;
	MSR_THRESH = config.msr_thresh;
	TOPDOWN_LEARN = config.topdown_learn;
	TOPDOWN_SEARCH = config.topdown_search;
	setVOCUSConfigFromROSConfig(_cfg, config);
	_vocus.setCfg(_cfg);
	lock.unlock();

}

void VOCUS_ROS::setVOCUSConfigFromROSConfig(VOCUS2_Cfg& vocus_cfg, const vocus2_ros::vocus2_rosConfig &config)
{
	cfg_mutex.lock();
	vocus_cfg.fuse_feature = (FusionOperation) config.fuse_feature;  
	vocus_cfg.fuse_conspicuity = (FusionOperation) config.fuse_conspicuity;
	vocus_cfg.c_space = (ColorSpace) config.c_space;
	vocus_cfg.start_layer = config.start_layer;
    vocus_cfg.stop_layer = max(vocus_cfg.start_layer,config.stop_layer); // prevent stop_layer < start_layer
    vocus_cfg.center_sigma = config.center_sigma;
    vocus_cfg.surround_sigma = config.surround_sigma;
    vocus_cfg.n_scales = config.n_scales;
    vocus_cfg.normalize = config.normalize;
    vocus_cfg.orientation = config.orientation;
    vocus_cfg.combined_features = config.combined_features;
    vocus_cfg.descriptorFile = "topdown_descriptor";

    // individual weights?
    if (config.fuse_conspicuity == 3)
    {
    	vocus_cfg.weights[10] = config.consp_intensity_on_off_weight;
    	vocus_cfg.weights[11] = config.color_channel_1_weight;
    	vocus_cfg.weights[12] = config.color_channel_2_weight;
    	vocus_cfg.weights[13] = config.orientation_channel_weight;
    }
    if (config.fuse_feature == 3)
    {
    	vocus_cfg.weights[0] = config.intensity_on_off_weight;
    	vocus_cfg.weights[1] = config.intensity_off_on_weight;
    	vocus_cfg.weights[2] = config.color_a_on_off_weight;
    	vocus_cfg.weights[3] = config.color_a_off_on_weight;
    	vocus_cfg.weights[4] = config.color_b_on_off_weight;
    	vocus_cfg.weights[5] = config.color_b_off_on_weight;
    	vocus_cfg.weights[6] = config.orientation_1_weight;
    	vocus_cfg.weights[7] = config.orientation_2_weight;
    	vocus_cfg.weights[8] = config.orientation_3_weight;
    	vocus_cfg.weights[9] = config.orientation_4_weight;

    }

    cfg_mutex.unlock();
}


void VOCUS_ROS::imageCb(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& info_msg)
{
    _cam.fromCameraInfo(info_msg);
    if(RESTORE_DEFAULT) // if the reset checkbox has been ticked, we restore the default configuration
	{
		restoreDefaultConfiguration();
		RESTORE_DEFAULT = false;
	}

	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}


	Mat img;
        _cam.rectifyImage(cv_ptr->image, img);
	Mat salmap;
    _vocus.process(img);

	if (TOPDOWN_LEARN == 1)
	{
		Rect ROI = annotateROI(img);

    //compute feature vector
		_vocus.td_learn_featurevector(ROI, _cfg.descriptorFile);

    // to turn off learning after we've learned
    // we disable it in the ros_configuration
    // and set TOPDOWN_LEARN to -1
		cfg_mutex.lock();
		_config.topdown_learn = false;
		_server.updateConfig(_config);
		cfg_mutex.unlock();
		TOPDOWN_LEARN = -1;
		HAS_LEARNED = true;
		return;
	}
	if (TOPDOWN_SEARCH == 1 && HAS_LEARNED)
	{
		double alpha = 0;
		salmap = _vocus.td_search(alpha);
		if(_cfg.normalize){

			double mi, ma;
			minMaxLoc(salmap, &mi, &ma);
			cout << "saliency map min " << mi << " max " << ma << "\n";
			salmap = (salmap-mi)/(ma-mi);
		}
	}
	else
	{

		salmap = _vocus.compute_salmap();
        if(COMPUTE_CBIAS)
			salmap = _vocus.add_center_bias(CENTER_BIAS);
	}
     vector<vector<Point>> msrs = computeMSR(salmap,MSR_THRESH, NUM_FOCI);

	for (const auto& msr : msrs)
	{
        if (msr.size() < 3000)
        { // if the MSR is really large, minEnclosingCircle sometimes runs for more than 10 seconds,
          // freezing the whole proram. Thus, if it is very large, we 'fall back' to the efficiently
          // computable bounding rectangle
            Point2f center;
            float rad=0;
            minEnclosingCircle(msr, center, rad);
            if(rad >= 5 && rad <= max(img.cols, img.rows)){
                circle(img, center, (int)rad, Scalar(0,0,255), 3);
            }
        }
        else
        {
            Rect rect = boundingRect(msr);
            rectangle(img, rect, Scalar(0,0,255),3);
        }
	}

    // Output modified video stream
    cv_ptr->image= img;
    _image_pub.publish(cv_ptr->toImageMsg());

    // Output saliency map
    salmap *= 255.0;
    salmap.convertTo(salmap, CV_8UC1);
    cv_ptr->image = salmap;
    cv_ptr->encoding = sensor_msgs::image_encodings::MONO8;
    _image_sal_pub.publish(cv_ptr->toImageMsg());

    // Output 3D point in the direction of the first MSR
    if( msrs.size() > 0 ){
       geometry_msgs::PointStamped point;
       point.header = info_msg->header;
       cv::Point3d cvPoint = _cam.projectPixelTo3dRay(msrs[0][0]);
       point.point.x = cvPoint.x;
       point.point.y = cvPoint.y;
       point.point.z = cvPoint.z;
       _poi_pub.publish(point);
    }

}

void VOCUS_ROS::callback(vocus2_ros::vocus2_rosConfig &config, uint32_t level) 
{
	setVOCUSConfigFromROSConfig(_cfg, config);
	COMPUTE_CBIAS = config.center_bias;
	CENTER_BIAS = config.center_bias_value;

	NUM_FOCI = config.num_foci;
	MSR_THRESH = config.msr_thresh;
	TOPDOWN_LEARN = config.topdown_learn;
	if (config.restore_default) // restore default parameters before the next image is processed
		RESTORE_DEFAULT = true;
	TOPDOWN_SEARCH = config.topdown_search;
	_vocus.setCfg(_cfg);
	_config = config;
}
