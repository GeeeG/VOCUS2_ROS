/*****************************************************************************
*
* VOCUS2.cpp file for the saliency program VOCUS2. 
* A detailed description of the algorithm can be found in the paper: "Traditional Saliency Reloaded: A Good Old Model in New Shape", S. Frintrop, T. Werner, G. Martin Garcia, in Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2015.  
* Please cite this paper if you use our method.
*
* Design and supervision: Simone Frintrop (frintrop@iai.uni-bonn.de)
* @author Thomas Werner   (wernert@cs.uni-bonn.de)
* @author Johannes Teutrine
*
* Version 1.2
*
* This code is published under the MIT License 
* (see file LICENSE.txt for details)
*
******************************************************************************/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <omp.h>
#include <algorithm>

#include "VOCUS2.h"
#include "HelperFunctions.h"
#include "ImageFunctions.h"

VOCUS2::VOCUS2(){
	// set up a default config
	this->cfg = VOCUS2_Cfg();

	// set flags indicating status of intermediate steps
	this->salmap_ready = false;
	this->splitted_ready = false;
	this->processed = false;
}

VOCUS2::VOCUS2(const VOCUS2_Cfg& cfg) {
	// use given config
	this->cfg = cfg;

	// set flags
	this->salmap_ready = false;
	this->splitted_ready = false;
	this->processed = false;
}
/*! Destructor
 */ 
VOCUS2::~VOCUS2() {}

/*!\brief Set Configuration
 *	
 * Sets Configuration of VOCUS2.
 * @param
 */ 
void VOCUS2::setCfg(const VOCUS2_Cfg& cfg) {
  this->cfg = cfg;
  this->salmap_ready = false;
  this->splitted_ready = false;
}

// write all intermediate results to the given directory
void VOCUS2::write_out(string dir){
	if(!salmap_ready) return;

	double mi, ma;

	for(int i = 0; i < (int)pyr_center_L.size(); i++){
		for(int j = 0; j < (int)pyr_center_L[i].size(); j++){
			minMaxLoc(pyr_center_L[i][j], &mi, &ma);
			imwrite(dir + "/pyr_center_L_" + to_string(i) + "_" + to_string(j) + ".png", (pyr_center_L[i][j]-mi)/(ma-mi)*255.f);
			std::cout << "write intermediate to path: " << dir + "/pyr_center_L_" + to_string(i) + "_" + to_string(j) + ".png" << endl;
			minMaxLoc(pyr_center_a[i][j], &mi, &ma);
			imwrite(dir + "/pyr_center_a_" + to_string(i) + "_" + to_string(j) + ".png", (pyr_center_a[i][j]-mi)/(ma-mi)*255.f);

			minMaxLoc(pyr_center_b[i][j], &mi, &ma);
			imwrite(dir + "/pyr_center_b_" + to_string(i) + "_" + to_string(j) + ".png", (pyr_center_b[i][j]-mi)/(ma-mi)*255.f);

			minMaxLoc(pyr_surround_L[i][j], &mi, &ma);
			imwrite(dir + "/pyr_surround_L_" + to_string(i) + "_" + to_string(j) + ".png", (pyr_surround_L[i][j]-mi)/(ma-mi)*255.f);

			minMaxLoc(pyr_surround_a[i][j], &mi, &ma);
			imwrite(dir + "/pyr_surround_a_" + to_string(i) + "_" + to_string(j) + ".png", (pyr_surround_a[i][j]-mi)/(ma-mi)*255.f);

			minMaxLoc(pyr_surround_b[i][j], &mi, &ma);
			imwrite(dir + "/pyr_surround_b_" + to_string(i) + "_" + to_string(j) + ".png", (pyr_surround_b[i][j]-mi)/(ma-mi)*255.f);
		}
	}

	for(int i = 0; i < (int)on_off_L.size(); i++){
		minMaxLoc(on_off_L[i], &mi, &ma);
		imwrite(dir + "/on_off_L_" + to_string(i) + ".png", (on_off_L[i]-mi)/(ma-mi)*255.f);
		
		minMaxLoc(on_off_a[i], &mi, &ma);
		imwrite(dir + "/on_off_a_" + to_string(i) + ".png", (on_off_a[i]-mi)/(ma-mi)*255.f);
		
		minMaxLoc(on_off_b[i], &mi, &ma);
		imwrite(dir + "/on_off_b_" + to_string(i) + ".png", (on_off_b[i]-mi)/(ma-mi)*255.f);
		
		minMaxLoc(off_on_L[i], &mi, &ma);
		imwrite(dir + "/off_on_L_" + to_string(i) + ".png", (off_on_L[i]-mi)/(ma-mi)*255.f);
		
		minMaxLoc(off_on_a[i], &mi, &ma);
		imwrite(dir + "/off_on_a_" + to_string(i) + ".png", (off_on_a[i]-mi)/(ma-mi)*255.f);
		
		minMaxLoc(off_on_b[i], &mi, &ma);
		imwrite(dir + "/off_on_b_" + to_string(i) + ".png", (off_on_b[i]-mi)/(ma-mi)*255.f);

		
	}

	vector<Mat> tmp(6);

	tmp[0] = fuse(on_off_L, cfg.fuse_feature);
	minMaxLoc(tmp[0], &mi, &ma);
	imwrite(dir + "/feat_on_off_L.png", (tmp[0]-mi)/(ma-mi)*255.f);
	
	tmp[1] = fuse(on_off_a, cfg.fuse_feature);
	minMaxLoc(tmp[1], &mi, &ma);
	imwrite(dir + "/feat_on_off_a.png", (tmp[1]-mi)/(ma-mi)*255.f);

	tmp[2] = fuse(on_off_b, cfg.fuse_feature);
	minMaxLoc(tmp[2], &mi, &ma);
	imwrite(dir + "/feat_on_off_b.png", (tmp[2]-mi)/(ma-mi)*255.f);

	tmp[3] = fuse(off_on_L, cfg.fuse_feature);
	minMaxLoc(tmp[3], &mi, &ma);
	imwrite(dir + "/feat_off_on_L.png", (tmp[3]-mi)/(ma-mi)*255.f);
	
	tmp[4] = fuse(off_on_a, cfg.fuse_feature);
	minMaxLoc(tmp[4], &mi, &ma);
	imwrite(dir + "/feat_off_on_a.png", (tmp[4]-mi)/(ma-mi)*255.f);

	tmp[5] = fuse(off_on_b, cfg.fuse_feature);
	minMaxLoc(tmp[5], &mi, &ma);
	imwrite(dir + "/feat_off_on_b.png", (tmp[5]-mi)/(ma-mi)*255.f);
	
	for(int i = 0; i < 3; i++){
		vector<Mat> tmp2;
		tmp2.push_back(tmp[i]);
		tmp2.push_back(tmp[i+3]);

		string ch = "";

		switch(i){
		case 0: ch = "L"; break;
		case 1: ch = "a"; break;
		case 2: ch = "b"; break;
		}

		Mat tmp3 = fuse(tmp2, cfg.fuse_feature);

		minMaxLoc(tmp3, &mi, &ma);
		imwrite(dir + "/conspicuity_" + ch + ".png", (tmp3-mi)/(ma-mi)*255.f);
	}
	
	//orientation
	if(cfg.orientation){
		cout << "write orientation maps: "<< feat_orientation.size() << endl;
		for(unsigned int i=0;i<feat_orientation.size();i++){
			stringstream ss;
			ss << i;
			string ori = ss.str();
			//cout << feat_orientation[0] << endl;
			minMaxLoc(feat_orientation[i], &mi, &ma);			
			imwrite(dir + "/orientation_" + ori + ".png",  (feat_orientation[i]-mi)/(ma-mi)*255.f );
		}
	}
	
	minMaxLoc(salmap, &mi, &ma);
	imwrite(dir + "/salmap.png", (salmap-mi)/(ma-mi)*255.f);	
}
	
	

 // convert img to Lab colorspace and compute pyramids
void VOCUS2::process(const Mat& img){
	// clone the input image, not really neccessary
	input = img.clone();
	
	//make the weight vector
	makeWeight();
	
	// prepare input image (convert colorspace + split channels)
	planes = prepare_input(img);

	// call process for desired pyramid strcture
	if(cfg.pyr_struct == NEW) pyramid_new(img);  // default
	else if(cfg.pyr_struct == CODI) pyramid_codi(img);
	// else if(cfg.pyr_struct == SINGLE) pyramid_itti(img);
	else pyramid_classic(img);
 
	// set flag indicating that the pyramids are present
	this->processed = true;

	// compute center surround contrast
	center_surround_diff();		

	if(cfg.orientation)	orientation();

	// for(int i = 0; i < 4; i++){
	// 	for(int o = 0; o < gabor[i].size(); o++){
	// 		for(int s = 0; s < cfg.n_scales; s++){
	// 			int pos = o*cfg.n_scales+s;
	// 			imshow(gabor

}

// void VOCUS2::pyramid_itti(const Mat& img){
	
// 	// clear previous results
// 	clear();

// 	// set flags
// 	salmap_ready = false;
// 	splitted_ready = false;

// 	// prepare input image (convert colorspace + split planes)
// 	planes = prepare_input(img);

// 	// special itti settings
// 	float itti_sigma = 1.f;
// 	cfg.start_layer = 0;
// 	cfg.stop_layer = 8;
// 	cfg.n_scales = 1;

// 	// layers used by iNVT
// 	int starts[5] = {0,1,2};
// 	int steps[2] = {3, 4};

// 	// if itti color space is used (seperate R and G, and B and Y)
// 	if(cfg.c_space == ITTI){

// 		// build pyramids
// 		vector<vector<Mat> > pyr_base_L, pyr_base_r, pyr_base_g, pyr_base_b, pyr_base_y;

// #pragma omp parallel sections
// 		{
// #pragma omp section
// 			pyr_base_L = build_multiscale_pyr(planes[0], itti_sigma);
// #pragma omp section
// 			pyr_base_r = build_multiscale_pyr(planes[1], itti_sigma);
// #pragma omp section
// 			pyr_base_g = build_multiscale_pyr(planes[2], itti_sigma);
// #pragma omp section
// 			pyr_base_b = build_multiscale_pyr(planes[3], itti_sigma);
// #pragma omp section
// 			pyr_base_y = build_multiscale_pyr(planes[4], itti_sigma);
// 		}

// 		// for all layer and for all step sizes compute DoG and put results into 
// 		// contrast pyramids
// 		for(int i = 0; i < 4; i++){
// 			for(int j = 0; j < 2; j++){

// 				// if number of possible layers is too low => skip
// 				if(starts[i]+steps[j] >= pyr_base_L.size()) continue;
			
// 				Mat diff, tmp1, tmp2;

// 				// --- L channel ----
// 				//
// 				// resize to to larger layer
// 				resize(pyr_base_L[starts[i]+steps[j]][0], tmp1, pyr_base_L[starts[i]][0].size());
// 				diff = abs(pyr_base_L[starts[i]][0] - tmp1);
// 				on_off_L.push_back(diff);
// 				off_on_L.push_back(diff);

// 				// a channel
// 				tmp1 = pyr_base_r[starts[i]][0]-pyr_base_g[starts[i]][0];
// 				tmp2 = pyr_base_r[starts[i]+steps[j]][0]-pyr_base_g[starts[i]+steps[j]][0];
			
// 				resize(tmp2, tmp2, tmp1.size());

// 				diff = abs(tmp1-tmp2);
// 				on_off_a.push_back(diff);
// 				off_on_a.push_back(diff);

// 				// b channel
// 				tmp1 = pyr_base_b[starts[i]][0]-pyr_base_y[starts[i]][0];
// 				tmp2 = pyr_base_b[starts[i]+steps[j]][0]-pyr_base_y[starts[i]+steps[j]][0];
			
// 				resize(tmp2, tmp2, tmp1.size());

// 				diff = abs(tmp1-tmp2);
// 				on_off_b.push_back(diff);
// 				off_on_b.push_back(diff);
// 			}
// 		}
// 	}

// 	else{
// 		vector<vector<Mat> > pyr_base_L, pyr_base_a, pyr_base_b;

// #pragma omp parallel sections
// 		{
// #pragma omp section
// 			pyr_base_L = build_multiscale_pyr(planes[0], itti_sigma);
// #pragma omp section
// 			pyr_base_a = build_multiscale_pyr(planes[1], itti_sigma);
// #pragma omp section
// 			pyr_base_b = build_multiscale_pyr(planes[2], itti_sigma);
// 		}

// 		for(int i = 0; i < 3; i++){
// 			for(int j = 0; j < 2; j++){

// 				if(starts[i]+steps[j] >= pyr_base_L.size()) continue;
			
// 				Mat diff, tmp1;

// 				// L channel
// 				resize(pyr_base_L[starts[i]+steps[j]][0], tmp1, pyr_base_L[starts[i]][0].size());
// 				diff = abs(pyr_base_L[starts[i]][0] - tmp1);
// 				on_off_L.push_back(diff);
// 				off_on_L.push_back(diff);

// 				// a channel
// 				resize(pyr_base_a[starts[i]+steps[j]][0], tmp1, pyr_base_a[starts[i]][0].size());
// 				diff = abs(pyr_base_a[starts[i]][0] - tmp1);
// 				on_off_a.push_back(diff);
// 				off_on_a.push_back(diff);

// 				// b channel
// 				resize(pyr_base_b[starts[i]+steps[j]][0], tmp1, pyr_base_b[starts[i]][0].size());
// 				diff = abs(pyr_base_b[starts[i]][0] - tmp1);
// 				on_off_b.push_back(diff);
// 				off_on_b.push_back(diff);
// 			}
// 		}
// 	}

// }


//Codi pyramid
void VOCUS2::pyramid_codi(const Mat& img){
	// clear previous results
	clear();

	// set flags
	salmap_ready = false;
	splitted_ready = false;

	// create base pyramids
	vector<vector<Mat> > pyr_base_L, pyr_base_a, pyr_base_b;
#pragma omp parallel sections
	{
#pragma omp section
	pyr_base_L = build_multiscale_pyr(planes[0], 1.f);
#pragma omp section
	pyr_base_a = build_multiscale_pyr(planes[1], 1.f);
#pragma omp section
	pyr_base_b = build_multiscale_pyr(planes[2], 1.f);
	}

	// recompute sigmas that are needed to reach the desired
	// smoothing for center and surround
	float adapted_center_sigma = sqrt(pow(cfg.center_sigma,2)-1);
	float adapted_surround_sigma = sqrt(pow(cfg.surround_sigma,2)-1);

	// reserve space
	pyr_center_L.resize(pyr_base_L.size());
	pyr_center_a.resize(pyr_base_L.size());
	pyr_center_b.resize(pyr_base_L.size());
	pyr_surround_L.resize(pyr_base_L.size());
	pyr_surround_a.resize(pyr_base_L.size());
	pyr_surround_b.resize(pyr_base_L.size());

	// for every layer of the pyramid
	for(int o = 0; o < (int)pyr_base_L.size(); o++){
		pyr_center_L[o].resize(cfg.n_scales);
		pyr_center_a[o].resize(cfg.n_scales);
		pyr_center_b[o].resize(cfg.n_scales);
		pyr_surround_L[o].resize(cfg.n_scales);
		pyr_surround_a[o].resize(cfg.n_scales);
		pyr_surround_b[o].resize(cfg.n_scales);

		// for all scales build the center and surround pyramids independently
#pragma omp parallel for
		for(int s = 0; s < cfg.n_scales; s++){

			float scaled_center_sigma = adapted_center_sigma*pow(2.0, (double)s/(double)cfg.n_scales);
			float scaled_surround_sigma = adapted_surround_sigma*pow(2.0, (double)s/(double)cfg.n_scales);

			GaussianBlur(pyr_base_L[o][s], pyr_center_L[o][s], Size(), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_base_L[o][s], pyr_surround_L[o][s], Size(), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);

			GaussianBlur(pyr_base_a[o][s], pyr_center_a[o][s], Size(), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_base_a[o][s], pyr_surround_a[o][s], Size(), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);

			GaussianBlur(pyr_base_b[o][s], pyr_center_b[o][s], Size(), scaled_center_sigma, scaled_center_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_base_b[o][s], pyr_surround_b[o][s], Size(), scaled_surround_sigma, scaled_surround_sigma, BORDER_REPLICATE);
		}
	}
}



//new pyramids
void VOCUS2::pyramid_new(const Mat& img){
	// clear previous results
	clear();

	salmap_ready = false;
	splitted_ready = false;

	// build center pyramid
#pragma omp parallel sections
	{
#pragma omp section
	pyr_center_L = build_multiscale_pyr(planes[0], (float)cfg.center_sigma);
#pragma omp section
	pyr_center_a = build_multiscale_pyr(planes[1], (float)cfg.center_sigma);
#pragma omp section
	pyr_center_b = build_multiscale_pyr(planes[2], (float)cfg.center_sigma);
	}

	// compute new surround sigma (paper: sigma_x)
	float adapted_sigma = sqrt(pow(cfg.surround_sigma,2)-pow(cfg.center_sigma,2));

	// reserve space
	pyr_surround_L.resize(pyr_center_L.size());
	pyr_surround_a.resize(pyr_center_a.size());
	pyr_surround_b.resize(pyr_center_b.size());

	// for all layers (octaves)
	for(int o = 0; o < (int)pyr_center_L.size(); o++){
		pyr_surround_L[o].resize(cfg.n_scales);
		pyr_surround_a[o].resize(cfg.n_scales);
		pyr_surround_b[o].resize(cfg.n_scales);

		// for all scales, compute surround counterpart
#pragma omp parallel for
		for(int s = 0; s < cfg.n_scales; s++){
			float scaled_sigma = adapted_sigma*pow(2.0, (double)s/(double)cfg.n_scales);

			GaussianBlur(pyr_center_L[o][s], pyr_surround_L[o][s], Size(), scaled_sigma, scaled_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_center_a[o][s], pyr_surround_a[o][s], Size(), scaled_sigma, scaled_sigma, BORDER_REPLICATE);
			GaussianBlur(pyr_center_b[o][s], pyr_surround_b[o][s], Size(), scaled_sigma, scaled_sigma, BORDER_REPLICATE);
		}
	}
}

//classic pyramid
void VOCUS2::pyramid_classic(const Mat& img){
	// clear previous results
	clear();

	salmap_ready = false;
	splitted_ready = false;

	// compute center and surround pyramid directly but independent
#pragma omp parallel sections
{
#pragma omp section
	pyr_center_L = build_multiscale_pyr(planes[0], (float)cfg.center_sigma);

#pragma omp section
	pyr_center_a = build_multiscale_pyr(planes[1], (float)cfg.center_sigma);

#pragma omp section
	pyr_center_b = build_multiscale_pyr(planes[2], (float)cfg.center_sigma);

#pragma omp section
	pyr_surround_L = build_multiscale_pyr(planes[0], (float)cfg.surround_sigma);

#pragma omp section
	pyr_surround_a = build_multiscale_pyr(planes[1], (float)cfg.surround_sigma);

#pragma omp section
	pyr_surround_b = build_multiscale_pyr(planes[2], (float)cfg.surround_sigma);
}
}



 // center surround difference
void VOCUS2::center_surround_diff(){
	int on_off_size = pyr_center_L.size()*cfg.n_scales;

	on_off_L.resize(on_off_size); off_on_L.resize(on_off_size);
	on_off_a.resize(on_off_size); off_on_a.resize(on_off_size);
	on_off_b.resize(on_off_size); off_on_b.resize(on_off_size);

	// the following code could be faster on processors with more than 4 cores:

 // #pragma omp parallel for
 // 	for(int i = 0; i < on_off_size; i++){
 // 		int o = i/cfg.n_scales;
 // 		int s = i % cfg.n_scales;

 // 		Mat diff;

 // 		// ========== L channel ==========
 // 		diff = pyr_center_L[o][s]-pyr_surround_L[o][s];
 // 		threshold(diff, on_off_L[i], 0, 1, THRESH_TOZERO);
 // 		diff *= -1.f;
 // 		threshold(diff, off_on_L[i], 0, 1, THRESH_TOZERO);

 // 		// ========== a channel ==========
 // 		diff = pyr_center_a[o][s]-pyr_surround_a[o][s];
 // 		threshold(diff, on_off_a[i], 0, 1, THRESH_TOZERO);
 // 		diff *= -1.f;
 // 		threshold(diff, off_on_a[i], 0, 1, THRESH_TOZERO);

 // 		// ========== b channel ==========
 // 		diff = pyr_center_b[o][s]-pyr_surround_b[o][s];
 // 		threshold(diff, on_off_b[i], 0, 1, THRESH_TOZERO);
 // 		diff *= -1.f;
 // 		threshold(diff, off_on_b[i], 0, 1, THRESH_TOZERO);
 // 	}

	// compute DoG by subtracting layers of two pyramids
	for(int o = 0; o < (int)pyr_center_L.size(); o++){
#pragma omp parallel for
		for(int s = 0; s < cfg.n_scales; s++){
			Mat diff;
			int pos = o*cfg.n_scales+s;

			// ========== L channel ==========
			diff = pyr_center_L[o][s]-pyr_surround_L[o][s];
			threshold(diff, on_off_L[pos], 0, 1, THRESH_TOZERO);
			diff *= -1.f;
			threshold(diff, off_on_L[pos], 0, 1, THRESH_TOZERO);

			// ========== a channel ==========
			diff = pyr_center_a[o][s]-pyr_surround_a[o][s];
			threshold(diff, on_off_a[pos], 0, 1, THRESH_TOZERO);
			diff *= -1.f;
			threshold(diff, off_on_a[pos], 0, 1, THRESH_TOZERO);

			// ========== b channel ==========
			diff = pyr_center_b[o][s]-pyr_surround_b[o][s];
			threshold(diff, on_off_b[pos], 0, 1, THRESH_TOZERO);
			diff *= -1.f;
			threshold(diff, off_on_b[pos], 0, 1, THRESH_TOZERO);
		}
	}
}



//orientation channel
void VOCUS2::orientation(){

	// reserve space
	pyr_laplace.resize(pyr_center_L.size());
	for(int o = 0; o < (int)pyr_center_L.size(); o++){
		pyr_laplace[o].resize(pyr_center_L[o].size());
	}

	gabor.resize(4);
	for(int i = 0; i < 4; i++) gabor[i].resize(pyr_center_L.size()*cfg.n_scales);


	// build all layers of laplace pyramid except the last one
#pragma omp parallel for
	for(int o = 0; o < (int)pyr_center_L.size()-1; o++){
		for(int s = 0; s < (int)pyr_center_L[o].size(); s++){
			Mat& src1 = pyr_center_L[o][s];
			Mat& src2 = pyr_center_L[o+1][s];

			Mat tmp;
			resize(src2, tmp, src1.size(), INTER_NEAREST);

			// pyr_laplace[o][s] = (src1-tmp+1)/2;
			pyr_laplace[o][s] = src1-tmp;
		}
	}

	// copy last layer
	for(int s = 0; s < cfg.n_scales; s++){
		pyr_laplace[pyr_center_L.size()-1][s] = pyr_center_L[pyr_center_L.size()-1][s];
	}

	int filter_size = 11*cfg.center_sigma+1;

	for(int ori = 0; ori < 4; ori++){
		Mat gaborKernel = getGaborKernel(Size(filter_size,filter_size), 2*cfg.center_sigma, (ori*M_PI)/4, 10, .5, 2*CV_PI);

		float k_sum =  sum(sum(abs(gaborKernel)))[0];
		gaborKernel /= k_sum;
		
		double mi, ma;
		minMaxLoc(gaborKernel, &mi, &ma);
		// imshow("gaborKernel streched", (gaborKernel-mi)/(ma-mi)); waitKey(0);

		for(int o = 0; o < (int)pyr_laplace.size(); o++){
			for(int s = 0; s < cfg.n_scales; s++){
				int pos = o*cfg.n_scales+s;
				
				Mat& src = pyr_laplace[o][s];
				Mat& dst = gabor[ori][pos];

				minMaxLoc(src, &mi, &ma);
				// imshow("laplace streched", (src-mi)/(ma-mi));

				filter2D(src, dst, -1, gaborKernel);

				// minMaxLoc(dst, &mi, &ma);
				// cout << "mi: " << mi << " ma: " << ma << endl;

				dst = abs(dst);

				 //imshow("response streched", dst);
				 //waitKey(0);
			}
		}
	}
}




//if image is processed, this returns saliency, feature maps and consp. maps.
//this is used to compute Top Down Learning.
/*
Mat VOCUS2::get_TD_essentials(vector<Mat>& f_intensity, vector<Mat>& f_color1, vector<Mat>& f_color2, vector<Mat>& f_or, vector<Mat>& consp){

	// check if center surround contrasts are computed
	if(!processed){
		cout << "Image not yet processed. Call process(Mat)." << endl;
		return Mat();
	}	
	
	// if saliency map is already present => return it
	//if(salmap_ready) return salmap;

	// intensity feature maps
	vector<Mat> feature_intensity;
	feature_intensity.push_back(fuse(on_off_L, cfg.fuse_feature));
	feature_intensity.push_back(fuse(off_on_L, cfg.fuse_feature));
	f_intensity = feature_intensity;
	// color feature maps
	vector<Mat> feature_color1, feature_color2;

	if(cfg.combined_features){
		feature_color1.push_back(fuse(on_off_a, cfg.fuse_feature));
		feature_color1.push_back(fuse(off_on_a, cfg.fuse_feature));
		feature_color1.push_back(fuse(on_off_b, cfg.fuse_feature)); 
		feature_color1.push_back(fuse(off_on_b, cfg.fuse_feature));
	}
	else{
		feature_color1.push_back(fuse(on_off_a, cfg.fuse_feature));
		feature_color1.push_back(fuse(off_on_a, cfg.fuse_feature));
		feature_color2.push_back(fuse(on_off_b, cfg.fuse_feature)); 
		feature_color2.push_back(fuse(off_on_b, cfg.fuse_feature));

	}
	f_color1=feature_color1;
	f_color2=feature_color2;

	vector<Mat> feature_orientation;
	if(cfg.orientation && cfg.combined_features){
		for(int i = 0; i < 4; i++){		
			feature_orientation.push_back(fuse(gabor[i], cfg.fuse_feature));
		}
	}
	f_or = feature_orientation;
	
	// conspicuity maps
	vector<Mat> conspicuity_maps;
	conspicuity_maps.push_back(fuse(feature_intensity, cfg.fuse_conspicuity)); //hier 
	out.push_back(conspicuity_maps[0]);
	if(cfg.combined_features){
		conspicuity_maps.push_back(fuse(feature_color1, cfg.fuse_conspicuity)); //hier
		if(cfg.orientation){
			 conspicuity_maps.push_back(fuse(feature_orientation, cfg.fuse_conspicuity)); //hier ft_or
		 }
	}
	else{
		conspicuity_maps.push_back(fuse(feature_color1, cfg.fuse_conspicuity)); //hier col1
		conspicuity_maps.push_back(fuse(feature_color2, cfg.fuse_conspicuity)); //hier col2
		if(cfg.orientation){
			for(int i = 0; i < 4; i++){		
				conspicuity_maps.push_back(fuse(gabor[i], cfg.fuse_feature)); //hier or
			}
		}
	}
	consp = conspicuity_maps;

	// saliency map
	salmap = fuse(conspicuity_maps, cfg.fuse_conspicuity); //hier consp1, consp2, consp3

	// normalize output to [0,1]
	if(cfg.normalize){
		double mi, ma;
		minMaxLoc(salmap, &mi, &ma);
		salmap = (salmap-mi)/(ma-mi);
	}

	// resize to original image size
	resize(salmap, salmap, input.size(), 0, 0, INTER_CUBIC);
	
	salmap_ready = true;

	return salmap;	
}
*/


//initializes the weights, if not specified by the user, to standart values. 
void VOCUS2::makeWeight(){
	// *********************** DANIEL EDIT START *******************
	// instead of pushing all weights to the end, we directly put them into their index
	// this has two advantages:
	// 1. we dont keep increasing the vector size and new weights will be ignored (because access is on index basis)
	// 2. more efficient
	if(weightValues.size() == 0)
		weightValues.resize(cfg.weights.size());
	for (int i = 0; i < cfg.weights.size(); i++)
	{
		if (i < 6)
			cfg.weights[i]==-1 ? weightValues[i] = 0.5 : weightValues[i] = cfg.weights[i];
		else
			cfg.weights[i]==-1 ? weightValues[i] = 0.25 : weightValues[i] = cfg.weights[i];

	}
	/*cfg.weights[0]==-1 ? weightValues.push_back(0.5) : weightValues.push_back(cfg.weights[0]);
	cfg.weights[1]==-1 ? weightValues.push_back(0.5) : weightValues.push_back(cfg.weights[1]);
	cfg.weights[2]==-1 ? weightValues.push_back(0.5) : weightValues.push_back(cfg.weights[2]);
	cfg.weights[3]==-1 ? weightValues.push_back(0.5) : weightValues.push_back(cfg.weights[3]);
	cfg.weights[4]==-1 ? weightValues.push_back(0.5) : weightValues.push_back(cfg.weights[4]);
	cfg.weights[5]==-1 ? weightValues.push_back(0.5) : weightValues.push_back(cfg.weights[5]);
	cfg.weights[6]==-1 ? weightValues.push_back(0.25) : weightValues.push_back(cfg.weights[6]);
	cfg.weights[7]==-1 ? weightValues.push_back(0.25) : weightValues.push_back(cfg.weights[7]);
	cfg.weights[8]==-1 ? weightValues.push_back(0.25) : weightValues.push_back(cfg.weights[8]);
	cfg.weights[9]==-1 ? weightValues.push_back(0.25) : weightValues.push_back(cfg.weights[9]);
	cfg.weights[10]==-1 ? weightValues.push_back(0.25) : weightValues.push_back(cfg.weights[10]);
	cfg.weights[11]==-1 ? weightValues.push_back(0.25) : weightValues.push_back(cfg.weights[11]);
	cfg.weights[12]==-1 ? weightValues.push_back(0.25) : weightValues.push_back(cfg.weights[12]);
	cfg.weights[13]==-1 ? weightValues.push_back(0.25) : weightValues.push_back(cfg.weights[13]);*/
	// *********************** DANIEL EDIT END *******************


	///1 - int on off  -1
	///2 - int off on  -2
	///3 - col on off a  -3
	///4 - col off on a  -4
	///5 - col on off b  -5
	///6 - col of onn b  -6
	///7 - or 1  -7
	///8 - or 2  -8 
	///9 - or 3  -9
	///10 - or 4  -10 
	///11 - consp int  -u
	///12 - consp col 1  -i
	///13 - consp col 2  -j
	///14 - consp or  -k 
}

//returns the weight from the config file or computes a default weight value when
//no command line weight was given (default value depends on other given weights)
//Each weight constellation has a string identifier
vector<double> VOCUS2::getWeight(string IDString){
	vector<double> weights;	
	if(!strcmp(IDString.c_str(),"CONSP")){
		//cout << "conspweights" << endl;
		//cout << weightValues[10] << weightValues[11] << weightValues[12] << weightValues[13] <<endl;
		weights.push_back(weightValues[10]);
		weights.push_back(weightValues[11]);		
		weights.push_back(weightValues[12]);	
		return weights;
	}
	else if(!strcmp(IDString.c_str(),"CONSP_COMB")){
		weights.push_back(weightValues[10]);
		weights.push_back(weightValues[11]);	
		return weights;
	}
	else if(!strcmp(IDString.c_str(),"CONSP_OR")){
		weights.push_back(weightValues[10]);
		weights.push_back(weightValues[11]);
		weights.push_back(weightValues[12]);
		weights.push_back(weightValues[13]);
		return weights;
	}
	else if(!strcmp(IDString.c_str(),"CONSP_COMB_OR")){
		weights.push_back(weightValues[10]);
		weights.push_back(weightValues[11]);	
		weights.push_back(weightValues[13]);
		return weights;
	}
	else if(!strcmp(IDString.c_str(),"FT_INT")){
		//cout << "intweights" << endl;
		//cout << weightValues[0] << endl;
		weights.push_back(weightValues[0]);
		weights.push_back(weightValues[1]);	
		return weights;
	}
	else if(!strcmp(IDString.c_str(),"FT_COL1")){
		//cout << "col1weights" << endl;
		weights.push_back(weightValues[2]);
		weights.push_back(weightValues[3]);	
		return weights;
	}
	else if(!strcmp(IDString.c_str(),"FT_COL2")){
		//cout << "col2weights" << endl;
		weights.push_back(weightValues[4]);
		weights.push_back(weightValues[5]);	
		return weights;
	}
	else if(!strcmp(IDString.c_str(),"FT_COL")){
		weights.push_back(weightValues[2]);
		weights.push_back(weightValues[3]);	
		weights.push_back(weightValues[4]);
		weights.push_back(weightValues[5]);	
		return weights;
	}
	
	
	
	std::cout << "No matching channel identified." << endl;
	return weights;
}




Mat VOCUS2::compute_tdmap(){
	// check if center surround contrasts are computed
	if(!processed){
		cout << "Image not yet processed. Call process(Mat)." << endl;
		return Mat();
	}
	
	parseDescriptorFile();
	
	// if saliency map is already present => return it
	// intensity feature maps
	vector<Mat> feature_intensity;
	feature_intensity.push_back(fuse(on_off_L, cfg.fuse_feature));
	//exitation and inhibition map
	exmap = Mat::zeros(feature_intensity[0].size(), CV_32F);
	inmap = Mat::zeros(feature_intensity[0].size(), CV_32F);
	
	int n_exmap=0;
	int n_inmap=0;
	//compute ex and in map
	if(descriptorWeights[0].size()==2){
		if(descriptorWeights[0][0]>=1){ //exite
			cv::add(exmap, descriptorWeights[0][0]*feature_intensity[0], exmap, Mat(), CV_32F);
			n_exmap++;
		}
		else{ //inhibit
			if(descriptorWeights[0][0]<1 && descriptorWeights[0][0]>0){
				//if(descriptorWeights[0][0]==0) descriptorWeights[0][0] = 
				double v = (1./descriptorWeights[0][0]);
				cv::add(inmap, v*feature_intensity[0], inmap, Mat(), CV_32F);
				n_inmap++;
			}
		}
	}
	else{
		return Mat();
	}

	feature_intensity.push_back(fuse(off_on_L, cfg.fuse_feature));
	if(descriptorWeights[0][1]>1){
			cv::add(exmap, descriptorWeights[0][1]*feature_intensity[1], exmap, Mat(), CV_32F);
			n_exmap++;
	}
	else{
		if(descriptorWeights[0][1]!=0){
			double v = (1./descriptorWeights[0][1]);
			cv::add(inmap, v*feature_intensity[1], inmap, Mat(), CV_32F);
			n_inmap++;
		}
	}

	
	// color feature maps
	vector<Mat> feature_color1, feature_color2;
	if(cfg.combined_features){
		feature_color1.push_back(fuse(on_off_a, cfg.fuse_feature));
		feature_color1.push_back(fuse(off_on_a, cfg.fuse_feature));
		feature_color1.push_back(fuse(on_off_b, cfg.fuse_feature)); 
		feature_color1.push_back(fuse(off_on_b, cfg.fuse_feature));
		
		if(descriptorWeights[1].size()!=4){
			cout << "color1 does not match in combined features" << endl;
			return Mat();
		}
		for(int i=0;i<4;i++){
			if(descriptorWeights[1][i]>1){
				cv::add(exmap, descriptorWeights[1][i]*feature_color1[i], exmap, Mat(), CV_32F);
				n_exmap++;
			}
			else{
				if(descriptorWeights[1][i]!=0){

					double v = (1./descriptorWeights[1][i]);
					cv::add(inmap, v*feature_color1[i], inmap, Mat(), CV_32F);
					n_inmap++;
				}
			}
		}
		
	}
	else{
		feature_color1.push_back(fuse(on_off_a, cfg.fuse_feature));
		//cout << feature_color1[0] << endl;
		feature_color1.push_back(fuse(off_on_a, cfg.fuse_feature));
		if(descriptorWeights[1].size()!=2){
			cout << "color1 does not match in uncombined features" << endl;
			return Mat();
		}
		if(descriptorWeights[1][0]>1){
			cv::add(exmap, descriptorWeights[1][0]*feature_color1[0], exmap, Mat(), CV_32F);
			n_exmap++;
		}
		else{
			if(descriptorWeights[1][0]!=0){
				double v = (1./descriptorWeights[1][0]);
				cv::add(inmap, v*feature_color1[0], inmap, Mat(), CV_32F);
				n_inmap++;
			}
		}
		if(descriptorWeights[1][1]>1){
			cv::add(exmap, descriptorWeights[1][1]*feature_color1[1], exmap, Mat(), CV_32F);
			n_exmap++;
		}
		else{
			if(descriptorWeights[1][1]!=0){
				double v = (1./descriptorWeights[1][1]);
				cv::add(inmap, v*feature_color1[1], inmap, Mat(), CV_32F);
				n_inmap++;
			}
		}
		
		
		
		feature_color2.push_back(fuse(on_off_b, cfg.fuse_feature)); 
		feature_color2.push_back(fuse(off_on_b, cfg.fuse_feature));
		
		if(descriptorWeights[2].size()!=2){
			cout << "color1 does not match in uncombined features" << endl;
			return Mat();
		}
		if(descriptorWeights[2][0]>1){
			cv::add(exmap, descriptorWeights[2][0]*feature_color2[0], exmap, Mat(), CV_32F);
			n_exmap++;
		}
		else{	
			if(descriptorWeights[2][0]!=0){
				double v = (1./descriptorWeights[2][0]);
				cv::add(inmap, v*feature_color2[0], inmap, Mat(), CV_32F);
				n_inmap++;
			}
		}
		if(descriptorWeights[2][1]>1){
			cv::add(exmap, descriptorWeights[2][1]*feature_color2[1], exmap, Mat(), CV_32F);
			n_exmap++;
		}
		else{
			if(descriptorWeights[2][1]!=0){
				double v = (1./descriptorWeights[2][1]);
				cv::add(inmap, v*feature_color2[1], inmap, Mat(), CV_32F);
				n_inmap++;
			}
		}
	}

	vector<Mat> feature_orientation;
	if(cfg.orientation && cfg.combined_features && descriptorWeights[3].size()==4){
		cout << "Use orientation feature" << endl;
		for(unsigned int i = 0; i < 4; i++){		
			feature_orientation.push_back(fuse(gabor[i], cfg.fuse_feature));
			if(descriptorWeights[3][i]>1){
				cv::add(exmap, descriptorWeights[3][i]*feature_orientation[i], exmap, Mat(), CV_32F);
				n_exmap++;
			}
			else{
				if(descriptorWeights[3][i]!=0){
					double v = (1./descriptorWeights[3][i]);
					cv::add(inmap, v*feature_orientation[i], inmap, Mat(), CV_32F);
					n_inmap++;
				}
			}
		}
	}
	else if (descriptorWeights[3].size()!=4){
		//cout << "no orientation features used" << endl;
	}

	// conspicuity maps
	vector<Mat> conspicuity_maps;
	conspicuity_maps.push_back(fuse(feature_intensity, cfg.fuse_conspicuity));

	if(cfg.combined_features){
		conspicuity_maps.push_back(fuse(feature_color1, cfg.fuse_conspicuity));
		if(cfg.orientation){
			 conspicuity_maps.push_back(fuse(feature_orientation, cfg.fuse_conspicuity));
		 }
	}
	else{
		conspicuity_maps.push_back(fuse(feature_color1, cfg.fuse_conspicuity));
		conspicuity_maps.push_back(fuse(feature_color2, cfg.fuse_conspicuity));
		if(cfg.orientation){
			for(int i = 0; i < 4; i++){		
				conspicuity_maps.push_back(fuse(gabor[i], cfg.fuse_feature));
			}
		}
	}
	
	//weight consp maps 
	if(conspicuity_maps.size()!=descriptorWeights[4].size()){
		cout << "consp maps size does not match descriptorWeights size!" << endl;
	}
		
		
	for(unsigned int i=0;i<conspicuity_maps.size();i++){
		if(descriptorWeights[4][i]>1){
			cv::add(exmap, descriptorWeights[4][i]*conspicuity_maps[i], exmap, Mat(), CV_32F);
			n_exmap++;
		}
		else{
			if(descriptorWeights[4][i]!=0){
				double v = (1./descriptorWeights[4][i]);
				cv::add(inmap, v*conspicuity_maps[i], inmap, Mat(), CV_32F);
				n_inmap++;
			}
		}
	}	

	// scale ex- and in map and compute tdmap
	exmap = exmap/n_exmap;
	double mi1, ma1;
	minMaxLoc(exmap, &mi1, &ma1);
	exmap = (exmap-mi1)/(ma1-mi1);
	
	double mi2, ma2;
	minMaxLoc(inmap, &mi2, &ma2);
	inmap = (inmap-mi2)/(ma2-mi2);	
	inmap = inmap/n_inmap;
	tdmap = exmap-inmap;

	// normalize output to [0,1]
	if(cfg.normalize){
		double mi, ma;
		minMaxLoc(salmap, &mi, &ma);
		salmap = (salmap-mi)/(ma-mi);
	}

	// resize to original image size
	resize(tdmap, tdmap, input.size(), 0, 0, INTER_CUBIC);
	
	td_ready = true;
	
	return tdmap;
}



Mat VOCUS2::compute_salmap(){
	//cout << "getting salmap" << endl;
	// check if center surround contrasts are computed
	
	if(!processed){
		cout << "Image not yet processed. Call process(Mat)." << endl;
		return Mat();
	}

	// if saliency map is already present => return it
	if(salmap_ready) return salmap;

	// intensity feature maps
	vector<Mat> feature_intensity;
	feature_intensity.push_back(fuse(on_off_L, cfg.fuse_feature));
	feature_intensity.push_back(fuse(off_on_L, cfg.fuse_feature));
	//cout << "ft intensity size in computation" << feature_intensity.size() << endl;
	// color feature maps
	vector<Mat> feature_color1, feature_color2;

	if(cfg.combined_features){
		feature_color1.push_back(fuse(on_off_a, cfg.fuse_feature));
		feature_color1.push_back(fuse(off_on_a, cfg.fuse_feature));
		feature_color1.push_back(fuse(on_off_b, cfg.fuse_feature)); 
		feature_color1.push_back(fuse(off_on_b, cfg.fuse_feature));
	}
	else{
		feature_color1.push_back(fuse(on_off_a, cfg.fuse_feature));
		feature_color1.push_back(fuse(off_on_a, cfg.fuse_feature));
		feature_color2.push_back(fuse(on_off_b, cfg.fuse_feature)); 
		feature_color2.push_back(fuse(off_on_b, cfg.fuse_feature));
	}

	vector<Mat> feature_orientation;
	if(cfg.orientation && cfg.combined_features){
		for(int i = 0; i < 4; i++){		
			feature_orientation.push_back(fuse(gabor[i], cfg.fuse_feature));
		}
	}
	else if(cfg.orientation && !cfg.combined_features){
		for(int i = 0; i < 4; i++){		
			feature_orientation.push_back(fuse(gabor[i], cfg.fuse_feature));
		}
	}
		

	// conspicuity maps
	//************** DANIEL EDIT START **************
	// since we are fusing feature maps to form conspicuity maps
	// I believe we have to use the setting cfg.fuse_feature instead of cfg.fuse_conspicuity.

	vector<Mat> conspicuity_maps;
	conspicuity_maps.push_back(fuse(feature_intensity, cfg.fuse_feature, getWeight("FT_INT")));

	if(cfg.combined_features){
		conspicuity_maps.push_back(fuse(feature_color1, cfg.fuse_feature, getWeight("FT_COL")));
		if(cfg.orientation){
			 conspicuity_maps.push_back(fuse(feature_orientation, cfg.fuse_feature));
		 }
	}
	else{
		conspicuity_maps.push_back(fuse(feature_color1, cfg.fuse_feature, getWeight("FT_COL1")));
		conspicuity_maps.push_back(fuse(feature_color2, cfg.fuse_feature, getWeight("FT_COL2")));
		if(cfg.orientation){
			for(int i = 0; i < 4; i++){		
				conspicuity_maps.push_back(fuse(gabor[i], cfg.fuse_feature, getWeight("FT_OR")));
			}
		}
	}
	//************** DANIEL EDIT END   **************
	string conspstr("CONSP");
	if(cfg.combined_features) conspstr+="_COMB"; 
	if(cfg.orientation) conspstr+="_OR";
	salmap = fuse(conspicuity_maps, cfg.fuse_conspicuity, getWeight(conspstr));

	// normalize output to [0,1]
	if(cfg.normalize){
		double mi, ma;
		minMaxLoc(salmap, &mi, &ma);
		salmap = (salmap-mi)/(ma-mi);
	}

	// resize to original image size
	resize(salmap, salmap, input.size(), 0, 0, INTER_CUBIC);
	
	salmap_ready = true;
	
	//cout << "leaving this part" << endl;
	
	//copy vectors to class members
	consp_maps = conspicuity_maps;
	feat_color1 = feature_color1;
	feat_color2 = feature_color2;
	feat_intensity = feature_intensity;
	feat_orientation = feature_orientation;
	
	return salmap;
}


//center bias
Mat VOCUS2::add_center_bias(float size){
	if(!salmap_ready){
		cerr << "add_center_bias: Failed adding bias, Saliency Map not computed." << endl;
		exit(EXIT_FAILURE);
	}
	const double sigmaX = -log2(size)*0.5*salmap.cols;
	const double sigmaY = -log2(size)*0.5*salmap.rows;
	Mat gaussianX = getGaussianKernel(salmap.cols, sigmaX, salmap.depth());
	Mat gaussianY = getGaussianKernel(salmap.rows, sigmaY, salmap.depth());
	salmap = salmap.mul(gaussianY * gaussianX.t());

	// normalize to [0,1]
	if(cfg.normalize){
		double mi, ma;
		minMaxLoc(salmap, &mi, &ma);
		salmap = (salmap-mi)/(ma-mi);
	}
	return salmap;
} 


//returns the weight from the config file or computes a default weight value when
//no command line weight was given (default value depends on other given weights)
vector<Mat> VOCUS2::get_splitted_salmap(){
	if(!processed){
		cout << "Image not yet processed. Call process(Mat)." << endl;
		return vector<Mat>(1, Mat());
	}
	if(splitted_ready) return salmap_splitted;

	salmap_splitted.resize(on_off_L.size());

	for(int o = 0; o < (int)on_off_L.size(); o++){
		Mat tmp = Mat::zeros(on_off_L[o].size(), CV_32F);

		tmp += on_off_L[o];
		tmp += off_on_L[o];
		tmp += on_off_a[o];
		tmp += off_on_a[o];
		tmp += on_off_b[o];
		tmp += off_on_b[o];

		tmp /= 6.f;

		if(cfg.normalize){
			double mi, ma;
			minMaxLoc(tmp, &mi, &ma);
			tmp = (tmp-mi)/(ma-mi);
		}

		salmap_splitted[o] = tmp;
	}
	
	splitted_ready = true;

	return salmap_splitted;
}

// void VOCUS2::apply_segmentation(float sigma, float k, float min_size, int method){
// 	if(!salmap_ready){
// 		compute_salmap();
// 	}

// 	GraphBasedImageSegmentation segmenter;
// 	cv::Mat seg = segmenter.segmentImage(input, sigma, k, min_size);

// 	set<Component*> segments = segmenter.getComponents();

// 	set<Component*>::iterator seg_it;
// 	vector<Component*> segments_list;
// 	int num_segments = 0;

// 	// re-arrange segments in a list to use an index
// 	for(seg_it = segments.begin(); seg_it != segments.end(); ++seg_it){
// 		segments_list.push_back(*seg_it);
// 		num_segments++;
// 	}
	
// 	vector<float> sal(num_segments, 0.f);

// 	// collect values
// #pragma omp parallel for
// 	for(int s = 0; s < num_segments; s++){
// 		set<Vertex*> vertices = segments_list[s]->getVertices();
// 		set<Vertex*>::iterator vert_it;

// 		// cout << "round: " << s << endl;

// 		for(vert_it = vertices.begin(); vert_it != vertices.end(); ++vert_it){;
// 			Point p = (*vert_it)->getPixelLocation();
// 			if(method == 0) sal[s] += salmap.ptr<float>(p.x)[p.y];
// 			else sal[s] = max(sal[s], salmap.ptr<float>(p.x)[p.y]);
// 		}

// 		if(method == 0) sal[s] /= segments_list[s]->getComponentSize();

// 		for(vert_it = vertices.begin(); vert_it != vertices.end(); ++vert_it){;
// 			Point p = (*vert_it)->getPixelLocation();
// 			salmap.ptr<float>(p.x)[p.y] = sal[s];
// 		}
// 	}

// 	segmenter.cleanData();
// }


vector<vector<Mat> > VOCUS2::build_multiscale_pyr(Mat& mat, float sigma){

	// maximum layer = how often can the image by halfed in the smaller dimension
	// a 320x256 can produce at most 8 layers because 2^8=256
	int max_octaves = min((int)log2(min(mat.rows, mat.cols)), cfg.stop_layer)+1;

	Mat tmp = mat.clone();

	// fast compute unused first layers with one scale per layer
	for(int o = 0; o < cfg.start_layer; o++){
		GaussianBlur(tmp, tmp, Size(), 2.f*sigma, 2.f*sigma, BORDER_REPLICATE);
		resize(tmp, tmp, Size(), 0.5, 0.5, INTER_NEAREST);
	}
	
	// reserve space
	vector<vector<Mat> > pyr;
	pyr.resize(max_octaves-cfg.start_layer);
	
	// compute pyramid as it is done in [Lowe2004]
	float sig_prev = 0.f, sig_total = 0.f;
	
	for(int o = 0; o < max_octaves-cfg.start_layer; o++){
		pyr[o].resize(cfg.n_scales+1);

		// compute an additional scale that is used as the first scale of the next octave
		for(int s = 0; s <= cfg.n_scales; s++){
			Mat& dst = pyr[o][s];

			// if first scale of first used octave => just smooth tmp
			if(o == 0 && s == 0){
				Mat& src = tmp;

				sig_total = pow(2.0, ((double)s/(double)cfg.n_scales))*sigma;
				GaussianBlur(src, dst, Size(), sig_total, sig_total, BORDER_REPLICATE);
				sig_prev = sig_total;
			}

			// if first scale of any other octave => subsample additional scale of previous layer
			else if(o != 0 && s == 0){
				Mat& src = pyr[o-1][cfg.n_scales];					
				resize(src, dst, Size(src.cols/2, src.rows/2), 0, 0, INTER_NEAREST);
				sig_prev = sigma;
			}

			// else => smooth an intermediate step
			else{
				sig_total = pow(2.0, ((double)s/(double)cfg.n_scales))*sigma;
				float sig_diff = sqrt(sig_total*sig_total - sig_prev*sig_prev);

				Mat& src = pyr[o][s-1];
				GaussianBlur(src, dst, Size(), sig_diff, sig_diff, BORDER_REPLICATE);
				sig_prev = sig_total;
			}
		}
	}

	// erase all the additional scale of each layer
	for(auto& o : pyr){
		o.erase(o.begin()+cfg.n_scales);
	}

	return pyr;
}

/*
float VOCUS2::maximaDetection(Mat& img, float t = 0.5){
	
	// hold maximal points
	vector<Point> point_maxima;

	// hold maximal blobs
	vector<vector<Point> > blob_maxima;

	CV_Assert(img.channels() == 1);

	// find maximum
	double ma;
	minMaxLoc(img, nullptr, &ma);

	// ignore map if global max is too small
	if(ma < 0.05) return 0.f;

	// ignore values < some portion t of the maximal value
	float thresh = ma*t;
	Mat mask;
	threshold(img, mask, thresh, 1, THRESH_BINARY_INV);
	mask.convertTo(mask, CV_8U);

	// number of maxima
	int n_max = 0;

	// for each image pixel
	for(int r = 0; r < img.rows; r++){
		for(int c = 0; c < img.cols; c++){
			
			// skip marked pixel
			if(mask.ptr<uchar>(r)[c] != 0) continue;

			float val = img.ptr<float>(r)[c];

			vector<Point> lower, greater, equal;

			// investigate neighborhood for pixel of values
			// greater, lower or equal to the current pixel
			for(int dr = -1; dr <= 1; dr++){
				for(int dc = -1; dc <= 1; dc++){
					// skip current pixel
					if(dr == 0 && dc == 0) continue;

					// skip out of bound pixels
					if(r+dr < 0 || r+dr >= img.rows) continue;
					if(c+dc < 0 || c+dc >= img.cols) continue;

					float tmp = img.ptr<float>(r+dr)[c+dc];
					Point p = Point(c+dc, r+dr);

					if(tmp < val) lower.push_back(p);
					else if(tmp > val) greater.push_back(p);
					else equal.push_back(p);
				}
			}

			// case 1: isolated point
			if(equal.size() == 0){

				// current point is done
				mask.ptr<uchar>(r)[c] = 1;
				
				// all smaller neighbours are definitive no maxima
				for(Point& p : lower) mask.ptr<uchar>(p.y)[p.x] = 1;

				// if no greater neighbours => maximum
				if(greater.size() == 0){
					// add as maximum
					point_maxima.push_back(Point(c,r));
					n_max++;
				}
			}

			// case 2: blob
			else{
				Mat considered = Mat::zeros(img.size(), CV_8U);
				
				// mark all pixel as considered
				for(Point& p : lower) considered.ptr<uchar>(p.y)[p.x] = 1;
				for(Point& p : equal) considered.ptr<uchar>(p.y)[p.x] = 1;
				for(Point& p : greater) considered.ptr<uchar>(p.y)[p.x] = 1;
				considered.ptr<uchar>(r)[c] = 1;

				// extent point to blob
				int pos = 0;
				while(pos < (int)equal.size()){
					int nr = equal[pos].y;
					int nc = equal[pos].x;

					for(int dr = -1; dr <= 1; dr++){
						for(int dc = -1; dc <= 1; dc++){
							// skip current pixel
							if(dr == 0 && dc == 0) continue;

							// skip out of bound pixels
							if(nr+dr < 0 || nr+dr >= img.rows) continue;
							if(nc+dc < 0 || nc+dc >= img.cols) continue;

							// skip considered pixels
							if(considered.ptr<uchar>(nr+dr)[nc+dc] == 1) continue;

							float tmp = img.ptr<float>(nr+dr)[nc+dc];
							Point p = Point(nc+dc, nr+dr);

							if(tmp < val) lower.push_back(p);
							else if(tmp > val) greater.push_back(p);
							else equal.push_back(p);

							considered.ptr<uchar>(p.y)[p.x] = 1;
						}
					}
					pos++;
				}

				// mark all lower neighbours (definitive no maxima)
				for(Point& p : lower) mask.ptr<uchar>(p.y)[p.x] = 1.f;

				// mark all blob pixels (maxima)
				equal.push_back(Point(c,r));
				for(Point& p : equal) mask.ptr<uchar>(p.y)[p.x] = 1.f;

				// case 2.1: all neighbours are lower
				if(greater.size() == 0){
					blob_maxima.push_back(equal);
					n_max++;
				}
			}
		}
	}
//	return n_max;
}*/


float VOCUS2::compute_uniqueness_weight(Mat& img, float t = 0.5){

	// hold maximal points
	vector<Point> point_maxima;

	// hold maximal blobs
	vector<vector<Point> > blob_maxima;

	CV_Assert(img.channels() == 1);

	// find maximum
	double ma;
	minMaxLoc(img, nullptr, &ma);

	// ignore map if global max is too small
	if(ma < 0.05) return 0.f;

	// ignore values < some portion t of the maximal value
	float thresh = ma*t;
	Mat mask;
	threshold(img, mask, thresh, 1, THRESH_BINARY_INV);
	mask.convertTo(mask, CV_8U);

	// number of maxima
	int n_max = 0;

	// for each image pixel
	for(int r = 0; r < img.rows; r++){
		for(int c = 0; c < img.cols; c++){
			
			// skip marked pixel
			if(mask.ptr<uchar>(r)[c] != 0) continue;

			float val = img.ptr<float>(r)[c];

			vector<Point> lower, greater, equal;

			// investigate neighborhood for pixel of values
			// greater, lower or equal to the current pixel
			for(int dr = -1; dr <= 1; dr++){
				for(int dc = -1; dc <= 1; dc++){
					// skip current pixel
					if(dr == 0 && dc == 0) continue;

					// skip out of bound pixels
					if(r+dr < 0 || r+dr >= img.rows) continue;
					if(c+dc < 0 || c+dc >= img.cols) continue;

					float tmp = img.ptr<float>(r+dr)[c+dc];
					Point p = Point(c+dc, r+dr);

					if(tmp < val) lower.push_back(p);
					else if(tmp > val) greater.push_back(p);
					else equal.push_back(p);
				}
			}

			// case 1: isolated point
			if(equal.size() == 0){

				// current point is done
				mask.ptr<uchar>(r)[c] = 1;
				
				// all smaller neighbours are definitive no maxima
				for(Point& p : lower) mask.ptr<uchar>(p.y)[p.x] = 1;

				// if no greater neighbours => maximum
				if(greater.size() == 0){
					// add as maximum
					point_maxima.push_back(Point(c,r));
					n_max++;
				}
			}

			// case 2: blob
			else{
				Mat considered = Mat::zeros(img.size(), CV_8U);
				
				// mark all pixel as considered
				for(Point& p : lower) considered.ptr<uchar>(p.y)[p.x] = 1;
				for(Point& p : equal) considered.ptr<uchar>(p.y)[p.x] = 1;
				for(Point& p : greater) considered.ptr<uchar>(p.y)[p.x] = 1;
				considered.ptr<uchar>(r)[c] = 1;

				// extent point to blob
				int pos = 0;
				while(pos < (int)equal.size()){
					int nr = equal[pos].y;
					int nc = equal[pos].x;

					for(int dr = -1; dr <= 1; dr++){
						for(int dc = -1; dc <= 1; dc++){
							// skip current pixel
							if(dr == 0 && dc == 0) continue;

							// skip out of bound pixels
							if(nr+dr < 0 || nr+dr >= img.rows) continue;
							if(nc+dc < 0 || nc+dc >= img.cols) continue;

							// skip considered pixels
							if(considered.ptr<uchar>(nr+dr)[nc+dc] == 1) continue;

							float tmp = img.ptr<float>(nr+dr)[nc+dc];
							Point p = Point(nc+dc, nr+dr);

							if(tmp < val) lower.push_back(p);
							else if(tmp > val) greater.push_back(p);
							else equal.push_back(p);

							considered.ptr<uchar>(p.y)[p.x] = 1;
						}
					}
					pos++;
				}

				// mark all lower neighbours (definitive no maxima)
				for(Point& p : lower) mask.ptr<uchar>(p.y)[p.x] = 1.f;

				// mark all blob pixels (maxima)
				equal.push_back(Point(c,r));
				for(Point& p : equal) mask.ptr<uchar>(p.y)[p.x] = 1.f;

				// case 2.1: all neighbours are lower
				if(greater.size() == 0){
					blob_maxima.push_back(equal);
					n_max++;
				}
			}
		}
	}

	// cout << n_max << endl;

	// Mat tmp = Mat::zeros(img.size(), CV_8UC3);
	// for(int r = 0; r < img.rows; r++){
	// 	for(int c = 0; c < img.cols; c++){
	// 		int val = round(img.ptr<float>(r)[c]*255);

	// 		tmp.ptr<Vec3b>(r)[c] = Vec3b(val, val, val);
	// 	}
	// }

	// for(auto& b : blob_maxima){
	// 	for(auto& p : b) tmp.ptr<Vec3b>(p.y)[p.x] = Vec3b(255,0,0);
	// }
	// for(auto& p : point_maxima) tmp.ptr<Vec3b>(p.y)[p.x] = Vec3b(0,0,255);

	// imshow("max tmp", tmp); waitKey(0);
	
	if(n_max == 0) return 0.f;
	else return 1/sqrt(n_max);
}

// void VOCUS2::mark_equal_neighbours(int r, int c, float value, Mat& map, Mat& marked){
// 	// marked.ptr<uchar>(r)[c] = 255;
// 	marked.ptr<uchar>(r)[c] = 0;

// 	// for all neighbours
// 	// note: all checks are done within the loop to reduce overhead of
// 	// creating stackframes for recursive calls for pixels that anyway
// 	// will not be labeled
// 	for(int i = -1; i <= 1; i++){
// 		for(int j = -1; j <= 1; j++){
// 			// skip current pixel
// 			if(j == 0 && i == 0) continue;
// 			// skip neighbours that are out of bound
// 			if(r+i < 0 || r+i >= map.rows) continue;
// 			if(c+j < 0 || c+j >= map.cols) continue;
// 			// skip neighbours that are already labeld
// 			if(marked.ptr<uchar>(r+i)[c+j] > 0) continue;
// 			// skip neighbours that not of the same value
// 			if(map.ptr<float>(r+i)[c+j] != value) continue;

// 			mark_equal_neighbours(r+1, c+j, value, map, marked);
// 		}
// 	}
// }

//fuse maps. For the fusing operations that depend on previouslive given Weights, optWeight has to be set.
Mat VOCUS2::fuse(vector<Mat> maps, FusionOperation op_in, vector<double> weights){
	
	FusionOperation op = op_in;

	//if no weights given but weighting option was chosen -> switch to mean.
	if(op == INDIVIDUAL_WEIGHTING && weights.size()==0){
		op = ARITHMETIC_MEAN;
	}
	
	// resulting map that is returned
	Mat fused = Mat::zeros(maps[0].size(), CV_32F);
	int n_maps = maps.size();	// no. of maps to fuse
	vector<Mat> resized;		// temp. array to hold the resized maps
	resized.resize(n_maps);		// reserve space (needed to use openmp for parallel resizing)

	// ========== ARTIMETIC MEAN ==========
	if(op == ARITHMETIC_MEAN){
#pragma omp parallel for schedule(dynamic, 1)
		for(int i = 0; i < n_maps; i++){
			if(fused.size() != maps[i].size()){
				resize(maps[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
			}
			else{
				resized[i] = maps[i];
			}
		}

		for(int i = 0; i < n_maps; i++){
			cv::add(fused, resized[i], fused, Mat(), CV_32F);
		}

		fused /= (float)n_maps;
	}

	// ========== MAX ==========

	else if(op == MAX){
	
#pragma omp parallel for schedule(dynamic, 1)
		for(int i = 0; i < n_maps; i++){
			resize(maps[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
		}

		for(int i = 0; i < n_maps; i++){
#pragma omp parallel for
			for(int r = 0; r < fused.rows; r++){
				float* row_tmp = resized[i].ptr<float>(r);
				float* row_fused = fused.ptr<float>(r);
				for(int c = 0; c < fused.cols; c++){
					row_fused[c] = max(row_fused[c], row_tmp[c]);
				}
			}
		}
	}

	// ========== UNIQUENESS WEIGHTING ==========

	else if(op == UNIQUENESS_WEIGHT){
		float weight[n_maps];
		// todo: openmp
// #pragma omp parallel for
		for(int i = 0; i < n_maps; i++){
			 weight[i] = compute_uniqueness_weight(maps[i]);

			if(weight[i] > 0){
				resize(maps[i]*weight[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
			}
		}

		float sum_weights = 0;

		for(int i = 0; i < n_maps; i++){
			if(weight[i] > 0){
				sum_weights += weight[i];
				cv::add(fused, resized[i], fused, Mat(), CV_32F);
			}
		}

		if(sum_weights > 0) fused /= sum_weights;
	}
	
	
	// ========== h i j - weights ==========

	else if(op == INDIVIDUAL_WEIGHTING  && weights.size()!=0){
		cout << "indiweight" << endl;
		float weight[n_maps];
		// todo: openmp
		// #pragma omp parallel for
		for(int i = 0; i < n_maps; i++){
			
			 weight[i] = weights[i];
				cout << weight[i] << endl;
			if(weight[i] > 0){
				resize(maps[i]*weight[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
			}
		}
		for(int i = 0; i < n_maps; i++){
			if(weight[i] > 0){
				cv::add(fused, resized[i], fused, Mat(), CV_32F);
			}
		}
	}
	
	
	
	
	
	// ========== DESCRIPTOR WEIGHTS ==========

	else if ( op == DESCRIPTOR_WEIGHT){		
		parseDescriptorFile(); //read from the decsriptor file 
		cout << "Read "<< descriptorWeights[0].size() << "intensity data and " << descriptorWeights[1].size() << "c1 data AND " <<  descriptorWeights[2].size() << "c2 data AND "<<descriptorWeights[3].size() << "o data AND " << descriptorWeights[4].size() << " conspicuity data."  << endl;

		//get the current feature weights vector from the object
		vector< double > weights;
		getCurrentFeatureSet(weights); //get the weights for the upcoming maps
				
		//process the maps with the weights
		for(int i = 0; i < n_maps; i++){
			if(fused.size() != maps[i].size()){
				resize(maps[i], resized[i], fused.size(), 0, 0, INTER_CUBIC);
			}
			else{
				resized[i] = maps[i];
			}
		}
		//cout << "weights:" << weights.size() << " " << n_maps << endl;
		if(weights.size()!=(unsigned int) n_maps){
			cout << "something went wrong:" << weights.size() << " " << n_maps << endl;
			exit(0);
		}
		
		//adding the maps up 	
		for(int i=0; i<n_maps;i++){
			cv::add(fused, weights[i]*resized[i], fused, Mat(), CV_32F);
		}
	}
	
	return fused;
}



// converts the image to the destination colorspace
// and splits the color channels
vector<Mat> VOCUS2::prepare_input(const Mat& img){

	CV_Assert(img.channels() == 3);

	vector<Mat> planes;
	
	// if(cfg.c_space == ITTI) planes.resize(5);
	// else planes.resize(3);

	planes.resize(3);

	// convert colorspace and split to single planes
	if(cfg.c_space == LAB){
		Mat converted;
		// convert colorspace (important: before conversion to float to keep range [0:255])
		cvtColor(img, converted, CV_BGR2Lab);
		// convert to float
		converted.convertTo(converted, CV_32FC3);
		// scale down to range [0:1]
		converted /= 255.f;
		split(converted, planes);

	}

	// opponent color as in CoDi (todo: maybe faster with openmp)
	else if(cfg.c_space == OPPONENT_CODI){
		Mat converted;
		img.convertTo(converted, CV_32FC3);

		vector<Mat> planes_bgr;
		split(converted, planes_bgr);

		planes[0] = planes_bgr[0] + planes_bgr[1] + planes_bgr[2];
		planes[0] /= 3*255.f;

		planes[1] = planes_bgr[2] - planes_bgr[1];
		planes[1] /= 255.f;

		planes[2] = planes_bgr[0] - (planes_bgr[1] + planes_bgr[2])/2.f;
		planes[2] /= 255.f;
	}
	else if(cfg.c_space == OPPONENT){
		Mat converted;
		img.convertTo(converted, CV_32FC3);

		vector<Mat> planes_bgr;
		split(converted, planes_bgr);

		planes[0] = planes_bgr[0] + planes_bgr[1] + planes_bgr[2];
		planes[0] /= 3*255.f;

		planes[1] = planes_bgr[2] - planes_bgr[1] + 255.f;
		planes[1] /= 2*255.f;
		
		planes[2] = planes_bgr[0] - (planes_bgr[1] + planes_bgr[2])/2.f + 255.f;
		planes[2] /= 2*255.f;
	}

	else{
		Mat converted;
		img.convertTo(converted, CV_32FC3);
		converted /= 255.f;
		split(converted, planes);
	}

	return planes;
}

/*! Computes the featurevector for visual search
 * 
 *  @param[in] ROI Region of Interest
 * 	@param[in] descriptorName Descriptor Filename
 */
void VOCUS2::td_learn_featurevector(Rect ROI, string descriptorName){
	//compute feature/consp. maps and extract them from vocus system
	vector<Mat> f_intensity;
	vector<Mat> f_color1;
	vector<Mat> f_color2;
	vector<Mat> f_or;
	vector<Mat> consp;
	Mat xsal = compute_salmap();
	getFeatureAndConspicuityMaps(f_intensity, f_color1, f_color2, f_or, consp);

	//weight values
	vector<double> weightValues;
							
	//take the saliency map and truncate everything to zero except the annotated rectangle
	Mat trunc = truncToZero(salmap, ROI); 

	//get a white mask from the saliency map using the truncated map
	Mat MSRmask = salmap.clone();
	int number_pixels_msr = computeMSRmask(MSRmask, trunc, 0.7);
		
	//rest of pixels
	int number_pixels_rest = (salmap.rows*salmap.cols)-number_pixels_msr;
	stringstream ss;
	
	//compute all weights from the feature/consp maps
	//code them, write them into the stringstream
	for(unsigned int i=0; i<f_intensity.size(); i++)
		{
			Mat ftmap = f_intensity[i].clone();
			float weight = computeFeatureValue(ftmap, MSRmask, number_pixels_msr, number_pixels_rest, ROI);
				cout << endl << std::endl;
				ss << "i: " << weight << endl;
				weightValues.push_back(weight);
			}
			
			for(unsigned int i=0; i<f_color1.size(); i++)
			{
				Mat ftmap = f_color1[i].clone();
				float weight = computeFeatureValue(ftmap, MSRmask, number_pixels_msr, number_pixels_rest,  ROI);
				ss << "c1: " << weight << endl;
				weightValues.push_back(weight);
			}
			
			   for(unsigned int i=0; i<f_color2.size(); i++)
			{
				Mat ftmap = f_color2[i].clone();
				float weight = computeFeatureValue(ftmap, MSRmask, number_pixels_msr, number_pixels_rest,  ROI);
				ss << "c2: " << weight << endl;
				weightValues.push_back(weight);
			}
			   
			for(unsigned int i=0; i<f_or.size(); i++)
			{
				Mat ftmap = f_or[i].clone();
				float weight = computeFeatureValue(ftmap, MSRmask, number_pixels_msr, number_pixels_rest,  ROI);
				ss << "o: " << weight << endl;
				weightValues.push_back(weight);
			} 
			 
			for(unsigned int i=0; i<consp.size(); i++)
			{
				Mat ftmap = consp[i].clone();
				float weight = computeFeatureValue(ftmap, MSRmask, number_pixels_msr, number_pixels_rest,  ROI);
				ss << "con: " << weight << endl;
				weightValues.push_back(weight);
			}	
		cout << "Writing descriptor file ";
		//now write stream to file (descriptors/image_name.desc
		ofstream out;
		string s = "descriptors/";
//************** DANIEL EDIT START **************
		s = "";
//************** DANIEL EDIT END   **************
		s += descriptorName; //files[i];
		string seperator = string("/");
		size_t p = s.rfind(seperator,s.length());
		if(p!=string::npos)
			s = s.substr(p);
//************** DANIEL EDIT START **************
		//s = string("descriptors") + s + ".desc";
//************** DANIEL EDIT END   **************	

		cout << s << endl;
		out.open(s.c_str());
		out << ss.str();
		cout << ss.str() << endl;
		out.close();		
}


Mat VOCUS2::td_search(double alpha){
	Mat bu_salmap; 
	Mat td_salmap; 
	bu_salmap = compute_salmap(); 
	td_salmap = compute_tdmap();  
	td_salmap = alpha*bu_salmap + (1-alpha)*td_salmap;
	return td_salmap;
}


 // clear all datastructures from previous results
void VOCUS2::clear(){
	salmap.release();
	on_off_L.clear();
	off_on_L.clear();
	on_off_a.clear();
	off_on_a.clear();
	on_off_b.clear();
	off_on_b.clear();

	pyr_center_L.clear();
	pyr_surround_L.clear();
	pyr_center_a.clear();
	pyr_surround_a.clear();
	pyr_center_b.clear();
	pyr_surround_b.clear();
}




