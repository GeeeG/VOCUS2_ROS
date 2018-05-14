/*****************************************************************************
*
* ImageFunctions.cpp file for the saliency program VOCUS2. 
* A detailed description of the algorithm can be found in the paper: "Traditional Saliency Reloaded: A Good Old Model in New Shape", S. Frintrop, T. Werner, G. Martin Garcia, in Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2015.  
* Please cite this paper if you use our method.
*
* Design and supervision: Simone Frintrop (frintrop@iai.uni-bonn.de)
* @author Thomas Werner   (wernert@cs.uni-bonn.de)
* @author Johannes Teutrine 
*
* @since 1.2
*
* This code is published under the MIT License 
* (see file LICENSE.txt for details)
*
******************************************************************************/


#include "ImageFunctions.h"


using namespace cv;
using namespace std;


void floodfill_r(Mat& src, int x, int y, double bot, vector < vector<bool> >& vis, double* flood_minmax){
 for(int i=-1;i<2;i++){
    for(int j=-1;j<2;j++){
      
      if(x+i>=0&&y+j>=0&&x+i<src.rows&&y+j<src.cols){
      if(  !vis[x+i][y+j] && src.at<float>(x+i,y+j)>=bot){
		vis[x+i][y+j]=true;		
		if(flood_minmax[0]>x+i) flood_minmax[0]=x+i;
		if(flood_minmax[1]<x+i) flood_minmax[1]=x+i;
		if(flood_minmax[2]>y+j) flood_minmax[2]=y+j;
		if(flood_minmax[3]<y+j) flood_minmax[3]=y+j;
		src.at<float>(x+i,y+j)=1;
		floodfill_r(src,x+i,y+j,bot,vis, flood_minmax);
      }
    }
	}
 }
}



void floodfill(Mat& src, int x, int y, double eps, double* flood_minmax){
	vector< vector<bool> > visited;
	for(int i=0;i<src.rows;i++){
		vector<bool> v;
		for(int j=0;j<src.cols;j++){
			v.push_back(false);
		}
		visited.push_back ( v );
	}
	float bot = src.at<float>(y,x) * eps;
	flood_minmax[0]=flood_minmax[1]=x; //min_x, max_x
	flood_minmax[2]=flood_minmax[3]=y; //min_y, max_y
	floodfill_r(src,y,x,bot,visited, flood_minmax);
}



Mat truncToZero(Mat map, Rect roi){	
	Mat truncated_map = Mat::zeros(map.rows, map.cols, map.type());
	Mat source_ROI = map(roi);
	Mat truncated_map_ROI = truncated_map(roi);
	source_ROI.copyTo(truncated_map_ROI);
	return truncated_map;
}


double calcMeanVal(Mat& map, int pixels){
		double mean = 0;
		for(int i=0; i<map.rows; i++){
			for(int j=0; j<map.cols; j++){
				mean += map.at<float>(i,j);				
			}
		}
		
		mean = ((double)mean/(double)pixels);

		return mean;
}


void calcXminMsr(Mat& msr, Mat& xminmsr){
		for(int i=0; i<xminmsr.rows; i++){
			for(int j=0; j<xminmsr.cols; j++){
				if(msr.at<float>(i,j)>0)
					xminmsr.at<float>(i,j) = 0;
			}	
		}
}


/*
//get most salient region in saliency map 
vector<Point> get_msr(Mat& salmap, double threshold){
	double ma;
	Point p_ma;
	minMaxLoc(salmap, nullptr, &ma, nullptr, &p_ma);
	vector<Point> msr;
	msr.push_back(p_ma);

	int pos = 0;
	float thresh = threshold*ma; //bis zu welchem Unterschied darf region wachsen?
	Mat considered = Mat::zeros(salmap.size(), CV_8U);
	considered.at<uchar>(p_ma) = 1;

	while(pos < (int)msr.size()){
		int r = msr[pos].y;
		int c = msr[pos].x;
		for(int dr = -1; dr <= 1; dr++){
			for(int dc = -1; dc <= 1; dc++){
				if(dc == 0 && dr == 0) continue;
				if(considered.ptr<uchar>(r+dr)[c+dc] != 0) continue;
				if(r+dr < 0 || r+dr >= salmap.rows) continue;
				if(c+dc < 0 || c+dc >= salmap.cols) continue;
				if(salmap.ptr<float>(r+dr)[c+dc] >= thresh){
					msr.push_back(Point(c+dc, r+dr));
					considered.ptr<uchar>(r+dr)[c+dc] = 1;
				}
			}
		}
		pos++;
	}
	return msr;
}
* */


vector< vector<Point> > computeMSR(Mat& salmap, double threshold, int maxRegions){
	double ma; //maximum value
	Point p_ma; //point of max value
	vector < vector<Point> > msrvec; //salient regions
	Mat considered = Mat::ones(salmap.size(), CV_8U); //unconsidered: 1
	Mat region_mask = Mat::zeros(salmap.size(), CV_8U); //0, if NOT belonging to region

	//start computation at max value
	minMaxLoc(salmap, nullptr, &ma, nullptr, &p_ma);
	Mat maskedSalmap = salmap.clone();
	bool goon = true;
	bool okay = true;
	int maxis=0;
	
	while(goon){
		maxis++;
		minMaxLoc(maskedSalmap, nullptr, &ma, nullptr, &p_ma, considered);  //maximum of unconsidered points

		if(ma<0.05) goon=false; //return if value is too low
		vector<Point> msr; //the salient region to compute
		msr.push_back(p_ma); //start growing process with max value pt

		int pos = 0;
		float thresh = threshold*ma; //threshold for allowed value distance to max value
		considered.at<uchar>(p_ma) = 2; //2: belongs to THIS region
		okay = true;
		while(pos < (int)msr.size() && okay ){
			if(okay){
			int r = msr[pos].y;
			int c = msr[pos].x;
			for(int dr = -1; dr <= 1; dr++){
				for(int dc = -1; dc <= 1; dc++){
					if(dc == 0 && dr == 0) continue;
					if(considered.ptr<uchar>(r+dr)[c+dc] == 2) continue; 
					if(r+dr < 0 || r+dr >= salmap.rows) continue;
					if(c+dc < 0 || c+dc >= salmap.cols) continue;
					
					if(salmap.ptr<float>(r+dr)[c+dc] >= thresh && considered.ptr<uchar>(r+dr)[c+dc] == 1 ){
						msr.push_back(Point(c+dc, r+dr));
						considered.ptr<uchar>(r+dr)[c+dc] = 2; //considered in this run
					}
					else if(salmap.ptr<float>(r+dr)[c+dc] >= thresh && considered.ptr<uchar>(r+dr)[c+dc] == 0 ){ 
						//point was considered before in bigger value region. adding illegal -> reject region.
						okay = false;
					}
					
				}
			}
			pos++;
			}
		}
		if(okay){ //region rejected or not?
			msrvec.push_back(msr);
		}
		
		//mark region as considered
		for(unsigned int p=0; p<msr.size();p++){
			int rp = msr[p].y; //row
			int cp = msr[p].x; //col
			considered.ptr<uchar>(rp)[cp] = 0;
		}
		if(msrvec.size()>=(unsigned int)maxRegions) goon=false; //stop if enough regions found
	}		
	return msrvec;
}


Mat msrVecToMat(vector<Point> msr, Mat saliencyMap){
	Mat binMSR = Mat::zeros(saliencyMap.rows, saliencyMap.cols, CV_8UC1);
	for(unsigned int i=0;i<msr.size();i++){
		binMSR.at<uchar>(msr[i].y,msr[i].x) = 255;
	}
	return binMSR;
}


int computeMSRmask(Mat& mask, Mat& map, double thresh){
	double global_max;	
	Point seed(0,0);
	minMaxLoc(map,NULL,&global_max, NULL, &seed);
	double* flood=new double[4]	;
	int x = seed.x;
	int y = seed.y; 
	
	floodfill(map, x, y, thresh, flood);	
	int number_pixels=0;		

	//extract white region
	for( int i=0;i<mask.rows;i++){
		for( int j=0;j<mask.cols;j++){
			if(map.at<float>(i,j)==1){
				mask.at<float>(i,j)=1;
				number_pixels++;
			}
			else 
				mask.at<float>(i,j)=0;
		}
	}
	return number_pixels;	
}


void extractMSR(Mat& msr, Mat mask, Mat roi){
	for(int i=0;i<mask.rows;i++){
		for(int j=0;j<mask.cols;j++){
			if(mask.at<float>(i,j)==1){ 
				msr.at<float>(i,j)=roi.at<float>(i,j);
			}
			else 
				msr.at<float>(i,j)=0;
		}
	}
}


float computeFeatureValue(Mat& ftmap, Mat& mask, int pixels_mask, int pixels_rest, Rect inpRoi){
	double x1=1, x2=1, m = 0;
	Mat roi = truncToZero(ftmap, inpRoi);
	Mat msr=roi.clone();		
	extractMSR(msr, mask, roi);
	Mat xminmsr = ftmap.clone();
	calcXminMsr(msr, xminmsr);
	x1 = calcMeanVal(msr, pixels_mask);   				
	x2 = calcMeanVal(xminmsr,pixels_rest);
	cout << "x1="<<x1 << " x2="<<x2;	
	if(x1==0) return 0.00000001;	
	if(x2==0){
		return 0.000000000001;	 
	}
	m = x1/x2;
	if(m<0){
		return 0.000000000001;
	}
	cout << "  m=" << m << endl;
	return m;
}


