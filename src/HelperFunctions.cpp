/*****************************************************************************
*
* HelperFunctions.cpp file for the saliency program VOCUS2. 
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

#include <fstream>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/stat.h>
#include <string>

#include "HelperFunctions.h"


using namespace cv;
using namespace std;

struct stat sb;
	
	
/* global variables for mouse handling in learning*/
Point2d point1, point2; /* vertical points of the bounding box */
int drag = 0;
Rect rect; /* bounding box */
Mat handlerSrc, roiImg; /* roiImg - the part of the image in the bounding box */
int select_flag = 0;
	
//mouse handler for the TOPDOWN Learning
void mouseHandler(int event, int x, int y, int flags, void* param){
    if (event == CV_EVENT_LBUTTONDOWN && !drag)
    {
        /* left button clicked. ROI selection begins */
        point1 = Point2d(x, y);
		cout << "Clicked " << point1 << endl; //report clicked point
        drag = 1;
    }
    
    if (event == CV_EVENT_MOUSEMOVE && drag)
    {
        // ROI selection
        Mat img1 = handlerSrc.clone();
        point2 = Point2d(x, y);
        rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
        imshow("Annotate", img1);
    }
    
    if (event == CV_EVENT_LBUTTONUP && drag)
    {
        point2 = Point2d(x, y);
		cout << "Bottom_right:" << point2 << endl;
        drag = 0;
    }
    
    if (event == CV_EVENT_LBUTTONUP)
    {
	//Roi selected
        select_flag = 1;
        drag = 0;
    }
}	
	
	
Rect annotateROI(Mat& src){
	//Create a window
	namedWindow("Annotate", WINDOW_AUTOSIZE);
	//set the callback function for any mouse event
	handlerSrc = src.clone();
	setMouseCallback("Annotate", mouseHandler, NULL);		
	//show the image
	imshow("Annotate", src);
	// Wait until key pressed
	waitKey(0);
	//flip points to correct order
	Point point3(point1.x,point1.y);
	if(point1.x>point2.x && point1.y>point2.y){
		point1.x=point2.x;
		point1.y=point2.y;
		point2.x=point3.x;
		point2.y=point3.y;	
	}
	else if(point1.x>point2.x && point1.y<point2.y){
		point1.x=point2.x;
		point2.x=point3.x;	
	}
	else if(point1.x<point2.x && point1.y>point2.y){
		point1.y=point2.y;
		point2.y=point3.y;
	}
	drag = 0; select_flag = 0; 
	destroyWindow("Annotate");			
	//annotation done. 
	cout << "annotation done" << endl;
	return Rect(point1,point2);
}		
	
	
	
void print_usage(char* argv[]){
	cout << "\nUsage: " << argv[0] << " [OPTIONS] <image-data>" << endl << endl;

	cout << "<image-data> can be an image, a list of images, a video, or webcam input. See Readme for details" << endl << endl;


	cout << "===== Options =====" << endl << endl;

	cout << "   -x <path>" << "\t\t" << "Config file (is loaded first, additional options have higher priority)" << endl << endl;

	cout << "   -C <value>" << "\t\t" << "Used colorspace [default: 1]:" << endl;
	cout << "\t\t   " << "0: LAB" << endl;
	cout << "\t\t   " << "1: Opponent (CoDi)" << endl;
	cout << "\t\t   " << "2: Opponent (Equal domains)\n" << endl;
	// cout << "\t\t   " << "3: Itti\n" << endl;

	cout << "   -f <value>" << "\t\t" << "Fusing operation (Feature maps) [default: 0]:" << endl;
	cout << "   -F <value>" << "\t\t" << "Fusing operation (Conspicuity/Saliency maps) [default: 0]:" << endl;

	cout << "\t\t   " << "0: Arithmetic mean" << endl;
	cout << "\t\t   " << "1: Max" << endl;
	cout << "\t\t   " << "2: Uniqueness weight\n" << endl;
	cout << "\t\t   " << "3: Weight Channel by command line parameter (ONLY -F)\n" << endl;
	cout << "\t\t   " << "4: Descriptor Weights from .desc file\n" << endl;

	cout << "   -p <value>" << "\t\t" << "Pyramidal structure [default: 2]:" << endl;

	cout << "\t\t   " << "0: Two independent pyramids (Classic)" << endl;
	cout << "\t\t   " << "1: Two pyramids derived from a base pyramid (CoDi-like)" << endl;
	cout << "\t\t   " << "2: Surround pyramid derived from center pyramid (New)\n" << endl;
	// cout << "\t\t   " << "3: Single pyrmamide (Itti style)\n" << endl;

	cout << "   -l <value>" << "\t\t" << "Start layer (included) [default: 0]" << endl << endl;

	cout << "   -L <value>" << "\t\t" << "Stop layer (included) [default: 4]" << endl << endl;

	cout << "   -S <value>" << "\t\t" << "No. of scales [default: 2]" << endl << endl;

	cout << "   -c <value>" << "\t\t" << "Center sigma [default: 2]" << endl << endl;

	cout << "   -s" << "\t\t" << "Surround sigma [default: 10]" << endl 
	     << "\t\t(the effective sigma value)" << endl << endl;

	// cout << "   -n" << "\t\t" << "Normalize output to [0:1]" << endl << endl;
	
	cout << "   -r" << "\t\t" << "Use orientation channel [default: off]  " << endl << endl;

	cout << "   -e" << "\t\t" << "Use Combined Feature [default: off]" << endl << endl;

	cout << "   -u <value>" << "\t\t" << "Weight for -f 3: Intensity channel Conspicuity to Saliency fusion" << endl << endl;

	cout << "   -i <value>" << "\t\t" << "Weight for -f 3: Color Features 1 Conspicuity to Saliency fusion" << endl << endl;

	cout << "   -j <value>" << "\t\t" << "Weight for -f 3 (or -F 3): Color Features 2 Conspicuity to Saliency fusion (please see Readme)" << endl << endl;
	cout << "   -1 ..-0 <value>" << "\t\t" << "Weights for -f 3 (or -F 3): Fusion of Feature maps to Conspicuity maps (please see Readme)" << endl << endl;


	cout << "===== MISC (NOT INCLUDED IN A CONFIG FILE) =====" << endl << endl;

	cout << "   -v" << "\t\t" << "Webcam source" << endl << endl;

	cout << "   -V" << "\t\t" << "Video files" << endl << endl;

	cout << "   -t <value>" << "\t\t" << "MSR threshold (percentage of fixation) [default: 0.75]" << endl << endl;

	cout << "   -N" << "\t\t" << "No visualization" << endl << endl;
	
	cout << "   -o <path>" << "\t\t" << "WRITE results to specified path [default: <input_path>/saliency/*]" << endl << endl;

	cout << "   -w <path>" << "\t\t" << "WRITE all intermediate maps to an existing folder" << endl << endl;

	cout << "   -b <value>" << "\t\t" << "Add center bias to the saliency map\n" << endl << endl;
	
	cout << "   -y " << "\t\t" << "Learning Mode. Computes a feature descriptor from a region of interest. Usage: -y " << endl << endl;

	cout << "   -z <descriptorFile>" << "\t\t" << "Visual Search. Uses a feature descriptor (descriptor computation with -y)" << endl << endl;

	cout << "   -a <int>" << "\t\t" << "The number of Foci of Attention to attend [default: 1]" << endl << endl;
	
	cout << "   -A <path>" << "\t\t" << "Path for saving the Foci of Attention" << endl << endl;



	// cout << "   -G" << "\t\t" << "Apply EGBISegmentation and assign saliency to segments" << endl;
	// cout << "\t\t   " << "Parameterformat:" << endl;
	// cout << "\t\t   \t" << "sigma:k:min_size:method\n" << endl;
	// cout << "\t\t   " << "sigma: Smoothing factor [0:1]" << endl;
	// cout << "\t\t   " << "k: Something EGBIS specific [300-1000]" << endl;
	// cout << "\t\t   " << "min_size: Minimum allowed size of segments" << endl;
	// cout << "\t\t   " << "method: Computation of segment saliency [0: Avg., 1: Max.]" << endl;

}



//writes the saliency map to a path specified with -o
int writeSaliencyToPath(string base_path, Mat salmap, string WRITE_PATH, bool WRITEPATHSET){
	if(stat(WRITE_PATH.c_str(), &sb)!=0){
		if(WRITEPATHSET){ 
			std::cout << "Creating directory..." << std::endl;
			mkdir(WRITE_PATH.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		}		
	}
	//else return -1;
	
	//WRITE resulting saliency maps to path/directory
		string pathStr(base_path);
		int found = 0;
		found = pathStr.find_last_of("/\\");
		string path = pathStr.substr(0,found);
		string filename_plus = pathStr.substr(found+1);
		string filename = filename_plus.substr(0,filename_plus.find_last_of("."));
		string WRITE_NAME = filename + "_saliency";
		string WRITE_result_PATH;
		//if no path set, write to /saliency directory
		if(!WRITEPATHSET){
			//get the folder, if possible 
			if(found>=0){
				string WRITE_DIR = "/saliency/";
				if ( !(stat(WRITE_PATH.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))){
					mkdir((path+WRITE_DIR).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
				}
				WRITE_result_PATH = path + WRITE_DIR + WRITE_NAME + ".png";
			}
			else{					//if images are in the source folder 
				string WRITE_DIR = "saliency/";
				if ( !(stat(WRITE_PATH.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode))){
					mkdir((WRITE_DIR).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
				}
				WRITE_result_PATH = WRITE_DIR + WRITE_NAME + ".png";
			}
		}
		else{ 
			WRITE_result_PATH = WRITE_PATH+"/"+WRITE_NAME+".png";
		}
		std::cout << "WRITE result to " << WRITE_result_PATH << std::endl << std::endl;
		imwrite(WRITE_result_PATH.c_str(), salmap*255.f);
			
		return 0;
}


void showMSR(Mat img, vector< vector<Point> > maxima, int NUM_FOCI){
		
	//show NUM_FOCI 
	for(int i=0; i<NUM_FOCI;i++){
		if(maxima.size()<=(unsigned)i) break;
			//get the i-th msr
			vector<Point> msr = maxima[i];
			Point2f center;
			float rad;
			//compute circles enclosing the msr
			minEnclosingCircle(msr, center, rad);
			if(rad >= 5 && rad <= max(img.cols, img.rows)){
				circle(img, center, (int)rad, Scalar(0+40*i,0+30*i,255-20*i), 3);
			}
			//show the circles
			imshow("Region", img);
			waitKey(0);		
		}
	
}

void saveFoci(string filePath, vector< vector<Point> > maxima, int NUM_FOCI){
		ofstream myfile;
		myfile.open(filePath.c_str(), ios::app);
		//show NUM_FOCI 
		for(int i=0; i<NUM_FOCI;i++){
			if(maxima.size()<=(unsigned)i) break;
			//get the i-th msr
			vector<Point> msr = maxima[i];
			Point2f center;
			float rad;
			//compute circles enclosing the msr
			minEnclosingCircle(msr, center, rad);
			myfile << " " << center.x << " " << center.y;

		}	
		myfile << endl;
		myfile.close();

}
