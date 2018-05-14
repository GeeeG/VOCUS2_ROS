/*****************************************************************************
*
* main.cpp file for the saliency program VOCUS2. 
* A detailed description of the algorithm can be found in the paper: "Traditional Saliency Reloaded: A Good Old Model in New Shape", S. Frintrop, T. Werner, G. Martin Garcia, in Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2015.  
* Please cite this paper if you use our method.
*
* Design and supervision: Simone Frintrop (frintrop@iai.uni-bonn.de)
* @author Thomas Werner   (wernert@cs.uni-bonn.de)
* @author Johannes Teutrine
* @since 1.0
*
* This code is published under the MIT License 
* (see file LICENSE.txt for details)
*
******************************************************************************/


#include <iostream>
#include <sstream>
#include <unistd.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/stat.h>
#include <string>

#include "VOCUS2.h"
#include "ImageFunctions.h"
#include "HelperFunctions.h"

using namespace std;
using namespace cv;

struct stat sb_;


int WEBCAM_MODE = -1;
int TOPDOWN_SEARCH = -1;
int TOPDOWN_LEARN = -1;
bool VIDEO_MODE = false;
float MSR_THRESH = 0.75; // most salient region
bool SHOW_OUTPUT = true;
string WRITE_OUT = "";
string WRITE_PATH = "";
string OUTOUT = "";
bool WRITEPATHSET = false;
double CENTER_BIAS = 0.000005;
bool COMPUTE_CBIAS = false;
int NUM_FOCI = 1;
bool SAVE_FOCI = false;
string FOCI_FILENAME = "";


// string EGBIS = "";
float SIGMA, K;
int MIN_SIZE, METHOD;



/*! String Splitter
 *	Splits a string s by an delimeter delim.
 * @param[in] s string to split
 * @param[in] delim Delimeter
 * @return Vector of string parts
 */ 
vector<string> split_string(const string &s, char delim) {
    vector<string> elems;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}



/*! 
 *  Command Line Input Processing
 *	Boost-Based Input processing
 * 
 * 	@param[in] argc command line input argument size
 * 	@param[in] argv command line input arguments
 *  @param[out] cfg VOCUS2_Cfg to store arguments
 * 	@param[out] files Images to process
 */ 
bool process_arguments(int argc, char* argv[], VOCUS2_Cfg& cfg, vector<char*>& files){
	if(argc == 1) return false;

	int c;

	while((c = getopt(argc, argv, "nreNhVyC:x:f:F:v:l:o:L:S:c:s:t:p:w:z:G:b:h:u:i:j:k:1:2:3:4:5:6:7:8:9:0:a:A:")) != -1){
		switch(c){
		case 'h': return false;
		case 'C': cfg.c_space = ColorSpace(atoi(optarg)); break;
		case 'f': cfg.fuse_feature = FusionOperation(atoi(optarg)); break;
		case 'F': cfg.fuse_conspicuity = FusionOperation(atoi(optarg)); break;
		case 'v': WEBCAM_MODE = atoi(optarg) ; break;
		case 'l': cfg.start_layer = atoi(optarg); break;
		case 'L': cfg.stop_layer = atoi(optarg); break;
		case 'S': cfg.n_scales = atoi(optarg); break;
		case 'c': cfg.center_sigma = atof(optarg); break;
		case 's': cfg.surround_sigma = atof(optarg); break;
		case 'V': VIDEO_MODE = true; break;
		case 't': MSR_THRESH = atof(optarg); break;
		case 'N': SHOW_OUTPUT = false; break;
		case 'o': WRITEPATHSET = true; WRITE_PATH = string(optarg); break;
		case 'n': cfg.normalize = true; break;
		case 'r': cfg.orientation = true; break;
		case 'e': cfg.combined_features = true; break;
		case 'p': cfg.pyr_struct = PyrStructure(atoi(optarg));
		case 'w': WRITE_OUT = string(optarg); break;
		case 'y': TOPDOWN_LEARN=1; break;
		case 'z': TOPDOWN_SEARCH=1; cfg.setDescriptorFile( string(optarg) ); break;
		// case 'G': EGBIS = string(optarg); break;
		case 'b': CENTER_BIAS = atof(optarg); COMPUTE_CBIAS = true; break;
		case 'u': cfg.weights[10] = atof(optarg); break; 
		case 'i': cfg.weights[11] = atof(optarg); break;
		case 'j': cfg.weights[12] = atof(optarg); break;
		case 'k': cfg.weights[13] = atof(optarg); break;
		case '1': cfg.weights[0] = atof(optarg); break;
		case '2': cfg.weights[1] = atof(optarg); break;
		case '3': cfg.weights[2] = atof(optarg); break;
		case '4': cfg.weights[3] = atof(optarg); break;
		case '5': cfg.weights[4] = atof(optarg); break;
		case '6': cfg.weights[5] = atof(optarg); break;
		case '7': cfg.weights[6] = atof(optarg); break;
		case '8': cfg.weights[7] = atof(optarg); break;
		case '9': cfg.weights[8] = atof(optarg); break;
		case '0': cfg.weights[9] = atof(optarg); break;		
		case 'a': NUM_FOCI = atoi(optarg); break;
		case 'A': FOCI_FILENAME = string(optarg); SAVE_FOCI=true; break;
		case 'x': break;
			
		default:
			return false;

		}
	}

	cout << NUM_FOCI << endl;
	
	if( (cfg.fuse_feature==4 || cfg.fuse_conspicuity==4) && TOPDOWN_SEARCH!=1     ){
		cout << "No descriptor file given for fusing operation 3" << endl;
		return false;
	} 
	
	if(MSR_THRESH < 0 || MSR_THRESH > 1){
		cerr << "MSR threshold must be in the range [0,1]" << endl;
		return false;
	}

	if(cfg.start_layer < 0){
		cerr << "Start layer must be positive" << endl;
		return false;
	}

	if(cfg.start_layer > cfg.stop_layer){
		cerr << "Start layer cannot be larger than stop layer" << endl;
		return false;
	}

	if(cfg.n_scales <= 0){
		cerr << "Number of scales must be > 0" << endl;
		return false;
	}

	if(cfg.center_sigma <= 0){
		cerr << "Center sigma must be positive" << endl;
		return false;
	}

	if(cfg.surround_sigma <= cfg.center_sigma){
		cerr << "Surround sigma must be positive and larger than center sigma" << endl;
		return false;
	}

	
    for (int i = optind; i < argc; i++)
    	files.push_back(argv[i]);
    if (files.size() == 0 && WEBCAM_MODE < 0) {
    	return false;
    }
	if(files.size() == 0 && VIDEO_MODE){
		return false;
	}
	if(WEBCAM_MODE >= 0 && VIDEO_MODE){
		return false;
	}

	
	return true;
}

// if parameter x specified, load config file
VOCUS2_Cfg create_base_config(int argc, char* argv[]){

	VOCUS2_Cfg cfg;

	int c;

	while((c = getopt(argc, argv, "NbnrehVyC:x:f:F:v:l:L:S:o:c:s:t:p:w:z:G:u:i:j:k:1:2:3:4:5:6:7:8:9:10:a:A:")) != -1){
		if(c == 'x') cfg.load(optarg);
	}

	optind = 1;

	return cfg;
}




int main(int argc, char* argv[]) {
	VOCUS2_Cfg cfg = create_base_config(argc, argv);
	vector<char*> files; // names of input images or video files

	bool correct = process_arguments(argc, argv, cfg, files);
	if(!correct){
		print_usage(argv);
		return EXIT_FAILURE;
	}

	VOCUS2 vocus2(cfg);

	bool IMAGE_MODE=true;
	if(WEBCAM_MODE > -1 || VIDEO_MODE)
		IMAGE_MODE = false;


	//test data
	if(IMAGE_MODE){
		if(files.size()>=1){
			Mat img = imread(files[0]); 
			//basically a test if data is an image or not.
			if( !(img.channels()==3) ){ 
				std::cout << "ABORT: Incorrect Image Data"<< img.channels() << std::endl;
				exit(-1);
			}
		}
	}


	//LEARNING MODE
	if( TOPDOWN_LEARN==1){
		std::cout << "LEARNING" << std::endl;
	
		if(!IMAGE_MODE){
			cerr << "Topdown Mode conflicts with Video Mode"<< endl;
			exit(1);
		}

		for(size_t i = 0; i < files.size(); i++){
			cout << "Opening " << files[i] << " (" << i+1 << "/" << files.size() << "), ";
			
			Mat img = imread(files[i],CV_LOAD_IMAGE_COLOR);
			
			//basically a test if data is an image or not.
			if( !(img.channels()==3)  || ! img.data ){ 
				std::cerr << "Abort image " << i << endl;
				continue;		
			}
			
			
			//compute saliency map (bottom up)
			Mat salmap;
			vocus2.process(img);
			salmap = vocus2.compute_salmap();
			if(COMPUTE_CBIAS) {
			  salmap = vocus2.add_center_bias(CENTER_BIAS);
			}
			
			
		//Annotate ROI in source image
		Rect ROI = annotateROI(img);
		
		//compute feature vector
		vocus2.td_learn_featurevector(ROI, files[i]);
			
		}	
			
		return EXIT_SUCCESS;
	}


	//Visual Search
	if(TOPDOWN_SEARCH==1){ 
		double overall_time = 0;
		
		cout << "Starting Visual Search" << endl;
		if(stat(WRITE_PATH.c_str(), &sb_)!=0){
			if(WRITEPATHSET){ 
				std::cout << "Creating directory..." << std::endl;
				mkdir(WRITE_PATH.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
			}		
		}

		//for all input images: compute top-down saliency map 
		for(size_t i = 0; i < files.size(); i++){
			cout << "Opening " << files[i] << " (" << i+1 << "/" << files.size() << "), ";
			
			Mat img = imread(files[i], 1);
				
			//basically a test if data is an image or not.
			if( !(img.channels()==3)  || ! img.data ){ 
				std::cerr << "Abort image " << i << endl;
				continue;		
			}
			
			long long start = getTickCount();
			
			// compute image pyramids, feature and conspicuity maps
			vocus2.process(img);
			
			// compute a global saliency map by alpha*BU + (1-alpha)*TD;
			double alpha = 0; // 1: only BU, 0: only TD
			Mat salmap = vocus2.td_search(alpha);

			long long stop = getTickCount();
			double elapsed_time = (stop-start)/getTickFrequency();
			cout << elapsed_time << "sec" << endl;
			overall_time += elapsed_time;
			
			int verify = writeSaliencyToPath(files[i], salmap, WRITE_PATH, WRITEPATHSET);
			if(verify==-1) cout << "problem occured in writing to path " << WRITE_PATH << endl;
			if(WRITE_OUT.compare("") != 0) vocus2.write_out(WRITE_OUT);
		
		
			//normalise saliency map
			double mi, ma;
			cv::minMaxLoc(salmap, &mi, &ma);
			salmap = (salmap-mi)/(ma-mi);
			
			if(SHOW_OUTPUT){
				imshow("Saliency normalized", salmap);
				imshow("Inhibition", vocus2.getInhibition());
				imshow("Excitation", vocus2.getExcitation());
				waitKey(0);
			}
			
			destroyAllWindows();
			cout << "Avg. runtime per image: " << overall_time/(double)files.size() << endl;

			
			//get all maxima
			vector< vector<Point> > maxima;
			maxima = computeMSR(salmap,0.7,10);
			
			//show foci
			showMSR(img, maxima, NUM_FOCI);
			
			if(SAVE_FOCI)
				saveFoci(vocus2.getFociPath(),  maxima, NUM_FOCI);
			
			img.release();
			
		}
		return EXIT_SUCCESS;
	}


	
	if(IMAGE_MODE){ // if normal image
		double overall_time = 0;
		
		//compute saliency
		for(size_t i = 0; i < files.size(); i++){
			cout << "Opening " << files[i] << " (" << i+1 << "/" << files.size() << "), ";
			Mat img = imread(files[i], 1);
			
			//basically a test if data is an image or not.
			if( !(img.channels()==3)  || ! img.data ){ 
				std::cerr << "Abort image " << i << endl;
				continue;		
			}

			long long start = getTickCount();
			Mat salmap;
			vocus2.process(img);

			if(COMPUTE_CBIAS) 
			  salmap = vocus2.add_center_bias(CENTER_BIAS);
			else 
			  salmap = vocus2.compute_salmap();

			long long stop = getTickCount();
			double elapsed_time = (stop-start)/getTickFrequency();
			cout << elapsed_time << "sec" << endl;
			overall_time += elapsed_time;
						
			
			if(SHOW_OUTPUT){
				imshow("Input", img);
				imshow("Saliency normalized", salmap);
				showMSR(img,computeMSR(salmap,MSR_THRESH, NUM_FOCI),NUM_FOCI);
				Mat bin = msrVecToMat(computeMSR(salmap,MSR_THRESH, 1)[0], salmap);
				imshow("binMSR", bin);
				waitKey(0);
			}			
			
			int verify = writeSaliencyToPath(files[i], salmap, WRITE_PATH, WRITEPATHSET);
			if(verify==-1) cout << "problem occured in writing to path " << WRITE_PATH << endl;
			if(WRITE_OUT.compare("") != 0) vocus2.write_out(WRITE_OUT);

			img.release();
		}

		cout << "Avg. runtime per image: " << overall_time/(double)files.size() << endl;

	}

	else if(WEBCAM_MODE >= 0 || !VIDEO_MODE){ // data from webcam
		double overall_time = 0;
		int n_frames = 0;

		VideoCapture vc(WEBCAM_MODE);
		if(!vc.isOpened()) return EXIT_FAILURE;

		Mat img;

		while(vc.read(img)){
			n_frames++;
			long start = getTickCount();

			Mat salmap;
			
			vocus2.process(img);

			if(COMPUTE_CBIAS) 
			  salmap = vocus2.add_center_bias(CENTER_BIAS);
			else 
			  salmap = vocus2.compute_salmap();


			long stop = getTickCount();
			double elapsed_time = (stop-start)/getTickFrequency();
			cout << "frame " << n_frames << ": " << elapsed_time << "sec" << endl;
			overall_time += elapsed_time;
			vector<Point> msr = (computeMSR(salmap,MSR_THRESH, 1))[0];


			Point2f center;
			float rad;
			minEnclosingCircle(msr, center, rad);

			if(rad >= 5 && rad <= max(img.cols, img.rows)){
			  //		circle(img, center, (int)rad, Scalar(0,0,255), 3);
			}
			
			if(SHOW_OUTPUT){
				imshow("input (ESC to exit)", img);
				imshow("saliency streched (ESC to exit)", salmap);
				int key_code = waitKey(30);

				if(key_code == 113 || key_code == 27) break;
			}
			img.release();
				
		}
		vc.release();
		cout << "Avg. runtime per frame: " << overall_time/(double)n_frames << endl;
	}
	else{ // Video data
	  for(size_t i = 0; i < files.size(); i++){ // loop over video files
			double overall_time = 0;
			int n_frames = 0;

			VideoCapture vc(files[i]);
		
			if(!vc.isOpened()) return EXIT_FAILURE;

			Mat img;

			while(vc.read(img)){
				n_frames++;
				long start = getTickCount();

				Mat salmap;
			
				vocus2.process(img);

				if(!COMPUTE_CBIAS) salmap = vocus2.compute_salmap();
				else salmap = vocus2.add_center_bias(CENTER_BIAS);

				long stop = getTickCount();
				double elapsed_time = (stop-start)/getTickFrequency();
				cout << "frame " << n_frames << ": " << elapsed_time << "sec" << endl;
				overall_time += elapsed_time;

				vector<Point> msr = (computeMSR(salmap,MSR_THRESH, 1))[0];

				Point2f center;
				float rad;
				minEnclosingCircle(msr, center, rad);

				if(rad >= 5 && rad <= max(img.cols, img.rows)){
				  //	circle(img, center, (int)rad, Scalar(0,0,255), 3);
				}

				if(SHOW_OUTPUT){
					imshow("input (ESC to exit)", img);
					imshow("saliency streched (ESC to exit)", salmap);
					int key_code = waitKey(30);

					if(key_code == 113 || key_code == 27) break;
				}

				img.release();
				
			}
			vc.release();
			cout << "Avg. runtime per frame: " << overall_time/(double)n_frames << endl;
		}
	}		

	return EXIT_SUCCESS;
}
