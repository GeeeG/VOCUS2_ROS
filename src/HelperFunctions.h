/*****************************************************************************
*
* HelperFunctions.h file for the saliency program VOCUS2. 
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

#ifndef HelperFunctions_H_
#define HelperFunctions_H_

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/stat.h>
#include <string>
	
#include "ImageFunctions.h"


/** Annotate ROI in image
 * 
 * The input image is shown to the user and the user can draw a rectangle around the region of interest.
 * 
 * @param[in] src input image
 * @return Rect annotated region as rectangle
 */
Rect annotateROI(Mat& src);

/** Prints Help
 * 
 * @param[in] argv command line char*
 */ 
void print_usage(char* argv[]);

/** Mouse Handler
 * 
 * @see annotateROI()
 * 
 * @param[in] event Event code
 * @param[in] x clicked x coordinate
 * @param[in] y clicked y coordinate
 * @param[in] flags global click/drag flag
 * @param[in] param 
 */
void mouseHandler(int event, int x, int y, int flags, void* param);

/** Writes Saliency map
 * 
 * Writes saliency map to a given path
 * 
 * @param[in] base_path Path
 * @param[in] salmap Saliency map
 * @param[in] WRITE_PATH Path configured by user in command line
 * @param[in] WRITEPATHSET Boolean Path-set value
 * @return 0 on correct written map
 */
int writeSaliencyToPath(string base_path, Mat salmap, string WRITE_PATH, bool WRITEPATHSET);

/** show local maxima
 * 
 * Show local maxima marked by a circle in the image.
 * 
 * @param[in] img Input image
 * @param[in] maxima all maxima as point vectors
 * @param[in] NUM_FOCI the maximum number of foci to show
 */
void showMSR(Mat img, vector< vector<Point> > maxima, int NUM_FOCI);

/** Save Foci points to a file
 * 
 * @param[in] filePath path of the Foci file
 * @param[in] maxima all maxima as point vectors
 * @param[in] NUM_FOCI Number of Foci to write
 */
void saveFoci(string filePath, vector< vector<Point> > maxima, int NUM_FOCI);


#endif
