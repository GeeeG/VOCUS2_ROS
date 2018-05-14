/*****************************************************************************
*
* ImageFunctions.h file for the saliency program VOCUS2. 
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

#ifndef ImageFunctions_H_
#define ImageFunctions_H_

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/stat.h>
#include <string>
	
#include "ImageFunctions.h"

using namespace cv;
using namespace std;



/** Floodfill Recursion
 *  
 * Recursion step of floodfile subroutine. 
 * Takes a pixel as an input and checks for each neighbor in the 4-neighborhood of the input pixel,
 * if the neighbor was attended before and if the value of the neighbor is lower as the bottom value.
 * If both is not the case, the neighbor pixel is marked as visited and is input to the next recursion step. 
 * If not, the recursion ends.
 * @see floodfill()
 * 
 * @param[in] src Source Matrix
 * @param[in] x x coodinate of current pixel
 * @param[in] y y coordinate of current pixel
 * @param[in] bot bottom value for filling the pixel
 * @param[in] vis Visited flag
 * @param[in] flood_minmax neighbors of the input pixel
 */
void floodfill_r(Mat& src, int x, int y, double bot, vector < vector<bool> >& vis, double* flood_minmax);




/** Floodfill
 * 
 *  Initialization Step of floodfill subroutine.
 *  Calls @see floodfill()
 *  @param[in] src Source Matrix
 *  @param[in] x x coodinate of current pixel
 *  @param[in] y y coordinate of current pixel
 *  @param[in] eps threshold for the filling process
 *  @param[in] flood_minmax 
 */
void floodfill(Mat& src, int x, int y, double eps, double* flood_minmax);

/** Truncated Map 
 * 
 * The Roi is copied from the input image to the output Mat. 
 * All other pixels of the output map are set to zero. 
 * Type, number of columns and number of rows of the output mat 
 * are the same as of the input image. 
 * 
 * @param[in] map Input Image
 * @param[in] roi Region of Interest in map
 * @return Mat with ROI
 */
Mat truncToZero(Mat map, Rect roi);

/** Mean Value Calculation
 * 
 * Calculates Mean of all pixels in the map.
 * 
 * @param[in] map Truncated input map
 * @param[in] pixels Number of pixels to divide by
 * @return Mean Value
 */
double calcMeanVal(Mat& map, int pixels);



/** Sets msr pixels to zero in output image patch
 * 
 * Sets all pixels in the input image patch to zero, when they occur (are not zero) in the msr Mat.
 * 
 * @param[in] msr Most salient region in patch
 * @param[out] xminmsr input image patch
 */ 
void calcXminMsr(Mat& msr, Mat& xminmsr);

/** Compute Most Salient Region(s)
 * 
 * Calculates local maxima and returns them as point vectors. 
 * 
 * 
 * @param[in] salmap Saliency map
 * @param[in] threshold Threshold for regions
 * @param[in] maxRegions Calculate at most maxRegions 
 * @return Vectors of local maxima pixels
 */
vector< vector<Point> > computeMSR(Mat& salmap, double threshold, int maxRegions);

/** Transfer local maximum point vector to a Mat
 * 
 * Calculates a binary image for a local maximum, where all maximum points correspond to white points in the image while the 
 * other pixels are black. 
 * The Saliency map is used as input to get type, number of rows and number of cols for the output image.
 * 
 * @param[in] msr Local maximum point vector
 * @param[in] saliencyMap Saliency map
 * @return binary image with white local maximum region 
 */
Mat msrVecToMat(vector<Point> msr, Mat saliencyMap);

/** Calculates a white mask that covers the most salient region in an image patch
 * 
 * Stores the white mask in parameter mask. 
 * Stores the most salient region in parameter map.
 * Uses a region growing to extract the msr. 
 * @see floodfill()
 * 
 * @param[out] mask white region mask
 * @param[out] input/output image patch
 * @param[in] thresh Growing Threshold
 * @return count of pixels in mask
 */
int computeMSRmask(Mat& mask, Mat& map, double thresh);

/** Extract MSR from the roi 
 *  
 * Copies all pixels from the roi that are white in the mask to msr.  
 * @see computeMSRmask()  
 * 
 * @param[out] msr output most salient region
 * @param[in] mask image patch mask 
 * @param[in] roi Region of interest
 */
void extractMSR(Mat& msr, Mat mask, Mat roi);


/** Compute Feature Value
 * 
 * For Calculation details see
 * Simone Frintrop: "VOCUS: A Visual Attention System for Object Detection and Goal-directed Search"
 * 
 * @param[out] ftmap Feature map
 * @param[in] mask White mask
 * @param[in] pixels_mask pixel count of the mask
 * @param[in] pixels_rest pixel count of msr without mask
 * @param[in] inpRoi The region of interest
 * @return descriptor Value
 */ 
float computeFeatureValue(Mat& ftmap, Mat& mask, int pixels_mask, int pixels_rest, Rect inpRoi);


#endif







