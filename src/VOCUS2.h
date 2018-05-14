/*****************************************************************************
*
* VOCUS2.h file for the saliency program VOCUS2. 
* A detailed description of the algorithm can be found in the paper: "Traditional Saliency Reloaded: A Good Old Model in New Shape", S. Frintrop, T. Werner, G. Martin Garcia, in Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2015.  
* Please cite this paper if you use our method.
*
* Design and supervision: Simone Frintrop (frintrop@iai.uni-bonn.de)
* @author Thomas Werner   (wernert@cs.uni-bonn.de)
* @author Johannes Teutrine 
*
* @since 1.2
*
* Version 1.2
*
* This code is published under the MIT License 
* (see file LICENSE.txt for details)
*
******************************************************************************/

#ifndef VOCUS2_H_
#define VOCUS2_H_

#include <opencv2/core/core.hpp>

#include <string>
#include <fstream>

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/extended_type_info.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/assume_abstract.hpp>

using namespace std;
using namespace cv;


/** Color Spaces
 *  Lab, Opponent_Codi, Opponent, Itti
 */ 
enum ColorSpace{
	// CIELab
	LAB = 0, /**< L*a*b Colorspace */
	OPPONENT_CODI = 1, /**< like in Klein/Frintrop DAGM 2012 */
	OPPONENT = 2, 	/**< like in Klein/Frintrop DAGM 2012 but shifted and scaled to [0,1]*/
	ITTI = 3 /**< splitted RG and BY channels*/
};

/** Fusion Operation
 * Arithmetic Mean, Max, Uniquess Weight, Manual Weights, Descriptor Weights
 */
enum FusionOperation{
	ARITHMETIC_MEAN = 0, /**< Arithmetic Mean of the maps to fuse*/
	MAX = 1, /**< Maximum */
	UNIQUENESS_WEIGHT = 2, /**< As in Simone Frintrop: VOCUS: A Visual Attention System for Object Detection and Goal-directed Search, PhD thesis 2005 */
	INDIVIDUAL_WEIGHTING = 3, /**< Set weights manualy (See Readme) */
	DESCRIPTOR_WEIGHT = 4, /**< Use weights from a Descriptor */
};

/** Pyramid Structure
*/
enum PyrStructure{
	CLASSIC = 0, /**< Two independent pyramids*/
	CODI = 1, /**< Two independent pyramids derived from a base pyramid*/
	NEW = 2, /**< Derive surround Pyramid from center pyramid */
	SINGLE = 3 /**< Use iNVT pyramids */
};

/** Configuration for VOCUS2
 *  This class holds all parameters for the VOCUS2 computation.
 */
class VOCUS2_Cfg{
public:
	/** default constructor, default parameters */
	VOCUS2_Cfg(){
		c_space = OPPONENT_CODI;
		fuse_feature =  ARITHMETIC_MEAN;
		fuse_conspicuity = ARITHMETIC_MEAN;
		start_layer = 0;
		stop_layer = 4;

		center_sigma = 3;
		surround_sigma = 13; // this is the effective sigma value 
		// the real sigma with which the center image will be smoothed is computed by new_sigma = sqrt(surround_sigma^2 - center_sigma^2)

		n_scales = 2;
		normalize = true;
		pyr_struct = NEW;
		orientation = false;
		combined_features = false;
		numFoci = 2;
		fileFoci="";
		descriptorFile = "";
		for(auto i=0;i<14;i++){
			weights.push_back(-1);
		}
	};

	
	/** constructor for a given config file
	 * @param[in] f_name config file path
	 */
	VOCUS2_Cfg(string f_name){
		for(auto i=0;i<14;i++){
			weights.push_back(0);
		}
		load(f_name);
	}
	
	/** Destructor */
	virtual ~VOCUS2_Cfg(){};
	
	/** Enumarator for Colorspace */
	ColorSpace c_space;
	
	/** Enumerator for Fusing Operation*/
	FusionOperation fuse_feature, fuse_conspicuity;
	
	/** Enumerator for Pyramid Structure*/
	PyrStructure pyr_struct;
	
	/** weight vector for channel weighting
	 */
	vector<double> weights;


	int start_layer /**< Start Layer*/, stop_layer /**< Stop Layer*/, n_scales /**< Number of Scales */;
	float center_sigma /**<Sigma size for center*/, surround_sigma /**<Sigma size for surround*/;
	string descriptorFile; /**< descriptor File path */
	
	bool normalize /**< Toogle output normalization, orientation*/, orientation /**< Toogle use of orientation channel*/, combined_features /**<Toogle combine Color Feature Channels*/;
	
	int numFoci; /**< Number of Foci of Attention to attend */
	string fileFoci; /**< Path of File to store Foci of Attention (append to file) */
	
	/** Load XML Config File
	 * 
	 */
	bool load(string f_name){
		std::ifstream conf_file(f_name);
		if (conf_file.good()) {
			boost::archive::xml_iarchive ia(conf_file);
			ia >> boost::serialization::make_nvp("VOCUS2_Cfg", *this);
			conf_file.close();
			return true;
		}
		else cout << "Config file: " << f_name << " not found. Using defaults." << endl;
		return false;
	}

	/**
	 * Save XML Config File
	 */
	bool save(string f_name){
		std::ofstream conf_file(f_name);
		if (conf_file.good()) {
			boost::archive::xml_oarchive oa(conf_file);
			oa << boost::serialization::make_nvp("VOCUS2_Cfg", *this);
			conf_file.close();
			return true;
		}
		return false;
	}
	
	
	/**
	 * Set Descriptor File member variable to input descriptor file 
	 * @param[in] f descriptor file path
	 */
	void setDescriptorFile(string f){
		descriptorFile = f;
	}
	


private:
    friend class boost::serialization::access;
    template<class Archive>
    
    /** Boost Serialization */
    void serialize(Archive & ar, const unsigned int version){
    	ar & BOOST_SERIALIZATION_NVP(c_space);
    	ar & BOOST_SERIALIZATION_NVP(fuse_feature);
    	ar & BOOST_SERIALIZATION_NVP(fuse_conspicuity);
		ar & BOOST_SERIALIZATION_NVP(pyr_struct);
    	ar & BOOST_SERIALIZATION_NVP(start_layer);
    	ar & BOOST_SERIALIZATION_NVP(stop_layer);
    	ar & BOOST_SERIALIZATION_NVP(center_sigma);
    	ar & BOOST_SERIALIZATION_NVP(surround_sigma);
    	ar & BOOST_SERIALIZATION_NVP(n_scales);
		ar & BOOST_SERIALIZATION_NVP(normalize);
		// ar & BOOST_SERIALIZATION_NVP(orientation);
		// ar & BOOST_SERIALIZATION_NVP(combined_features);	
    }
};


/** VOCUS2 Main Class
 * 
 * Implements the Algorithm of 
 * QUELLE
 * 
 */
class VOCUS2 {
public:
	/** VOCUS2 Constructor 
	 * Construct with default values 
	 */
	VOCUS2();
	
	/** VOCUS2 Constructor
	 * Set Parameters to configurated values
	 * @param[in] cfg Configuration
	 */
	VOCUS2(const VOCUS2_Cfg& cfg);
	
	/**virtual Destrucotr*/
	virtual ~VOCUS2();
	
	/** setCfg 
	 * 
	 * @param[in] cfg Set Configuration to cfg
	 */
	void setCfg(const VOCUS2_Cfg& cfg);
	

	/*!\brief Process Image
	* 
	*	Computes the Image Pyramids from the input image.
	* 	Computes Laplace Pyramid, if orientation channel is toogled.
	* 	Initializes the weight vector for manual weighted feature/conspicuity maps.
	*
	* 	@see pyramid_classic(img)
	*  @see pyramid_new(img)
	*  @see orientation()
	* Does not produce the final saliency map!
	* Has to be called first (before @see compute_salmap)
	* 
	* 	@param[in] img Input Image
	*/	 
	void process(const Mat& image);

	
	/*!\brief Add Center Bias
	* 	
	* 	Adds Center Bias to saliency map
	* 
	*  @param[in] size size of bias
	*	@return Saliency Map with Center Bias
	*/ 
	Mat add_center_bias(float lambda);

	/** Compute Saliency Map
	* Computes Bottom Up Saliency Map with the procedure described in
	* Simone Frintrop, Thomas Werner, and Germán Martín García: "Traditional Saliency Reloaded: A Good Old Model in New Shape".
	*
	* Uses the contrast pyramids and fuses them to feature maps, and fuses feature maps to conspicuity maps, and fuses the conspicuity maps to a single saliency map.
	* The method @see process() has to be called beforehand!
	* 
	*	@return Saliency Map
	*/ 
  	Mat compute_salmap();
  	
  	/*!\brief Compute Top Down Map using a feature descriptor 
	* 	
	* 	Loads a feature descriptor and computes Inhibition and Excitation as 
	*  described in QUELLE
	* 
	*	@return Top-Down Map
	*/ 
  	Mat compute_tdmap();
  	
  	
  	/** Learn Feature Vector
  	 * 
  	 * Learns a feature vector with the algorithm described in 
  	 * QUELLE PHDTHESIS
  	 * 
  	 * @param[in] ROI Region of Interest to learn from
  	 * @param[in] descriptorName Name of the descriptor to load from descriptor directory
  	 */ 
  	void td_learn_featurevector(Rect ROI, string descriptorName);
  	
 
 	/*! \brief Compute the Saliency Map for Visual Search.
	* 	
	*  Alpha is used to balance between Top-Down and Bottom-Up Saliency.
	*  alpha=0: Purely Top-Down
	* 	alpha=1: Purely Bottom-Up
	* 
	*  @param[in] alpha Balance Factor
	* 	@return Weighted Saliency Map
	*/
  	Mat td_search(double alpha);
 
	/** Collects the Feature and Conspicuity Maps in Vectors
	 * 
	 * -Depending on the systems configuration, Vectors might have size 0.
	 * (e.g. feature not used)
	 * -If combined features are used for Color channels, all color feature maps are stored in f_color1.
	 * 
	 * @param[out] f_intensity Intensity feature maps
	 * @param[out] f_color1 Color feature channel 1
	 * @param[out] f_color2 Color feature channel 2
	 * @param[out] f_or orienation feature channel
	 * @param[out] consp Conspicuity maps 
	 */
	void getFeatureAndConspicuityMaps(vector<Mat>& f_intensity, vector<Mat>& f_color1, vector<Mat>& f_color2, vector<Mat>& f_or, vector<Mat>& consp){
		//cout << "feature intensity size " << feat_intensity.size() <<  endl;
		f_intensity = feat_intensity;
		f_color1 = feat_color1;
		f_color2 = feat_color2;
		f_or = feat_orientation;
		consp = consp_maps;
	};
	
	/** Detects local maxima in an image 
	 * @param[out] Input/output image
	 * @param[in] t Region Growing Threshold
	 */
	float maximaDetection(Mat& img, float t = 0.5);
	
	
	/** Get splitted salmap
	 */
	vector<Mat> get_splitted_salmap(); 
	
	
	/*!\brief Write intermediate results
	* 
	* Writes Pyramids, Feature Maps, Conspicuity Maps to a directory.
	* @param[in] dir Destination directory
	*/ 
	void write_out(string dir);
	
	/** Inhibition map
	 *  
	 *  @return inhibition map
	 */
	Mat getInhibition(){
		return inmap;
	}
	
	/** Exciation map
	 *  
	 *  @return exitation map
	 */
	Mat getExcitation(){
		return exmap;
	}
	
	/** Number of Foci points from Configuration
	 * 
	 * @return Number of Foci points to process
	 */
	int getFociNum(){
		return cfg.numFoci;
	}
	
	/** Foci path from configuration
	 * 
	 * @return Path to Foci File from configuration
	 */
	string getFociPath(){
		return cfg.fileFoci;
	}
	

private:
	VOCUS2_Cfg cfg; /**< Vocus configuration */
	Mat input; /**< input image */

	Mat salmap; /**< saliency map */
	Mat exmap; /**< exitation  map */
	Mat inmap; /**< inhibition map */
	Mat tdmap; /**< top down map */
	Mat salmap_global; /** weighted saliency map */
	vector<Mat> salmap_splitted /**< splitted saliency map */, 
	            planes /**< one plane for each color channel  */;
	vector<double> weightValues /**< weight values (manual configuration)  */;
	vector < vector<double> > descriptorWeights; /**< vector of weights from feature descriptor (optional)*/
	unsigned int flagFuseVector=0; /**< if using a descriptor file, this flag marks the vector that is used for the next fusing-operation */
	
	// vectors to hold conspicuity maps and feature maps
	vector<Mat> consp_maps; /**< conspicuity maps */
	vector<Mat> feat_color1 /**< color map channel 1 maps*/, feat_color2 /**< color map channel 2 maps*/;
	vector<Mat> feat_intensity /**< intensity channel maps */;
	vector<Mat> feat_orientation /**< orientation channel */;

	// vectors to hold contrast pyramids as arrays
	vector<Mat> on_off_L /**< contrast pyramids on off L */, off_on_L /**< contrast pyramids off on L */;
	vector<Mat> on_off_a /**< contrast pyramids on off a */, off_on_a /**< contrast pyramids off on a */;
	vector<Mat> on_off_b /**< contrast pyramids on off b */, off_on_b /**< contrast pyramids off on b */;

	// vector to hold the gabor pyramids
	vector<vector<Mat> > gabor /**< gabor filtered pyramids for orientation*/;

	// vectors to hold center and surround gaussian pyramids
	vector<vector<Mat> > pyr_center_L /**< center pyramid L */, pyr_surround_L /**< surround pyramid L */;
	vector<vector<Mat> > pyr_center_a /**< center pyramid a */, pyr_surround_a /**< surround pyramid a */;
	vector<vector<Mat> > pyr_center_b /**< center pyramid b */, pyr_surround_b /**< surround pyramid b */;

	// vector to hold the edge (laplace) pyramid
	vector<vector<Mat> > pyr_laplace /**< laplace pyramid for orientation channel*/;

	bool salmap_ready /**< ready flag for saliency map */, td_ready /**< ready flag for top down map */, splitted_ready /**< ready flag for splitted saliency */, processed /**< ready flag for processed, @see process() */;

	// process image wrt. the desired pyramid structure
	/*!\brief Classic Pyramids
	* 
	*	Computes Classic Pyramids, used in
	*  QUELLE
	* 
	* 	@param[in] img Input Image
	*/ 
	void pyramid_classic(const Mat& image);
	
	/*!\brief Center Surround Difference
	* 
	*	Computes Center Surround Difference for L*a*b Colorspace Image, 
	*  using the Pre-Computed Image Pyramids
	*/ 
	void pyramid_new(const Mat& image);
	
	/*!\brief Pyramid Codi
	* 
	*	Computes Image Pyramids in the style of
	*  QUELLE Codi
	* 
	* 	@param[in] img Input Image
	*/ 
	void pyramid_codi(const Mat& image);


	/** Calculate Weight Vector
	 * 
	 * @param[out] weights Weights for Feature set
	 */
	void getCurrentFeatureSet(vector<double>& weights){
		cout << descriptorWeights.size() << endl;
		cout << flagFuseVector << endl;
		cout << descriptorWeights[0].size() << endl;
		for(unsigned int i=0;i<descriptorWeights[flagFuseVector].size();i++){
			weights.push_back( descriptorWeights[flagFuseVector][i] );
		}
	}
	

	/** Return descriptor file path 
	 *  	
	 * @return descriptor file path
	 */
	string getDescriptorFilePath(){
		string x = cfg.descriptorFile;
		return x;
	}


	/*!\brief Prepares source image for processing in VOCUS2
	* 	
	* 	Checks for valid data, converts to the specified colorspace.
	* 
	*  @param[in] img Input image
	*	@return input image converted to the correct colorspace 
	*/
	vector<Mat> prepare_input(const Mat& img);

	/*!\brief Returns weight configuration
	* 	
	* 	Returns weight configuration for Feature/Conspicuity Maps.
	*  The IDString controls the set of weights that is reported back.
	* 
	* 	@param[in] IDString Identifier for Weight Set
	*	@return Vector of Weights
	*/ 
	vector<double> getWeight(string IDString);
	
	/*!\brief Initializes weight Values for Feature Channels
	* 	
	* 	If not specified by the user, default values are used.  
	*/ 
	void makeWeight();
	
	/*! \brief Clears VOCUS2 member vectors
	*/
	void clear();

	
	/*!\brief Multiscale Pyramids 
	* 	
	* 	Builds Multiscale Gaussian Pyramids as described in [Lowe2004]
	*  @param[in] Image
	* 	@param[in] sigma Sigma Value for Gaussian Pyramids
	*	@return vector<vector<Mat>> pyramids
	*/ 
	vector<vector<Mat> > build_multiscale_pyr(Mat& img, float sigma = 1.f);

	/*!\brief Map Fusion 
	* 	
	* 	Fuses a vector of maps by a specified FusionOperation. 
	* 	Each map can be weighted in the fusion process by a value specified in weights, if the 
	* 	manual weighting flag of VOCUS2 is set.
	* 
	*  @param[in] maps Vector of Feature or Conspicuity Maps to fuse
	*  @param[in] op_in FusionOperation
	*  @param[in] weights Vector of weights
	*	@return Fused Mat
	*/ 
	Mat fuse(vector<Mat> mat_array, FusionOperation op_in, vector<double> weights);

	/*!\brief Map Fusion 
	* 	
	* 	Fuses a vector of feature or conspicuity maps by a specified FusionOperation. 
	* 	Each map can be weighted in the fusion process by a value specified in weights.
	* 
	*  @param[in] maps Vector of Feature or Conspicuity Maps to fuse
	*  @param[in] op_in FusionOperation
	*  @return Fused Mat
	*/ 
	Mat fuse(vector<Mat> mat_array, FusionOperation op_in){
		vector<double> weights;	
		return fuse(mat_array, op_in, weights);
	};
	
	/** Parse descriptor file
	 *  Note: Descriptor file and configuration have to match! 
	 *	This works only if the learning and and the search are working on the same configuration (fusion, Colspaces, pyrs).
	 *  @return 0 on correct parse
	*/
	int parseDescriptorFile(){
		ifstream in;
		cout << "im opening " << getDescriptorFilePath() << endl;
			string  path;
			path = getDescriptorFilePath();
			in.open(path.c_str());

		//vectors to store the descriptor weights
		vector<double> descriptor_int;
		vector<double> descriptor_c1;
		vector<double> descriptor_c2;
		vector<double> descriptor_o;
		vector<double> descriptor_consp;
		
		
		string fst, secnd;
		while(in >> fst ){
			cout << fst;
			in >> secnd;
			cout << secnd <<endl;
			//sort to different vectors for intensity, color, etc	
			if( !(fst.compare("i:"))) //push in intensity vec
				descriptor_int.push_back(atof(secnd.c_str()));
				
			else if( !(fst.compare("c1:"))) //color 1
				descriptor_c1.push_back(atof(secnd.c_str()));
			else if( !(fst.compare("c2:")) ) //color 2
				descriptor_c2.push_back(atof(secnd.c_str()));
			else if( !(fst.compare("or:")) ) //orientation
				descriptor_o.push_back(atof(secnd.c_str()));
			else if( !(fst.compare("con:")) ) //conspicuity
				descriptor_consp.push_back(atof(secnd.c_str())); 
		}

//************** DANIEL EDIT START **************
		// clear weights so that previously learned weights are overwritten
		descriptorWeights.clear();
//************** DANIEL EDIT END   **************

		descriptorWeights.push_back(descriptor_int);
		descriptorWeights.push_back(descriptor_c1);
		descriptorWeights.push_back(descriptor_c2);
		descriptorWeights.push_back(descriptor_o);
		descriptorWeights.push_back(descriptor_consp);
//************** DANIEL EDIT START **************
		//cout << "Report: "<< descriptorWeights.size()<<" and "<< descriptorWeights[0].size() << "intensity data and " << descriptorWeights[1].size() << "c1 data AND " <<  descriptorWeights[2].size() << "c2 data AND "<<descriptorWeights[3].size() << "o data AND " << descriptorWeights[4].size() << " conspicuity data."  << endl;

//************** DANIEL EDIT END   **************
				return 0;
	}


	/** Center Surround Difference
	* 
	*	Computes Center Surround Difference for Lab Colorspace Image, 
	*   using the Pre-Computed Image Pyramids. 
	* 	Results are the contrast pyramids on_off_L, off_on_L, on_off_a, off_on_a, on_off_b, off_on_b.
	*/ 
	void center_surround_diff();
	
	/** \brief Orientation Channel
	* 	
	*  Computes Laplace Pyramid and gabor filters for four orientations. 
	*/ 
	void orientation();

	/*!\brief Uniqueness Weight Measure 
	* 	
	*  Weight the feature maps according to the number of local maxima in the map.
	*  See
	*  Simone Frintrop: "VOCUS: A Visual Attention System for Object Detection and Goal-directed Search"
	*  for further information.
	*  @param[in] img Input map
	*  @param[in] t Threshold for region growing
	*  @return Uniqueness Weight for the input map
	*/ 	
	float compute_uniqueness_weight(Mat& map, float t);

};


#endif /* VOCUS2_H_ */
