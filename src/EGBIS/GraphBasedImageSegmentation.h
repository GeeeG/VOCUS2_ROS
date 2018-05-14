#ifndef GRAPHBASEDIMAGESEGMENTATION_H
#define GRAPHBASEDIMAGESEGMENTATION_H


#include "Component.h"
#include <algorithm>
#include "Vertex.h"
#include "Edge.h"

/**
 *	@author Fahrettin Gökgöz
 *	@link
 *
 *
 *
 *
 *
 */
class GraphBasedImageSegmentation{
private: //Data Elements

	typedef  int vertexIDs; 	// Point.x * Image.cols + Point.y

	//
	cv::Mat rawImage;
	//
	cv::Mat segmentedImage;

	std::vector<Component*> IncludedComponent;
	//
	std::set<Component*> components;
	//
	std::vector<Vertex> vertices;
	//
	std::vector<Edge> edges;

	//Algorithm's Parameters

	int min_size;

	float sigma;

	float k;







private: //functions



	void smoothImage(	const cv::Mat& image,
						cv::Mat &smoothedImage,
						float sigma
	);



	cv::Mat createNeighborhoodMask(int number);



 	void buildGraph( 	const cv::Mat& smoothedImage,
 						const cv::Mat& neighborMask,
 						std::vector<Edge> &edges,
 						std::vector<Vertex> &vertices
 	);



 	void createComponentsAndMap(	std::vector<Vertex> &vertices,
 									std::set<Component*> &components,
 									float k,
 								std::vector<Component*> &IncludedComponent
 	);



 	void segment(	std::vector<Vertex> &vertices,
 					std::vector<Edge> &edges,
 					std::set<Component*> &components,
 					std::vector<Component*> &IncludedComponent,float k
 	);



	void merge(	Component *f, Component *t,float weight,
					std::set<Component*> &components,
					std::vector<Component*> &IncludedComponent
	);

	void joinSmallComponents();



 	cv::Vec3b randomColor();


	cv::Mat paintOutput();



	int calculateIndex(int x, int y, int maxY);


	cv::Point2d calculatePoint(int ID,int maxY);


public:
	cv::Mat segmentImage(cv::Mat iImage,float
			sigma,float k, int min_size);

	std::set<Component*>& getComponents(){
		return components;
	}

	void cleanData(){
		std::set<Component*>::iterator it;

		for(it = components.begin();it != components.end();++it){
			//delete Remaining components
			delete (*it);
		}
	}
};



/**
 * Given the postion of vertex, it calculates the vertex ID
 * @param x 	: X postion of vertex
 * @param y 	: Y postion of vertex
 * @param maxY	: Maximum Possible column number
 */
int GraphBasedImageSegmentation::calculateIndex(
		int x, int y, int maxY){

	return x*maxY + y;
}




/**
 * 	Given the Vertex ID and Maximum Column, it calulates
 * 	the position of vertex
 *  @param ID		: Vertex ID
 *  @Param maxY		: Maximum possible column from image
 */
cv::Point2d GraphBasedImageSegmentation::calculatePoint(
		int ID,int maxY){

	int x= ID/maxY;
	int y=ID-(x*maxY);
	return cv::Point2d(x,y);
}



/**
 *	Given the Input image and sigma for the Gaussian and threshold for the
 *	components and minimum size of them, it segments the Input image using
 *	graph and then produce a segmented image
 *
 *	@Param iImage	: Input Image
 *	@Param sigmaG	: Sigma for the Gaussian Filter
 *	@Param kG		: Threshold for Components
 *	@Param min_sizeG: Minimum size of each components
 *	@return sImage	: Segmented Image
 */
cv::Mat GraphBasedImageSegmentation::segmentImage(
		cv::Mat iImage,float sigmaG,float kG, int min_sizeG){

	//copy parameters
	this->rawImage = iImage.clone();
	this->k = kG;
	this->min_size = min_sizeG;
	this->sigma = sigmaG;
	cv::Mat smoothedImage;

	///***Pre-processing Part***///
	smoothImage( rawImage, smoothedImage,  sigma);

	//Create a neighborhood mask to create the graph
	cv::Mat nMask = createNeighborhoodMask(1);

	//Build graph from input Image
	buildGraph( smoothedImage, nMask, edges, vertices);

	///***Algorithm Part***///
	segment(vertices, edges, components, IncludedComponent, kG);

	///***Post-processing Part***///
 	joinSmallComponents();

 	//Produce Segmented Image
 	cv::Mat sImage = paintOutput();

 	//clean Used space
 	//cleanData();

 	return sImage;
}






/***************************************************************************\
*						Pre-Processing Functions							*
\***************************************************************************/

/**
 * Smoothes the image with Gaussian filter with given sigma
 * and makes sure Sigma is bigger then or equal to 0.01
 * @param image
 * @param smoothedImage
 * @param sigma
 */
void GraphBasedImageSegmentation::smoothImage(
		const cv::Mat& image, cv::Mat &smoothedImage, float sigma){

	//set sigma for both dimenasion
	float cSigma = std::max(sigma,0.01f);

	//set kernel size
	int len = (int)std::ceil(sigma * 4) + 1;
	len = (len - 1) * 2 + 1;

	//Smooth image
	cv::GaussianBlur(	image, smoothedImage,
						cv::Size(len,len), cSigma, cv::BORDER_CONSTANT
					  );
}


/**
 * 	Currently: Only produce a one type of neighbor to test the program
 *	@Param number : Neighborhood number
 */
cv::Mat GraphBasedImageSegmentation::createNeighborhoodMask(
		int number){

	//Neighborhood mask
	cv::Mat nMask(3,3,CV_8UC1);

	// 1 1 1
	nMask.at<uchar>(0,0)=1;nMask.at<uchar>(0,1)=1;nMask.at<uchar>(0,2)=1;
	// 1 0 0
	nMask.at<uchar>(1,0)=1;nMask.at<uchar>(1,1)=0;nMask.at<uchar>(1,2)=0;
	// 0 0 0
	nMask.at<uchar>(2,0)=0;nMask.at<uchar>(2,1)=0;nMask.at<uchar>(2,2)=0;

	return nMask;
}


/**
 * Builds a graph from image using pixels as vertexes and
 * Euclidian distance between neighbor pixels as edges
 * @param smoothedImage
 * @param neighborhoodMask
 * @param edges
 * @param vertices
 * @return
 */
void GraphBasedImageSegmentation::buildGraph(
		const cv::Mat& smoothedImage, const cv::Mat& neighborMask,
		std::vector<Edge> &edges, std::vector<Vertex> &vertices){

	int numberOfRows=smoothedImage.rows;
	int numberOfCols=smoothedImage.cols;
	int neighborRows = neighborMask.rows;
	int neighborCols = neighborMask.cols;
	int maskXRadius = neighborRows/2;
	int maskYRadius = neighborCols/2;

	//set vertex size to image pixel size
	vertices.resize(smoothedImage.rows*smoothedImage.cols);
	int numberOfEdges = 8*	 smoothedImage.rows*smoothedImage.cols;/* -
							(2*(smoothedImage.rows+smoothedImage.cols));*/
	int edgeCount=0;
	edges.resize(numberOfEdges);

	Vertex fromV,toV;

	//For each pixel in the picture
	for(unsigned int i=0;i<numberOfRows;i++){
		for(unsigned int j=0;j<numberOfCols;j++){

			//Add each available neighbor to graph
			for(unsigned int k=0;k<neighborRows;k++){
				for(unsigned int l=0;l<neighborCols;l++){

					//if the mask is set and the neighbor available
					if( neighborMask.at<uchar>(k,l) && 		// if mask is set
						(i+(k-maskXRadius))>=0 &&			// Up available
						(i+(k-maskXRadius))<numberOfRows &&	// down available
						(j+(l-maskYRadius))>=0 &&			// left available
						(j+(l-maskYRadius))<numberOfCols	// right avail.
					){


					//Vertice Locations
					cv::Point2d from(i,j);
					cv::Point2d to(i+(k-maskXRadius), j+(l-maskYRadius));


					//add new vertices

					//From Vertex
					fromV.setVerticeProperty(
						 calculateIndex(i,j, smoothedImage.cols),from
					);

					vertices.at(fromV.getPixelID()) = fromV;

					//To Vertex
					toV.setVerticeProperty(
						 calculateIndex( to.x, to.y, smoothedImage.cols),to
					);

					vertices.at(toV.getPixelID()) = toV;


					//add edge

					//V(from) <=> V(to)  = distance
					edges.at(edgeCount) = (
						Edge(
							&vertices.at(fromV.getPixelID()),
							&vertices.at(toV.getPixelID()),
							//Calculate distance between pixels
							cv::norm(
								smoothedImage.at<cv::Vec3b>(from.x,from.y),
								smoothedImage.at<cv::Vec3b>( to.x, to.y ),
								cv::NORM_L2
							)
						)
					);

					edgeCount++;

				  }
				}
			}
		}
	}
	edges.resize(edgeCount);
}

/***************************************************************************\
*							Algorithm Functions								*
\***************************************************************************/

void GraphBasedImageSegmentation::segment(std::vector<Vertex> &vertices,
		std::vector<Edge> &edges, std::set<Component*> &components,
		std::vector<Component*> &IncludedComponent,float k)
{
	//step 0 : 	Sort E into π = (o1 , ... , om ),
	//			by non-decreasing edge weight.
	std::sort (edges.begin(), edges.end());

	//Step 1 : 	Start with a segmentation S0 ,
	//			where each vertex Vi is in its own component.
	createComponentsAndMap(vertices, components, k, IncludedComponent);

	//Step 2 : 	Repeat step 3 for q = 1, ... , m.

	//Step 3 : 	If Ci^(q-1) != Cj^(q-1) and w(oq) <= MInt(Ci^(q-1),Cj^(q-1))
	//			Then merge components
	for(int i=0;i<edges.size();i++){
		// From Index
		int fIndex = edges.at(i).getCopyOfConnections(0).getPixelID();
		// To Index
		int tIndex = edges.at(i).getCopyOfConnections(1).getPixelID();
		// Edge Weight
		float weight = edges.at(i).getWeight();

		//Components that have the vertices
		Component *f=IncludedComponent[fIndex],*t= IncludedComponent[tIndex];

		//If components are not equal
		if(f != t &&
			//If minimum of internal Differences is greater then weight
			weight < f->getInternalDifference() &&
			weight < t->getInternalDifference()
		){
			//Find small component to move the data
			if(f->getComponentSize()<t->getComponentSize()){
				//merge Components
				merge( t,  f,  weight, components, IncludedComponent);
			}else{
				//merge Components
				merge( f,  t,  weight, components, IncludedComponent);
			}
		}
	}
}


/**
 * This function Takes Vertices of created graph and given threshold
 * then creates components and maps each individiual vertex to
 * individual component
 *
 * @param vertices			: Vertices of Created Graph
 * @param components		: Component list to fill
 * @param k					: Threshold Constant
 * @param IncludedComponent	: VertexID -> Component Map
 */
void GraphBasedImageSegmentation::createComponentsAndMap(
		std::vector<Vertex> &vertices,
		std::set<Component*> &components,float k,
		std::vector<Component*> &IncludedComponent)
{
	//Reserve Place
	IncludedComponent.resize(vertices.size());

	//Add each individual vertex to individuals components
	for(int i=0;i<vertices.size();i++){

		//create new component
		Component * cTemp = new Component(&vertices.at(i));

		// set threshold to given parameter
		cTemp->setThresholdconstant(k);

		//Insert to component list
		components.insert(cTemp);

		//to keep track of connected components
		IncludedComponent[i] = cTemp;

	}

}











/**
 * This function merges two components. To do that it moves the
 * pointers of vertices from one component to other.Then it updates
 * the map for vertexID to component and weight, then finally it
 * clears the second component
 *
 * @Param f					: Component to move data
 * @Param t					: Component to take data
 * @Param weight			: New weight for component
 * @Param components		: To clear the merged component
 * @Param IncludedComponent : To update the map
 *
 */
void GraphBasedImageSegmentation::merge(Component *f, Component *t,
		float weight, std::set<Component*> &components,
		std::vector<Component*> &IncludedComponent)
{
	//merge
	(*f) += (*t);
	// Set Max weight in component
	f->setMaxWeight(weight);

	//update mapping for vertices
	const std::set<Vertex*> iVertices = t->getVertices();

	std::set<Vertex*>::iterator it;


	for(it = iVertices.begin();it != iVertices.end();++it){
		IncludedComponent[(vertexIDs)(*it)->getPixelID()] = f;
	}
	//delete old component
	components.erase(t);
	delete t;
}












/***************************************************************************\
*						Post-Processing Functions							*
\***************************************************************************/


/**
 *
 *
 *
 */
void GraphBasedImageSegmentation::joinSmallComponents(){

	for(int i=0;i<edges.size();i++){

		// From Index
		int fIndex = edges.at(i).getCopyOfConnections(0).getPixelID();
		// To Index
		int tIndex = edges.at(i).getCopyOfConnections(1).getPixelID();
		// Edge Weight
		float weight = edges.at(i).getWeight();

		//Components that have the vertices
		Component *f=IncludedComponent[fIndex],*t= IncludedComponent[tIndex];

		//If components are not equal
		if(f != t &&
					(
						//and one of them smaller then the minimum size
						min_size > f->getComponentSize() ||
						min_size > t->getComponentSize()
					)
		){

			//Find small component to move the data
			if(f->getComponentSize()<t->getComponentSize()){
				//merge Components
				merge( t,  f,  weight, components, IncludedComponent);
			}else{
				//merge Components
				merge( f,  t,  weight, components, IncludedComponent);
			}
		}
	}
}




/**
 *	Given the segmentes components, it creates the segmented image
 */
cv::Mat GraphBasedImageSegmentation::paintOutput(){
	std::set<Component*>::iterator itC;
	std::set<Vertex*>::iterator itV;
	cv::Vec3b clr;

	//Set Size of segmented image to input image
	segmentedImage = cv::Mat(rawImage.rows,rawImage.cols,rawImage.type());

	for(	itC = this->components.begin();
			itC!= this->components.end();
			++itC)
	{
		//pick a random color for component
		clr = randomColor();

		//Paint each vertex in the component with selected color
		for(	itV = (*itC)->getVertices().begin();
				itV !=(*itC)->getVertices().end();
				++itV)
		{
			segmentedImage.at<cv::Vec3b>(
					(*itV)->getPixelLocation().x,
					(*itV)->getPixelLocation().y
			) = clr;
		}
	}

	return segmentedImage;
}


/**
 * Creates 8 bit 3 channel random color
 */
cv::Vec3b GraphBasedImageSegmentation::randomColor(){
	cv::Vec3b color;
	uchar r=(uchar)random(),g=(uchar)random(),b=(uchar)random();
	color[0] = r; color[1] = g; color[2] = b;
	return color;
}


#endif
