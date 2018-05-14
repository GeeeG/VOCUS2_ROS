#ifndef VERTEX_H
#define VERTEX_H

#include <iostream>
#include <opencv2/opencv.hpp>


/*
 *	Vertex
 *
 *
 *	@Author: Fahrettin Gökgöz
 */
class Vertex{
		//	Must be unique for each vertex to compare
		//
		//	e.g. For the same image
		//  [X position * image.columns + Y position]
		//	is Unique to pixel

		unsigned int pixelID;

		cv::Point2d pixelLocation;

	    int compare(const Vertex& b)const;

	    void copy(const Vertex& cpy);

	    void init(const int id, const cv::Point2d pLoc);

	    friend std::ostream & operator<<(
	    						std::ostream & screen,
	    						const Vertex &out
	    );

	public:

	    //Set Functions
	    void setPixelLabel(int ID);

		void setPixelLocation(cv::Point2d location);

		void setVerticeProperty( int ID,cv::Point pLoc);

		//Get Functions
		unsigned int getPixelID()const;

		cv::Point2d getPixelLocation() const;



		//Operators
		bool operator<(const Vertex& b)const;

		bool operator>(const Vertex& b)const;

		bool operator>=(const Vertex& b)const;

		bool operator<=(const Vertex& b)const;

		bool operator==(const Vertex& b)const;

		bool operator!=(const Vertex& b)const;

		Vertex& operator=(const Vertex &b);

		//Constructors
		Vertex();

		Vertex(int pID,cv::Point2d pLoc);

		Vertex(const Vertex &cpy);

};



/***************************************************************************\
*							Private Functions								*
\***************************************************************************/



/**
 * 	Compares Vertex with given Vertex, Then returns
 * 	-1 for smaller
 * 	0 for equal
 * 	1 for greater
 * 	@Param b
 * 	@Return
 */
int Vertex::compare(const Vertex& b)const{
	if(pixelID<b.pixelID){return -1;}
	else if(pixelID==b.pixelID){return 0;}
	else{return 1;}
}

/**
 *
 *
 */
void Vertex::copy(const Vertex& cpy){
	this->pixelID = cpy.pixelID;
	this->pixelLocation= cpy.pixelLocation;
}


/**
 *
 *
 */
void Vertex::init(const int id, const cv::Point2d pLoc){
	this->pixelID = id;
	this->pixelLocation = pLoc;
}



/***************************************************************************\
*							Public Functions								*
\***************************************************************************/



/**
 * 	Sets the pixel ID
 *	@Param ID
 */
void Vertex::setPixelLabel(int ID){
	pixelID = ID;
}



/**
 *	Sets the pixel location
 *	@Param location
 */
void Vertex::setPixelLocation(cv::Point2d location){
	pixelLocation = location;
}



/**
 * 	Returns the Pixel ID
 *	@Return pixelID
 */
unsigned int Vertex::getPixelID()const{
	return pixelID;
}

/**
 * 	Returns the pixel location
 *	@Return pixelLocation
 */
cv::Point2d Vertex::getPixelLocation() const{
	return pixelLocation;
}


/**
 *	Checks smaller
 *	@Param b
 */
bool Vertex::operator<(const Vertex& b)const{
	return compare(b)==-1?true:false;
}

/**
 *	Checks greater
 *	@Param b
 */
bool Vertex::operator>(const Vertex& b)const{
	return compare(b)==1?true:false;
}

/**
 *	Checks greater or equal
 *	@Param b
 */
bool Vertex::operator>=(const Vertex& b)const{
	return compare(b)!=-1?true:false;
}


/**
 *	Checks smaller or equal
 *	@Param b
 */
bool Vertex::operator<=(const Vertex& b)const{
	return compare(b)!=1?true:false;
}


/**
 * 	Checks equality with the given Vertex
 *	@Param b
 */
bool Vertex::operator==(const Vertex& b)const{
	return compare(b)==0?true:false;
}


/**
 * 	Checks not equality with the given Vertex
 *	@Param b
 */
bool Vertex::operator!=(const Vertex& b)const{
	return compare(b)==0?false:true;
}


/**
 *
 */
Vertex& Vertex::operator=(const Vertex &b) {
	copy(b);
	return *this;
}

/**
 *	Creates a vertex with pixel ID is equal to 0
 *	and pixel location is equal to (0,0)
 */
Vertex::Vertex(){
	init(0, cv::Point2d(0,0));
}




/**
 *	Creates a vertex with given pixel ID and pixel Location
 *	@Param pID
 *	@Param pLoc
 */
Vertex::Vertex(int pID,cv::Point2d pLoc){
	setPixelLabel(pID);
	setPixelLocation(pLoc);
}


/**
 * 	Copy Constructor
 *	@Param cpy
 */
Vertex::Vertex(const Vertex &cpy){
	copy(cpy);
}


/**
 *
 */
void Vertex::setVerticeProperty( int ID,cv::Point pLoc){
	setPixelLabel(ID);
	setPixelLocation(pLoc);
}

/***************************************************************************\
*							Utility Functions								*
\***************************************************************************/


/**
 *	Produce a human readable information for the Vertex
 *	@Param screen
 *	@Param out
 */
std::ostream & operator<<(std::ostream &screen,const Vertex &out) {

	screen	<<"PixelID: ["<<out.pixelID<<"] "
			"Coordinates : ("
						<< out.getPixelLocation().x<<","
						<< out.getPixelLocation().y<<") "
			<<std::endl;
	return screen;
}
#endif
