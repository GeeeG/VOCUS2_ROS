#ifndef EDGE_H
#define EDGE_H

#include "Vertex.h"


/*
 *	Class Edge
 *	Author: Fahrettin Gökgöz
 *
 *
 *
 *
 *
 */
class Edge{
	Vertex *a,*b;

	float weight;

	int compare(const Edge& b)const;

	void copy(const Edge& b);

	friend std::ostream & operator<<(
				std::ostream & screen,
				const Edge &out
	);

public:

	//Set Functions
	inline void setWeight(float w);

	inline void setConnections(Vertex *a,Vertex *b);

	//Get Functions
	inline float getWeight();

	Vertex getCopyOfConnections(int con);

	const Vertex* getPointerToConnections(int con);

	//Operators
	bool operator<(const Edge& b)const;

	bool operator>(const Edge& b)const;

	bool operator>=(const Edge& b)const;

	bool operator<=(const Edge& b)const;

	bool operator==(const Edge& b)const;

	bool operator!=(const Edge& b)const;

	Edge& operator=(const Edge &b);

	//Constructors
	Edge();

	Edge(Vertex *a,Vertex *b, float w);

	Edge(const Edge& cpy);
};



/***************************************************************************\
*							Private Functions								*
\***************************************************************************/


/**
 *
 *
 */
int Edge::compare(const Edge& b)const{
	if(weight<b.weight){return -1;}
	else if(weight==b.weight){return 0;}
	else{return 1;}
}

/**
 *
 *
 */
void Edge::copy(const Edge& b){
	this->a = b.a;
	this->b = b.b;
	this->weight = b.weight;
}




/***************************************************************************\
*							Public Functions								*
\***************************************************************************/



/**
 * Returns the edge weight
 * @Return weight
 */
inline float Edge::getWeight(){
	return weight;
}

/**
 * This function returns the required vertex
 * 0 for the first one, rest will return the second one
 * @param con
 */
Vertex Edge::getCopyOfConnections(int con){
	if(con==0){
		return Vertex(a->getPixelID(),a->getPixelLocation());
	}
	return Vertex(b->getPixelID(),b->getPixelLocation());
}



/**
 * Sets the weight in between vertices
 *	@Param w
 */
inline void Edge::setWeight(float w){
	weight = w;
}




/**
 * 	Sets the pointers to vertices
 *	@Param a
 *	@Param b
 */
inline void Edge::setConnections(Vertex *a,Vertex *b){
	this->a = a; this->b = b;
}


//Operators

/**
 *
 *
 */
bool Edge::operator<(const Edge& b)const{
	return compare(b)==-1?true:false;
}


/**
 *
 *
 */
bool Edge::operator>(const Edge& b)const{
	return compare(b)==1?true:false;
}


/**
 *
 *
 */
bool Edge::operator>=(const Edge& b)const{
	return compare(b)==-1?false:true;
}


/**
 *
 *
 */
bool Edge::operator<=(const Edge& b)const{
	return compare(b)==1?false:true;
}


/**
 *
 *
 */
bool Edge::operator==(const Edge& b)const{
	return compare(b)==0?true:false;
}


/**
 *
 *
 */
bool Edge::operator!=(const Edge& b)const{
	return compare(b)==0?false:true;
}


/**
 *
 *
 */
Edge& Edge::operator=(const Edge &b){
	copy(b);
	return *this;
}


//Constructors

/**
 *
 *
 */
Edge::Edge(){
	this->a = NULL;
	this->b = NULL;
	this->weight = 0.0f;
}



/**
 *
 *
 */
Edge::Edge(Vertex *a,Vertex *b, float w){
	setWeight(w);
	setConnections(a,b);
}



/**
 *
 *
 */
Edge::Edge(const Edge& cpy){
	copy(cpy);
}


/***************************************************************************\
*							Utility Functions								*
\***************************************************************************/




/**
 *	Produce a human readable information for the Edge
 *	@Param screen
 *	@Param out
 */
std::ostream & operator<<(std::ostream & screen,const Edge &out){
	screen	<<"Edge Connect Points "
			"[ 	("	<<out.a->getPixelLocation().x <<" , "
					<<out.a->getPixelLocation().y<<") , "
			   "(" 	<<out.b->getPixelLocation().x<<" , "
			   	   	<<out.b->getPixelLocation().y<<")   ]"
			   "	with weight "<<out.weight<<std::endl;
	return screen;
}


#endif
