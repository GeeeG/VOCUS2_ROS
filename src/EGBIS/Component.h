#ifndef COMPONENT_H
#define COMPONENT_H



#include <iostream>
#include <set>
#include <math.h>
#include "Vertex.h"



/*
 *	Component.h
 *	Author: Fahrettin Gökgöz
 *
 *
 *
 *
 */
class Component{
	private:

 		std::set<Vertex*> includedVertices;	//	Pointer to included vertices

 		float minInternalDifference;		//	MInt(C) = Int(C) + T(C)

 		float thresholdconstant;			//	k

 		float maxWeight;					//	max w(e) in MST


		void calculateMInt();


		void join(const Component& c2);

		void copy(const Component& c2);

		bool isEqual(const Component& c2);


		void addVertice(Vertex* newVertice);


		friend std::ostream & operator<<(
				std::ostream & screen,const Component &out
		);

	public:

		//Set Functions
		void setMaxWeight(double w);

		void setThresholdconstant(float constant);


		//Get Functions
		unsigned int getComponentSize();

		inline float getInternalDifference();

		const std::set<Vertex*> & getVertices();

		//Operators
		void operator+=(const Component &c2);

		bool operator==(const Component &c2);

		bool operator<(const float weightOfNewEdge)const;

		Component& operator=(const Component &c2);

		//
		void add(Vertex* v0);

		void joinComponents(Component c2);


		//Constructors
		Component();

		Component(Vertex& v0);

		Component(Vertex* v0);

		Component(Component& cmp);


};







/***************************************************************************\
*							Private Functions								*
\***************************************************************************/





/**
 *	Checks all data in given Component
 *	@Param c2
 *	@Return
 */
bool Component::isEqual(const Component& c2){
	if(minInternalDifference!=c2.minInternalDifference){
		return false;
	}

	if(maxWeight!=c2.maxWeight){
		return false;
	}

	if(thresholdconstant!=c2.thresholdconstant){
		return false;
	}

	if(includedVertices.size()!= c2.includedVertices.size()){
		return false;
	}

	std::set<Vertex*>::iterator iter,finish;

	finish = this->includedVertices.end();

	for(iter = c2.includedVertices.begin();
		iter!=c2.includedVertices.end();
		++iter)
	{
		if(finish == this->includedVertices.find(*iter)){
			return false;
		}

	}

	//if all passed return true;
	return true;
}


/**
 *
 */
void Component::copy(const Component& c2){
	includedVertices.clear();
	join(c2);
	minInternalDifference= c2.minInternalDifference;
	thresholdconstant = c2.thresholdconstant;
	maxWeight=c2.maxWeight;
}


/**
 *	joins two set of elements
 *	@Param c2
 */
void Component::join(const Component& c2){
	std::set<Vertex*>::iterator iter;

	for(iter = c2.includedVertices.begin();
		iter!=c2.includedVertices.end();
		++iter)
	{
		addVertice(*iter);
	}
}



/**
 *	Calculates the minimum internal difference
 */
void Component::calculateMInt(){
	minInternalDifference =
			maxWeight +
			(thresholdconstant/(float)includedVertices.size());
}


/**
 *	Adds new vertex to vertex set
 *	@Param newVertice
 */
void Component::addVertice(Vertex* newVertice){
	includedVertices.insert(newVertice);
}





/***************************************************************************\
*							Public Functions								*
\***************************************************************************/


/**
 *	set new Maximum egde weight
 *	@Param
 */
void Component::setMaxWeight(double w){
	maxWeight = w;
}




/**
 *	Set constant K
 *	@Param
 */
void Component::setThresholdconstant(float constant){
	thresholdconstant=constant;
}



/**
 *	Adds vertice to set
 *	@Param
 */
void Component::add(Vertex* v0){
	addVertice(v0);
}



/**
 *	MInt(C) = Int(C) + T(C)
 *	@Param
 */
inline float Component::getInternalDifference(){
	calculateMInt();
	return minInternalDifference;
}


/**
 *	Set Union
 *	@Param
 */
void Component::operator+=(const Component &c2){
	join(c2);
}




/**
 *	True if All components are equal
 *	@Param
 */
bool Component::operator==(const Component &c2){
	return isEqual(c2);
}




/**
 *	If w(e) <= MINT(C) true, false otherwise
 *	@Param
 */
bool Component::operator<(const float weightOfNewEdge)const{
	return weightOfNewEdge>minInternalDifference?false:true;
}



Component& Component::operator=(const Component &c2){
	copy(c2);
	return *this;
}


/**
 *	Set Union
 *	@Param
 */
void Component::joinComponents(Component c2){
	join(c2);
}


/**
 *
 */
unsigned int Component::getComponentSize(){
	return includedVertices.size();
}


/**
 *
 *
 */
const std::set<Vertex*> & Component::getVertices(){
	return includedVertices;
}

//empty constructor
/**
 *
 *
 */
Component::Component(){
	setMaxWeight(0.0f);
}

//reference
/**
 *
 *
 */
Component::Component(Vertex& v0){
	setMaxWeight(0.0f);
	includedVertices.insert(&v0);
}

//pointer
/**
 *
 */
Component::Component(Vertex* v0){
	setMaxWeight(0.0f);
	includedVertices.insert(v0);
}

Component::Component(Component& cmp){
	copy(cmp);
}




/***************************************************************************\
*							Utility Functions								*
\***************************************************************************/


/**
 *	Produce a human readable information for the Component
 *	@Param screen
 *	@Param out
 */
std::ostream & operator<<(std::ostream & screen,const Component &out){
	screen<<"Included Vertices \n";
	std::set<Vertex*>::iterator it;
	for(it = out.includedVertices.begin();it!=out.includedVertices.end();++it){
		screen<<*(*it);
	}
	return screen;
}

#endif
