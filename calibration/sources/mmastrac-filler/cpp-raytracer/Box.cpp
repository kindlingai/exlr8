/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The routines for the box.
 *****************************************************************/

#include "Box.h"

bool Box::getIntersection(const Ray& ray, IntersectionList& il) const
{
}

void Box::getNormal(Intersection& intr) const
{ 
}

void Box::init() 
{
  #define NORMAL(x, y, z) (cross(y - x, z - x))
  #define x vector3(1, 0, 0)
  #define y vector3(1, 0, 0)
  #define z vector3(1, 0, 0)
  /*
  _n[0] = NORMAL(
  _n[1] =
  _n[2] = 
  _n[3] =   
  _n[4] = 
  _n[5] = */
}














