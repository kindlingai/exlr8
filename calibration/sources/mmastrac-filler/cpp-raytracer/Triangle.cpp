/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The routines for the triangle.
 *****************************************************************/

#include "Triangle.h"

bool Triangle::getIntersection(const Ray& ray, IntersectionList& il) const
{
  /* Transform ray and startpoint into object space to make collision
     detection easier */
  vector3 xformStartPoint, xformDirection;
  xformStartPoint = ray._start * _inverse;
  // Is this necessary?  There must be a more accurate way...
  xformDirection = (ray._start + 400 * ray._direction) * _inverse 
    - xformStartPoint;
  xformDirection.normalize();

  if (fabs(xformDirection[1]) > EPSILON)
  {
    Intersection intr;
    intr._distance = -xformStartPoint[1] / xformDirection[1];
    if (intr._distance < EPSILON)
      return false;
    //    cerr << xformStartPoint << " " << xformDirection << endl;
    intr._point 
      = vector3(xformStartPoint[0] + xformDirection[0] * intr._distance,
                0,
                xformStartPoint[2] + xformDirection[2] * intr._distance);
    if (intr._point[0] > -0.5 && intr._point[2] > -0.5 
        && intr._point[0] + intr._point[2] < 0)
    {
      if (xformStartPoint[1] > 0.0)
        intr._normal = _npos;
      else
        intr._normal = _nneg;
      intr._point = intr._point * _matrix;
      intr._distance = vector3(intr._point - ray._start).length();
      intr._object = this;
      intr._inside = false;
      il.insert(il.end(), intr);
      return true;
    }
  }
  
  return false;
}

void Triangle::getNormal(Intersection& intr) const
{ 
}

void Triangle::init()
{
  vector3 p0 = vector3(0, 0, 0) * _matrix;
  vector3 p1 = vector3(1, 0, 0) * _matrix;
  vector3 p2 = vector3(0, 0, 1) * _matrix;
  vector3 v0 = p1 - p0;
  vector3 v1 = p2 - p0;
  _npos = -cross(v0, v1);
  _npos.normalize();
  _nneg = -_npos;
}















