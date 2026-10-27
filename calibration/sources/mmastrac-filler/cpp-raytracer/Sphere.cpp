/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The routines for the sphere.
 *****************************************************************/

#include "Sphere.h"

/*
                
           P  OOOOO
      v  ___.OOOOOOOO   
     ___/  OOOOOOOOOOO   
 E ./___________*OOOOOO 
       c   OOOOOOOOOOO
            OOOOOOOOO
              OOOOO

 */

bool Sphere::getIntersection(const Ray& ray, IntersectionList& il) const
{
  /* Transform ray and startpoint into object space to make collision
     detection easier */
  vector3 xformStartPoint, xformDirection;
  xformStartPoint = ray._start * _inverse;
  // Is this necessary?  There must be a more accurate way...
  xformDirection = (ray._start + 400 * ray._direction) * _inverse 
    - xformStartPoint;
  xformDirection.normalize();

  /*
    By solving the set of equations for intersection of a ray with a sphere,
    we are left with a quadratic in t (the distance).  Factoring out all the
    fun stuff leaves us with an optimized discriminant calculation telling us
    whether we have hit the sphere or not.
   */
  double b = xformStartPoint * xformDirection;
  double c = xformStartPoint * xformStartPoint - 1;
  double D = b * b - c;

  // Check if the ray completely misses the sphere (optimized case)
  if (D < 0.0)
    return false;

  double root = sqrt(D);
  double t;

  // Closest root to gaze origin 
  t = -b - root;
  if (t >= 0.0)
  {
    Intersection intr;
    intr._distance = t;
    vector3 point = xformStartPoint + xformDirection * intr._distance;   
    intr._point = point * _matrix; 
    intr._distance = vector3(intr._point - ray._start).length();
    intr._inside = false;
    if (intr._distance > EPSILON)
    {
      intr._object = this;
      il.insert(il.end(), intr);
    }
  }

  t = -b + root;
  if (t >= 0.0)
  {
    Intersection intr;
    intr._distance = t;
    vector3 point = xformStartPoint + xformDirection * intr._distance;
    intr._point = point * _matrix; 
    intr._distance = vector3(intr._point - ray._start).length();
    intr._inside = false;
    if (intr._distance > EPSILON)
    {
      intr._object = this;
      il.insert(il.end(), intr);
    }
  }

  // Ray intersects sphere, but it is closer than minimum distance
  return (il.size() > 0);
}

void Sphere::getNormal(Intersection& intr) const
{ 
  // Take sphere normal in sphere space and transform to world space
  vector3 sphereNormal = intr._point * _inverse;
  sphereNormal[0] /= _a;
  sphereNormal[1] /= _b;
  sphereNormal[2] /= _c;
  
  intr._normal = _x * sphereNormal[0] + _y * sphereNormal[1] + _z * sphereNormal[2];
  intr._normal.normalize();
}

void Sphere::init()
{
  vector3 center = vector3(0, 0, 0) * _matrix;
  _x = vector3(1, 0, 0) * _matrix - center;
  _a = _x.length();
  _x.normalize();
  _y = vector3(0, 1, 0) * _matrix - center;
  _b = _y.length();
  _y.normalize();
  _z = vector3(0, 0, 1) * _matrix - center;
  _c = _z.length();
  _z.normalize();
}











