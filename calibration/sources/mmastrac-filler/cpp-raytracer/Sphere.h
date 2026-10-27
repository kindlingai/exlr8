/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The routines for the sphere.
 *****************************************************************/

#ifndef SPHERE_H
#define SPHERE_H

#include "global.h"
#include "SceneObject.h"
#include "Intersection.h"

class Sphere : public SceneObject {
public:
  virtual bool getIntersection(const Ray& ray, IntersectionList& il) const;
  virtual void getNormal(Intersection& intr) const;
  virtual const char* getName() const
    { return "sphere"; }
  virtual bool isPrimitive() const
    { return true; }
  virtual SceneObject* create() const
    { return new Sphere(); }
  virtual void init();

private:
  double _a, _b, _c;
  vector3 _x, _y, _z;
};

#endif
