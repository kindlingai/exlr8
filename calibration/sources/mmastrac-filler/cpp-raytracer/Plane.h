/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The routines for the plane.
 *****************************************************************/

#ifndef PLANE_H
#define PLANE_H

#include "global.h"
#include "SceneObject.h"
#include "Intersection.h"

class Plane : public SceneObject {
public:
  virtual bool getIntersection(const Ray& ray, IntersectionList& il) const;
  virtual void getNormal(Intersection& intr) const;
  virtual const char* getName() const
    { return "plane"; }
  virtual bool isPrimitive() const
    { return true; }
  virtual bool isInfinite() const 
    { return true; }
  virtual SceneObject* create() const
    { return new Plane(); }
  virtual void init();

  vector3 _npos, _nneg;
};

#endif


