/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The routines for the triangle.
 *****************************************************************/

#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "global.h"
#include "SceneObject.h"
#include "Intersection.h"

class Triangle : public SceneObject {
public:
  virtual bool getIntersection(const Ray& ray, IntersectionList& il) const;
  virtual void getNormal(Intersection& intr) const;
  virtual const char* getName() const
    { return "triangle"; }
  virtual bool isPrimitive() const
    { return true; }
  virtual SceneObject* create() const
    { return new Triangle(); }
  virtual void init();

  vector3 _npos, _nneg;
};

#endif


