/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The routines for the square.
 *****************************************************************/

#ifndef SQUARE_H
#define SQUARE_H

#include "global.h"
#include "SceneObject.h"
#include "Intersection.h"

class Square : public SceneObject {
public:
  virtual bool getIntersection(const Ray& ray, IntersectionList& il) const;
  virtual void getNormal(Intersection& intr) const;
  virtual const char* getName() const
    { return "square"; }
  virtual bool isPrimitive() const
    { return true; }
  virtual SceneObject* create() const
    { return new Square(); }
  virtual void init();

  vector3 _npos, _nneg;
};

#endif


