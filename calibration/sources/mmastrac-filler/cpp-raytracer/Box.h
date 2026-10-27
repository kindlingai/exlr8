/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The routines for the box.
 *****************************************************************/

#ifndef BOX_H
#define BOX_H

#include "global.h"
#include "SceneObject.h"
#include "Intersection.h"

class Box : public SceneObject {
public:
  virtual bool getIntersection(const Ray& ray, IntersectionList& il) const;
  virtual void getNormal(Intersection& intr) const;
  virtual const char* getName() const
    { return "box"; }
  virtual bool isPrimitive() const
    { return true; }
  virtual SceneObject* create() const
    { return new Box(); }
  virtual void init();
  
  vector3 _n[6];
};

#endif


