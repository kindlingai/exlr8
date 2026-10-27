/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The abstract interface for a SceneTreeNode class.
 *****************************************************************/

#ifndef SCENETREENODE_H
#define SCENETREENODE_H

#include "global.h"

class Intersection;
class Light;
typedef vector<Intersection> IntersectionList;
typedef vector<Light*> LightList;

class SceneTreeNode {
public:
  virtual bool getIntersection(const Ray& ray, IntersectionList& il) const 
    { return false; } 
  virtual void getColor(const SceneTreeNode& root, 
                        const LightList& ll,
                        Intersection& intr, 
                        vector3& color) const 
    { }
  virtual void init()
    { }
  
  virtual void getNormal(Intersection& intr) const 
    { cerr << "SceneTreeNode::getNormal() called!" << endl; }
  virtual const char* getName() const
    { return "generic_scene_object"; }
  virtual bool isPrimitive() const
    { return false; }
  virtual bool isInfinite() const
    { return false; }
};

#endif
