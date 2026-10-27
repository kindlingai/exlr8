/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The default implementation of the SceneObject hierarchy.
 *****************************************************************/

#ifndef SCENEOBJECT_H
#define SCENEOBJECT_H

#include "Light.h"
#include "SceneTreeNode.h"
#include "global.h"

class SceneObject : public SceneTreeNode
{
public:
  SceneObject();
  virtual SceneObject* create() const
    { cerr << "SceneObject::create() called!" << endl; return NULL; }  
  virtual void getColor(const SceneTreeNode& root, 
                        const LightList& ll,
                        Intersection& intr, 
                        vector<double>& interiorStack, 
                        vector3& color) const;

  virtual bool refract(const SceneTreeNode& root, 
                       const LightList& ll,
                       Intersection& intr, 
                       vector<double>& interiorStack, 
                       vector3& color, double filter, double transmit) const;

  virtual void reflect(const SceneTreeNode& root, 
                       const LightList& ll,
                       Intersection& intr, 
                       vector<double>& interiorStack, 
                       vector3& color, double reflect) const;

  double _ior;
  double _ambient, _diffuse, _specular;
  /* 
     Color of object is essentially:
     
     (1-_transmit-_reflect)*lightingSum.color 
       + (_reflect)*reflectedRay.color
       + (1-_filter)*refractedRay.color
  */
  double _transmit, _reflect, _filter;
  double _caustic;
  vector3  _color;
  matrix4  _matrix, _inverse;
};

#endif


