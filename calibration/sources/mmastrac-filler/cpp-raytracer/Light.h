/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The class holding properties of the lights.
 *****************************************************************/

#ifndef LIGHT_H
#define LIGHT_H

#include "global.h"
#include "SceneObject.h"
#include "SceneTreeNode.h"
#include "Intersection.h"

class Light
{
public:
  Light() : _position(ZERO_VECTOR), _color(DEFAULT_COLOR),
    _lookAt(ZERO_VECTOR), _castsShadows(true) { }
  vector3 _position;
  vector3 _color;
  vector3 _lookAt;

  bool _castsShadows;

  bool isShadowed(const SceneTreeNode& so, 
                  const Intersection& intr,
                  vector3& tint);
  void setupLighting(const Ray& gaze, 
                     const Intersection& intr);
  bool calcDiffuse(const Intersection& intr, vector3& color);
  bool calcSpecular(const Intersection& intr, vector3& color);
private:
  vector3 _n, _l, _v;
  double _fAtt;
};

#endif
