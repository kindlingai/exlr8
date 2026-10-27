/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The Intersection class.
 *****************************************************************/

#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "global.h"
#include "Light.h"

class SceneObject;

class Intersection
{
public:
  Intersection();

  vector3 _point;
  vector3 _normal;
  double _distance;
  Ray _originalRay;
  int _age;
  const SceneObject* _object;
  bool _inside;
};

typedef vector<Intersection> IntersectionList;

#endif
