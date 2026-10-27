/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The routines for the scene object list.
 *****************************************************************/

#ifndef SCENEOBJECTLIST_H
#define SCENEOBJECTLIST_H

#include "global.h"
#include "SceneObject.h"
#include "Intersection.h"

class SceneObjectList : public SceneObject {
public:
  virtual bool getIntersection(const Ray& ray, IntersectionList& il) const;

  virtual void getNormal(Intersection& intr) const;
  virtual const char* getName() const
    { return "scene_object_list"; }
  virtual void addSceneObject(const SceneObject* so);

private:
  vector<const SceneObject*> _objects;
};

#endif
