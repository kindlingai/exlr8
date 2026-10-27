/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The routines for the scene object list.
 *****************************************************************/

#include "SceneObjectList.h"

bool SceneObjectList::getIntersection(const Ray& ray, 
                                      IntersectionList& il) const
{
  vector<const SceneObject*>::const_iterator i = _objects.begin();
  bool found = false;

  while (i < _objects.end())
  {
    if ((*i)->getIntersection(ray, il))
      found = true;
    i++;
  }

  return found;
}

void SceneObjectList::getNormal(Intersection& intr) const
{ 
  if (intr._object)
    intr._object->getNormal(intr);
}

void SceneObjectList::addSceneObject(const SceneObject* so)
{
  _objects.insert(_objects.begin(), so);
}














