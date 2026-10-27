/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The Intersection class.
 *****************************************************************/

#include "Intersection.h"
#include "SceneTreeNode.h"
#include "SceneObject.h"

Intersection::Intersection() : _point(ZERO_VECTOR), _normal(ZERO_VECTOR), 
  _distance(0), _originalRay(), _object(NULL), _age(0)
{ }

