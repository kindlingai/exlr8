/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The default implementation of the SceneObject hierarchy.
 *****************************************************************/

#include "SceneObject.h"
#include "Intersection.h"
#include "Light.h"

SceneObject::SceneObject() : 
  _ior(DEFAULT_IOR), 
  _ambient(DEFAULT_AMBIENT), 
  _diffuse(DEFAULT_DIFFUSE), 
  _specular(DEFAULT_SPECULAR),
  _color(DEFAULT_COLOR),
  _filter(DEFAULT_FILTER),
  _transmit(DEFAULT_TRANSMIT),
  _reflect(DEFAULT_REFLECT),
  _caustic(DEFAULT_CAUSTIC),
  _matrix(IDENTITY),
  _inverse(IDENTITY)
{
}

void SceneObject::reflect(const SceneTreeNode& root,
                          const LightList& ll,
                          Intersection& intr,
                          vector<double>& interiorStack, 
                          vector3& color,
                          double reflect) const
{
  IntersectionList il;
  Ray r;
  r._start = intr._point;
  // Calculate reflection vector
  r._direction = (2 * intr._normal * -intr._originalRay._direction) 
    * intr._normal + intr._originalRay._direction;
  r._direction.normalize();
  if (root.getIntersection(r, il))
  {      
    // Determine closest intersection point
    IntersectionList::iterator i;
    IntersectionList::iterator closest = il.begin();
    for (i = il.begin(); i < il.end(); i++)
    {
      if (i->_distance < closest->_distance)
        closest = i;
    }
    closest->_originalRay = r;
    // Increase the age for maximum ray level
    closest->_age = intr._age + 1;
    // Calculate the color of the reflected vector using recursion
    vector3 reflected;
    closest->_object->getColor(root, ll, *closest, interiorStack, reflected);
    color += reflect * reflected;
  }    
}

bool SceneObject::refract(const SceneTreeNode& root,
                          const LightList& ll,
                          Intersection& intr,
                          vector<double>& interiorStack, 
                          vector3& color,
                          double filter, double transmit) const
{
  IntersectionList il;
  Ray r;
  bool tir = false;
  
  // N = ior.old / ior.new
  double N, ior;

  // Are we entering or leaving a surface?
  if (intr._inside) 
  {
    // Leaving
    ior = interiorStack.back();
    interiorStack.pop_back();
    N = _ior / interiorStack.back();
    //    cerr << "Leaving: " << _ior << " " << interiorStack.back() << " = " << N << endl;
  }
  else
  {
    // Entering
    N = interiorStack.back() / _ior;
    //    cerr << "Entering: "  << interiorStack.back() << " " << _ior << " = " << N << endl;
    interiorStack.push_back(_ior);
  }
  // cerr << intr._inside << ": " << N << " " << interiorStack.size() << endl;
  //  double N = intr._iorStart / _ior;
  //  double N = 1;
  vector3 n = intr._normal;
  vector3 i = -intr._originalRay._direction;

  /* 
     If we hit the surface from the inside, we find the dot 
     product of the normal and the incident vector is negative.
  */
  double dot = n * i;
  if (dot < 0.0)
  {
    n = -n;
    dot = -dot;
  }
  
  // Calculate discriminant to determine whether total internal reflection occurs
  double D = 1 - N * N * (1 - dot * dot);
  if (D < 0.0) 
  {
    reflect(root, ll, intr, interiorStack, color, 1);
    tir = true;
  }
  else
  {
    r._start = intr._point;
    r._direction =
      (N * dot - sqrt(D)) * n
      - N * i; 
    r._direction.normalize();
  }
  if (root.getIntersection(r, il))
  {      
    IntersectionList::iterator i;
    IntersectionList::iterator closest = il.begin();
    for (i = il.begin(); i < il.end(); i++)
    {
      if (i->_distance < closest->_distance)
        closest = i;
    }
    closest->_originalRay = r;
    closest->_age = intr._age + 1;
    vector3 refracted;
    closest->_object->getColor(root, ll, *closest, interiorStack, refracted);
    vector3 filtered;
    filtered[0] = refracted[0] * filter * _color[0];
    filtered[1] = refracted[1] * filter * _color[1];
    filtered[2] = refracted[2] * filter * _color[2];
    color += transmit * refracted + filtered;
  }    

  if (intr._inside) 
  {
    interiorStack.push_back(ior);
  }
  else
  {
    interiorStack.pop_back();
  }

  return tir;
}

void SceneObject::getColor(const SceneTreeNode& root, 
                           const LightList& ll,
                           Intersection& intr, 
                           vector<double>& interiorStack, 
                           vector3& color) const 
{
  if (intr._age > 6)
  {
    color = vector3(0, 0, 0);
    return;
  }

  intr._object->getNormal(intr);

  // Attenuate color if transparency condition exists
  double trans = 1 - (_filter + _transmit); 
  if (trans > 1.0)
    trans = 1.0;

  color = trans * _ambient * _color;

  LightList::const_iterator li;
  //  cerr << "calling " << intr._object->getName() << "::getNormal()" << endl;
  for (li = ll.begin(); li < ll.end(); li++)
  {
    Light* l = *li;
    vector3 shadowTint;
    if (l->isShadowed(root, intr, shadowTint))
    {
      vector3 component;
      l->setupLighting(intr._originalRay, intr);
      if (l->calcDiffuse(intr, component))
      {
        color[0] += trans * component[0] * shadowTint[0];
        color[1] += trans * component[1] * shadowTint[1];
        color[2] += trans * component[2] * shadowTint[2];
      }
      if (l->calcSpecular(intr, component))
      {
        color[0] += component[0] * shadowTint[0];
        color[1] += component[1] * shadowTint[1];
        color[2] += component[2] * shadowTint[2];
      }
    }
    else
    {
      vector3 component;
      l->setupLighting(intr._originalRay, intr);
      if (l->calcDiffuse(intr, component))
        color = color + trans * component;
      if (l->calcSpecular(intr, component))
        color = color + component;
    }
  }    
 	
  bool tir = false;
  if (_filter > 0.0 || _transmit > 0.0)
  {
    tir = refract(root, ll, intr, interiorStack, color, _filter, _transmit);
  }

  // Handle reflection, assuming no total internal reflection
  if (_reflect > 0.0 && !tir)
  {
    reflect(root, ll, intr, interiorStack, color, _reflect);
  }  
}









