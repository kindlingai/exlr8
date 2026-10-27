/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The class holding properties of the lights.
 *****************************************************************/

#include "Light.h"

bool Light::isShadowed(const SceneTreeNode& root, const Intersection& intr,
                       vector3& tint)
{
  if (_castsShadows)
  {
    tint = vector3(1, 1, 1);
    Ray r;
    r._start = intr._point;
    r._direction = _position - intr._point;
    r._direction.normalize();
  
    IntersectionList il;
    bool shadow = false;
    if (root.getIntersection(r, il))
    {
      IntersectionList::iterator i;
      IntersectionList::iterator closest = il.begin();      
      double closestDistance = closest->_distance;
      for (i = il.begin(); i < il.end(); i++)
        if ((i->_point - intr._point).length() > EPSILON)
        {
          if (i->_distance < closestDistance)
          {
            closestDistance = i->_distance;
            closest = i;
          }
          double filter = i->_object->_filter;
          double transmit = i->_object->_transmit;
          if (i->_object->_caustic > 0.0)
          {
            i->_object->getNormal(*i);
            double dot = fabs(closest->_normal * r._direction);
            double effect = 1.0 + pow(dot, i->_object->_caustic * 45);
            filter *= effect;
            transmit *= effect;
          }
          if (filter > 0.0) 
          {
            tint[0] *= i->_object->_color[0] * filter;
            tint[1] *= i->_object->_color[1] * filter;
            tint[2] *= i->_object->_color[2] * filter;
          }
          if (transmit > 0.0)
          {
            tint *= transmit;
          }
          shadow = true;
          if (filter == 0.0 && transmit == 0.0)
          {
            tint = vector3(0, 0, 0);
            break;
          }
        }
    }
    return shadow;
  }

  // Light doesn't cast shadows, so no shadows here.
  return false;
}

void Light::setupLighting(const Ray& gaze, const Intersection& intr)
{
  _n = intr._normal;
  _v = -gaze._direction;
  _l = vector3(_position - intr._point);
  _l.normalize();
  
  _fAtt = 1 / (.25 + 0.5 * intr._distance 
               + 0.5 * intr._distance * intr._distance);
  if (_fAtt < 1.0)
    _fAtt = 1.0;
}          

bool Light::calcDiffuse(const Intersection& intr, vector3& color)
{
  // If there is a diffuse component, add it
  if (_n * _l > 0.0)
  {
    double x = _fAtt * intr._object->_diffuse * _n * _l;
    color[0] = x * intr._object->_color[0] * _color[0];
    color[1] = x * intr._object->_color[1] * _color[1];
    color[2] = x * intr._object->_color[2] * _color[2];
    return true;
  }

  return false;
}
  
bool Light::calcSpecular(const Intersection& intr, vector3& color)
{
  // If we have a specular component, check if we should add it
  if (intr._object->_specular > 0.0)
  {
    // TODO: add a phong/specular selection
    /*
      Calculate the reflection vector to determine the specular
      component of the lighting.  
    */
    // vector3 r = (2 * n * v) * n - v;
    
    /*
      Calculate the halfway vector to do true specular.  It looks
      really, really cool.
    */
    vector3 h = _l + _v;
    h.normalize();
    
    /*
      Calculate the specular value, then test to see if we're examining the 
      interior surface of the object.
    */

    double spec = _n * h;
    if (_v * _n < 0.0)
      spec = -spec;

    // Ensure that the specular is actually > 0 before using
    if (spec > 0.0)
    {               
      // TODO: replace this with light->_color
      // TODO: replace 20 with specular->_power 
      color = _fAtt * (intr._object->_specular * pow(spec, 20) * _color);
      return true;
    }
  }

  return false;
}


