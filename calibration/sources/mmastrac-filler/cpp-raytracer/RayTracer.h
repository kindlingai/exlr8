/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The heart of the ray tracer.
 *****************************************************************/

#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "global.h"
#include "ostream.h"
#include "SceneTreeNode.h"
#include "Light.h"

typedef vector<Light*> LightList;

class TraceParameters
{
public:
  TraceParameters() : _width(0), _height(0), 
    _lookAt(ZERO_VECTOR), _eyePoint(ZERO_VECTOR), _up(ZERO_VECTOR), 
    _right(ZERO_VECTOR), _fov(0) { }

  int _width, _height;
  int _oversampling;
  double _fov;
  vector3 _lookAt, _eyePoint, _up, _right;
};

class TraceStats
{
public:
  TraceStats() : _raysCasted(0), _raysIntersected(0), 
    _startTime(0), _endTime(0) { }

  int _raysCasted;
  int _raysIntersected;
  time_t _startTime;
  time_t _endTime;
};

class RayTracer
{
public:
  trace(const SceneTreeNode& root, 
        const LightList& ll,
        const TraceParameters& tp, 
        TraceStats& ts,
        ostream& outs);
};

#endif






