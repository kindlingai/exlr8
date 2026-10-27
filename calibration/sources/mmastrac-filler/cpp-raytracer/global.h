/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * Global variables and includes.
 *****************************************************************/

#ifndef GLOBAL_H
#define GLOBAL_H

#define DEBUG 1
#define DEBUG_POINT(x, y, z) { extern int DEBUG_x, DEBUG_y; if (DEBUG && DEBUG_x == (x) && DEBUG_y == (y)) { z } }

#include <vector>
#include "linearalgebra.h"

typedef jVec3 vector3;
typedef jMat4 matrix4;

#define DEFAULT_IOR 1
#define DEFAULT_AMBIENT 0.2
#define DEFAULT_DIFFUSE 0.7
#define DEFAULT_SPECULAR 0
#define DEFAULT_COLOR vector3(1, 1, 1)
#define DEFAULT_FILTER 0
#define DEFAULT_TRANSMIT 0
#define DEFAULT_REFLECT 0
#define DEFAULT_CAUSTIC 0
#define IDENTITY jMat4()
#define MAX_COLOR 255
#define ZERO_VECTOR vector3(0, 0, 0)
#define DEG_TO_RAD(x) (((x)*pi) / 180)
#define RAD_TO_DEG(x) (((x)*180) / pi)

class Ray
{
public:
  Ray() : _start(ZERO_VECTOR), _direction(ZERO_VECTOR) { }

  vector3 _start;
  vector3 _direction;
};

#define EPSILON 0.0001
#define INVERSE_EPSILON 0.00000001

#endif


