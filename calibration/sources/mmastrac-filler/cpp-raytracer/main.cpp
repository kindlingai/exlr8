#include "RayTracer.h"
#include "RootNode.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Plane.h"
#include "Square.h"
#include "SceneObjectList.h"
#include "FileReader.h"

int main(int argc, char** argv)
{
  cerr << "oort (object-oriented raytracer) version 0.1" << endl;
  cerr << "(c) Copyright 1999 by Matthew Mastracci and Luigi Iuliano" << endl;
  cerr << "oort comes with ABSOLUTELY NO WARRANTY; for details type 'oort --license'" 
       << endl;
  cerr << endl;

  /*
    Should probably use GNU getarg for this.
   */
  if (argc == 1)
  {
    cerr << "Usage: oort filename" << endl;
    cerr << endl;
    cerr << "The script is read from the command-line and output to STDOUT in PNM format." << endl;
    cerr << endl;
    cerr << "Examples:" << endl;
    cerr << "  oort script | pnmtopng > output.png" << endl;
    cerr << "  oort script | xv -" << endl;
    cerr << endl;
    exit(0);
  }

  if (strcmp(argv[1], "--license") == 0)
  {
    cerr << "This program is distributed under the terms of the GNU General Public License." << endl;
    cerr << "See the file COPYING for license details." << endl;
    cerr << endl;
    exit(0);
  }

  FileReader fr;
  Sphere sp;
  Plane p;
  Square sq;
  Triangle tr;
  fr.RegisterPrimitive(&sp);
  fr.RegisterPrimitive(&p);
  fr.RegisterPrimitive(&sq);
  fr.RegisterPrimitive(&tr);

  if (strcmp(argv[1], "-") != 0)
  {
    ifstream ins(argv[1]);
    fr.ReadConfig(ins);
  }
  else
  {
    fr.ReadConfig(cin);
  }

  RayTracer rt;
  Sphere s1, s2, s3;
  Plane p1;
  TraceParameters tp = *fr._tp;
  /*  tp._width = 800;
  tp._height = 600;
  tp._fov = 60;
  tp._eyePoint = vector3(0, 2.5, -5);
  tp._lookAt = vector3(0.5, 1.5, 0);
  tp._oversampling = 1;
  tp._up = vector3(0, 1, 0);*/
  s1._matrix = 
    matrix4(matrix4::mScale, 0.5, .5, .5) *
    matrix4(matrix4::mTranslate, 0, 1.6, 0);
  s1._inverse = s1._matrix.inverse();
  s2._matrix = 
    matrix4(matrix4::mTranslate, 1.5, 2, 0);
    //      matrix4(matrix4::mScale, 0.5, .5, .5);
  s2._inverse = s2._matrix.inverse();
  s3._matrix = 
    matrix4(matrix4::mTranslate, 0.8, 1, .8);
  s3._inverse = s3._matrix.inverse();
  TraceStats ts;
  SceneObjectList sol;
  sol.addSceneObject(&s2);
  sol.addSceneObject(&s1);
//  sol.addSceneObject(&s3);
  sol.addSceneObject(&p1);
  rt.trace(*fr._so, *fr._ll, *fr._tp, ts, cout);
}

