/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The routines for reading the script files.
 *****************************************************************/

#include "RayTracer.h"
#include "SceneObject.h"
#include <string>
#include <vector>

typedef vector< pair< string, const SceneObject * > > ProtoVector;
typedef vector< SceneObject * > CreatedVector;

class FileReader
{
public:
  FileReader();
  ~FileReader();

  void ReadConfig(istream& ins);
  void ReadCamera(istream& ins);
  void ReadLight(istream& ins);
  void ReadPrimitive(istream& ins, const SceneObject* so);
  void RegisterPrimitive(const SceneObject* so);
   
  SceneObject* _so;
  TraceParameters* _tp;
  LightList* _ll;
  ProtoVector _proto;
  CreatedVector _created;
};





