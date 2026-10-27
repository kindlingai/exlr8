/*****************************************************************
 * Matthew Mastracci and Luigi Iuliano
 *
 * The routines for reading the script files.
 *****************************************************************/

#include <string>
#include "FileReader.h"
#include "SceneObjectList.h"
#include "SceneObject.h"

FileReader::FileReader()
{
  _so = new SceneObjectList;
  _tp = new TraceParameters;
  _ll = new LightList;
}

FileReader::~FileReader()
{
  delete _tp;
  delete _so;
  {
    LightList::iterator i;
    for (i = _ll->begin(); i < _ll->end(); i++)
    {
      delete *i;
    }
  }
  delete _ll;
  {
    CreatedVector::iterator i;
    for (i = _created.begin(); i < _created.end(); i++)
    {
      delete *i;
    }
  }
}

void FileReader::RegisterPrimitive(const SceneObject* so)
{
  _proto.insert(_proto.end(), pair< string, const SceneObject* >(so->getName(), so));
}

void FileReader::ReadLight(istream& ins)
{
  string token;
  int i;
  double d;
  vector3 vector;

  Light* l = new Light;
  _ll->insert(_ll->end(), l);

  while (ins)
  {
    ins >> token;
    if (token == "position")
    {
      ins >> vector;
      l->_position = vector;
    }
    if (token == "color")
    {
      ins >> vector;
      l->_color = vector;
    }
    if (token == "lookat")
    {
      ins >> vector;
      l->_lookAt = vector;
    }
    if (token == "end" || !ins)
      break;
  }
}

void FileReader::ReadCamera(istream& ins)
{
  string token;
  int i;
  double d;
  vector3 vector;

  while (ins)
  {
    ins >> token;
    if (token == "end" || !ins)
      break;
    if (token == "eyepoint")
    {
      ins >> vector;
      _tp->_eyePoint = vector;
    }
    if (token == "lookat")
    {
      ins >> vector;
      _tp->_lookAt = vector;
    }
    if (token == "up")
    {
      ins >> vector;
      _tp->_up = vector;
    }
    if (token == "right")
    {
      ins >> vector;
      _tp->_right = vector;
    }
    if (token == "fov")
    {
      ins >> d;
      _tp->_fov = d;
    }
    if (token == "oversampling")
    {
      ins >> i;
      _tp->_oversampling = i;
    }
    if (token == "res")
    {
      ins >> i;
      _tp->_width = i;
      ins >> i;
      _tp->_height = i;
    }
  }
}

void FileReader::ReadPrimitive(istream& ins, const SceneObject* so)
{
  string token;
  int i;
  double d;
  vector3 vector;
  
  SceneObject* obj = so->create();
  ((SceneObjectList*)_so)->addSceneObject(obj);
  _created.insert(_created.end(), obj);

  while (ins)
  {
    ins >> token;
    if (token == "end" || !ins)
      break;
    if (token == "color")
    {
      ins >> vector;
      obj->_color = vector;
    }
    if (token == "diffuse")
    {
      ins >> d;
      obj->_diffuse = d;
    }
    if (token == "specular")
    {
      ins >> d;
      obj->_specular = d;
    }
    if (token == "ambient")
    {
      ins >> d;
      obj->_ambient = d;
    }
    if (token == "transmit")
    {
      ins >> d;
      obj->_transmit = d;
    }
    if (token == "filter")
    {
      ins >> d;
      obj->_filter = d;
    }
    if (token == "reflect")
    {
      ins >> d;
      obj->_reflect = d;
    }
    if (token == "caustic")
    {
      ins >> d;
      obj->_caustic = d;
    }
    if (token == "ior")
    {
      ins >> d;
      obj->_ior = d;
    }
    if (token == "translate")
    {
      ins >> vector;
      obj->_matrix = matrix4(matrix4::mTranslate, vector[0], vector[1], vector[2]) * obj->_matrix;
      obj->_inverse = obj->_matrix.inverse();
    }    
    if (token == "scale")
    {
      ins >> vector;
      obj->_matrix = matrix4(matrix4::mScale, vector[0], vector[1], vector[2]) * obj->_matrix;
      obj->_inverse = obj->_matrix.inverse();
    }    
    if (token == "rotatex")
    {
      ins >> d;
      obj->_matrix = matrix4(matrix4::mRotateX, DEG_TO_RAD(d)) * obj->_matrix;
      obj->_inverse = obj->_inverse * matrix4(matrix4::mRotateX, DEG_TO_RAD(-d));
    }
    if (token == "rotatey")
    {
      ins >> d;
      obj->_matrix = matrix4(matrix4::mRotateY, DEG_TO_RAD(d)) * obj->_matrix;
      obj->_inverse = obj->_inverse * matrix4(matrix4::mRotateY, DEG_TO_RAD(-d));
    }
    if (token == "rotatez")
    {
      ins >> d;
      obj->_matrix = matrix4(matrix4::mRotateZ, DEG_TO_RAD(d)) * obj->_matrix;
      obj->_inverse = obj->_inverse * matrix4(matrix4::mRotateZ, DEG_TO_RAD(-d));
    }
    if (token == "matrix")
    {
      matrix4 m;
      ins 
        >> m.m[0][0] >> m.m[0][1] >> m.m[0][2] >> m.m[0][3]
        >> m.m[1][0] >> m.m[1][1] >> m.m[1][2] >> m.m[1][3] 
        >> m.m[2][0] >> m.m[2][1] >> m.m[2][2] >> m.m[2][3] 
        >> m.m[3][0] >> m.m[3][1] >> m.m[3][2] >> m.m[3][3];
      obj->_matrix = m;
      obj->_inverse = obj->_matrix.inverse();
    }    
  }

  obj->init();
}

void FileReader::ReadConfig(istream& ins)
{
  string token;
  char s[500];
  vector3 vector;
  int i;
  double d;
  while (ins)
  {
    ins >> token;
    if (token == "#")
    {
      ins.getline(s, 500);
      continue;
    }
    if (!ins) 
      break;

    if (token == "camera")
    {
      ReadCamera(ins);
      continue;
    }
    if (token == "light")
    {
      ReadLight(ins);
      continue;
    }
    ProtoVector::iterator i;
    for (i = _proto.begin(); 
         i != _proto.end(); 
         i++)
    {
      if (token == i->first)
      {
        ReadPrimitive(ins, i->second);
        break;
      }
    }
  }
}
