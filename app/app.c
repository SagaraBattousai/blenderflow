#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Model/stl_reader.h>

int main(void)
{
  stl_model_t stl_model;
  get_stl_model(&stl_model, "assets/Colon/Colon.stl");

  vertex v = *(stl_model.normals);
  printf("%f, %f, %f\n", v.x, v.y, v.z);

  triangle3D_t t = *(stl_model.triangels + 0);
  printf("%f, %f, %f\n", t.i.x, t.i.y, t.i.z);
  printf("%f, %f, %f\n", t.j.x, t.j.y, t.j.z);
  printf("%f, %f, %f\n", t.k.x, t.k.y, t.k.z);


  return 0;
}