#ifndef __STL_READER_H__
#define __STL_READER_H__

#include <stdio.h>
#include <Primatives/vec.h>
#include <Primatives/tri.h>
#include <Endianness/endianness.h>



typedef struct stl_model
{
  unsigned int triangle_count;
  vertex *normals;
  triangle3D_t *triangels;
}stl_model_t;

int get_stl_model(stl_model_t *model, const char *filename);

void free_stl_model(stl_model_t *model);

#endif