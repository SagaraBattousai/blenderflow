#ifndef __PLY_READER_H__
#define __PLY_READER_H__


#include <stdio.h>
#include <Primatives/vec.h>
#include <Endianness/endianness.h>

typedef struct ply_model
{
  size_t vertex_count;
  size_t face_count;
  endianness_t endian;
  vecf3_t *verticies;
  int **face_indicies;
}ply_model_t;

int get_ply_model(ply_model_t *model, const char *filename);

void free_ply_model(ply_model_t *model);

#endif