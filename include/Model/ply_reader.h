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

void get_model(ply_model_t *model, const char *filename);

void free_model(ply_model_t *model);

//void get_counts(ply_model_t *model, FILE *file);//size_t *vertices, size_t *faces, endianness_t *endian,

#endif