#ifndef __TRI_H__
#define __TRI_H__

#include <Primatives/vec.h>

typedef struct triangle3D
{
  vecf3_t i;
  vecf3_t j;
  vecf3_t k;
}triangle3D_t;

typedef struct triangle2D
{
  vecf2_t i;
  vecf2_t j;
  vecf2_t k;
}triangle2D_t;

#endif
