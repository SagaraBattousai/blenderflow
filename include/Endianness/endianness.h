#ifndef __ENDIANNESS_H__
#define __ENDIANNESS_H__

#include <Primatives/vec.h>

typedef enum endianness { BIG = 0, LITTLE = 1 } endianness_t;

endianness_t base_endianness(void);

float endianness_castf(endianness_t dst_endianness,
  endianness_t src_endianness, char *data);

vertex endianness_cast_vertex(endianness_t dst_endianness,
  endianness_t src_endianness, char *data);


#endif