#ifndef __ENDIANNESS_H__
#define __ENDIANNESS_H__

#include <stdlib.h>
#include <Primatives/vec.h>

typedef enum endianness { BIG = 0, LITTLE = 1 } endianness_t;

endianness_t system_endianness(void);

void endianness_cast(endianness_t dst_endianness, endianness_t src_endianness,
  char *data, void *dst, size_t dst_size);

void endianness_to_system_cast(endianness_t src_endianness, char *data,
  void *dst, size_t dst_size);

void endianness_from_system_cast(endianness_t dst_endianness, char *data,
  void *dst, size_t dst_size);

#endif