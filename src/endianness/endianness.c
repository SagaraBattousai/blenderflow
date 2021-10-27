#include <Endianness/endianness.h>

#include <stdlib.h>


endianness_t system_endianness(void)
{
  int data = 1;
  char byte_zero = ((char *)&data)[0];
  return (endianness_t)byte_zero;
}

void endianness_cast(endianness_t dst_endianness, endianness_t src_endianness,
  char *data, void *dst, size_t dst_size)
{
  if (dst_endianness == src_endianness)
  {
    memcpy(dst, data, dst_size);
  }
  else
  {
    char *dst_bytes = (char *)dst;
    for (size_t byte_it = 0; byte_it < dst_size; byte_it++)
    {
      dst_bytes[dst_size - byte_it - 1] = data[byte_it];
    }
  }
}

void endianness_to_system_cast(endianness_t src_endianness, char *data,
  void *dst, size_t dst_size)
{
  endianness_cast(system_endianness(), src_endianness, data, dst, dst_size);
}

void endianness_from_system_cast(endianness_t dst_endianness, char *data,
  void *dst, size_t dst_size)
{
  endianness_cast(dst_endianness, system_endianness(), data, dst, dst_size);
}
