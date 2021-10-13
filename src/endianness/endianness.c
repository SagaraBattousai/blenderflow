#include <Endianness/endianness.h>

#include <stdlib.h>


endianness_t base_endianness(void)
{
  int data = 1;
  char byte_zero = ((char *)&data)[0];
  return (endianness_t)byte_zero;
}

float endianness_castf(endianness_t dst_endianness,
  endianness_t src_endianness, char *data)
{
  float ret = -1;
  if (dst_endianness == src_endianness)
  {
    memcpy(&ret, data, sizeof(float));
  }
  else
  {
    char *ret_bytes = (char *)&ret;

    ret_bytes[3] = data[0];
    ret_bytes[2] = data[1];
    ret_bytes[1] = data[2];
    ret_bytes[0] = data[3];
  }
  return ret;
}

vertex endianness_cast_vertex(endianness_t dst_endianness,
  endianness_t src_endianness, char *data)
{
  vertex ret = { 0, 0, 0 };

  if (dst_endianness == src_endianness)
  {
    memcpy(&ret, data, sizeof(vertex));
  }
  else
  {
    char *ret_bytes = (char *)&ret;
    for (size_t byte_it = 0; byte_it < sizeof(vertex); byte_it++)
    {
      ret_bytes[sizeof(vertex) - byte_it - 1] = data[byte_it];
    }
  }
  return ret;

}
