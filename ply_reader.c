#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define true 1
#define ELEMENT_LENGTH 7
#define FORMAT_LENGTH 6
#define END_HEADER_LEN 10

typedef enum endianness {BIG = 0, LITTLE = 1} endianness_t;

endianness_t base_endianness(void)
{
  int data = 1;
  char byte_zero = ((char *) &data)[0];
  return (endianness_t) byte_zero;
}

float endianness_castf(endianness_t dst_endianness,
                       endianness_t src_endianness, char* data)
{
  float ret;
  if (dst_endianness == src_endianness)
  {
    memcpy(&ret, data, sizeof(float));
  }
  else
  {
    char *ret_bytes = (char *) &ret;

    ret_bytes[3] = data[0];
    ret_bytes[2] = data[1];
    ret_bytes[1] = data[2];
    ret_bytes[0] = data[3];
  }
  return ret;
}

void get_counts(size_t *vertices, size_t *faces, endianness_t *endian,
                FILE *file)
{
  char *header = (char *) malloc(128 * sizeof(char));
  char *tok;
  while(true)
  {
    fgets(header, 128, file);

    if (!strncmp(header, "format", FORMAT_LENGTH))
    {
      tok = strtok(header, " "); //format
      tok = strtok(NULL, " "); //endianness
      if (!strncmp(tok, "binary_little_endian", 20))
      {
        *endian = LITTLE;
      } 
      else if (!strncmp(tok, "binary_big_endian", 17))
      {
        *endian = BIG;
      }
      //If it were ascii we should return an error but leave as TODO
    }
    else if (!strncmp(header, "element", ELEMENT_LENGTH))
    {
      tok = strtok(header, " ");
      tok = strtok(NULL, " ");
      char *count = strtok(NULL, " ");
      if (!strncmp(tok, "vertex", 6))
      {
        *vertices = (int) strtol(count, NULL, 10);
      }
      else if (!strncmp(tok, "face", 4))
      {
        *faces = (int) strtol(count, NULL, 10);
      }

    }
    else if (!strncmp(header, "end_header", END_HEADER_LEN))
    {
      break;
    }
  }
  free(header);
  return; 
}

int main(void)
{
  FILE *file = fopen("colon_slices_1.ply", "r");

  endianness_t base_endian = base_endianness();
  size_t vertices;
  size_t faces;
  endianness_t endian;
  get_counts(&vertices, &faces, &endian, file);

  char data[4];
  fread(data, 1, 4, file);

  float val = endianness_castf(base_endian, endian, data);

  printf("%f\n", val);

  return 0;
}
