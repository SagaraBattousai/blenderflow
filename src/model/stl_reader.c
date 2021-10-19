#include <Model/stl_reader.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEADER_LENGTH 80
static const endianness_t STL_ENDIANNESS = LITTLE;

/*
UINT8[80]    – Header                 -     80 bytes
UINT32       – Number of triangles    -      4 bytes

foreach triangle                      - 50 bytes:
    REAL32[3] – Normal vector             - 12 bytes
    REAL32[3] – Vertex 1                  - 12 bytes
    REAL32[3] – Vertex 2                  - 12 bytes
    REAL32[3] – Vertex 3                  - 12 bytes
    UINT16    – Attribute byte count      -  2 bytes
end
*/

int get_stl_model(stl_model_t *model, const char *filename)
{
  FILE *file = fopen(filename, "rb");
  if (file == NULL)
  {
    return -1;
  }
  //Strip header
  fseek(file, HEADER_LENGTH, SEEK_SET);

  unsigned char tri_count_data[4];
  fread(tri_count_data, 4, 1, file);
  endianness_to_system_cast(STL_ENDIANNESS, tri_count_data, &model->triangle_count, sizeof(unsigned int));

  model->normals = (vertex *)malloc(model->triangle_count * sizeof(vertex));
  if (model->normals == NULL)
  {
    return -1;
  }
  
  model->triangels = (triangle3D_t *)malloc(
    model->triangle_count * sizeof(triangle3D_t)
  );
  if (model->triangels == NULL)
  {
    return -2;
  }

  unsigned char normal_data[12];
  unsigned char triangle_data[36];
  unsigned int i = 0;
  while (i < model->triangle_count)
  {
    fread(normal_data, 4, 3, file);

    endianness_to_system_cast(STL_ENDIANNESS, normal_data,
      (model->normals + i), sizeof(vertex));

    fread(triangle_data, 12, 3, file);

    endianness_to_system_cast(STL_ENDIANNESS, triangle_data,
      (model->triangels + i), sizeof(triangle3D_t));

    fseek(file, 2, SEEK_CUR);
    i++;
  }

  return 0;
}

void free_stl_model(stl_model_t *model)
{
  free(model->normals);
  free(model->triangels);
}
