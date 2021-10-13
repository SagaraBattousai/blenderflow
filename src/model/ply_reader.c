#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Model/ply_reader.h>

#define true 1
#define ELEMENT_LENGTH 7
#define FORMAT_LENGTH 6
#define END_HEADER_LEN 10

static void get_counts(ply_model_t *model, FILE *file)//size_t *vertices, size_t *faces, endianness_t *endian,
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
        model->endian = LITTLE;
      } 
      else if (!strncmp(tok, "binary_big_endian", 17))
      {
        model->endian = BIG;
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
        model->vertex_count = (size_t) strtol(count, NULL, 10);
      }
      else if (!strncmp(tok, "face", 4))
      {
        model->face_count = (size_t) strtol(count, NULL, 10);
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

static int get_vertex_data(ply_model_t *model, endianness_t system_endianness, FILE *file)
{
  model->verticies = (vecf3_t *)malloc(model->vertex_count * sizeof(vecf3_t));
  if (model->verticies == NULL)
  {
    return -1;
  }

  char vertex_data[12];
  size_t i = 0;
  while (i < model->vertex_count)
  {
    fread(vertex_data, 4, 3, file);
    vertex vert = endianness_cast_vertex(system_endianness, model->endian, vertex_data);
    *(model->verticies + i) = vert;
    i++;
  }

  printf("Vertex{%f, %f, %f}\n", (model->verticies + 1)->x, (model->verticies + 1)->y, (model->verticies + 1)->z);

  return 0;
}

void get_model(ply_model_t *ply_model, const char *filename)
{
  FILE *file = fopen(filename, "r");

  endianness_t system_endianness = base_endianness();

  get_counts(ply_model, file);
  
  get_vertex_data(ply_model, system_endianness, file);

  fclose(file);
}

void free_model(ply_model_t *model)
{
  free(model->verticies);
  //TODO: Handle faces as double pointer so may cause issues but ignore for now
}
