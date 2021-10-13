#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Model/ply_reader.h>

int main(void)
{
  ply_model_t ply_model;
  get_model(&ply_model, "assets/Colon/Colon.ply");

  printf("%zi\n", sizeof(vertex));


  return 0;
}