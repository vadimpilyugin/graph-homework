#include "cstdlib"
#include "cstdio"
#include "cstdint"

#include "io.h"

graph read_graph(const std::string fn) {
  FILE *f = fopen(fn.c_str(), "rb");
  if (f == NULL) {
    perror("open graph file failed");
    exit(2);
  }

  uint32_t n;
  uint64_t m;
  fread(&n, sizeof(n), 1, f);
  fread(&m, sizeof(m), 1, f);
  
  char tmp;
  fread(&tmp, sizeof(tmp), 1, f);
  fread(&tmp, sizeof(tmp), 1, f);

  uint64_t *rowsIndices = (uint64_t*)malloc((n+1)*sizeof(uint64_t));
  if (rowsIndices == NULL) {
    perror("failed to allocate rowsIndices");
    exit(3);
  }
  fread(rowsIndices, sizeof(*rowsIndices), n+1, f);

  uint32_t *endV = (uint32_t*)malloc(m*sizeof(uint32_t));
  if (endV == NULL) {
    perror("failed to allocate endV");
    exit(3);
  }
  fread(endV, sizeof(*endV), m, f);

  double *weights = (double*)malloc(m*sizeof(double));
  if (weights == NULL) {
    perror("failed to allocate weights");
    exit(4);
  }
  fread(weights, sizeof(*weights), m, f);

  long curr_pos = ftell(f);
  fseek(f, 0, SEEK_END);
  long end_pos = ftell(f);
  if (curr_pos != end_pos) {
    fprintf(stderr, "Current file position: %ld != %ld\n",
      curr_pos, end_pos);
    exit(5);
  }

  fclose(f);

  graph g;
  g.n = n;
  g.m = m;
  g.rowsIndices = rowsIndices;
  g.endV = endV;
  g.weights = weights;

  return g;
}

// void free_graph(graph *g);