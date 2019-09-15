#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>

#include "graph.h"

extern bool verbose;

void out_weights(std::vector<double> v) {
  printf("[");
  for (uint64_t i = 0; i < v.size(); i++) {
    printf("%.3f, ", v[i]);
  }
  printf("]\n");
}

void out_indices(uint64_t *v, uint32_t size) {
  printf("[");
  for (uint32_t i = 0; i < size; i++) {
    printf("%lu, ", v[i]);
  }
  printf("]\n");
}

void out_indices_32(uint32_t *v, uint32_t size) {
  printf("[");
  for (uint32_t i = 0; i < size; i++) {
    printf("%u, ", v[i]);
  }
  printf("]\n");
}

void output_graph_info(graph g) {
  printf("\n\n========== graph ==========\n\n");
  printf("n = %d\n", g.n);
  printf("m = %ld\n", g.m);
  printf("rowsIndices [%d elems]\n", g.n+1);
  for (int i = 0; i < 10; i++) {
    printf("rowsIndices[%d] = %ld\n", i, g.rowsIndices[i]);
  }
  printf("\n");
  printf("endV [%ld elems]\n", g.m);
  for (int i = 0; i < 10; i++) {
    printf("endV[%d] = %d\n", i, g.endV[i]);
  }
  printf("\n");
  printf("weights [%ld elems]\n", g.m);
  for (int i = 0; i < 10; i++) {
    printf("weights[%d] = %f\n", i, g.weights[i]);
  }
  printf("\n");
  printf("===========================\n\n");
}
