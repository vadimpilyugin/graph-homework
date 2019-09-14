#pragma once

#include <vector>
#include <unordered_map>

typedef struct
{
  uint32_t n;
  uint64_t m;
  uint64_t* rowsIndices;
  uint32_t* endV;

  double* weights;
} graph;

typedef struct {
  uint32_t n;
  uint64_t m;
  uint64_t* rowsIndices;
  uint32_t* locEndV;
  double* locWeights;

  int myrank;
  int nproc;
  int leftover;
  uint64_t batch_size;
  uint64_t my_batch_size;
  uint64_t start_i;
  uint64_t end_i;
  uint32_t start_v;
} loc_graph;

void output_graph_info(graph g);
void is_undirected(graph g);
void is_multigraph(graph g);
void out_indices_32(uint32_t *v, uint32_t size);

template<typename Key, typename Value>
std::vector<Key> keys(std::unordered_map<Key,Value> mp) {
  std::vector<Key> v;
  for (auto &pair: mp) {
    v.push_back(std::get<0>(pair));
  }
  return v;
}