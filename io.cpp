#include "cstdlib"
#include "cstdio"
#include "cstdint"
#include "cassert"

#include <unordered_map>
#include <vector>
#include <algorithm>

#include "io.h"

extern bool verbose;

std::vector<uint32_t> keys(std::unordered_map<uint32_t,std::vector<uint64_t>> mp) {
  std::vector<uint32_t> v;
  std::unordered_map<uint32_t,std::vector<uint64_t>>::iterator it;
  for (it = mp.begin(); it != mp.end(); it++) {
    v.push_back(it -> first);
  }
  return v;
}

std::unordered_map <uint32_t, double> trees_weight;

bool weight_comparator(int i, int j) {
  return trees_weight[i] > trees_weight[j];
}

void write_tree(graph g, char *selected_edges, DSU &d, const std::string fn) {
  std::unordered_map <uint32_t, std::vector<uint64_t>> component_edges;
  uint64_t size = 0;
  for (uint32_t vertex = 0; vertex < g.n; vertex++) {
    for (uint64_t edge_id = g.rowsIndices[vertex]; edge_id < g.rowsIndices[vertex+1]; edge_id++) {

      if (selected_edges[edge_id]) {
        uint32_t comp_id = FindDSU(d,vertex);
        if (verbose)
          printf("Edge %u ~~> %u [%u] selected\n",
            vertex, g.endV[edge_id], comp_id);
        component_edges[comp_id].push_back(edge_id);
        trees_weight[comp_id] += g.weights[edge_id];
        size++;
      }
    }
  }
  FILE *f = fopen(fn.c_str(), "wb");
  fwrite(&d.NComponents, sizeof(uint32_t), 1, f);
  printf("Number of trees [%u] = %lu bytes\n",
    d.NComponents, sizeof(uint32_t));
  fwrite(&size, sizeof(uint64_t), 1, f);
  printf("Number of edges [%lu] = %lu bytes\n",
    size, sizeof(size));

  // insert empty trees corresponding to single-node trees
  if (component_edges.size() < d.NComponents) {
    uint32_t add_comps = d.NComponents - component_edges.size();
    for (uint32_t i = 0; i < add_comps; i++) {
      component_edges[g.n+i] = std::vector<uint64_t>();
      trees_weight[g.n+i] = 0;
    }
  }


  uint64_t start = 0;
  uint64_t end;
  std::vector<uint32_t> ks = keys(component_edges);
  std::sort(ks.begin(), ks.end(), weight_comparator);
  for (uint32_t i = 0; i < ks.size(); i++) {
    std::vector<uint64_t> edges = component_edges[ks[i]];
    end = start + edges.size();
    fwrite(&start, sizeof(uint64_t), 1, f);
    fwrite(&end, sizeof(uint64_t), 1, f);
    if (verbose) {
      printf("start [%lu] = %lu bytes\n", start, sizeof(start));
      printf("end [%lu] = %lu bytes\n", end, sizeof(end));
    }
    start = end;
  }
  for (uint32_t i = 0; i < ks.size(); i++) {
    std::vector<uint64_t> edges = component_edges[ks[i]];
    if (edges.empty()) {
      continue;
    }
    fwrite(edges.data(), sizeof(uint64_t), edges.size(), f);
    if (verbose) {
      printf("Comp %u: [", ks[i]);
      for (uint64_t eid = 0; eid < edges.size(); eid++) {
        printf("%lu, ", edges[eid]);
      }
      printf("]\n");
    }

  }
  fclose(f);
}

void free_graph(graph g) {
  free(g.rowsIndices);
  free(g.endV);
  free(g.weights);
}

graph read_graph(const std::string fn) {
  FILE *f = fopen(fn.c_str(), "rb");
  if (f == NULL) {
    perror("open graph file failed");
    exit(2);
  }

  uint32_t n;
  uint64_t m;
  size_t n_items = fread(&n, sizeof(n), 1, f);
  assert(n_items == 1 && "failed to read g.n");
  n_items = fread(&m, sizeof(m), 1, f);
  assert(n_items == 1 && "failed to read g.n");
  
  char tmp;
  n_items = fread(&tmp, sizeof(tmp), 1, f);
  assert(n_items == 1 && "failed to read align");
  n_items = fread(&tmp, sizeof(tmp), 1, f);
  assert(n_items == 1 && "failed to read directed");

  uint64_t *rowsIndices = (uint64_t*)malloc((n+1)*sizeof(uint64_t));
  if (rowsIndices == NULL) {
    perror("failed to allocate rowsIndices");
    exit(3);
  }
  n_items = fread(rowsIndices, sizeof(*rowsIndices), n+1, f);
  assert(n_items == n+1 && "failed to read rowsIndices");

  uint32_t *endV = (uint32_t*)malloc(m*sizeof(uint32_t));
  if (endV == NULL) {
    perror("failed to allocate endV");
    exit(3);
  }
  n_items = fread(endV, sizeof(*endV), m, f);
  assert(n_items == m && "failed to read endV");

  double *weights = (double*)malloc(m*sizeof(double));
  if (weights == NULL) {
    perror("failed to allocate weights");
    exit(4);
  }
  n_items = fread(weights, sizeof(double), m, f);
  assert(n_items == m && "failed to read weights");

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