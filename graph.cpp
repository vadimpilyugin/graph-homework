#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>

#include "graph.h"

extern bool verbose;

void out_weights(std::vector<double> v) {
  printf("[");
  for (auto &d: v) {
    printf("%.3lf, ", d);
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
  if (verbose) {
    for (uint32_t i = 10; i < g.m; i++) {
      printf("endV[%d] = %d\n", i, g.endV[i]);
    } 
  }
  printf("\n");
  printf("weights [%ld elems]\n", g.m);
  for (int i = 0; i < 10; i++) {
    printf("weights[%d] = %lf\n", i, g.weights[i]);
  }
  printf("\n");
  printf("===========================\n\n");
}

void is_undirected(graph g) {

  printf("========== is_undirected ==========\n\n");
  std::unordered_map<uint32_t, std::unordered_map<uint32_t,std::vector<double> > > all_adj;
  for (uint32_t start_v = 0; start_v < g.n; start_v++) {
    std::unordered_map<uint32_t,std::vector<double> > adj;

    for (uint64_t edge_id = g.rowsIndices[start_v];
      edge_id < g.rowsIndices[start_v+1]; edge_id++) {
      adj[g.endV[edge_id]].push_back(g.weights[edge_id]);
    }

    for (auto &e_adj: adj) {
      auto &start_w = std::get<1>(e_adj);
      std::sort(start_w.begin(), start_w.end());
    }

    all_adj[start_v] = adj;
  }

  for (uint32_t start_v = 0; start_v < g.n; start_v++) {
    auto adj = all_adj[start_v];
    auto ks = keys(adj);
    for (auto &end_v: ks) {
      auto start_w = adj[end_v];
      auto end_adj = all_adj[end_v];
      if (start_w == end_adj[start_v]) {
        if (verbose) {
          printf("Erasing %d ~~> %d: ", start_v, end_v);
          out_weights(adj[end_v]);
          printf("Erasing %d ~~> %d: ", end_v, start_v);
          out_weights(end_adj[start_v]);
          printf("\n");
        }
        all_adj[start_v].erase(end_v);
        all_adj[end_v].erase(start_v);
      } else {
        printf("%d ~~> %d: ", start_v, end_v);
        out_weights(adj[end_v]);
        printf("%d ~~> %d: ", end_v, start_v);
        out_weights(end_adj[start_v]);
        printf("\n");
      }
    }
  }

  printf("\n===================================\n");
}

void is_multigraph(graph g) {
  printf("========== is_multigraph ==========\n\n");
  for (uint32_t vertex = 0; vertex < g.n; vertex++) {
    std::unordered_map<uint32_t,int> adj;
    for (uint64_t edge_id = g.rowsIndices[vertex];
      edge_id < g.rowsIndices[vertex+1]; edge_id++) {
      adj[g.endV[edge_id]]++;
    }
    for (auto &pair: adj) {
      uint32_t end_v = std::get<0>(pair);
      int n_edges = std::get<1>(pair);
      if (n_edges == 1 && end_v == vertex) {
        printf("%d ~~> %d [single loop]\n", vertex, end_v);
      } else if (end_v == vertex) {
        printf("%d ~~> %d [multi loop: %d edges]\n",
          vertex, end_v, n_edges);
      } else if (n_edges > 1) {
        printf("%d ~~> %d [%d edges]\n",
          vertex, end_v, n_edges);
      }
    }
  }
  printf("\n\n===================================\n");
}