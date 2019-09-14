#pragma once

#include "graph.h"
#include "dsu.h"
#define MASTER 0

loc_graph scatter_graph(graph g);
uint64_t start_idx(int myrank, int nproc, uint64_t m);
void print_local_info(loc_graph loc_g);
void print_local_vertices(loc_graph g);
char *iterations(loc_graph g, DSU &d);
char *gather_selected(loc_graph g, char *loc_selected_edges);