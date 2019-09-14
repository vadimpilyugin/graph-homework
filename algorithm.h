#pragma once

#include "graph.h"

loc_graph scatter_graph(graph g);
uint64_t start_idx(int myrank, int nproc, uint64_t m);
void print_local_info(loc_graph loc_g);
void print_local_vertices(loc_graph g);
char *iterations(loc_graph g);