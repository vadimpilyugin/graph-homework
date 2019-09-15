#pragma once
#include "string"
#include "graph.h"

#include "dsu.h"

graph read_graph(const std::string fn);
void free_graph(graph g);
void write_tree(graph g, char *selected_edges, DSU &d, const std::string fn);
void free_graph(graph g);