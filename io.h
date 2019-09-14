#pragma once
#include "string"
#include "graph.h"

graph read_graph(const std::string fn);
void free_graph(graph g);