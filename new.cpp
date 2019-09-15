#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <algorithm>

#include "mpi.h"

#include "io.h"
#include "dsu.h"
#include "graph.h"
#include "algorithm.h"

bool verbose;

void usage() {
  printf("Usage: ./main -input <rmat-x> [-verbose] -output <rmat-x.mst>\n");
}

typedef struct {
  double weight;
  uint32_t my_root;
  uint32_t foreign_root;
  uint64_t edge_id;
} MinConn;

int main(int argc, char **argv)
{
  if (argc < 5) {
    usage();
    return 1;
  }

  const char *input_fn = NULL;
  const char *output_fn = NULL;
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "-input")) {
      input_fn = argv[i+1];
    }
    if (!strcmp(argv[i], "-output")) {
      output_fn = argv[i+1];
    }
    if (!strcmp(argv[i], "-verbose")) {
      verbose = true;
    }
  }

  int myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  graph g;

  if (myrank == MASTER) {
    g = read_graph(input_fn);
    output_graph_info(g);
    // is_multigraph(g);
    // is_undirected(g);
  }

  loc_graph loc_g = scatter_graph(g);

  // union-find structure for all n vertices
  DSU d = InitDSU(loc_g.n);

  char *loc_selected_edges = iterations(loc_g, d);
  char *selected_edges = gather_selected(loc_g, loc_selected_edges);
  free(loc_selected_edges);

  if (myrank == MASTER) {
    write_tree(g, selected_edges, d, output_fn);
    free_graph(g);
    FreeDSU(d);
    free(selected_edges);
  }

  MPI_Finalize();

  return 0;
}