#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <algorithm>

#include "mpi.h"

#include "io.h"
#include "graph.h"
#include "algorithm.h"

bool verbose;

void usage() {
  printf("Usage: ./main -input <rmat-x> [-verbose] -nproc <1-64>\n");
}

void model_process(int myrank, int nproc, uint32_t n, uint64_t m,
  uint32_t *endV, double *weights, uint64_t *rowsIndices) {

  uint64_t batch_size = m / nproc;
  uint64_t leftover = m % nproc;

  uint64_t my_batch_size = batch_size;
  if (uint32_t(myrank) < leftover) {
    my_batch_size++;
  }

  uint64_t start_i = start_idx(myrank, nproc, m);
  uint32_t start_v = 0;

  // не ошибка: i, i+1 => < n
  for (uint32_t i = 0; i < n; i++) {
    if (rowsIndices[i] <= start_i && rowsIndices[i+1] > start_i) {
      start_v = i;
      break;
    }
  }

  printf("My rank = %d\nMy batch size: %ld\nLeftover: %ld\n",
    myrank, my_batch_size, leftover);
  printf("Start vertex: %u\n", start_v);
  // printf("my endV: ");
  // out_indices_32(endV, my_batch_size);
  printf("\n");
}

typedef struct {
  double weight;
  uint32_t my_root;
  uint32_t foreign_root;
  uint64_t edge_id;
} MinConn;

int main(int argc, char **argv)
{
  if (argc < 2) {
    usage();
    return 1;
  }

  const char *input_fn;
  int nproc = 1;
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "-input")) {
      input_fn = argv[i+1];
    }
    if (!strcmp(argv[i], "-verbose")) {
      verbose = true;
    }
    if (!strcmp(argv[i], "-nproc")) {
      nproc = atoi(argv[i+1]);
      if (nproc == 0) {
        perror("wrong value for nproc");
        return 5;
      }
    }
  }

  int myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  graph g;

  if (myrank == MASTER) {
    g = read_graph(input_fn);
    // output_graph_info(g);
    // is_multigraph(g);
    // is_undirected(g);
  }

  loc_graph loc_g = scatter_graph(g);
  // print_local_info(loc_g);
  // print_local_vertices(loc_g);
  char *loc_selected_edges = iterations(loc_g);
  char *selected_edges = gather_selected(loc_g, loc_selected_edges);

  return 0;
}