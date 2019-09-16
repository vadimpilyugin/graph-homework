#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdarg>
#include <cassert>
#include <cfloat>
#include <map>
#include <algorithm>

#include "mpi.h"

#include "io.h"
#include "dsu.h"
#include "graph.h"
#include "algorithm.h"

#define MILLION 1000000

bool verbose;

void usage() {
  printf("Usage: ./main -input <rmat-x> [-verbose] -output <rmat-x.mst> -nIter <1-100>\n");
}

typedef struct {
  double weight;
  uint32_t my_root;
  uint32_t foreign_root;
  uint64_t edge_id;
} MinConn;

void print0(int rank, const char* format, ...)
{
    if (rank == 0) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
}

char *repeated_boruvka(int n_iter, loc_graph g, DSU &d, const char *input_fn) {

  print0(g.myrank, "\n======= Boruvka's algorithm =======\n\n");
  char *loc_selected_edges = NULL;
  double *perf = NULL;
  MPI_Alloc_mem(n_iter * sizeof(double), MPI_INFO_NULL, &perf);
  assert(perf && "perf malloc failed");
  double *run_times = NULL;
  MPI_Alloc_mem(n_iter * sizeof(double), MPI_INFO_NULL, &run_times);
  assert(run_times && "run_times malloc failed");

  for (int i = 0; i < n_iter; i++) {
    if (g.myrank == MASTER) {
      printf("iteration %d: ",i);
    }
    // reset the dsu
    ResetDSU(d);
    
    MPI_Barrier(MPI_COMM_WORLD);

    double start_ts = MPI_Wtime();
    char *tmp = iterations(g, d);
    double finish_ts = MPI_Wtime();

    if (loc_selected_edges != NULL) {
      // check if the answers are the same
      for (uint32_t j = 0; j < g.my_batch_size; j++) {
        if (loc_selected_edges[i] != tmp[i]) {
          printf("%d: loc_selected_edges[%u] = %d\ntmp[%u] = %d\n",
            g.myrank, i, loc_selected_edges[i], i, tmp[i]);
          assert(0 && "new answer is different");
        }
      }
      MPI_Free_mem(loc_selected_edges);
    }
    loc_selected_edges = tmp;

    double time = finish_ts - start_ts;
    perf[i] = time * double(g.m) / MILLION;
    run_times[i] = time;
    print0(g.myrank,"\tfinished. Time is %.4f secs\n", time);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  print0(g.myrank,"\n----------------\nalgorithm iterations finished.\n");

  double min_perf = 0, max_perf = 0, avg_perf = 0;
  double avg_run_time = 0;
  double global_min_perf = 0, global_max_perf = 0, global_avg_perf = 0;
  double global_avg_run_time = 0;
  max_perf = avg_perf = 0;
  min_perf = DBL_MAX;     
  for (int i = 0; i < n_iter; ++i) {  
      avg_perf += perf[i];
      avg_run_time += run_times[i];
      if (perf[i] < min_perf) min_perf = perf[i]; 
      if (perf[i] > max_perf) max_perf = perf[i]; 
  }
  avg_perf /= n_iter;
  avg_run_time /= n_iter;
  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Reduce(&min_perf, &global_min_perf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&max_perf, &global_max_perf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&avg_perf, &global_avg_perf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&avg_run_time, &global_avg_run_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  print0(g.myrank, "%s: nproc = %d; time = %.4f; vertices = %u; edges = %lu; trees = %u; n_iter = %d; MST performance min = %.4f; avg = %4f; max = %.4f\n",
    input_fn, g.nproc, global_avg_run_time, g.n, g.m, d.NComponents,
    n_iter, min_perf, avg_perf, max_perf);
  print0(g.myrank,"Performance = %.4f MTEPS\n", avg_perf);
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Free_mem(perf);
  MPI_Free_mem(run_times);
  return loc_selected_edges;
}

int main(int argc, char **argv)
{
  if (argc < 7) {
    usage();
    return 1;
  }

  const char *input_fn = NULL;
  const char *output_fn = NULL;
  int n_iter = 0;
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
    if (!strcmp(argv[i], "-nIter")) {
      n_iter = atoi(argv[i+1]);
    }
  }

  int myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  graph g;

  if (myrank == MASTER) {
    g = read_graph(input_fn);
    output_graph_info(g);
  }

  // scatter the graph across processes
  loc_graph loc_g = scatter_graph(g);

  // allocate dsu on every process
  DSU d = InitDSU(loc_g.n);

  // do a number of iterations of the algorithm
  char *loc_selected_edges = repeated_boruvka(n_iter, loc_g, d, input_fn);

  if (myrank != MASTER) {
    // master has g.rowsIndices == loc_g.rowsIndices, so it frees it later
    MPI_Free_mem(loc_g.rowsIndices);
    // master needs dsu to output the answer, so it frees it later
    FreeDSU(d);
  }
  // scattered graph is not needed anymore, only the selected edges are
  MPI_Free_mem(loc_g.locEndV);
  MPI_Free_mem(loc_g.locWeights);

  char *selected_edges = gather_selected(loc_g, loc_selected_edges);
  MPI_Free_mem(loc_selected_edges);

  if (myrank == MASTER) {
    write_tree(g, selected_edges, d, output_fn);
    free_graph(g);
    FreeDSU(d);
    free(selected_edges);
  }

  MPI_Finalize();

  return 0;
}