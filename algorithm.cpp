#include <cassert>
#include <cstdlib>
#include <cfloat>
#include "mpi.h"

#include "algorithm.h"
#include "dsu.h"

#define TAG_0 0
#define COLOR_0 0
#define COLOR_1 1
#define SELECTED 1
#define UNSELECTED 0

MPI_Comm first_group;
MPI_Comm second_group;

extern bool verbose;

uint64_t start_idx(int myrank, int nproc, uint64_t m) {

  uint64_t batch_size = m / nproc;
  uint64_t leftover = m % nproc;

  if (uint32_t(myrank) < leftover) {
    return (batch_size+1) * myrank;
  }

  return (batch_size+1) * leftover + batch_size * (myrank - leftover);
}

loc_graph scatter_graph(graph g) {
  // input - graph g on 0 process
  // output - loc_graph on every process
  
  int success;

  // determine rank and # of processes
  int nproc;
  success = MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  assert(success == MPI_SUCCESS && "comm size failed");

  int myrank;
  success = MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  assert(success == MPI_SUCCESS && "comm rank failed");

  bool i_am_the_master = myrank == MASTER;

  // broadcast global parameters: n,m,rowsIndices
  uint32_t n = g.n;
  uint64_t m = g.m;
  uint64_t *rowsIndices = g.rowsIndices;

  success = MPI_Bcast(&n, 1, MPI_UINT32_T, MASTER, MPI_COMM_WORLD);
  assert(success == MPI_SUCCESS && "broadcast of g.n failed");
  
  success = MPI_Bcast(&m, 1, MPI_UINT32_T, MASTER, MPI_COMM_WORLD);
  assert(success == MPI_SUCCESS && "broadcast of g.m failed");
  
  if (!i_am_the_master) {
    rowsIndices = (uint64_t *)malloc((n+1) * sizeof(uint64_t));
    assert(rowsIndices && "malloc of rowsIndices failed");
  }
  
  success = MPI_Bcast(rowsIndices, n+1, MPI_UINT64_T, MASTER, MPI_COMM_WORLD);
  assert(success == MPI_SUCCESS && "broadcast of g.rowsIndices failed");

  // calculate batch_size and leftover
  uint64_t batch_size = m / nproc;
  int leftover = m % nproc;
  uint64_t my_batch_size = batch_size;
  if (myrank < leftover) {
    my_batch_size++;
  }

  // allocate space for parts of endV and weights
  uint32_t *locEndV = (uint32_t *)malloc(
    my_batch_size * sizeof(uint32_t));
  assert(locEndV && "malloc of locEndV failed");
  double *locWeights = (double *)malloc(
    my_batch_size * sizeof(double));
  assert(locWeights && "malloc of locWeights failed");

  // scatter endV and weights vectors to processes

  if (leftover == 0) {
    // if number of edges is divisible by nproc
    success = MPI_Scatter(
      g.endV, my_batch_size, MPI_UINT32_T,
      locEndV, my_batch_size, MPI_UINT32_T,
      MASTER, MPI_COMM_WORLD);
    assert(success == MPI_SUCCESS && "scatter of g.endV failed");
    success = MPI_Scatter(
      g.weights, my_batch_size, MPI_DOUBLE,
      locWeights, my_batch_size, MPI_DOUBLE,
      MASTER, MPI_COMM_WORLD);
    assert(success == MPI_SUCCESS && "scatter of g.weights failed");
    // and that's it!
  } else {
    // else if the number of edges is not divisible by nproc
    uint32_t *newEndV;
    double *newWeights;
    uint64_t first_part_len = leftover * my_batch_size;
    uint64_t second_part_len = (nproc - leftover) * batch_size;

    // perform two-step scatter
    // first step: #0 ~~> #leftover
    if (i_am_the_master) {
      success = MPI_Send(
        g.endV + first_part_len, second_part_len, MPI_UINT32_T,
        leftover, TAG_0, MPI_COMM_WORLD);
      assert(success == MPI_SUCCESS && "send of g.endV failed");
      success = MPI_Send(
        g.weights + first_part_len, second_part_len, MPI_DOUBLE,
        leftover, TAG_0, MPI_COMM_WORLD);
      assert(success == MPI_SUCCESS && "send of g.weights failed");
    }
    if (myrank == leftover) {
      newEndV = (uint32_t *)malloc(second_part_len * sizeof(uint32_t));
      assert(newEndV && "malloc of newEndV failed");
      newWeights = (double *)malloc(second_part_len * sizeof(double));
      assert(newWeights && "malloc of newWeights failed");
      success = MPI_Recv(
        newEndV, second_part_len, MPI_UINT32_T,
        MASTER, TAG_0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      assert(success == MPI_SUCCESS && "receive of second part endV failed");
      success = MPI_Recv(
        newWeights, second_part_len, MPI_DOUBLE,
        MASTER, TAG_0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      assert(success == MPI_SUCCESS && "receive of second part weights failed");
    }

    // second step: #0 ~~> #0..#leftover-1, #leftover ~~> #leftover..#nproc-1
    if (myrank < leftover) {
      success = MPI_Comm_split(MPI_COMM_WORLD, COLOR_0, myrank, &first_group);
      assert(success == MPI_SUCCESS && "first_group create for first scatter step failed");

      success = MPI_Scatter(
        g.endV, my_batch_size, MPI_UINT32_T,
        locEndV, my_batch_size, MPI_UINT32_T,
        MASTER, first_group);
      assert(success == MPI_SUCCESS && "scatter of g.endV failed");

      success = MPI_Scatter(
        g.weights, my_batch_size, MPI_DOUBLE,
        locWeights, my_batch_size, MPI_DOUBLE,
        MASTER, first_group);
      assert(success == MPI_SUCCESS && "scatter of g.weights failed");

    } else {
      success = MPI_Comm_split(MPI_COMM_WORLD, COLOR_1, myrank-leftover, &second_group);
      assert(success == MPI_SUCCESS && "second_group create for first scatter step failed");

      success = MPI_Scatter(
        newEndV, my_batch_size, MPI_UINT32_T,
        locEndV, my_batch_size, MPI_UINT32_T,
        MASTER, second_group);
      assert(success == MPI_SUCCESS && "scatter of newEndV failed");

      success = MPI_Scatter(
        newWeights, my_batch_size, MPI_DOUBLE,
        locWeights, my_batch_size, MPI_DOUBLE,
        MASTER, second_group);
      assert(success == MPI_SUCCESS && "scatter of newWeights failed");
    }

    if (myrank == leftover) {
      free(newEndV);
      free(newWeights);
    }
  }


  // fill in broadcasted parameters
  loc_graph loc_g;
  loc_g.n = n;
  loc_g.m = m;
  loc_g.rowsIndices = rowsIndices;
  loc_g.locEndV = locEndV;
  loc_g.locWeights = locWeights;
  
  loc_g.myrank = myrank;
  loc_g.nproc = nproc;
  loc_g.leftover = leftover;
  loc_g.batch_size = batch_size;
  loc_g.my_batch_size = my_batch_size;
  loc_g.start_i = start_idx(myrank, nproc, m);
  loc_g.end_i = start_idx(myrank+1, nproc, m);
  
  // determine the first vertex no
  // complexity: O(n)
  loc_g.start_v = 0;
  for (uint32_t i = 0; i < loc_g.n; i++) {
    if (rowsIndices[i] <= loc_g.start_i &&
      rowsIndices[i+1] > loc_g.start_i) {

      loc_g.start_v = i;
      break;
    }
  }

  return loc_g;
}

void print_local_info(loc_graph loc_g) {
  for (int i = 0; i < loc_g.nproc; i++) {
    if (loc_g.myrank == i) {
      printf("My rank: %d\n", loc_g.myrank);
      printf("My nproc: %d\n", loc_g.nproc);
      printf("------------\n");
      printf("My leftover: %d\n", loc_g.leftover);
      printf("Batch size: %lu\n", loc_g.batch_size);
      printf("My batch size: %lu\n", loc_g.my_batch_size);
      printf("start_i: %lu\n", loc_g.start_i);
      printf("end_i: %lu\n", loc_g.end_i);
      printf("start_v: %u\n", loc_g.start_v);
      printf("------------\n");
      printf("n = %u\n", loc_g.n);
      printf("m = %lu\n", loc_g.m);
      printf("rowsIndices pointer: %p\n", (void*)loc_g.rowsIndices);
      printf("locEndV pointer: %p\n", (void*)loc_g.locEndV);
      printf("locWeights pointer: %p\n", (void*)loc_g.locWeights);
      printf("\n\n==================\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void print_local_vertices(loc_graph g) {
  for (int i = 0; i < g.nproc; i++) {
    if (i == g.myrank) {
      printf("My rank: %d\n", g.myrank);
      printf("Start vertex: %u\n", g.start_v);
      printf("Start index: %lu\n", g.start_i);
      printf("Number of edges: %lu\n\n", g.my_batch_size);
      for (auto vertex = g.start_v;;vertex++) {
        auto edge_id = g.rowsIndices[vertex];
        if (vertex == g.start_v) {
          edge_id = g.start_i;
        }
        auto min_id = g.rowsIndices[vertex+1];
        if (min_id > g.end_i) {
          min_id = g.end_i;
        }
        uint64_t edges_num = 0;
        for (; edge_id < min_id; edge_id++) {
          edges_num++;
          continue;
        }
        
        if (vertex == g.start_v) {
          if (g.start_i != g.rowsIndices[g.start_v]) {
            printf("--- continuation: %u [%lu]\n", vertex, edges_num);
          } else {
            printf("--- clear start: %u [%lu]\n", vertex, edges_num);
          }
        } else {
          printf("%u [%lu]\n", vertex, edges_num);
        }

        if (edge_id == g.end_i) {
          break;
        }
      }
      printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

typedef struct {
  double weight;
  uint32_t my_root;
  uint32_t other_root;
  uint64_t edge_id;
} MinConn;

typedef struct {
  double w;
  int rank;
} MinLocItem;

char *iterations(loc_graph g, DSU &d) {
  MPI_Barrier(MPI_COMM_WORLD);
  uint32_t n = g.n;
  
  // allocate pointers array and array of shortest connections
  uint32_t *ptrs = (uint32_t *)malloc(n * sizeof(uint32_t));
  assert(ptrs && "malloc of ptrs failed");

  MinConn *mins = (MinConn *)malloc(n * sizeof(MinConn));
  assert(mins && "malloc of mins failed");

  uint32_t mins_len = 0;
  
  // pair of ptrs and mins acts like a hash table
  // mapping uint32_t my_root ~~> struct MinConn

  // array for determining the shortest edges across all processes
  MinLocItem *min_loc = (MinLocItem *)malloc(n * sizeof(MinLocItem));
  assert(min_loc && "malloc of min_loc failed");

  // array for receiving reduced min_loc
  MinLocItem *out_min_loc = (MinLocItem *)malloc(n * sizeof(MinLocItem));
  assert(out_min_loc && "malloc of out_min_loc failed");

  // array for determining the chosen components for each component on each iteration
  uint32_t *selected_comps = (uint32_t *)malloc(n * sizeof(uint32_t));
  assert(selected_comps && "malloc of selected_comps failed");
  
  // array for receiving selected_comps
  uint32_t *out_selected = (uint32_t *)malloc(n * sizeof(uint32_t));
  assert(out_selected && "malloc of out_selected failed");

  // array for keeping track of chosen connections
  char *selected_edges = (char *)calloc(g.my_batch_size, sizeof(char));
  assert(selected_edges && "calloc of selected_edges failed");

  for (uint32_t iter = 0; ; iter++) {
    if (verbose)
      printf("\n============== Iteration %u ==============\n\n", iter);
    for (uint32_t i = 0; i < n; i++) {
      ptrs[i] = n;
    }
    mins_len = 0;
    uint64_t edge_id = g.start_i;
    for (uint32_t vertex = g.start_v; edge_id < g.end_i; vertex++) {
      if (verbose)
        printf("%d: checking %u\n", g.myrank, vertex);
      MinConn curr_min;
      uint32_t struct_idx;
      uint32_t my_root = FindDSU(d, vertex);
      if (ptrs[my_root] == n) {
        if (verbose)
          printf("%d: Found new component: %u\n",
            g.myrank, my_root);
        ptrs[my_root] = mins_len;

        mins[mins_len].weight = DBL_MAX;
        mins[mins_len].my_root = my_root;
        mins[mins_len].other_root = n;
        mins[mins_len].edge_id = g.m;

        mins_len++;
      }
      struct_idx = ptrs[my_root];
      curr_min = mins[struct_idx];
      uint64_t id_max = g.rowsIndices[vertex+1];
      if (id_max > g.end_i) {
        id_max = g.end_i;
      }
      for (; edge_id < id_max; edge_id++) {
        uint64_t loc_edge_id = edge_id-g.start_i;
        uint32_t other_v = g.locEndV[loc_edge_id];
        uint32_t other_root = FindDSU(d, other_v);
        if (other_root == my_root) {
          continue;
        }
        double w = g.locWeights[loc_edge_id];
        if (w < curr_min.weight) {
          curr_min.weight = w;
          curr_min.my_root = my_root;
          curr_min.other_root = other_root;
          curr_min.edge_id = loc_edge_id;
        }
      }
      if (verbose) {
        if (curr_min.weight == DBL_MAX) {
          printf("%d: setting mins[%d] = (+inf, %u ~~> %u, %lu)\n",
            g.myrank,
            struct_idx, curr_min.my_root,
            curr_min.other_root, curr_min.edge_id);
        } else {
          printf("%d: setting mins[%d] = (%lf, %u ~~> %u [%u], %lu)\n",
            g.myrank,
            struct_idx, curr_min.weight, curr_min.my_root,
            curr_min.other_root, g.locEndV[curr_min.edge_id], curr_min.edge_id);
        }
      }
      mins[struct_idx] = curr_min;
    }

    if (verbose)
      MPI_Barrier(MPI_COMM_WORLD);

    // exchange section
    // unset selected components
    for (uint32_t i = 0; i < n; i++) {
      selected_comps[i] = n;
    }

    // unset all edges
    for (uint32_t i = 0; i < n; i++) {
      min_loc[i].w = DBL_MAX;
      min_loc[i].rank = g.nproc;
    }

    // set found minimum edges
    for (uint32_t i = 0; i < mins_len; i++) {
      min_loc[mins[i].my_root].w = mins[i].weight;
      min_loc[mins[i].my_root].rank = g.myrank;
    }

    // find overall minimum
    int success = MPI_Allreduce(
      min_loc, out_min_loc,
      n, MPI_DOUBLE_INT, MPI_MINLOC,
      MPI_COMM_WORLD);
    assert(success == MPI_SUCCESS && "allreduce of min_loc failed");

    if (g.myrank == MASTER && verbose) {
      printf("This is where the true minimum lies:\n");
      for (uint32_t i = 0; i < n; i++) {
        if (out_min_loc[i].w == DBL_MAX) {
          printf("Component %u: +inf, on proc %d\n",
            i, out_min_loc[i].rank);
        } else {
          printf("Component %u: %lf, on proc %d\n",
            i, out_min_loc[i].w, out_min_loc[i].rank);
        }
      }
    }
    if (verbose)
      MPI_Barrier(MPI_COMM_WORLD);
    

    // set minimum components
    if (verbose)
      for (uint32_t i = 0; i < mins_len; i++) {
        auto my_root = mins[i].my_root;
        if (out_min_loc[my_root].rank != g.myrank && 
          out_min_loc[my_root].w != DBL_MAX) {

          printf("%d: minimum edge %u ~~> %u [weight %lf] was not selected, %d != %d\n",
            g.myrank, mins[i].my_root, mins[i].other_root,
            mins[i].weight, out_min_loc[mins[i].my_root].rank, g.myrank);

        } else if (out_min_loc[my_root].rank == g.myrank) {

          if (mins[i].weight == DBL_MAX) {
            printf("%d: minimum edge %u ~~> %u [weight +inf] was selected\n",
              g.myrank, mins[i].my_root, mins[i].other_root);
          } else {
            printf("%d: minimum edge %u ~~> %u [weight %lf] was selected\n",
              g.myrank, mins[i].my_root, mins[i].other_root, mins[i].weight);
          }

        }
      }
    for (uint32_t i = 0; i < mins_len; i++) {
      auto my_root = mins[i].my_root;
      if (out_min_loc[my_root].rank == g.myrank &&
        out_min_loc[my_root].w != DBL_MAX) {

        // my batch has a minimal edge, mins[i]
        // set my component's choice to other_root
        selected_comps[my_root] = mins[i].other_root;
        // mark this edge as selected
        selected_edges[mins[i].edge_id] = SELECTED;
        // note: if there is a loop #c1 ~~> #c2, #c2 ~~> #c1,
        // it is resolved later
      }
    }

    // find out each component's choice of another component
    success = MPI_Allreduce(
      selected_comps, out_selected,
      n, MPI_UINT32_T, MPI_MIN,
      MPI_COMM_WORLD);
    assert(success == MPI_SUCCESS && "allreduce of selected_comps failed");

    if (g.myrank == MASTER && verbose) {
      printf("True selected components:\n");
      for (uint32_t i = 0; i < n; i++) {
        if (out_selected[i] == n) {
          printf("Component %u: none selected\n", i);
        } else {
          printf("Component %u: %u\n", i, out_selected[i]);
        }
      }
    }
    if (verbose)
      MPI_Barrier(MPI_COMM_WORLD);

    // break loops
    for (uint32_t i = 0; i < mins_len; i++) {
      auto my_root = mins[i].my_root;
      if (out_min_loc[my_root].rank == g.myrank &&
        out_min_loc[my_root].w != DBL_MAX) {

        auto other_root = mins[i].other_root;
        auto other_sel = out_selected[other_root];
        if (other_sel == my_root) {
          auto min_sel = my_root;
          if (my_root > other_root) {
            min_sel = other_root;
          }
          if (verbose)
            printf("%d: found loop %u ~~> %u, %u ~~> %u\n",
              g.myrank, my_root, other_root, other_root, my_root);
          out_selected[my_root] = min_sel;
          out_selected[other_root] = min_sel;
          if (my_root == min_sel) {
            // break the loop
            if (verbose)
              printf("%d: breaking the loop: %u ~/~> %u\n",
                g.myrank, my_root, other_root);
            selected_edges[mins[i].edge_id] = UNSELECTED;
          }
        }
      }
    }

    if (verbose)
      MPI_Barrier(MPI_COMM_WORLD);

    // broadcast correct selections
    success = MPI_Allreduce(
      out_selected, selected_comps,
      n, MPI_UINT32_T, MPI_MIN,
      MPI_COMM_WORLD);
    assert(success == MPI_SUCCESS && "allreduce of out_selected failed");

    if (g.myrank == MASTER && verbose) {
      printf("True pre-DSU without cycles:\n");
      for (uint32_t i = 0; i < n; i++) {
        if (selected_comps[i] == n) {
          printf("Component %u: none selected\n", i);
        } else if (selected_comps[i] == i) {
          printf("Component %u: is a root\n", i);
        } else {
          printf("Component %u: %u\n", i, selected_comps[i]);
        }
      }
    }

    if (verbose)
      MPI_Barrier(MPI_COMM_WORLD);

    // join selected components
    bool changed = false;
    for (uint32_t i = 0; i < n; i++) {
      if (selected_comps[i] != n && selected_comps[i] != i) {
        UnionDSU(d, i, selected_comps[i]);
        changed = true;
      }
    }
    if (g.myrank == MASTER && verbose) {
      printf("After DSU union-find:\n");
      for (uint32_t i = 0; i < n; i++) {
        printf("Component %u: %u\n", i, FindDSU(d, i));
      }
    }
    if (verbose)
      MPI_Barrier(MPI_COMM_WORLD);
    if (!changed) {
      break;
    }
  }

  free(ptrs);
  free(mins);
  free(min_loc);
  free(out_min_loc);
  free(selected_comps);
  free(out_selected);

  return selected_edges;
}

char *gather_selected(loc_graph g, char *loc_selected_edges) {
  // collect locally selected edges on #0

  int success;

  uint64_t m = g.m;
  auto myrank = g.myrank;
  auto nproc = g.nproc;
  uint64_t batch_size = m / nproc;
  int leftover = m % nproc;
  uint64_t my_batch_size = batch_size;
  if (myrank < leftover) {
    my_batch_size++;
  }

  char *selected_edges;
  if (myrank == MASTER) {
    selected_edges = (char *)malloc(m * sizeof(char));
    assert(selected_edges && "malloc of selected_edges failed");
  }

  if (leftover == 0) {
    // if number of edges is divisible by nproc
    success = MPI_Gather(
      loc_selected_edges, my_batch_size, MPI_CHAR,
      selected_edges, my_batch_size, MPI_CHAR,
      MASTER, MPI_COMM_WORLD);
    assert(success == MPI_SUCCESS && "gather of selected_edges failed");
    // and that's it!
  } else {
    // else if the number of edges is not divisible by nproc
    char *newSelectedEdges;
    uint64_t first_part_len = leftover * my_batch_size;
    uint64_t second_part_len = (nproc - leftover) * batch_size;

    // perform two-step gather
    // allocate second-part vector
    if (myrank == leftover) {
      newSelectedEdges = (char *)malloc(second_part_len * sizeof(char));
      assert(newSelectedEdges && "malloc of newSelectedEdges failed");
    }
    // first step: #leftover..#nproc-1 ~~> #leftover
    if (myrank >= leftover) {
      success = MPI_Gather(
        loc_selected_edges, my_batch_size, MPI_CHAR,
        newSelectedEdges, my_batch_size, MPI_CHAR,
        MASTER, second_group);
      assert(success == MPI_SUCCESS && "gather of newSelectedEdges failed");
    }
    // second step: #0..#leftover-1 ~~> #0
    if (myrank < leftover) {
      success = MPI_Gather(
        loc_selected_edges, my_batch_size, MPI_CHAR,
        selected_edges, my_batch_size, MPI_CHAR,
        MASTER, first_group);
      assert(success == MPI_SUCCESS && "gather of selected_edges from first_group failed");
    }
    // last step: #leftover ~~> #0
    if (myrank == leftover) {
      success = MPI_Send(
        newSelectedEdges, second_part_len, MPI_CHAR,
        MASTER, TAG_0, MPI_COMM_WORLD);
      assert(success == MPI_SUCCESS && "send of newSelectedEdges failed");
    } else if (myrank == MASTER) {
      success = MPI_Recv(
        selected_edges + first_part_len, second_part_len, MPI_CHAR,
        leftover, TAG_0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      assert(success == MPI_SUCCESS && "receive of newSelectedEdges failed");
    }

    if (myrank == leftover) {
      free(newSelectedEdges);
    }
  }
  return selected_edges;
}