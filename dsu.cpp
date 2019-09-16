#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstdint>

#include "dsu.h"
#include "mpi.h"

DSU InitDSU(uint32_t n) {
  Elem *ptr = NULL;
  MPI_Alloc_mem(n * sizeof(Elem), MPI_INFO_NULL, &ptr);
  assert(ptr && "malloc of dsu failed");
  DSU d;
  d.dsu = ptr;
  d.size = n;
  ResetDSU(d);
  return d;
}

void ResetDSU(DSU &d) {
  d.NComponents = d.size;
  for (uint32_t i = 0; i < d.size; i++) {
    d.dsu[i].parent = i;
    d.dsu[i].rank = 0;
  }
}

void FreeDSU(DSU &d) {
  MPI_Free_mem(d.dsu);
}

void UnionDSU(DSU &d, uint32_t elem1, uint32_t elem2) {
  uint32_t root1 = FindDSU(d, elem1);
  uint32_t root2 = FindDSU(d, elem2);
  if (root1 == root2) {
    fprintf(stderr, "Already in the same component\n");
    return;
  }
  d.NComponents--;
  if (d.dsu[root1].rank == d.dsu[root2].rank) {
    uint32_t lesser_root = root1 < root2 ? root1 : root2;
    uint32_t greater_root = root1 < root2 ? root2 : root1;
    d.dsu[greater_root].parent = lesser_root;
    d.dsu[lesser_root].rank = d.dsu[lesser_root].rank + 1;
    return;
  }
  if (d.dsu[root1].rank > d.dsu[root2].rank) {
    d.dsu[root2].parent = root1;
  } else {
    d.dsu[root1].parent = root2;
  }
}

uint32_t FindDSU(DSU &d, uint32_t elem) {
  uint32_t i;
  for (i = elem; d.dsu[i].parent != i; i = d.dsu[i].parent) {
    // do nothing
  }
  // dsu[i] == i
  uint32_t j = elem;
  while (j != i) {
    uint32_t tmp = d.dsu[j].parent;
    d.dsu[j].parent = i;
    j = tmp;
  }
  return i;
}