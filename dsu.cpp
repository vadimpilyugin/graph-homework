#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstdint>

#include "dsu.h"

DSU InitDSU(uint32_t n) {
  Elem *ptr = (Elem *)malloc(n * sizeof(Elem));
  assert(ptr && "malloc of dsu failed");
  for (uint32_t i = 0; i < n; i++) {
    ptr[i].parent = i;
    ptr[i].rank = 0;
  }

  DSU dsu;
  dsu.dsu = ptr;
  dsu.NComponents = n;
  return dsu;
}

void FreeDSU(DSU &d) {
  free(d.dsu);
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