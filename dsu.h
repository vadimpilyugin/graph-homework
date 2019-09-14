#pragma once

typedef struct {
  uint32_t rank;
  uint32_t parent;
} Elem;

typedef struct {
  Elem *dsu;
  uint32_t NComponents;
} DSU;

DSU InitDSU(uint32_t n);
void FreeDSU(DSU &d);
void UnionDSU(DSU &d, uint32_t elem1, uint32_t elem2);
uint32_t FindDSU(DSU &d, uint32_t elem);