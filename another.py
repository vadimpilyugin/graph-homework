import struct
import array
import io
from functools import reduce
import disjoint_set as su

def slice_buffer(content, lengths):
  slices = []
  for l in lengths:
    slices.append(content[:l])
    content = content[l:]
  return slices

class Graph:
  def __init__(self):
    self.isInitialized = False

  def get_edges(self, start, end):
    edges = []
    for edge_id in range(g.rowsIndices[start], g.rowsIndices[start+1]):
      if self.endV[edge_id] == end:
        edges.append(edge_id)
    return edges

  def read_from_buffer(self, content, header_size):
    
    n,m,directed,align = struct.unpack('=IQ?B', content[:header_size])
    
    array_str = "QId"
    array_sizes = list(map(lambda c: array.array(c).itemsize, array_str))
    array_lengths = [
      header_size,
      (n+1)*array_sizes[0],
      (m)*array_sizes[1],
      (m)*array_sizes[2],
    ]
    print(f"Array lengths: {array_lengths}")
    print(f"Sum: {reduce(lambda acc,i: i+acc, array_lengths)}")
    print(f"Content length: {len(content)}")
    
    slices = slice_buffer(content, array_lengths)[1:]
    arrays = []
    for i,c in enumerate(array_str):
      arrays.append(array.array(c, slices[i]))

    self.n = n
    self.m = m
    self.directed = directed
    self.align = align

    self.rowsIndices = arrays[0]
    self.endV = arrays[1]
    self.weights = arrays[2]
    self.isInitialized = True

  def read_from_file(self, fn, header_size=14):
    
    with open(fn, 'rb') as f:
      content = f.read()

    return self.read_from_buffer(content, header_size)

  def edges(self, vertex):
    i1, i2 = self.rowsIndices[vertex], self.rowsIndices[vertex+1]
    return self.endV[i1:i2], self.weights[i1:i2]

  def vertices(self):
    return list(range(self.n))

  def __repr__(self):

    if not self.isInitialized:
      return "Graph not initialized"
    
    buf = io.StringIO()
    buf.write(f"N = {self.n}\n")
    buf.write(f"M = {self.m}\n")
    buf.write(f"Directed = {self.directed}\n")
    buf.write(f"Align = {self.align}\n")
    
    buf.write("rowsIndices:\n")
    for i in self.rowsIndices[:10]:
      buf.write(f"{i}\n")
    buf.write("endV:\n")
    for i in self.endV[:10]:
      buf.write(f"{i}\n")
    buf.write("weights:\n")
    for i in self.weights[:10]:
      buf.write(f"{i}\n")
    return buf.getvalue()

def read_graph(fn):
  g = Graph()
  g.read_from_file(fn)
  print(g)
  vertices = g.vertices()
  v = vertices[0]
  ends, weights = g.edges(v)
  print(f"Vertex {v}:")
  for i,v in enumerate(ends):
    print(f"Connected to {v} [weight {weights[i]:.3f}]")
  return g

def nearest_edge_alg(g):
  tree = {0: True}
  while len(tree) < g.n:
    min_edge = None
    for vertex in tree:
      for edge_id in range(g.rowsIndices[vertex], g.rowsIndices[vertex+1]):
        if g.endV[edge_id] not in tree:
          if min_edge is None or min_edge[2] > g.weights[edge_id]:
            min_edge = (vertex, g.endV[edge_id], g.weights[edge_id], edge_id)
    if min_edge is not None:
      print(f"marked [{min_edge[1]}], start [{min_edge[0]}], weight [{min_edge[2]}]\n")
      tree[min_edge[1]] = True

def boruvka_alg(g):
  u = su.DisjointSet(g.n)
  prev_count = None
  it = 0
  selected_ids = {}
  while prev_count is None or prev_count != u.count:
    it += 1
    print(f"\n\n------------- Iteration {it} ----------------\n")
    print(u.count, prev_count)
    prev_count = u.count
    component_min = {}
    for vertex in range(g.n):
      for edge_id in range(g.rowsIndices[vertex], g.rowsIndices[vertex+1]):
        start_root = u.find(vertex)
        end_root = u.find(g.endV[edge_id])
        if start_root == end_root:
          continue
        weight = g.weights[edge_id]
        if start_root not in component_min or weight < component_min[start_root][2]:
          component_min[start_root] = (vertex, g.endV[edge_id], weight, edge_id)
        if end_root not in component_min or weight < component_min[end_root][2]:
          component_min[end_root] = (vertex, g.endV[edge_id], weight, edge_id)
    for root, min_edge in component_min.items():
      print(f"marked [{min_edge[1]}], start [{min_edge[0]}], weight [{min_edge[2]}] [id {min_edge[3]}]\n")
      u.union(min_edge[0], min_edge[1])
      selected_ids[min_edge[3]] = True
  total_weight = 0
  edges = list(sorted(list(selected_ids)))
  remap_edges(g, edges)
  for edge_id in selected_ids:
    total_weight += g.weights[edge_id]
  print(f"Total weight: {total_weight}")
  return edges

def list_and_sort(g, edges):
  for i in range(len(edges)):
    e = edges[i]
    edges[i] = (e, g.weights[e])
  edges.sort(key=lambda x: x[1])
  return edges

def remap_edges(g, edges):
  eq = equiv(g)
  for i in range(len(edges)):
    edges[i] = eq[edges[i]]
  edges.sort()

def find_reverse_edge(g, start_v, edge_id):
  forward_edges = list_and_sort(g, g.get_edges(start_v, g.endV[edge_id]))
  idx = forward_edges.index((edge_id, g.weights[edge_id]))
  if idx is None:
    raise f"Fatal error: {edge_id} is not in {forward_edges}"
  backward_edges = list_and_sort(g, g.get_edges(g.endV[edge_id], start_v))
  return backward_edges[idx][0]

def equiv(g):
  hsh = {}
  for vertex in range(g.n):
    for edge_id in range(g.rowsIndices[vertex], g.rowsIndices[vertex+1]):
      other_id = find_reverse_edge(g, vertex, edge_id)
      min_id = edge_id
      if other_id < edge_id:
        min_id = other_id
      hsh[edge_id] = min_id
  return hsh

def read_solution(fn):
  with open(fn, 'rb') as f:
    content = f.read()
  numTrees, numEdges = struct.unpack('=IQ', content[:12])
  print(f"numTrees: {numTrees}")
  print(f"numEdges: {numEdges}")
  p_edge_list = array.array('Q', content[12:12+2*numTrees*8])
  print(f"p_edge_list[{len(p_edge_list)} elements]: {p_edge_list}")
  edge_id = list(array.array('Q', content[12+2*numTrees*8:]))
  print(f"edge_id[{len(edge_id)} items]")
  return edge_id

i=12
graphFn = f"graph-homework/rmat-{i}"
g = read_graph(graphFn)
b_edges = boruvka_alg(g)
edges = read_solution(f"graph-homework/rmat-{i}.mst")
remap_edges(g, edges)
# print(edges)
# print(b_edges)
print(edges == b_edges)