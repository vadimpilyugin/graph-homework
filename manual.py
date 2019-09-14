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

# def gen_validation(num_trees=1):


def argmax(ar):
  i = None
  for j in range(len(ar)):
    if i is None or ar[j] > ar[i]:
      i = j
  return i

def argmin(ar):
  i = None
  for j in range(len(ar)):
    if i is None or ar[j] < ar[i]:
      i = j
  return i

graphFn = "graph-homework/rmat-5"
# graphFn = "graph-homework/rmat-12"
# graphFn = "reference/test-mat-1"
g = Graph()
print(g)
g.read_from_file(graphFn)
print(g)
vertices = g.vertices()
v = vertices[0]
ends, weights = g.edges(v)
print(f"Vertex {v}:")
for i,v in enumerate(ends):
  print(f"Connected to {v} [weight {weights[i]:.3f}]")

# u = su.DisjointSet(len(vertices))
# it = 0
# while u.count > 1:
#   comp_choices = {}
#   print(f"\n\n------------- Iteration {it} ----------------\n")
#   for v in vertices:
#     my_root = u.find(v)
#     ends, weights = g.edges(v)
#     new_ends = []
#     new_weights = []
#     for i,e in enumerate(ends):
#       other_end_root = u.find(e)
#       if my_root != other_end_root:
#         new_ends.append(other_end_root)
#         new_weights.append(weights[i])
#     weights = new_weights
#     ends = new_ends
#     min_idx = argmin(weights)
#     if min_idx is None:
#       continue
#     root_min = ends[min_idx]
#     w_min = weights[min_idx]
#     if my_root not in comp_choices:
#       comp_choices[my_root] = (root_min, w_min)
#     else:
#       prev_root_min, prev_w_min = comp_choices[my_root]
#       if prev_w_min > w_min:
#         comp_choices[my_root] = (root_min, w_min)
#   for my_root, min_tuple in comp_choices.items():
#     root_min, w_min = min_tuple
#     print(f"Root {my_root} chooses {root_min} [weight {w_min:.3f}]")
#     u.union(my_root, root_min)
#   it = it+1

u = su.DisjointSet(len(vertices))
selected_edges = {}
it = 0
prev_count = None
while prev_count is None or u.count != prev_count:
  prev_count = u.count
  print(f"\n\n------------- Iteration {it} ----------------\n")
  print(f"Number of components: {u.count}")
  hsh = {}
  for v1 in range(g.n):
    first_root = u.find(v1)
    for j2 in range(g.rowsIndices[v1], g.rowsIndices[v1+1]):
      v2 = g.endV[j2]
      second_root = u.find(v2)
      conn_weight = g.weights[j2]
      conn_id = ((v1,v2),j2)
      if v1 > v2:
        conn_id = ((v2,v1),j2)
      if first_root == second_root:
        continue
      if first_root not in hsh:
        hsh[first_root] = (conn_weight, conn_id, second_root)
      else:
        if conn_weight < hsh[first_root][0]:
          hsh[first_root] = (conn_weight, conn_id, second_root)
  for first_root, conn in hsh.items():
    conn_id = conn[1]
    print(f"Component {first_root} chooses {conn[2]} [id {conn_id}] [weight {conn[0]:.3f}]")
    selected_edges[conn_id[0]] = conn_id[1]
    u.union(first_root, conn[2])
  it += 1

total_weight = 0
selected_list = [(g.weights[i],i) for _,i in selected_edges.items()]
selected_list.sort(key=lambda x: x[1])
for _,conn_id in selected_edges.items():
  total_weight += g.weights[conn_id]
with open("graph-homework/rmat-5.vinfo", 'rb') as f:
  content = f.read()
valid_num_trees = struct.unpack('=L', content[:4])[0]
valid_weight_trees = array.array('d', content[4:])
print(f"valid_num_trees: {valid_num_trees}")
print(f"valid_weight_trees[{len(valid_weight_trees)}]: {valid_weight_trees}")

print("\n\n----------------\n")
print(f"Total weight: {total_weight}")

# numTrees [1]: 1 x 4 = 4 bytes
# numEdges [1023]: 1 x 8 = 8 bytes
# p_edge_list: 2 x 8 = 16 bytes
# p_edge_list[0] = 0
# p_edge_list[1] = 1023
# edge_id: 1023 x 8 = 8184 bytes

with open("graph-homework/rmat-5.mst", 'rb') as f:
  content = f.read()
numTrees, numEdges, _, p_edge_list = struct.unpack('=IQQQ', content[:28])
print(f"numTrees: {numTrees}")
print(f"numEdges: {numEdges}")
print(f"p_edge_list[1]: {numEdges}")
edge_id = list(array.array('Q', content[28:]))
print(f"edge_id[{len(edge_id)} items]")
edge_id.sort()
# for i in edge_id:
#   print(i)

print(f"selected_list[{len(selected_list)} items]")
for i in range(len(selected_list)):
  print(edge_id[i], selected_list[i][1])