# G->n [1024]: 4 bytes
# G->m [32768]: 8 bytes
# G->directed [0]: 1 bytes
# align [0]: 1 bytes
# G->rowsIndices: 8 bytes x 1025 items = 8200 bytes total
# G->endV: 4 bytes x 32768 items = 131072 bytes total
# G->weights: 8 bytes x 32768 items = 262144 bytes total

import struct

with open("reference/rmat-10", 'rb') as f:
  content = f.read()
n = struct.unpack("@I", content[:4])[0]
m = struct.unpack("@L", content[4:12])[0]
directed = struct.unpack("@?", content[12:13])[0]
align = struct.unpack("@B", content[13:14])[0]

binary_end = 14+8*(n+1)
rowsIndices = [0 for i in range(n+1)]
binary_items = content[14:binary_end]
j = 0

for i in range(n+1):
  rowsIndices[i] = struct.unpack("@L", binary_items[j:j+8])[0]
  j = j + 8
  # print(f"rowsIndices[{i}] = {rowsIndices[i]}")

endV = [0 for i in range(m)]
new_end = binary_end + (4 * m)
binary_items = content[binary_end:new_end]
binary_end = new_end
j = 0

for i in range(m):
  endV[i] = struct.unpack("@I", binary_items[j:j+4])[0]
  j = j + 4
  # print(f"endV[{i}] = {endV[i]}")

weights = [0 for i in range(m)]
new_end = binary_end + 8 * m
binary_items = content[binary_end:new_end]
binary_end = new_end
j = 0

for i in range(m):
  weights[i] = struct.unpack("@d", binary_items[j:j+8])[0]
  j = j + 8
  # print(f"weights[{i}] = {weights[i]}")

print(f"n = {n}\nm = {m}\ndirected = {directed}\n", end="")
print(f"align = {align}\n", end="")
print(f"rowsIndices: {len(rowsIndices)} items")
print(f"endV: {len(endV)}")
print(f"weights: {len(weights)}")
print()
print("rowsIndices:")
for i in rowsIndices[:10]:
  print(i)
print("endV:")
for i in endV[:10]:
  print(i)
print("weights:")
for i in weights[:10]:
  print(i)
print()

def reroot(parent):
  for i in range(len(parent)):
    j = i
    while parent[j] != j:
      j = parent[j]
    k = i
    while k != j:
      tmp = parent[k]
      parent[k] = j
      k = tmp

parent = [i for i in range(n)]
n_components = n

while True:
  s = input(f"Vertex No(max {n}, min 1): ")
  if s == "choose":
    for i in range(n):
      ind = (rowsIndices[i], rowsIndices[i+1])
      min_edge = None
      min_weight = None
      for j in range(ind[0], ind[1]):
        if min_edge == None or weights[j] < min_weight:
          min_edge = j
          min_weight = weights[j]
      print(f"{i+1}. Vertex chooses {endV[min_edge]} [weight {min_weight:.3f}]")
      if parent[i] != parent[endV[min_edge]]:
        n_components -= 1
        parent[i] = parent[endV[min_edge]]
    print(f"Number of components left: {n_components}")
    reroot(parent)
    print()
  else:
    vertex_no = int(s)-1
    ind = (rowsIndices[vertex_no], rowsIndices[vertex_no+1])
    for i in range(ind[0], ind[1]):
      print(f"Connected to {endV[i]} [weight {weights[i]:.3f}]")
    print()
    print(f"Connected to {ind[1] - ind[0]} vertices")