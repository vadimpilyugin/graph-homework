# Python3 port of http://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/UF.java.html
class DisjointSet:
  """
  Execution:    python disjoint_set.py < input.txt
  Data files:   http://algs4.cs.princeton.edu/15uf/tinyUF.txt
                  http://algs4.cs.princeton.edu/15uf/mediumUF.txt
                  http://algs4.cs.princeton.edu/15uf/largeUF.txt
  
  Weighted quick-union by rank with path compression.
  
  % python disjoint_set.py < tinyUF.txt
  4 3
  3 8
  6 5
  9 4
  2 1
  5 0
  7 2
  6 1
  2 components
  """
  def __init__(self, n):
    """Initializes an empty union–find data structure with n sites
    0 through n-1. Each site is initially in its own component.
    
    Args:
      n (int): the number of sites
    """
    # number of components
    self.count = n 
    # parent[i] = parent of i
    self.parent = list(range(n))
    # rank[i] = rank of subtree rooted at i (never more than 31)
    self.rank = [0] * n  
    
  def find(self, p):
    """Returns the component identifier for the component 
    containing site p.
    
    Args:
      p (int): the integer representing one site
    
    Returns:
      int. the component identifier for the component containing site p
    """
    self.validate(p)
    if p != self.parent[p]:
      self.parent[p] = self.find(self.parent[p]) # path compression
    return self.parent[p]

  
  def connected(self, p, q):
    """Returns true if the the two sites are in the same component.
    
    Args:
      p (int): the integer representing one site
      q (int): the integer representing the other site
      
    Returns:
      Boolean. true if the two sites p and q are in the same component;
        false otherwise.
    """
    return self.find(p) == self.find(q)
    
  def union(self, p, q):
    """Merges the component containing site p with the 
    the component containing site q.
    
    Args:
      p (int): the integer representing one site
      q (int): the integer representing the other site
    """
    rootP = self.find(p)
    rootQ = self.find(q)
    if rootP == rootQ: return
    
    # make root of smaller rank point to root of larger rank
    if   self.rank[rootP] < self.rank[rootQ]: self.parent[rootP] = rootQ
    elif self.rank[rootP] > self.rank[rootQ]: self.parent[rootQ] = rootP
    else:
      self.parent[rootQ] = rootP
      self.rank[rootP] += 1
    self.count -= 1

  def validate(self, p):
    """validate that p is a valid index"""
    n = len(self.parent)
    if p < 0 or p >= n:
      raise IndexError("index {} is not between 0 and {}".format(p, n - 1)) 

if __name__ == '__main__':
  n = int(input())
  ds = DisjointSet(n)
  for i in range(n):
    p, q = map(int, input().split())
    if ds.connected(p, q):
      continue
    ds.union(p, q)
    print(p, q)
  print(ds.count, "components")