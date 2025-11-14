<h1>KBBQGraph - Barebones and Quick for k-partite graphs</h1>

This is a WIP for a memory-efficient replacement for NetworkX for a specific application.

I work with k-partite graphs when matching datasets, where each dataset is a partite. Most immediately, I need to handle 100s of partites each with 10,000's of nodes, and ~5,000 directed edges connecting each partite set. This quickly saturates NetworkX.

I also need efficient indexing based on partite number and feature number -- this is not offered by any other graphing library.

Someday, I would like to add some very specific homegrown k-partite clustering algorithms. These also assume one-to-one constraints when compressed to undirected, i.e. at most a node _u_ from partite _i_ can only be connected to one node from partite _j_.

License - Free to distribute and use as long as credit to the developers is given.
Also, if you make something better, please tell me.

In addition to the basic usage below, check out `compress`, `weakly_connected_components` and `find_cliques`.

```python
import networkx as nx
from kbbqgraph import KBBQGraph

# ------------------------------------------------------------
# Create a new graph
# ------------------------------------------------------------
g = KBBQGraph(isDiGraph=0, preAllocate=10, name="example")

# ------------------------------------------------------------
# Add nodes
# ------------------------------------------------------------
# Add nodes to “dataset/partite 0”
g.add_nodes(0, [0, 1, 2])

# Add nodes to “dataset/partite 1”
g.add_nodes(1, [0, 1])

# ------------------------------------------------------------
# Add edges (batched)
# ------------------------------------------------------------
# Each node is a (dataset, id) tuple, i.e. (partite, local_index)
source_nodes = [
    (0, 0),  # node 0 in dataset 0
    (0, 2),  # node 2 in dataset 0
]

target_nodes = [
    (1, 1),  # node 1 in dataset 1
    (1, 0),  # node 0 in dataset 1
]

weights = [1.0, 0.5]  # optional; can also pass None

g.add_edges(
    source_nodes=source_nodes,
    target_nodes=target_nodes,
    weights=weights,
    check_duplicates=True,
)

print("Nodes:", g.num_nodes)
print("Edges:", g.num_edges)

# ------------------------------------------------------------
# Convert to NetworkX
# ------------------------------------------------------------
nx_graph = g.to_networkx()

print("NetworkX edges (u, v, data):")
for u, v, data in nx_graph.edges(data=True):
    print(u, "→", v, "weight=", data.get("weight"))

```
