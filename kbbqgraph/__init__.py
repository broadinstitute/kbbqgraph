from ctypes import (
    CDLL,
    c_double,
    c_int32,
    c_void_p,
    c_int64,
    c_uint64,
    c_uint32,
    c_int,
    POINTER,
    Structure,
    byref,
    addressof,
)
from typing import Any, Iterable, List, Optional, Tuple

import platform
import os
import numpy as np

NodeTuple = Tuple[int, int]  # (dataset, id)

dir_path = os.path.dirname(os.path.realpath(__file__))
kbbqdll = None
__version__ = "0.1.1"

if platform.system() == "Windows" and os.path.exists(
    os.path.join(dir_path, "kbbqdll.dll")
):
    kbbqdll = CDLL(os.path.join(dir_path, "kbbqdll.dll"))

elif platform.system() == "Linux" and os.path.exists(
    os.path.join(dir_path, "kbbqdll.so")
):
    kbbqdll = CDLL(os.path.join(dir_path, "kbbqdll.so"))

elif platform.system() == "Darwin" and os.path.exists(
    os.path.join(dir_path, "kbbqdll.dylib")
):
    kbbqdll = CDLL(os.path.join(dir_path, "kbbqdll.dylib"))

else:
    raise FileNotFoundError("Could not locate kbbqdll shared library.")


class NodeId(Structure):  # pylint: disable=too-few-public-methods
    """
    C Struct for Scan
    """

    _fields_ = [
        ("dataset", c_int32),
        ("id", c_int32),
    ]


double_array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
int32_array = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
int64_array = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
p_node_id = POINTER(NodeId)


class NodeId(Structure):
    _fields_ = [("dataset", c_int32), ("id", c_int32)]


class WeightedEdge(Structure):
    pass


class Node(Structure):
    pass


WeightedEdge._fields_ = [("target", POINTER(Node)), ("weight", c_double)]

Node._fields_ = [
    ("dataset", c_int32),
    ("id", c_int32),
    ("index", c_int64),
    ("outgoing", POINTER(WeightedEdge)),
    ("incoming", POINTER(POINTER(Node))),
    ("undirected", POINTER(WeightedEdge)),
    ("outgoingLength", c_int64),
    ("incomingLength", c_int64),
    ("undirectedLength", c_int64),
]


class NodeList(Structure):
    _fields_ = [("nodes", POINTER(POINTER(Node))), ("length", c_int64)]


class Cliques(Structure):
    _fields_ = [("cliques", POINTER(NodeList)), ("length", c_int64)]


class Graph(Structure):  # pylint: disable=too-few-public-methods
    """
    C Struct for Scan
    """

    _fields_ = [
        ("nodes", POINTER(POINTER(Node))),
        ("numAllocated", c_int64),
        ("numNodes", c_int64),
        ("indexed", c_int),
        ("sorted", c_int),
        ("isDiGraph", c_int),
        ("numEdges", c_uint64),
    ]


kbbqdll.InitGraph.argtypes = [c_int, c_uint64]
kbbqdll.InitGraph.restype = POINTER(Graph)

kbbqdll.AddNodes.argtypes = [
    POINTER(Graph),
    c_uint32,
    POINTER(c_uint32),
    c_int64,
]
kbbqdll.AddNodes.restype = None

kbbqdll.AddNodesUnsafe.argtypes = [
    POINTER(Graph),
    c_uint32,
    POINTER(c_uint32),
    c_int64,
]
kbbqdll.AddNodesUnsafe.restype = None

kbbqdll.DeleteNodes.argtypes = [
    POINTER(Graph),
    POINTER(NodeId),
    c_int64,
]
kbbqdll.DeleteNodes.restype = None


kbbqdll.SubGraph.argtypes = [
    POINTER(Graph),
    POINTER(NodeId),
    c_int64,
]
kbbqdll.SubGraph.restype = POINTER(Graph)

kbbqdll.AddEdges.argtypes = [
    POINTER(Graph),
    POINTER(NodeId),
    POINTER(NodeId),
    c_uint64,
    POINTER(c_double),
    c_int,
]
kbbqdll.AddEdges.restype = None

kbbqdll.DeleteEdges.argtypes = [
    POINTER(Graph),
    POINTER(NodeId),
    POINTER(NodeId),
    c_uint64,
]
kbbqdll.DeleteEdges.restype = None

kbbqdll.CompressToUndirectedGraph.argtypes = [POINTER(Graph)]
kbbqdll.CompressToUndirectedGraph.restype = None

kbbqdll.ReIndex.argtypes = [POINTER(Graph)]
kbbqdll.ReIndex.restype = None

kbbqdll.Sort.argtypes = [POINTER(Graph)]
kbbqdll.Sort.restype = None

kbbqdll.FreeGraph.argtypes = [POINTER(Graph)]
kbbqdll.FreeGraph.restype = None

kbbqdll.FreeP.argtypes = [c_void_p]

kbbqdll._findNode.argtypes = [POINTER(Graph), c_int32, c_int32]
kbbqdll._findNode.restype = c_int64

kbbqdll.WeaklyConnectedComponents.argtypes = [POINTER(Graph), POINTER(c_int64)]
kbbqdll.WeaklyConnectedComponents.restype = POINTER(POINTER(NodeList))

kbbqdll.FindAllMaximalCliques.argtypes = [POINTER(Graph), c_int64]
kbbqdll.FindAllMaximalCliques.restype = Cliques


class KBBQGraph:
    def __init__(
        self,
        is_directed: int = 1,
        pre_allocate: int = 0,
        graph: Optional[Any] = None,
        name: str = "",
    ) -> None:
        """
        Initialize a new KBBQGraph instance or wrap an existing low-level graph pointer.

        Parameters
        ----------
        is_directed : int, optional
            Whether the graph is directed (1) or undirected (0).
            Passed directly to the underlying `InitGraph` function.
            Default is 1 (directed).
        pre_allocate : int, optional
            Number of nodes to pre-allocate space for. This hints the backend to
            reserve internal memory up front to reduce reallocations during node
            insertion. Default is 0 (no preallocation).
        graph : Any, optional
            An existing low-level graph handle/pointer returned from the C/C++ DLL.
            If provided, this instance will wrap it instead of calling `InitGraph`.
            Default is None.
        name : str, optional
            Optional human-readable name for the graph. Default is an empty string.

        Raises
        ------
        MemoryError
            If the underlying DLL fails to initialize a new graph object
            (only applicable when `graph` is None).
        """
        if graph is None:
            self.graph = kbbqdll.InitGraph(is_directed, c_uint64(pre_allocate))
        else:
            self.graph = graph

        self.name = name

        if not self.graph:
            raise MemoryError("Failed to initialize the graph.")

    def __del__(self):
        if self.graph:
            kbbqdll.FreeGraph(self.graph)

    def __len__(self):
        return self.graph.contents.numNodes

    @property
    def num_edges(self):
        return self.graph.contents.numEdges

    @property
    def num_nodes(self):
        return self.graph.contents.numNodes

    @property
    def is_sorted(self):
        return self.graph.contents.sorted

    @property
    def is_indexed(self):
        return self.graph.contents.indexed

    @property
    def is_directed(self):
        return self.graph.contents.isDiGraph

    def add_nodes(
        self,
        datasetNum: int,
        ids: Iterable[int],
        check_duplicates: bool = True,
    ) -> None:
        """
        Add a batch of nodes to a specific dataset/partite in the graph.

        Parameters
        ----------
        datasetNum : int
            Integer dataset/partite identifier into which nodes should be inserted.
            This maps directly to the `dataset` dimension in the underlying graph.
        ids : Iterable[int]
            Iterable of integer node IDs to add. These IDs are interpreted as
            belonging to `datasetNum`.
        check_duplicates : bool, optional
            If True (default), calls the safe version of the DLL method
            (`AddNodes`), which performs duplicate checks.
            If False, uses `AddNodesUnsafe` which does **not** check for duplicates
            and is faster but unsafe.

        Notes
        -----
        Internally, this converts the Python IDs into a ctypes `uint32` array
        before passing them to the C++ backend.
        """
        ids_list = list(ids)
        numNodes = len(ids_list)

        ids_array = (c_uint32 * numNodes)(*ids_list)

        if check_duplicates:
            kbbqdll.AddNodes(
                self.graph, c_uint32(datasetNum), ids_array, c_int64(numNodes)
            )
        else:
            kbbqdll.AddNodesUnsafe(
                self.graph, c_uint32(datasetNum), ids_array, c_int64(numNodes)
            )

    def delete_nodes(self, nodes_to_delete: Iterable[NodeTuple]) -> None:
        """
        Delete a batch of nodes from the graph.

        Parameters
        ----------
        nodes_to_delete : Iterable[Tuple[int, int]]
            Iterable of (dataset, id) tuples specifying the nodes to delete.

        Notes
        -----
        Constructs a ctypes array of `NodeId` structs before passing them to the
        DLL `DeleteNodes` function.
        """
        nodes_list = list(nodes_to_delete)
        numNodes = len(nodes_list)

        nodes_array = (NodeId * numNodes)(
            *(NodeId(dataset=ds, id=id_) for ds, id_ in nodes_list)
        )

        kbbqdll.DeleteNodes(self.graph, nodes_array, c_int64(numNodes))

    def add_edges(
        self,
        source_nodes: Iterable[NodeTuple],
        target_nodes: Iterable[NodeTuple],
        weights: Optional[Iterable[float]] = None,
        check_duplicates: bool = True,
    ) -> None:
        """
        Add weighted edges to the graph.

        Parameters
        ----------
        source_nodes : Iterable[Tuple[int, int]]
            Iterable of (dataset, id) tuples representing the source nodes.
        target_nodes : Iterable[Tuple[int, int]]
            Iterable of (dataset, id) tuples representing the target nodes.
        weights : Iterable[float], optional
            Iterable of edge weights. If None or empty, defaults to zero for each edge.
        check_duplicates : bool, optional
            If True (default), duplicate edges are checked and prevented by the
            backend. If False, edges are inserted without duplicate checks.

        Raises
        ------
        ValueError
            If input lists are not the same length.

        Notes
        -----
        Converts Python tuples into ctypes `NodeId` arrays and weights into
        `c_double` arrays before invoking the low-level DLL call.
        """
        sources = list(source_nodes)
        targets = list(target_nodes)

        if len(sources) != len(targets):
            raise ValueError("source_nodes and target_nodes must have the same length.")

        num_edges = len(sources)

        source_array = (NodeId * num_edges)(
            *(NodeId(dataset=s[0], id=s[1]) for s in sources)
        )
        target_array = (NodeId * num_edges)(
            *(NodeId(dataset=t[0], id=t[1]) for t in targets)
        )

        # Normalize weights
        if weights is None:
            weights = [0.0] * num_edges
        weights_list = list(weights)
        weights_array = (c_double * num_edges)(*weights_list)

        # Call the AddEdges function
        if check_duplicates:
            kbbqdll.AddEdges(
                self.graph,
                source_array,
                target_array,
                c_uint64(num_edges),
                weights_array,
                0,
            )
        else:
            kbbqdll.AddEdges(
                self.graph,
                source_array,
                target_array,
                c_uint64(num_edges),
                weights_array,
                1,
            )

    def delete_edges(
        self,
        source_nodes: Iterable[NodeTuple],
        target_nodes: Iterable[NodeTuple],
    ) -> None:
        """
        Delete a batch of edges from the graph.

        Parameters
        ----------
        source_nodes : Iterable[Tuple[int, int]]
            Iterable of (dataset, id) tuples for the source nodes of the edges
            to be deleted.
        target_nodes : Iterable[Tuple[int, int]]
            Iterable of (dataset, id) tuples for the target nodes of the edges
            to be deleted.

        Raises
        ------
        ValueError
            If `source_nodes` and `target_nodes` are not the same length.

        Notes
        -----
        This function constructs ctypes arrays of `NodeId` structures and forwards
        them to the backend DLL's `DeleteEdges` function, which removes matching
        directed edges. For undirected graphs, the backend is expected to handle
        symmetry.
        """
        sources = list(source_nodes)
        targets = list(target_nodes)

        if len(sources) != len(targets):
            raise ValueError("source_nodes and target_nodes must have the same length.")

        numEdges = len(sources)

        source_array = (NodeId * numEdges)(
            *(NodeId(dataset=s[0], id=s[1]) for s in sources)
        )
        target_array = (NodeId * numEdges)(
            *(NodeId(dataset=t[0], id=t[1]) for t in targets)
        )

        kbbqdll.DeleteEdges(self.graph, source_array, target_array, c_uint64(numEdges))

    def compress(self) -> None:
        """
        Convert a directed graph into an undirected one by merging reciprocal edges.

        Notes
        -----
        This operation is performed **in place** and modifies the current graph.

        The underlying DLL function `CompressToUndirectedGraph` typically:
            * Identifies pairs of edges u→v and v→u
            * Collapses them into a single undirected edge
            * Averages or combines their weights (implementation depends on the DLL)

        Raises
        ------
        RuntimeError
            If the backend reports a failure (only if the DLL implements error codes).
        """
        kbbqdll.CompressToUndirectedGraph(self.graph)

    def _print_node(self, node):
        # Print incoming edges
        print(f"{node.dataset}-{node.id}")
        if self.is_directed:
            # Print outgoing edges
            print("Outgoing:")
            if node.outgoingLength > 0:
                for j in range(node.outgoingLength):
                    outgoing_edge = node.outgoing[j]
                    outgoing_node = outgoing_edge.target.contents
                    print(
                        f"  -> {outgoing_node.dataset}-{outgoing_node.id} ({outgoing_edge.weight})"
                    )

            print("Incoming:")
            if node.incomingLength > 0:
                for j in range(node.incomingLength):
                    incoming_node = node.incoming[j].contents
                    print(f"  <- {incoming_node.dataset}-{incoming_node.id}")

        else:
            print("Undirected:")
            if node.undirectedLength > 0:
                for j in range(node.undirectedLength):
                    outgoing_edge = node.undirected[j]
                    outgoing_node = outgoing_edge.target.contents
                    print(
                        f"  -> {outgoing_node.dataset}-{outgoing_node.id} ({outgoing_edge.weight})"
                    )

    def print_graph(self, print_num=None):
        if print_num is None:
            print_num = self.graph.contents.numNodes
        num_nodes = self.graph.contents.numNodes
        nodes_array = self.graph.contents.nodes

        for i in range(num_nodes):
            if i >= print_num:
                break
            node = nodes_array[i].contents
            self._print_node(node)

    def print_node(self, datasetNum: int, id: int):
        nodes_array = self.graph.contents.nodes

        node_index = kbbqdll._findNode(self.graph, datasetNum, id)
        if node_index < 0:
            print(f"{datasetNum}-{id} not found...")
        node = nodes_array[node_index].contents
        self._print_node(node)

    def resort(self):
        """
        Sort the nodes based on datasetnum and nodeid, for faster binary search lookups.
        Adding nodes unsorts
        """
        kbbqdll.Sort(self.graph)

    def reindex(self):
        """
        resets the index for nodes, in case some were deleted
        """
        kbbqdll.ReIndex(self.graph)

    def subgraph(self, nodes: list):
        # Given node id's, returns a subgraph copy of the graph

        numNodes = len(nodes)
        nodes_array = (NodeId * numNodes)(
            *(NodeId(dataset=s[0], id=s[1]) for s in nodes)
        )
        graph = kbbqdll.SubGraph(self.graph, nodes_array, c_int64(numNodes))
        return type(self)(
            0,  # not used if graph is supplied
            0,  # not used if graph is supplied
            graph,
            name="subgraph",
        )

    def weakly_connected_components(self) -> list:
        """
        works with directed and undirected
        """
        # Define the variable to store the number of components
        numComponents = c_int64(0)

        # Call the WeaklyConnectedComponents function
        components_ptr = kbbqdll.WeaklyConnectedComponents(
            self.graph, byref(numComponents)
        )

        # # Prepare the result list
        components_list = []

        # # Iterate over each component
        for i in range(numComponents.value):
            component = components_ptr[i]
            component_nodes = []

            # Iterate over each node in the component
            # print("Component is ", component.contents.length)
            for j in range(component.contents.length):
                node = component.contents.nodes[j]
                contents = node.contents
                component_nodes.append([contents.dataset, contents.id])
            kbbqdll.FreeP(component.contents.nodes)

            kbbqdll.FreeP(component)  # frees *NodeList
            components_list.append(component_nodes)
        kbbqdll.FreeP(components_ptr)
        return components_list

    def connected_components(self) -> list:
        """
        convenience wrapper, works with directed and undirected
        """
        return self.weakly_connected_components()

    def find_cliques(self, i) -> list:
        components = kbbqdll.FindAllMaximalCliques(self.graph, c_int64(i))
        clique_list = []
        # iterate over Cliques->cliques
        for i in range(components.length):
            clique = components.cliques[i]
            clique_nodes = []
            for j in range(clique.length):
                node = clique.nodes[j]
                contents = node.contents
                clique_nodes.append([contents.dataset, contents.id])
            kbbqdll.FreeP(clique.nodes)  # frees **nodes
            clique_list.append(clique_nodes)
        kbbqdll.FreeP(components.cliques)  # frees Cliques.cliques -- *NodeList
        return clique_list

    def to_networkx(
        self, partite_names, node_names, delimiter="__", weight_name="weight"
    ):
        """
        Converts the internal KBBQ graph to a networkx graph.

        Parameters
        ----------
        partite_names : list or dict
            Names for each partite index (dataset index).
            Examples:
                {0: 'ds1', 1: 'ds2'}
                ['ds1', 'ds2']
        node_names : dict or list
            Mapping of partite index -> list/dict of node names.
            DO NOT USE DATASET NAMES, ONLY DATASET INDICES
            Examples:
                {0: ['alice', 'bob', 'charlie']}
                ['ds1_names_list', 'ds2_names_list']
            Access pattern is: node_names[dataset][node_id]
            If a name is missing, falls back to str(node_id).
        delimiter : str
            Separator for combining partite and node names into a NetworkX node label.
            Example: 'ds1__alice'
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("Optional dependency 'networkx' is not installed.")

        graph_struct = self.graph.contents

        # Choose directed or undirected NetworkX graph
        if self.is_directed:
            nx_graph = nx.DiGraph()
        else:
            nx_graph = nx.Graph()

        nodes_array = graph_struct.nodes
        num_nodes = int(graph_struct.numNodes)

        # Map C Node* address -> NetworkX node label
        ptr_to_label = {}

        # --- first pass: add nodes -----------------------------------------------
        for i in range(num_nodes):
            node_ptr = nodes_array[i]
            if not bool(node_ptr):
                continue  # skip null pointers, if any

            node = node_ptr.contents
            dataset_idx = int(node.dataset)
            node_id = int(node.id)

            if isinstance(partite_names, dict):
                try:
                    partite_label = partite_names[dataset_idx]
                except KeyError:
                    raise IndexError(f"Missing partite name for dataset {dataset_idx}")
            else:  # assume list-like
                try:
                    partite_label = partite_names[dataset_idx]
                except IndexError:
                    raise IndexError(
                        f"partite_names must have at least {dataset_idx + 1} entries"
                    )

            node_label = None
            if isinstance(node_names, dict):
                per_partite = node_names.get(dataset_idx)
                if per_partite is not None:
                    if isinstance(per_partite, dict):
                        node_label = per_partite.get(node_id)
                    else:  # assume list-like
                        if 0 <= node_id < len(per_partite):
                            node_label = per_partite[node_id]
            else:
                # assume list-like of per-partite containers
                if 0 <= dataset_idx < len(node_names):
                    per_partite = node_names[dataset_idx]
                    if isinstance(per_partite, dict):
                        node_label = per_partite.get(node_id)
                    else:
                        if 0 <= node_id < len(per_partite):
                            node_label = per_partite[node_id]

            if node_label is None:
                node_label = str(node_id)

            full_label = f"{partite_label}{delimiter}{node_label}"

            nx_graph.add_node(
                full_label,
                dataset=dataset_idx,
                id=node_id,
                index=int(node.index),
            )

            # use the C address as a stable key
            ptr_to_label[int(addressof(node))] = full_label

        # --- second pass: add edges ----------------------------------------------
        if self.is_directed:
            # Use outgoing edges only; incoming is redundant for building the graph.
            for i in range(num_nodes):
                node_ptr = nodes_array[i]
                if not bool(node_ptr):
                    continue

                node = node_ptr.contents
                u_label = ptr_to_label.get(int(addressof(node)))
                if u_label is None:
                    continue

                out_len = int(node.outgoingLength)
                if out_len <= 0 or not bool(node.outgoing):
                    continue

                edges_array = node.outgoing

                for j in range(out_len):
                    edge = edges_array[j]
                    if not bool(edge.target):
                        continue

                    target_node = edge.target.contents
                    v_label = ptr_to_label.get(int(addressof(target_node)))
                    if v_label is None:
                        continue

                    weight = float(edge.weight)
                    nx_graph.add_edge(u_label, v_label, **{weight_name: weight})

        else:
            # Undirected: deduplicate edges (adjacency likely stored on both ends)
            seen = set()

            for i in range(num_nodes):
                node_ptr = nodes_array[i]
                if not bool(node_ptr):
                    continue

                node = node_ptr.contents
                u_label = ptr_to_label.get(int(addressof(node)))
                if u_label is None:
                    continue

                undirected_len = int(node.undirectedLength)
                if undirected_len <= 0 or not bool(node.undirected):
                    continue

                edges_array = node.undirected

                for j in range(undirected_len):
                    edge = edges_array[j]
                    if not bool(edge.target):
                        continue

                    target_node = edge.target.contents
                    v_label = ptr_to_label.get(int(addressof(target_node)))
                    if v_label is None:
                        continue

                    # sort labels to get an undirected key
                    key = tuple(sorted((u_label, v_label)))
                    if key in seen:
                        continue
                    seen.add(key)

                    weight = float(edge.weight)
                    nx_graph.add_edge(u_label, v_label, **{weight_name: weight})

        return nx_graph


def test_graph():
    a = KBBQGraph(True, 100)
    nodes = [
        [n]
        for n in [
            "1__a",
            "2__a",
            "3__a",
            "4__a",
            "1__b",
            "2__b",
            "3__b",
            "4__b",
            "1__c",
        ]
    ]

    edges = [
        ("1__a", "2__a", {"score": 0.5, "rank": 0}),  # kept
        ("1__a", "3__a", {"score": 0.9, "rank": 0}),  # kept
        ("1__a", "4__a", {"score": 2, "rank": 0}),  # kept
        ("2__a", "1__a", {"score": 0.3, "rank": 0}),  # kept
        ("2__a", "3__a", {"score": 0.1, "rank": 0}),  # kept
        ("3__a", "1__a", {"score": 0.6, "rank": 0}),  # kept
        ("3__a", "2__a", {"score": 0.1, "rank": 0}),  # kept
        ("3__a", "4__a", {"score": 100, "rank": 0}),
        ("1__b", "2__a", {"score": 100, "rank": 0}),
        ("1__b", "3__b", {"score": 10, "rank": 0}),  # kept
        ("2__b", "1__b", {"score": 100, "rank": 0}),
        ("2__b", "3__a", {"score": 100, "rank": 0}),
        ("3__b", "1__b", {"score": 11, "rank": 0}),  # kept
        ("3__b", "2__b", {"score": 100, "rank": 0}),
        ("4__b", "2__b", {"score": 100, "rank": 0}),
        ("4__a", "1__a", {"score": 1, "rank": 0}),
        ("1__c", "4__a", {"score": 100, "rank": 0}),
    ]
    ds_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    idx_dict = {"a": 1, "b": 2, "c": 3, "d": 4}

    def rename_node(node):
        if isinstance(node, str):
            return [ds_dict[node.split("__")[0]], idx_dict[node.split("__")[1]]]
        return [ds_dict[node[0].split("__")[0]], idx_dict[node[0].split("__")[1]]]

    kbbq_nodes = []
    for node in nodes:
        # print(node)

        kbbq_nodes.append(rename_node(node))

    kbbq_edges = [[], [], []]
    for edge in edges:
        source = edge[0]
        target = edge[1]
        weight = edge[2]["score"]
        kbbq_edges[0].append(rename_node(source))
        kbbq_edges[1].append(rename_node(target))
        kbbq_edges[2].append(weight)

    # for i in range(len(kbbq_edges[0])):
    #     print(kbbq_edges[0][i], kbbq_edges[1][i], kbbq_edges[2][i])

    for node in kbbq_nodes:
        a.Add_Nodes(node[0], [node[1]], check_duplicates=True)

    a.Sort()
    a.ReIndex()
    # print(sources)
    # print(targets)
    # print(weights)
    a.Add_Edges(kbbq_edges[0], kbbq_edges[1], kbbq_edges[2], check_duplicates=True)

    # print("*****")
    # a.Sort()
    # a.ReIndex()

    # import random

    # random.seed(0)

    # sources = []
    # targets = []
    # weights = []
    # for source_dataset in nodes:
    #     for target_dataset in nodes:
    #         if source_dataset == target_dataset:
    #             continue
    #         for id in nodes[source_dataset]:
    #             if id in nodes[target_dataset]:
    #                 sources.append([source_dataset, id])
    #                 targets.append([target_dataset, id])
    #                 weights.append(random.random())

    # print(sources, targets, weights)
    # print(sources)
    # print(targets)
    # print(weights)

    # a.Add_Edges(sources, targets, weights)

    # a.Add_Edges([[2, 5]], [[3, 6]], [0.5])
    # a.Add_Edges([[2, 5]], [[3, 12]], [0.5])

    print("Full**********")
    a.Print_Graph()
    # print("**")

    # for i, component in enumerate(a.weakly_connected_components()):
    #     print(component)

    a.Compress()
    a.Sort()
    a.ReIndex()
    a.Print_Graph()
    # print("Compressed**********")
    # a.Print_Graph()
    # print("*******************")
    # # a.Print_Graph()/
    # # subgraphs = []

    for i, component in enumerate(a.find_cliques()):
        g = a.SubGraph(component)
        # subgraphs.append(a.SubGraph(component))
        print(component)
    print("Done")

    # # for i, subgraph in enumerate(subgraphs):
    # #     print("Subgraph ", i)
    # #     subgraph.Print_Graph()


def draw_graph(nx_graph):
    """
    Draws an NX Graph
    """
    pos = nx.spring_layout(nx_graph, seed=42)
    plt.figure(figsize=(10, 8))

    # Draw nodes
    nx.draw_networkx_nodes(
        nx_graph, pos, node_size=600, node_color="#4C72B0", alpha=0.9
    )

    # Draw edges
    nx.draw_networkx_edges(
        nx_graph,
        pos,
        arrowstyle="-|>" if nx_graph.is_directed() else "-",
        arrowsize=20,
        width=1.6,
        edge_color="#555555",
    )

    # Draw labels
    nx.draw_networkx_labels(nx_graph, pos, font_size=9, font_color="black")

    # Draw edge weights as small text
    edge_labels = {
        (u, v): f"{d['weight']:.2f}" for u, v, d in nx_graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        nx_graph, pos, edge_labels=edge_labels, font_color="red", font_size=7
    )

    plt.title("KBBQGraph → NetworkX Visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def graph_that_crashed():
    weights = []
    nodes = [
        "30-1",
        "30-866",
        "2-2867",
        "25-5173",
        "25-7367",
        "6-8890",
        "2-9310",
        "10-10888",
        "30-17986",
        "0-33681",
        "0-74522",
    ]

    sources = [
        # "30-1",
        # "30-1",
        # "30-1",
        # "30-1",
        # "30-1",
        # "2-2867",
        # "2-2867",
        # "2-2867",
        # "25-7367",
        # "25-7367",
        # "25-7367",
        # "25-7367",
        # "6-8890",
        # "6-8890",
    ]

    targets = [
        # "0-33681",
        # "10-10888",
        # "2-2867",
        # "25-7367",
        # "6-8890",
        # "0-33681",
        # "10-10888",
        # "6-8890",
        # "0-33681",
        # "10-10888",
        # "2-9310",
        # "6-8890",
        # "0-33681",
        # "10-10888",
    ]
    # weights = [1] * 14
    targets = [(int(item.split("-")[0]), int(item.split("-")[1])) for item in targets]
    sources = [(int(item.split("-")[0]), int(item.split("-")[1])) for item in sources]
    nodes = [(int(item.split("-")[0]), int(item.split("-")[1])) for item in nodes]

    return nodes, sources, targets, weights


if __name__ == "__main__":
    # Example using your KBBQGraph class

    # Create a directed graph with preallocation space
    import matplotlib.pyplot as plt
    import networkx as nx

    g = KBBQGraph(is_directed=1, pre_allocate=20, name="big_example")

    print("Initial:")
    print("  num_nodes =", g.num_nodes)
    print("  num_edges =", g.num_edges)
    print()

    # ---------------------------------------------------------------------
    # Add nodes to 4 different datasets (partites)
    # ---------------------------------------------------------------------

    # Dataset 0: 5 nodes
    g.add_nodes(datasetNum=0, ids=[0, 1, 2, 3, 4])

    # Dataset 1: 3 nodes
    g.add_nodes(datasetNum=1, ids=[0, 1, 2])

    # Dataset 2: 4 nodes
    g.add_nodes(datasetNum=2, ids=[0, 1, 2, 3])

    # Dataset 3: 2 nodes
    g.add_nodes(datasetNum=3, ids=[0, 1])

    print("After adding nodes:")
    print("  num_nodes =", g.num_nodes)  # should be 14
    print("  num_edges =", g.num_edges)
    print()

    # ---------------------------------------------------------------------
    # Add edges across partites
    # ---------------------------------------------------------------------

    # Connect dataset 0 → dataset 1
    sources_01 = [(0, 0), (0, 1), (0, 4)]
    targets_01 = [(1, 0), (1, 2), (1, 1)]
    weights_01 = [0.8, 0.55, 0.2]
    g.add_edges(sources_01, targets_01, weights_01)

    # Connect dataset 1 → dataset 2
    sources_12 = [(1, 0), (1, 2)]
    targets_12 = [(2, 3), (2, 1)]
    weights_12 = [1.0, 0.33]
    g.add_edges(sources_12, targets_12, weights_12)

    # Connect dataset 2 → dataset 3
    sources_23 = [(2, 0), (2, 3), (2, 1)]
    targets_23 = [(3, 1), (3, 0), (3, 1)]
    weights_23 = [0.99, 0.42, 0.11]
    g.add_edges(sources_23, targets_23, weights_23)

    # Add some criss-cross edges for complexity
    sources_mix = [(0, 2), (1, 1), (3, 0)]
    targets_mix = [(2, 1), (3, 1), (0, 3)]
    weights_mix = [0.77, 0.4, 0.25]
    g.add_edges(sources_mix, targets_mix, weights_mix)

    print("After adding edges:")
    print("  num_nodes =", g.num_nodes)
    print("  num_edges =", g.num_edges)
    print()

    # ---------------------------------------------------------------------
    # OPTIONAL: sort & reindex graph if your C library supports it
    # ---------------------------------------------------------------------
    # kbbqdll.Sort(g.graph)
    # kbbqdll.ReIndex(g.graph)

    print("Graph building complete.")

    # Names of the partites (datasets)
    partite_names = ["ds1", "ds2", "ds3", "ds4"]

    # Node names per dataset.
    # For this example we just name them n0, n1, n2, etc.
    node_names = {
        0: [f"n{i}" for i in range(5)],  # dataset 0 had 5 nodes
        1: [f"n{i}" for i in range(3)],  # dataset 1 had 3 nodes
        2: [f"n{i}" for i in range(4)],  # dataset 2 had 4 nodes
        3: [f"n{i}" for i in range(2)],  # dataset 3 had 2 nodes
    }

    # Convert to NetworkX
    nx_graph = g.to_networkx(
        partite_names=partite_names, node_names=node_names, delimiter="__"
    )

    print("Converted to NetworkX.")
    print(f"  NX nodes: {nx_graph.number_of_nodes()}")
    print(f"  NX edges: {nx_graph.number_of_edges()}")

    # Example: show a few edges
    print("\nSample edges:")
    for u, v, data in list(nx_graph.edges(data=True))[:10]:
        print(f"  {u}  →  {v}   weight={data.get('weight')}")
    # Layout — spring is usually the nicest
    pos = nx.spring_layout(nx_graph, seed=42)

    plt.figure(figsize=(10, 8))

    # Draw nodes
    nx.draw_networkx_nodes(
        nx_graph, pos, node_size=600, node_color="#4C72B0", alpha=0.9
    )

    # Draw edges
    nx.draw_networkx_edges(
        nx_graph,
        pos,
        arrowstyle="-|>" if nx_graph.is_directed() else "-",
        arrowsize=20,
        width=1.6,
        edge_color="#555555",
    )

    # Draw labels
    nx.draw_networkx_labels(nx_graph, pos, font_size=9, font_color="black")

    # Draw edge weights as small text
    edge_labels = {
        (u, v): f"{d['weight']:.2f}" for u, v, d in nx_graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        nx_graph, pos, edge_labels=edge_labels, font_color="red", font_size=7
    )

    plt.title("KBBQGraph → NetworkX Visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
