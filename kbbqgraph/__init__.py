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
    cast,
    byref,
)
import networkx as nx
import platform
import os
import numpy as np


dir_path = os.path.dirname(os.path.realpath(__file__))
bbqdll = None

if platform.system() == "Windows" and os.path.exists(
    os.path.join(dir_path, "bbqgraph.dll")
):
    bbqdll = CDLL(os.path.join(dir_path, "bbqgraph.dll"))
elif platform.system() == "Linux" and os.path.exists(
    os.path.join(dir_path, "bbqgraph.so")
):
    bbqdll = CDLL(os.path.join(dir_path, "bbqgraph.so"))


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


bbqdll.InitGraph.argtypes = [c_int, c_uint64]
bbqdll.InitGraph.restype = POINTER(Graph)

bbqdll.AddNodes.argtypes = [
    POINTER(Graph),
    c_uint32,
    POINTER(c_uint32),
    c_int64,
]
bbqdll.AddNodes.restype = None

bbqdll.AddNodesUnsafe.argtypes = [
    POINTER(Graph),
    c_uint32,
    POINTER(c_uint32),
    c_int64,
]
bbqdll.AddNodesUnsafe.restype = None

bbqdll.DeleteNodes.argtypes = [
    POINTER(Graph),
    POINTER(NodeId),
    c_int64,
]
bbqdll.DeleteNodes.restype = None


bbqdll.SubGraph.argtypes = [
    POINTER(Graph),
    POINTER(NodeId),
    c_int64,
]
bbqdll.SubGraph.restype = POINTER(Graph)

bbqdll.AddEdges.argtypes = [
    POINTER(Graph),
    POINTER(NodeId),
    POINTER(NodeId),
    c_uint64,
    POINTER(c_double),
    c_int,
]
bbqdll.AddEdges.restype = None

bbqdll.DeleteEdges.argtypes = [
    POINTER(Graph),
    POINTER(NodeId),
    POINTER(NodeId),
    c_uint64,
]
bbqdll.DeleteEdges.restype = None

bbqdll.CompressToUndirectedGraph.argtypes = [POINTER(Graph)]
bbqdll.CompressToUndirectedGraph.restype = None

bbqdll.ReIndex.argtypes = [POINTER(Graph)]
bbqdll.ReIndex.restype = None

bbqdll.Sort.argtypes = [POINTER(Graph)]
bbqdll.Sort.restype = None

bbqdll.FreeGraph.argtypes = [POINTER(Graph)]
bbqdll.FreeGraph.restype = None

bbqdll.FreeP.argtypes = [c_void_p]

bbqdll._findNode.argtypes = [POINTER(Graph), c_int32, c_int32]
bbqdll._findNode.restype = c_int64

bbqdll.WeaklyConnectedComponents.argtypes = [POINTER(Graph), POINTER(c_int64)]
bbqdll.WeaklyConnectedComponents.restype = POINTER(POINTER(NodeList))

bbqdll.FindAllMaximalCliques.argtypes = [POINTER(Graph), c_int64]
bbqdll.FindAllMaximalCliques.restype = Cliques


class KBBQGraph:
    def __init__(self, isDiGraph: int = 1, preAllocate: int = 0, graph=None, name=""):
        # Call the InitGraph function
        if graph is None:
            self.graph = bbqdll.InitGraph(isDiGraph, c_uint64(preAllocate))
        else:
            self.graph = graph
        self.name = name
        if not self.graph:
            raise MemoryError("Failed to initialize the graph.")

    def __del__(self):
        if self.graph:
            bbqdll.FreeGraph(self.graph)

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

    def Add_Nodes(self, datasetNum: int, ids: list, check_duplicates=True):
        numNodes = len(ids)
        ids_array = (c_uint32 * numNodes)(*ids)
        if check_duplicates:
            bbqdll.AddNodes(
                self.graph, c_uint32(datasetNum), ids_array, c_int64(numNodes)
            )
        else:
            # print("Adding unsafely with duplicates as False")
            bbqdll.AddNodesUnsafe(
                self.graph, c_uint32(datasetNum), ids_array, c_int64(numNodes)
            )

    def Delete_Nodes(self, nodes_to_delete: list):
        numNodes = len(nodes_to_delete)
        nodes_array = (NodeId * numNodes)(
            *(NodeId(dataset=s[0], id=s[1]) for s in nodes_to_delete)
        )
        bbqdll.DeleteNodes(self.graph, nodes_array, c_int64(numNodes))

    def Add_Edges(
        self,
        source_nodes: list,
        target_nodes: list,
        weights: list = [],
        check_duplicates=True,
    ):

        numEdges = len(source_nodes)
        source_array = (NodeId * numEdges)(
            *(NodeId(dataset=s[0], id=s[1]) for s in source_nodes)
        )
        target_array = (NodeId * numEdges)(
            *(NodeId(dataset=t[0], id=t[1]) for t in target_nodes)
        )
        if len(weights) == 0:
            weights = [0] * numEdges
        weights_array = (c_double * numEdges)(*weights)

        # Call the AddEdges function
        if check_duplicates:
            bbqdll.AddEdges(
                self.graph,
                source_array,
                target_array,
                c_uint64(numEdges),
                weights_array,
                0,
            )
        else:
            bbqdll.AddEdges(
                self.graph,
                source_array,
                target_array,
                c_uint64(numEdges),
                weights_array,
                1,
            )

    def Delete_Edges(self, source_nodes: list, target_nodes: list):
        numEdges = len(source_nodes)
        source_array = (NodeId * numEdges)(
            *(NodeId(dataset=s[0], id=s[1]) for s in source_nodes)
        )
        target_array = (NodeId * numEdges)(
            *(NodeId(dataset=t[0], id=t[1]) for t in target_nodes)
        )
        bbqdll.DeleteEdges(self.graph, source_array, target_array, c_uint64(numEdges))

    def Compress(self):
        bbqdll.CompressToUndirectedGraph(self.graph)

    def _print_node(self, node):
        # Print incoming edges
        print(f"{node.dataset}-{node.id}")
        if self.graph.contents.isDiGraph:
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

    def Print_Graph(self, print_num=None):
        if print_num is None:
            print_num = self.graph.contents.numNodes
        num_nodes = self.graph.contents.numNodes
        nodes_array = self.graph.contents.nodes

        for i in range(num_nodes):
            if i >= print_num:
                break
            node = nodes_array[i].contents
            self._print_node(node)

    def Print_Node(self, datasetNum: int, id: int):
        nodes_array = self.graph.contents.nodes

        node_index = bbqdll._findNode(self.graph, datasetNum, id)
        if node_index < 0:
            print(f"{datasetNum}-{id} not found...")
        node = nodes_array[node_index].contents
        self._print_node(node)

    def Sort(self):
        bbqdll.Sort(self.graph)

    def ReIndex(self):
        bbqdll.ReIndex(self.graph)

    def SubGraph(self, nodes: list):
        # Given node id's, returns a subgraph copy of the graph

        numNodes = len(nodes)
        nodes_array = (NodeId * numNodes)(
            *(NodeId(dataset=s[0], id=s[1]) for s in nodes)
        )
        graph = bbqdll.SubGraph(self.graph, nodes_array, c_int64(numNodes))
        return type(self)(
            0,  # not used if graph is supplied
            0,  # not used if graph is supplied
            graph,
            name="subgraph",
        )

    def weakly_connected_components(self) -> list:
        # Define the variable to store the number of components
        numComponents = c_int64(0)

        # Call the WeaklyConnectedComponents function
        components_ptr = bbqdll.WeaklyConnectedComponents(
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
            bbqdll.FreeP(component.contents.nodes)

            bbqdll.FreeP(component)  # frees *NodeList
            components_list.append(component_nodes)
        bbqdll.FreeP(components_ptr)
        return components_list

    def find_cliques(self, i) -> list:
        components = bbqdll.FindAllMaximalCliques(self.graph, c_int64(i))
        clique_list = []
        # iterate over Cliques->cliques
        for i in range(components.length):
            clique = components.cliques[i]
            clique_nodes = []
            for j in range(clique.length):
                node = clique.nodes[j]
                contents = node.contents
                clique_nodes.append([contents.dataset, contents.id])
            bbqdll.FreeP(clique.nodes)  # frees **nodes
            clique_list.append(clique_nodes)
        bbqdll.FreeP(components.cliques)  # frees Cliques.cliques -- *NodeList
        return clique_list


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

    bbq_nodes = []
    for node in nodes:
        # print(node)

        bbq_nodes.append(rename_node(node))

    bbq_edges = [[], [], []]
    for edge in edges:
        source = edge[0]
        target = edge[1]
        weight = edge[2]["score"]
        bbq_edges[0].append(rename_node(source))
        bbq_edges[1].append(rename_node(target))
        bbq_edges[2].append(weight)

    # for i in range(len(bbq_edges[0])):
    #     print(bbq_edges[0][i], bbq_edges[1][i], bbq_edges[2][i])

    for node in bbq_nodes:
        a.Add_Nodes(node[0], [node[1]], check_duplicates=True)

    a.Sort()
    a.ReIndex()
    # print(sources)
    # print(targets)
    # print(weights)
    a.Add_Edges(bbq_edges[0], bbq_edges[1], bbq_edges[2], check_duplicates=True)

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
    import pandas as pd
    from bmxp.eclipse import MSAligner
    from __init__ import KBBQGraph

    datasets = ["Col1.csv", "Col2.csv", "Col3.csv", "Col4.csv", "Col5.csv"]
    partites = {}
    num_nodes = 0

    for i, dataset in enumerate(datasets):
        ds = pd.read_csv(dataset)
        partites[i] = ds.index.tolist()
        num_nodes += len(ds)
    print(num_nodes)

    num_edges = 0
    edges = {i: {} for i in range(len(datasets))}
    for i, ds1 in enumerate(datasets):
        for j, ds2 in enumerate(datasets):
            if i == j:
                continue
            edge_csv = pd.read_csv(ds1 + ds2)
            edges[i][j] = (edge_csv.index.tolist(), edge_csv.iloc[:, 0].tolist())
            num_edges += len(edge_csv)
    print(num_edges)
    counted_edges = 0
    kgraph = KBBQGraph(preAllocate=num_nodes)
    for i in partites:
        kgraph.Add_Nodes(i, partites[i], False)
    kgraph.Sort()
    kgraph.ReIndex()
    for i in edges:
        for j in edges[i]:
            sources = [[i] * len(edges[i][j][0]), edges[i][j][0]]
            targets = [[j] * len(edges[i][j][0]), edges[i][j][1]]
            kgraph.Add_Edges(
                list(zip(*sources)), list(zip(*targets)), check_duplicates=False
            )
            counted_edges += len(sources[0])
            print(sources[0][0], sources[1][0], targets[0][0], targets[1][0])
            print(kgraph.num_edges, counted_edges)
