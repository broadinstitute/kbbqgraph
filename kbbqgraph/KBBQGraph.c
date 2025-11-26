// functions within leading underscore and lowercase name use index or raw pointer addresses
// Capital functions (like AddNodes) use NodeIDs
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct Node Node;

struct NodeId
{
  int32_t dataset;
  int32_t id;
} typedef NodeId;

struct WeightedEdge
{
  Node *target;
  double weight;
} typedef WeightedEdge;

struct Node
{
  int32_t dataset;
  int32_t id;
  int64_t index;
  WeightedEdge *outgoing; // used in DiGraph or MonoGraph
  Node **incoming;        // only used in DiGraph
  WeightedEdge *undirected;
  int64_t outgoingLength;
  int64_t incomingLength;
  int64_t undirectedLength;
};

typedef struct NodeList
{
  Node **nodes;   // Array of NodeIds in the component
  int64_t length; // Number of nodes in the component
} NodeList;

typedef struct Cliques
{
  NodeList *cliques;
  int64_t length;
} Cliques;

// typedef struct OldNodeList {
//   NodeId* nodes;   // Array of NodeIds in the component
//   uint64_t length; // Number of nodes in the component
// } OldNodeList;

struct Graph
{
  Node **nodes;
  int64_t numAllocated;
  int64_t numNodes;
  int indexed;
  int sorted;
  int isDiGraph;
  uint64_t numEdges;
} typedef Graph;

void FreeP(void *p) { free(p); }

int64_t _NodeIdxWeighted(WeightedEdge *list, int64_t length, Node *node)
{
  for (uint64_t i = 0; i < length; i++)
    if (list[i].target == node)
      return i; // Node is already in the list
  return -1;
}

// converts a node to a uint64_t node key
static inline uint64_t NodeToKey(const Node *n)
{
  return ((uint64_t)(uint32_t)n->dataset << 32) | (uint32_t)n->id;
}

// converts a NodeID struct to a uint64_t node key
static inline uint64_t NodeIdToKey(const NodeId n)
{
  return ((uint64_t)n.dataset << 32) | n.id;
}

// converts a NodeID struct to a uint64_t node key
static inline uint64_t NodeCoordsToKey(uint32_t dataset, uint32_t id)
{
  return ((uint64_t)dataset << 32) | id;
}

int _compareNodeId(const void *a, const void *b)
{
  const Node *nodeA = *(Node *const *)a;
  const Node *nodeB = *(Node *const *)b;

  uint64_t hashA = NodeToKey(nodeA);
  uint64_t hashB = NodeToKey(nodeB);

  if (hashA < hashB)
    return -1;
  if (hashA > hashB)
    return 1;
  return 0;
}

// Function to re-index (sort) the nodes in the graph
// adding nodes will set sorted=0
void Sort(Graph *graph)
{
  if (graph->nodes == NULL || graph->numNodes == 0)
  {
    return; // No nodes to sort
  }

  // Use qsort to sort the nodes based on their HashValue
  qsort(graph->nodes, graph->numNodes, sizeof(Node *), _compareNodeId);

  // Mark the graph as indexed
  graph->sorted = 1;
  graph->indexed = 0;
}

// updates each nodes index to its order in the graph->nodes
// sorting or deleting nodes will set indexed=0
void ReIndex(Graph *graph)
{
  if (graph->nodes == NULL || graph->numNodes == 0)
  {
    return; // No nodes to sort
  }
  for (int64_t i = 0; i < graph->numNodes; i++)
  {
    graph->nodes[i]->index = i;
  }

  // Mark the graph as indexed
  graph->indexed = 1;
}

// given a source and target, removes the node from the Source-outgoing and target incoming
void _removeDiEdge(Graph *graph, Node *source, Node *target)
{
  // Remove Target from Source->outgoing
  for (int64_t i = 0; i < source->outgoingLength; i++)
  {
    if (source->outgoing[i].target == target)
    {
      source->outgoing[i].target = source->outgoing[source->outgoingLength - 1].target;
      source->outgoing[i].weight = source->outgoing[source->outgoingLength - 1].weight;
      source->outgoingLength = source->outgoingLength - 1;
      source->outgoing = realloc(source->outgoing, (source->outgoingLength) * sizeof(WeightedEdge));
      graph->numEdges--;
      break;
    }
  }

  // remove Source from Target->incoming
  for (int64_t i = 0; i < target->incomingLength; i++)
  {
    if (target->incoming[i] == source)
    {
      target->incoming[i] = target->incoming[target->incomingLength - 1];
      target->incomingLength = target->incomingLength - 1;
      target->incoming = realloc(target->incoming, (target->incomingLength) * sizeof(Node *));
      break;
    }
  }
}

// given a source and target, edge from both source and target's undirected
void _removeUnEdge(Graph *graph, Node *source, Node *target)
{
  // Remove Target from Source->outgoing
  for (int64_t i = 0; i < source->undirectedLength; i++)
  {
    if (source->undirected[i].target == target)
    {
      source->undirected[i].target = source->undirected[source->undirectedLength - 1].target;
      source->undirected[i].weight = source->undirected[source->undirectedLength - 1].weight;
      source->undirectedLength = source->undirectedLength - 1;
      source->undirected = realloc(source->undirected, (source->undirectedLength) * sizeof(WeightedEdge));
      graph->numEdges--;
      break;
    }
  }

  // remove Source from Target->incoming
  for (int64_t i = 0; i < target->undirectedLength; i++)
  {
    if (target->undirected[i].target == source)
    {
      target->undirected[i].target = target->undirected[target->undirectedLength - 1].target;
      target->undirected[i].weight = source->undirected[target->undirectedLength - 1].weight;
      target->undirectedLength = target->undirectedLength - 1;
      target->undirected = realloc(target->undirected, (target->undirectedLength) * sizeof(WeightedEdge));
      break;
    }
  }
}

// Binary search for a node in a sorted array of nodes
int _binarySearch(Node **nodes, int64_t length, uint64_t nodeKey)
{
  int left = 0;
  int right = length - 1;
  while (left <= right)
  {
    int mid = left + (right - left) / 2;
    uint64_t midHash = NodeToKey(nodes[mid]);
    if (midHash == nodeKey)
      return mid; // Node found
    else if (midHash < nodeKey)
      left = mid + 1;
    else
      right = mid - 1;
  }
  return -1; // Node not found
}

// Helper function to delete a node from the graph
void _deleteNode(Graph *graph, int64_t nodeIndex)
{
  Node *nodeToDelete = graph->nodes[nodeIndex];

  // Delete the incoming edges from the nodes in the outgoing list
  for (int64_t i = 0; i < nodeToDelete->undirectedLength; i++)
  {
    Node *targetNode = nodeToDelete->outgoing[i].target;
    _removeDiEdge(graph, nodeToDelete, targetNode);
  }

  for (int64_t i = 0; i < nodeToDelete->incomingLength; i++)
  {
    Node *sourceNode = nodeToDelete->incoming[i];
    _removeDiEdge(graph, sourceNode, nodeToDelete);
  }

  // also remove from the undirected target lists
  for (int64_t i = 0; i < nodeToDelete->undirectedLength; i++)
  {
    Node *target = nodeToDelete->undirected[i].target;
    _removeUnEdge(graph, target, nodeToDelete);
  }

  // Free the node's name and its edges
  free(nodeToDelete->outgoing);
  free(nodeToDelete->incoming);
  free(nodeToDelete->undirected);

  size_t remainingSize = (graph->numNodes - nodeIndex - 1) * sizeof(Node *);
  if (remainingSize > 0)
  {
    memcpy(&graph->nodes[nodeIndex], &graph->nodes[nodeIndex + 1], remainingSize);
  }
  graph->indexed = 0;
  graph->numNodes--;
  graph->nodes = realloc(graph->nodes, graph->numNodes * sizeof(Node *));
  graph->numAllocated = graph->numNodes;
  free(nodeToDelete);
}

// Function to find the index of a node in the graph
int64_t _findNode(Graph *graph, uint32_t dataset, uint32_t id)
{
  // uint32_t nameHash = djb2(name);
  uint64_t combinedHash = NodeCoordsToKey(dataset, id);

  if (graph->sorted)
    return _binarySearch(graph->nodes, graph->numNodes, combinedHash);
  else
  {
    // Perform linear search if the graph is not indexed
    for (int64_t i = 0; i < graph->numNodes; i++)
      if (*(uint64_t *)graph->nodes[i] == combinedHash)
        return i; // Node found
    return -1;    // Node not found
  }
}

// Adds a directed edge to both source->outgoing and target->incoming
void _addDiEdge(Graph *graph, Node *source, Node *target, double weight, int unsafe)
{
  int64_t i;
  if (unsafe)
    i = -1;
  else
    i = _NodeIdxWeighted(source->outgoing, source->outgoingLength, target);
  if (i < 0)
  {
    // Add target node to source outgoing list
    source->outgoing = realloc(source->outgoing, (source->outgoingLength + 1) * sizeof(WeightedEdge));
    source->outgoing[source->outgoingLength].target = target;
    source->outgoing[source->outgoingLength].weight = weight;
    source->outgoingLength++;

    // add source node to target's incoming list
    target->incoming = realloc(target->incoming, (target->incomingLength + 1) * sizeof(Node *));
    target->incoming[target->incomingLength] = source;
    target->incomingLength++;

    graph->numEdges++;
  }
  else
  {
    // if already present, just update the weight and return
    source->outgoing[i].weight = weight;
    return;
  }
}

// Adds an undirected edge to both source and target
void _addUnEdge(Graph *graph, Node *source, Node *target, double weight, int unsafe)
{
  int64_t i;
  if (unsafe)
    i = -1;
  else
    i = _NodeIdxWeighted(source->undirected, source->undirectedLength, target);
  if (i < 0)
  {
    source->undirected =
        (WeightedEdge *)realloc(source->undirected, (source->undirectedLength + 1) * sizeof(WeightedEdge));
    source->undirected[source->undirectedLength].target = target;
    source->undirected[source->undirectedLength].weight = weight;
    source->undirectedLength++;

    target->undirected =
        (WeightedEdge *)realloc(target->undirected, (target->undirectedLength + 1) * sizeof(WeightedEdge));
    target->undirected[target->undirectedLength].target = source;
    target->undirected[target->undirectedLength].weight = weight;
    target->undirectedLength++;

    graph->numEdges++;
  }
  else
  {
    // if already present, just update the weight and return
    source->undirected[i].weight = weight;
    // find in targeted
    target->undirected[_NodeIdxWeighted(target->undirected, target->undirectedLength, source)].weight = weight;
    return;
  }
}

Graph *InitGraph(int isDiGraph, uint64_t preAllocate)
{
  Graph *graph = malloc(sizeof(Graph));
  graph->numAllocated = preAllocate;
  graph->sorted = 1;
  graph->indexed = 1;
  graph->isDiGraph = isDiGraph;
  graph->nodes = malloc(sizeof(Node *) * preAllocate);
  graph->numNodes = 0;
  graph->numEdges = 0;
  return graph;
}

// Add nodes and check for duplicates
void AddNodes(Graph *graph, uint32_t datasetNum, uint32_t *ids, int64_t numNodes)
{
  // make space for new nodes, sort, and check duplicates. If there are duplicates, it adds the last one
  Node **newNodes = malloc(sizeof(Node *) * numNodes);
  // printf("malloc'd %d..\n", numNodes);
  for (int64_t i = 0; i < numNodes; i++)
  {
    Node *newNode = malloc(sizeof(Node));
    newNodes[i] = newNode;
    newNode->index = 0;
    newNode->dataset = datasetNum;
    newNode->id = ids[i];
    newNode->outgoing = NULL;
    newNode->incoming = NULL;
    newNode->undirected = NULL;
    newNode->outgoingLength = 0;
    newNode->incomingLength = 0;
    newNode->undirectedLength = 0;
  }
  qsort(newNodes, numNodes, sizeof(Node *), _compareNodeId);

  if (!graph->sorted)
    Sort(graph);
  if (!graph->indexed)
    ReIndex(graph);

  // mark nodes for adding to database or freeing and calculate new space
  int64_t actualNewNodes = 0;
  for (int64_t i = 0; i < numNodes; i++)
  {
    uint64_t combinedHash = NodeToKey(newNodes[i]);
    if (_binarySearch(&newNodes[i + 1], numNodes - i - 1, combinedHash) < 0)
      if (_binarySearch(graph->nodes, graph->numNodes, combinedHash) < 0)
      {
        newNodes[i]->index = 1;
        actualNewNodes++;
      }
  }

  // printf("here..\n");
  // Reallocate memory for the new number of nodes
  if (graph->numNodes + actualNewNodes > graph->numAllocated)
  {
    graph->nodes = realloc(graph->nodes, (graph->numNodes + actualNewNodes) * sizeof(Node *));
    graph->numAllocated = graph->numNodes + actualNewNodes;
  }
  // printf("Allocated\n");
  // Add each node to the graph
  int64_t newNodeIdx = 0;
  for (int64_t i = 0; i < numNodes; i++)
  {
    Node *newNode = newNodes[i];
    // printf("node derefed\n");
    // if we marked the index as 1, add it. otherwise free
    if (newNode->index)
    {
      // printf("marking...");
      // printf(" %d %d %d\n", graph->numNodes, newNodeIdx);
      newNode->index = graph->numNodes + newNodeIdx;
      graph->nodes[graph->numNodes + newNodeIdx] = newNode;
      newNodeIdx++;
    }
    else
    {
      free(newNode);
    }
  }
  // printf("freeing..\n");
  free(newNodes);
  // Update the node length in the graph
  graph->numNodes += newNodeIdx;

  // Mark the graph as unsorted
  graph->sorted = 0;
}

// Add nodes without checking for duplicates
void AddNodesUnsafe(Graph *graph, uint32_t datasetNum, uint32_t *ids, int64_t numNodes)
{
  // Reallocate memory for the new number of nodes
  if (graph->numNodes + numNodes > graph->numAllocated)
  {
    graph->nodes = realloc(graph->nodes, (graph->numNodes + numNodes) * sizeof(Node *));
    graph->numAllocated = graph->numNodes + numNodes;
  }

  // Add each node to the graph
  for (int64_t i = 0; i < numNodes; i++)
  {
    Node *newNode = malloc(sizeof(Node));
    graph->nodes[graph->numNodes + i] = newNode;
    newNode->index = graph->numNodes;
    newNode->dataset = datasetNum;
    newNode->id = ids[i];
    newNode->outgoing = NULL;
    newNode->incoming = NULL;
    newNode->undirected = NULL;
    newNode->outgoingLength = 0;
    newNode->incomingLength = 0;
    newNode->undirectedLength = 0;
  }

  // Update the node length in the graph
  graph->numNodes += numNodes;
  // Mark the graph as unsorted
  graph->sorted = 0;
}

// Function to delete a list of nodes and their associated edges from the graph
void DeleteNodes(Graph *graph, NodeId *nodesToDelete, int64_t numNodes)
{
  for (int64_t i = 0; i < numNodes; i++)
  {
    int64_t nodeIndex = _findNode(graph, nodesToDelete[i].dataset, nodesToDelete[i].id);
    if (nodeIndex != -1)
      _deleteNode(graph, nodeIndex);
  }
}

// Add Edges, unsafe will skip duplicate check
void AddEdges(Graph *graph, NodeId *sourceNodes, NodeId *targetNodes, uint64_t numEdges, double *weights, int unsafe)
{
  if (!graph->sorted)
    Sort(graph);
  double weight = 0;
  for (uint64_t i = 0; i < numEdges; i++)
  {
    // if (sourceNodes[i].dataset == targetNodes[i].dataset) continue; // cannot add edges to own dataset
    int64_t sourceNodeIndex = _findNode(graph, sourceNodes[i].dataset, sourceNodes[i].id);
    int64_t targetNodeIndex = _findNode(graph, targetNodes[i].dataset, targetNodes[i].id);

    if (sourceNodeIndex >= 0 && targetNodeIndex >= 0 && targetNodeIndex != sourceNodeIndex)
    {
      if (weights != NULL)
        weight = weights[i];
      if (graph->isDiGraph)
        _addDiEdge(graph, graph->nodes[sourceNodeIndex], graph->nodes[targetNodeIndex], weight, unsafe);
      else
        _addUnEdge(graph, graph->nodes[sourceNodeIndex], graph->nodes[targetNodeIndex], weight, unsafe);
    }
    else
    {
      printf("Couldn't Find %d-%d or %d-%d (%d %d)\n", sourceNodes[i].dataset, sourceNodes[i].id, targetNodes[i].dataset, targetNodes[i].id, sourceNodeIndex, targetNodeIndex);
      return;
    }
  }
}

// Function to delete edges between a list of source nodes and target nodes
void DeleteEdges(Graph *graph, NodeId *sourceNodes, NodeId *targetNodes, uint64_t numEdges)
{
  if (!graph->sorted)
    Sort(graph);

  for (uint64_t i = 0; i < numEdges; i++)
  {
    int64_t sourceIndex = _findNode(graph, sourceNodes[i].dataset, sourceNodes[i].id);
    int64_t targetIndex = _findNode(graph, targetNodes[i].dataset, targetNodes[i].id);

    if (sourceIndex != -1 && targetIndex != -1)
    {
      Node *sourceNode = graph->nodes[sourceIndex];
      Node *targetNode = graph->nodes[targetIndex];
      // Remove the target node from the source node's targets
      _removeDiEdge(graph, sourceNode, targetNode);
      // Remove the source node from the target node's sources
      _removeUnEdge(graph, targetNode, sourceNode);
    }
  }
}

// Returns a copy of a subgraph
Graph *SubGraph(Graph *graph, NodeId *nodes, int64_t numNodes)
{
  if (!graph->sorted)
    Sort(graph);
  if (!graph->indexed)
    ReIndex(graph);
  // check nodes and get idxs
  int64_t toAllocate = 0;
  int64_t *ogIdxs = malloc(sizeof(int64_t) * numNodes);
  int64_t *newIdxs = malloc(sizeof(int64_t) * numNodes);
  for (int64_t i = 0; i < numNodes; i++)
  {
    int64_t ogIdx = _findNode(graph, nodes[i].dataset, nodes[i].id);
    if (ogIdx < 0)
    {
      ogIdxs[i] = -1;
      continue;
    }
    ogIdxs[i] = ogIdx;
    newIdxs[i] = toAllocate;
    toAllocate++;
  }
  Graph *newGraph = InitGraph(graph->isDiGraph, toAllocate);

  // make nodes
  for (int64_t i = 0; i < numNodes; i++)
  {
    int64_t ogIdx = ogIdxs[i];
    if (ogIdx < 0)
      continue;
    AddNodes(newGraph, nodes[i].dataset, (uint32_t[]){nodes[i].id}, 1);
  }

  Sort(newGraph);
  ReIndex(newGraph);
  int64_t newIdx = 0;

  // iterate through the new nodes
  for (int64_t i = 0; i < numNodes; i++)
  {
    int64_t ogIdx = ogIdxs[i];
    if (ogIdx < 0)
      continue;
    Node *ogSource = graph->nodes[ogIdx];
    Node *newSource = newGraph->nodes[_findNode(newGraph, nodes[newIdxs[newIdx]].dataset, nodes[newIdxs[newIdx]].id)];
    // iterate through the outgoing edges
    for (int64_t j = 0; j < ogSource->outgoingLength; j++)
    {
      Node *originalTarget = ogSource->outgoing[j].target;

      if (!graph->isDiGraph)
      {
        // to avoid duplicate processing, only process if source is less than target
        if (originalTarget->index <= ogIdx)
          continue;
        int64_t newTargetIdx = _findNode(newGraph, originalTarget->dataset, originalTarget->id);
        if (newTargetIdx < 0)
          continue;
        _addUnEdge(newGraph, newSource, newGraph->nodes[newTargetIdx], ogSource->outgoing[j].weight, 1);
      }
      else
      {
        // if it's directed, need to process every node
        int64_t newTargetIdx = _findNode(newGraph, originalTarget->dataset, originalTarget->id);
        if (newTargetIdx < 0)
          continue;
        _addDiEdge(newGraph, newSource, newGraph->nodes[newTargetIdx], ogSource->outgoing[j].weight, 1);
      }
    }
    newIdx++;
  }
  return newGraph;
}

// Function to compress the directed graph into an undirected graph, in place
void CompressToUndirectedGraph(Graph *graph)
{
  if (!graph->sorted)
    Sort(graph);
  if (!graph->indexed)
    ReIndex(graph);
  graph->numEdges = 0;
  // Iterate over each node in the graph
  for (int64_t i = 0; i < graph->numNodes; i++)
  {
    Node *node = graph->nodes[i];
    // for each node, scan the outgoing list
    for (int64_t j = 0; j < node->outgoingLength; j++)
    {
      // and see if there is a matching incoming node
      for (int64_t k = 0; k < node->incomingLength; k++)
      {

        // check if there is a match
        if (node->outgoing[j].target != node->incoming[k])
          continue;

        // make sure it hasn't been processed already
        Node *targetNode = node->outgoing[j].target;
        if (targetNode->index <= i)
          continue;
        double forwardWeight = node->outgoing[j].weight;

        // find the source node in the target list
        int64_t l = _NodeIdxWeighted(targetNode->outgoing, targetNode->outgoingLength, node);
        if (l > -1)
        {
          double reverseWeight = targetNode->outgoing[l].weight;
          double weight = (forwardWeight + reverseWeight);
          _addUnEdge(graph, node, targetNode, weight, 0);
        }
      }
    }
    free(node->incoming);
    free(node->outgoing);
    node->incoming = NULL;
    node->outgoing = NULL;
    node->incomingLength = 0;
    node->outgoingLength = 0;
  }
  graph->isDiGraph = false;
}

//
// Helper function to mark connected nodes as visited using DFS
void dfs(Graph *graph, Node *node, int *visited, NodeList *component)
{
  int64_t nodeIndex = node->index;

  if (visited[nodeIndex])
    return;

  visited[nodeIndex] = 1;

  // Add current node to the component
  component->nodes[component->length] = node;
  (component->length)++;

  // Traverse outgoing edges
  for (int64_t i = 0; i < node->outgoingLength; i++)
  {
    dfs(graph, node->outgoing[i].target, visited, component);
  }

  // Traverse incoming edges if it's a directed graph
  if (graph->isDiGraph)
  {
    for (int64_t i = 0; i < node->incomingLength; i++)
    {
      dfs(graph, node->incoming[i], visited, component);
    }
  }
}

// Function to find all weakly connected components in the graph
NodeList **WeaklyConnectedComponents(Graph *graph, int64_t *numComponents)
{
  if (!graph->sorted)
    Sort(graph);
  if (!graph->indexed)
    ReIndex(graph);
  *numComponents = 0;
  if (graph->numNodes == 0)
    return NULL;

  // temp holder for node list
  NodeList *component = malloc(sizeof(NodeList));
  component->nodes = malloc(graph->numNodes * sizeof(Node *));

  // the actual list
  NodeList **components = malloc(graph->numNodes * sizeof(NodeList *));

  // Initialize visited array to keep track of visited nodes
  int *visited = (int *)calloc(graph->numNodes, sizeof(int));

  for (int64_t i = 0; i < graph->numNodes; i++)
  {
    if (visited[i])
      continue;

    component->length = 0;
    dfs(graph, graph->nodes[i], visited, component);
    NodeList *newComponent = malloc(sizeof(NodeList));
    newComponent->nodes = malloc(component->length * sizeof(Node *));
    newComponent->length = component->length;
    memcpy(newComponent->nodes, component->nodes, component->length * sizeof(Node *));
    components[*numComponents] = newComponent;
    (*numComponents) += 1;
  }

  free(visited);
  free(component);
  components = realloc(components, sizeof(NodeList **) * (*numComponents));
  // printf("Found %d componenets...\n", *numComponents);
  return components;
}

void TestEdgeList(Graph *graph, bool print)
{
  uint32_t target_dataset;
  uint32_t target_id;
  uint32_t source_dataset;
  uint32_t source_id;
  uint32_t edgeIndex = 0;
  for (int64_t i = 0; i < graph->numNodes; i++)
  {
    Node *source = graph->nodes[i];
    source_dataset = source->dataset;
    source_id = source->id;

    if (print)
      printf("Node %d-%d (%p)\n", source->dataset, source->id, source);

    for (int64_t j = 0; j < source->outgoingLength; j++)
    {
      target_dataset = source->outgoing[j].target->dataset;
      target_id = source->outgoing[j].target->id;
      if (print)
      {
        printf("%d-%d -> ", source->dataset, source->id);
        printf("%d-%d (%p)\n", source->outgoing[j].target->dataset, source->outgoing[j].target->id,
               source->outgoing[j].target);
      }
      edgeIndex++;
    }

    for (int64_t j = 0; j < source->undirectedLength; j++)
    {
      target_dataset = source->undirected[j].target->dataset;
      target_id = source->undirected[j].target->id;
      if (print)
      {
        printf("%d-%d -> ", source->dataset, source->id);
        printf("%d-%d (%p)\n", source->undirected[j].target->dataset, source->undirected[j].target->id,
               source->undirected[j].target);
      }
      edgeIndex++;
    }
  }
  printf("Successfully derefed %d edges.\n", edgeIndex);
}
//
//
// printf(" %d<=%d ", cliques->length, graph->numNodes);
// for (uint64_t i = 0; i < depth; i++) {
//   printf("  ");
// }
// printf("Initial  ");
// printf("R (%d): {", rLen);
// for (uint64_t i = 0; i < rLen; i++) {
//   printf("%d-%d,", R[i]->dataset, R[i]->id);
// }
// printf("}, ");

// printf("P (%d): {", pLen);
// for (uint64_t i = 0; i < pLen; i++) {
//   printf("%d-%d,", P[i]->dataset, P[i]->id);
// }
// printf("}, ");

// printf("X (%d): {", xLen);
// for (uint64_t i = 0; i < xLen; i++) {
//   printf("%d-%d", X[i]->dataset, X[i]->id);
// }
// printf("}\n");

// printf("Setting clique %d to %d\n", cliques->length, cliques->length);
// for (uint64_t i = 0; i < depth; i++) {
//   printf("  ");
// }
// printf("Returning R as clique: ");
// printf("R: {");
// for (uint64_t i = 0; i < rLen; i++) {
//   printf("%d-%d,", newClique->nodes[i]->dataset, newClique->nodes[i]->id);
// }
// printf("}\n");

// for (uint64_t i = 0; i < *numCliques; i++) {
//   NodeList clique = (*cliques)[i]; // length
//   printf("Clique %d of length %d: {", i, clique.length);

//   //     for (uint64_t j = 0; j < clique.length; j++)
//   //     {
//   //         NodeId node = clique.nodes[j];
//   //         printf("%d,", node.dataset);
//   //     }
//   printf("}\n");
// }

//   // for (uint64_t i = 0; i < depth; i++)
// {
//     printf("  ");
// }
// printf("Selected pivot %d\n", pivot->dataset);

// find P nodes that are not neighbors of the pivot

// printf("Iterating through P-N(pivot): {", pivot->dataset);
// for (uint64_t i = 0; i < depth; i++) {
//   printf("  ");
// }
//
// for (uint64_t i = 0; i < depth; i++) {
//   printf("  ");
// }
// for (uint64_t i = 0; i < depth; i++) {
//   printf("  ");
// }
// printf("Selected Vertex %d\n", vertex->dataset);
// for (uint64_t i = 0; i < depth; i++) {
//   printf("  ");
// }
// printf("Iteration %d  ", i);
// printf("R (%d): {", rLen);
// for (uint64_t i = 0; i < rLen; i++) {
//   printf("%d,", R[i]->dataset);
// }
// printf("}, ");

// printf("P (%d): {", pLen);
// for (uint64_t i = 0; i < pLen; i++) {
//   printf("%d,", P[i]->dataset);
// }
// printf("}, ");

// printf("X (%d): {", xLen);
// for (uint64_t i = 0; i < xLen; i++) {
//   printf("%d,", X[i]->dataset);
// }
// printf("}\n");
// for (int i = 0; i < depth; i++) {
//   printf("  ");
// }
// printf("Allocated %p %p %p %p\n", rNew, pNew, xNew, pMinusNeighbors);
// for (int i = 0; i < depth; i++) {
//   printf("  ");
// }
// printf("P and X Updated    ");
// printf("P (%d): {", pLen);
// for (int i = 0; i < pLen; i++) {
//   printf("%d,", P[i]->dataset);
// }
// printf("}, ");

// printf("X (%d): {", xLen);
// for (int i = 0; i < xLen; i++) {
//   printf("%d", X[i]->dataset);
// }
// printf("}\n");

// for (int i = 0; i < depth; i++) {
//   printf("  ");
// }
// printf("Freeing %p %p %p %p\n", rNew, pNew, xNew, pMinusNeighbors);
// printf("Running...");
// printf("Iterating...");

// for (int i = 0; i < depth; i++) {
//   printf("  ");
// }
// printf("Freed...\n");
// for (int i = 0; i < depth; i++) {
//   printf("  ");
// }
// printf("Done with Iteration\n");
// Add node to X
// Clique Stuff is untested, and I dont even have a digraph->Ungraph yetNodeId*
// Helper function to check if two nodes are neighbors
int isNeighbor(Node *a, Node *b)
{
  for (int64_t i = 0; i < a->undirectedLength; i++)
  {
    if (a->undirected[i].target == b)
    {
      return 1;
    }
  }
  return 0;
}

// Recursive Bron-Kerbosch algorithm to find maximal cliques
void BronKerbosch(Graph *graph, Node **R, int64_t rLen, Node **P, int64_t pLen, Node **X, int64_t xLen,
                  Cliques *cliques, int64_t depth, int64_t externNum)
{

  // If P and X are empty, record R as a maximal clique
  if (pLen == 0 && xLen == 0)
  {
    NodeList newClique;
    newClique.length = rLen;
    newClique.nodes = (Node **)malloc(sizeof(Node *) * rLen);
    memcpy(newClique.nodes, R, sizeof(Node *) * rLen);
    // make more space if needed
    if (cliques->length >= graph->numNodes)
    {
      cliques->cliques = realloc(cliques->cliques, (cliques->length + 1) * sizeof(NodeList));
      if (!cliques->cliques)
      {
        printf("Memory could not be allocated.");
        exit(1);
      }
    }

    NodeList *clique = &cliques->cliques[cliques->length];
    clique->nodes = newClique.nodes;
    clique->length = newClique.length;
    cliques->length = cliques->length + 1;
    return;
  }

  // Choose a pivot (arbitrarily selecting the first node from P or X)
  Node *pivot = (pLen > 0) ? P[0] : X[0];
  Node **pMinusNeighbors = (Node **)malloc((pLen) * sizeof(Node *));
  int64_t pMN = 0;
  for (int64_t i = 0; i < pLen; i++)
  {
    Node *vertex = P[i];
    if (isNeighbor(pivot, vertex))
      continue;
    pMinusNeighbors[pMN] = P[i];
    pMN++;
  }

  // Exclude nodes that are neighbors of the pivot i.e. For each node in P that is not a neighbor
  for (int64_t i = 0; i < pMN; i++)
  {
    Node *vertex = pMinusNeighbors[i];
    int64_t pNewLen = 0, xNewLen = 0;

    // New R will have the pivot
    Node **rNew = malloc((rLen + 1) * sizeof(Node *));
    for (int64_t j = 0; j < rLen; j++)
      rNew[j] = R[j];
    rNew[rLen] = vertex;
    int64_t rNewLen = rLen + 1;

    // New P is the intersection of P (with nodes removed) and the neighbors
    Node **pNew = malloc(pLen * sizeof(Node *));
    for (int64_t j = 0; j < pLen; j++)
    {
      if (isNeighbor(vertex, P[j]))
      {
        pNew[pNewLen] = P[j];
        pNewLen++;
      }
    }

    // Allocate entire nodelist for X to avoid reallocs
    Node **xNew = (Node **)malloc((graph->numNodes) * sizeof(Node *));
    for (int64_t j = 0; j < xLen; j++)
    {
      if (isNeighbor(vertex, X[j]))
        xNew[xNewLen++] = X[j];
    }

    BronKerbosch(graph, rNew, rNewLen, pNew, pNewLen, xNew, xNewLen, cliques, depth + 1, externNum);

    xLen++;
    X[xLen - 1] = vertex;

    // replace it with the last node and decrement its length
    for (int64_t j = 0; j < pLen; j++)
    {
      if (vertex->id == P[j]->id && vertex->dataset == P[j]->dataset)
      {
        P[j] = P[pLen - 1];
        break;
      }
    }
    pLen--;

    free(rNew);
    free(pNew);
    free(xNew);
  }

  free(pMinusNeighbors);
}

Cliques FindAllMaximalCliques(Graph *graph, int64_t externNum)
{
  Cliques cliques = (Cliques){NULL, 0};
  if (graph->isDiGraph)
  {
    printf("Cliques are only available on undirected graphs...\n");
    return cliques;
  }

  // allocate enough space for maximum number of cliques
  cliques.cliques = (NodeList *)malloc(sizeof(NodeList) * (graph->numNodes + 1));

  for (int64_t i = 0; i < graph->numNodes + 1; i++)
  {
    NodeList *clique = &cliques.cliques[i];
    clique->nodes = NULL;
    clique->length = 0;
  }
  cliques.length = 0;

  Node **R = NULL; // Initially empty clique
  Node **P = (Node **)malloc((graph->numNodes + 1) * sizeof(Node *));
  Node **X = (Node **)malloc((graph->numNodes + 1) * sizeof(Node *));
  int64_t pLen = graph->numNodes;
  int64_t rLen = 0;
  int64_t xLen = 0;

  // Initialize P with all nodes
  for (int64_t i = 0; i < graph->numNodes; i++)
  {
    P[i] = graph->nodes[i];
  }

  // Start the Bron-Kerbosch algorithm
  BronKerbosch(graph, R, rLen, P, pLen, X, xLen, &cliques, 0, externNum);

  // Free temporary arrays
  free(P);
  free(X);
  free(R);

  return cliques;
}

void PrintIncoming(Graph *graph, uint32_t datasetNum, uint32_t id)
{
  int64_t index = _findNode(graph, datasetNum, id);
  if (index < 0)
    return;
  Node *node = graph->nodes[index];
  printf("Incoming: \n");
  for (int64_t i = 0; i < node->incomingLength; i++)
  {
    printf("%d-%d <- %d-%d\n", node->dataset, node->id, node->incoming[i]->dataset, node->incoming[i]->id);
  }
}

void PrintOutgoing(Graph *graph, uint32_t datasetNum, uint32_t id)
{
  int64_t index = _findNode(graph, datasetNum, id);
  if (index < 0)
    return;
  Node *node = graph->nodes[index];
  printf("Outgoing: \n");

  for (int64_t i = 0; i < node->outgoingLength; i++)
  {
    printf("%d-%d -> %d-%d (%f)\n", node->dataset, node->id, node->outgoing[i].target->dataset,
           node->outgoing[i].target->id, node->outgoing[i].weight);
  }
}

void FreeGraph(Graph *graph)
{
  for (int64_t i = 0; i < graph->numNodes; i++)
  {

    Node *node = graph->nodes[i];
    free(node->incoming);
    free(node->outgoing);
    free(node->undirected);
    free(node);
  }
  free(graph->nodes);
  free(graph);
}

void PrintNode(Graph *graph, uint32_t datasetNum, uint32_t id)
{

  int64_t nodeIndex = _findNode(graph, datasetNum, id);
  if (nodeIndex < 0)
  {
    printf("Node %d-%d not found...\n", datasetNum, id);
    return;
  }
  printf("Node %d-%d, Index: %d, Address: %p\n", datasetNum, id, nodeIndex, graph->nodes[nodeIndex]);
  PrintOutgoing(graph, datasetNum, id);
  PrintIncoming(graph, datasetNum, id);
}

void shuffle(uint32_t *array, size_t n)
{
  if (n > 1)
  {
    for (size_t i = n - 1; i > 0; i--)
    {
      size_t j = rand() % (i + 1);
      uint32_t temp = array[i];
      array[i] = array[j];
      array[j] = temp;
    }
  }
}

uint32_t *randomNumbers(uint32_t min, uint32_t max, size_t count)
{
  if (max - min + 1 < count)
  {
    printf("Range is too small to generate the required number of unique random numbers.\n");
    return NULL;
  }

  uint32_t *numbers = malloc((max - min + 1) * sizeof(uint32_t));
  if (!numbers)
  {
    printf("Memory allocation failed.\n");
    return NULL;
  }

  // Fill the array with all possible values within the range
  for (uint32_t i = 0; i <= max - min; i++)
  {
    numbers[i] = min + i;
  }

  // Shuffle the array
  shuffle(numbers, max - min + 1);

  // Allocate an array to hold the result
  uint32_t *result = malloc(count * sizeof(uint32_t));
  if (!result)
  {
    printf("Memory allocation failed.\n");
    free(numbers);
    return NULL;
  }

  // Copy the first `count` numbers from the shuffled array to the result
  for (size_t i = 0; i < count; i++)
  {
    result[i] = numbers[i];
  }

  free(numbers);
  return result;
}

#define NUM_FEATURES 1000
#define NUM_DATASETS 10
#define NUM_EDGES (NUM_FEATURES / 2)

void main()
{

  // random graph
  printf("Starting...\n");
  double time_taken;

  // generate random data
  Graph *graph = InitGraph(1, NUM_FEATURES * NUM_DATASETS);
  uint32_t data[NUM_FEATURES];
  for (int64_t i = 0; i < NUM_FEATURES; i++)
  {
    data[i] = i;
  }

  // add nodes
  clock_t t;
  t = clock();
  uint32_t *datasets = randomNumbers(0, NUM_DATASETS - 1, NUM_DATASETS);
  for (uint32_t i = 0; i < NUM_DATASETS; i++)
  {
    uint32_t *ids = randomNumbers(0, NUM_FEATURES - 1, NUM_FEATURES);
    AddNodes(graph, datasets[i], ids, NUM_FEATURES);
    free(ids);
  }
  free(datasets);

  for (int64_t i = 0; i < graph->numNodes; i++)
  {
    Node node = *graph->nodes[i];
  }
  t = clock() - t;
  time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
  printf("%d Nodes in %f seconds.\n", graph->numNodes, time_taken);

  // index benchmark
  t = clock();
  ReIndex(graph);
  t = clock() - t;
  time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
  printf("Sorted in %f seconds.\n", time_taken);

  t = clock();
  for (uint32_t i = 0; i < NUM_DATASETS; i++)
  {
    printf("Adding dataset %d edges...\n", i);
    for (uint32_t j = 0; j < NUM_DATASETS; j++)
    {
      if (j == i)
        continue;

      // add edges
      NodeId sources[NUM_EDGES];
      NodeId targets[NUM_EDGES];
      uint32_t *sourceIdxs = randomNumbers(0, NUM_FEATURES - 1, NUM_EDGES);
      uint32_t *targetIdxs = randomNumbers(0, NUM_FEATURES - 1, NUM_EDGES);
      for (uint32_t k = 0; k < NUM_EDGES; k++)
      {
        sources[k].id = sourceIdxs[k];
        targets[k].id = targetIdxs[k];
        sources[k].dataset = i;
        targets[k].dataset = j;
        // printf("Adding %d %d -> %d %d...\n", sources[k].dataset, sources[k].id, targets[k].dataset, targets[k].id);
      }

      AddEdges(graph, sources, targets, NUM_EDGES, NULL, 0);
      free(sourceIdxs);
      free(targetIdxs);
    }
  }
  t = clock() - t;
  time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
  printf("%ld edges added in %f.\n", graph->numEdges, time_taken);

  // iterate, delete, then iterate again
  // TestEdgeList(graph, true);
  // PrintNode(graph, 1, 1);
  // NodeId toDelete[] = {{1, 1}};
  // DeleteNodes(graph, toDelete, 1);
  // TestEdgeList(graph, false);
  // PrintNode(graph, 2, 3);
  printf("New Graph:\n");
  Graph *one_one = SubGraph(graph, (NodeId[]){{1, 1}, {0, 6}, {2, 3}, {3, 0}}, 4);
  int64_t numComponenents;
  t = clock();
  NodeList **components = WeaklyConnectedComponents(graph, &numComponenents);

  t = clock() - t;
  time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
  printf("Found in %f.\n", time_taken);

  // TestEdgeList(one_one, true);
  // PrintNode(one_one, 1, 1);
  printf("\nDone.");

  // // small graph
  // //
  // //

  // Graph *graph = InitGraph(1, 0);

  // AddNodes(graph, 0, (uint32_t[2]){1, 2}, 2);
  // AddNodes(graph, 10, (uint32_t[2]){11, 12}, 2);
  // AddNodes(graph, 20, (uint32_t[2]){21, 22}, 2);
  // AddNodes(graph, 30, (uint32_t[]){31, 32, 33, 34, 35, 36, 37, 38, 39, 40}, 10);
  // AddNodes(graph, 40, (uint32_t[]){31, 32, 33, 34, 35, 36, 37, 38, 39, 40}, 10);
  // NodeId sources[] = {
  //     {0, 1},
  //     {0, 1},
  //     {0, 2},
  //     {10, 11},
  //     {10, 11},
  //     {20, 21},

  // };

  // NodeId targets[] = {
  //     {10, 12},
  //     {20, 21},
  //     {10, 11},
  //     {0, 2},
  //     {20, 21},
  //     {0, 1},
  // };
  // AddEdgesByNames(graph, sources, targets, 6, NULL);
  // getEdgeList(graph);
  // printf("Done 1.\n*****\n");
  // deleteNodes(graph, (NodeId[1]){10, 12}, 1);
  // printf("*****\n");
  // getEdgeList(graph);
  // printf("Done 2.");
}
