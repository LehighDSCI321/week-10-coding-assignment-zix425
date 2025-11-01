""" TraversableDigraph, which inherits from SortableDigraph; and DAG"""

from typing import Dict, List, Tuple, Union
from collections import deque


class VersatileDigraph:
    """A flexible directed graph with nodes and edges management."""

    def __init__(self) -> None:
        """Initialize an empty directed graph."""
        self.nodes: Dict[str, Union[int, float]] = {}
        self.edges: Dict[Tuple[str, str], Tuple[str, Union[int, float]]] = {}

    # ---------- Node Management ----------
    def add_node(self, node_name: str, node_value: Union[int, float] = 0) -> None:
        """Add a node with a numeric value."""
        if not isinstance(node_name, str):
            raise TypeError("Node name must be a string.")
        if not isinstance(node_value, (int, float)):
            raise TypeError("Node value must be numeric.")
        if node_name in self.nodes:
            raise ValueError(f"Node '{node_name}' already exists.")
        self.nodes[node_name] = node_value

    def get_node_value(self, node_name: str) -> Union[int, float]:
        """Return the numeric value of the node."""
        if node_name not in self.nodes:
            raise KeyError(f"Node '{node_name}' not found.")
        return self.nodes[node_name]

    def get_nodes(self) -> List[str]:
        """Return list of all node names."""
        return list(self.nodes.keys())

    # ---------- Edge Management ----------
    def add_edge(
        self,
        source: str,
        destination: str,
        edge_name: str = "",
        edge_weight: Union[int, float] = 0,
    ) -> None:
        """Add a directed edge from source to destination."""
        if source not in self.nodes or destination not in self.nodes:
            raise KeyError("Source or destination node does not exist.")
        edge_key = (source, destination)
        if edge_key in self.edges:
            raise ValueError(f"Edge from '{source}' to '{destination}' already exists.")
        self.edges[edge_key] = (edge_name, edge_weight)

    def get_edge_weight(self, source: str, destination: str) -> Union[int, float]:
        """Return edge weight between two nodes."""
        key = (source, destination)
        if key not in self.edges:
            raise KeyError(f"Edge {source}->{destination} not found.")
        return self.edges[key][1]

    def predecessors(self, node_name: str) -> List[str]:
        """Return a list of predecessor nodes."""
        if node_name not in self.nodes:
            raise KeyError(f"Node '{node_name}' not found.")
        return [src for src, dst in self.edges if dst == node_name]

    def successors(self, node_name: str) -> List[str]:
        """Return a list of successor nodes."""
        if node_name not in self.nodes:
            raise KeyError(f"Node '{node_name}' not found.")
        return [dst for src, dst in self.edges if src == node_name]


class SortableDigraph(VersatileDigraph):
    """Directed acyclic graph supporting topological sorting."""

    def top_sort(self) -> List[str]:
        """Return a topologically sorted list of nodes using DFS recursion."""
        visited: Dict[str, bool] = {n: False for n in self.nodes}
        temp_mark: Dict[str, bool] = {n: False for n in self.nodes}
        result: List[str] = []

        def visit(node: str) -> None:
            """Recursive helper for topological sort."""
            if temp_mark[node]:
                raise ValueError("Graph contains a cycle.")
            if not visited[node]:
                temp_mark[node] = True
                for successor in self.successors(node):
                    visit(successor)
                temp_mark[node] = False
                visited[node] = True
                result.insert(0, node)

        for node in self.nodes:
            if not visited[node]:
                visit(node)
        return result


class TraversableDigraph(SortableDigraph):
    """A graph supporting both DFS and BFS traversals."""

    def dfs(self, start: str) -> List[str]:
        """Perform depth-first search and return list of reachable nodes (excluding start)."""
        if start not in self.nodes:
            raise KeyError(f"Start node '{start}' not found.")
        visited: List[str] = []
        seen = set()

        def visit(node: str) -> None:
            """Recursive DFS traversal."""
            seen.add(node)
            for successor in self.successors(node):
                if successor not in seen:
                    visited.append(successor)
                    visit(successor)

        visit(start)
        return visited

    def bfs(self, start: str):
        """Perform breadth-first search using a deque, yield nodes (excluding start)."""
        if start not in self.nodes:
            raise KeyError(f"Start node '{start}' not found.")
        visited = {start}
        queue = deque(self.successors(start))
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                yield node
                for successor in self.successors(node):
                    if successor not in visited:
                        queue.append(successor)


class DAG(TraversableDigraph):
    """Directed Acyclic Graph that prevents cycle creation."""

    def add_edge(
        self,
        source: str,
        destination: str,
        edge_name: str = "",
        edge_weight: Union[int, float] = 0,
    ) -> None:
        """Add edge only if it does not create a cycle."""
        if source not in self.nodes or destination not in self.nodes:
            raise KeyError("Source or destination node does not exist.")
        # Check for potential cycle: if destination can reach source
        for node in self.bfs(destination):
            if node == source:
                raise ValueError(
                    f"Adding edge {source}->{destination} would create a cycle."
                )
        super().add_edge(source, destination, edge_name, edge_weight)
