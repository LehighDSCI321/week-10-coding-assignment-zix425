import subprocess
import re
import pytest

from student_code import TraversableDigraph

def test_bfs():
    graph = TraversableDigraph()
    
    # Adding nodes
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_node("D")
    graph.add_node("E")
    
    # Adding edges
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    graph.add_edge("C", "D")
    graph.add_edge("D", "E")
    
    # Perform BFS from node "A"
    bfs_result = list(graph.bfs("A"))
    
    # Expected BFS result: ["B", "C", "D", "E"] (Start node "A" should not be included in the result)
    assert bfs_result == ["B", "C", "D", "E"]
    
    # Adding more complex scenario with additional edges
    graph.add_edge("C", "E")
    graph.add_edge("B", "E")
    
    # Perform BFS from node "A" again
    bfs_result = list(graph.bfs("A"))
    
    # Expected BFS result: ["B", "C", "D", "E"] (Order might vary but all nodes reachable from "A" should be included)
    valid_bfs_results = [
        ["B", "C", "D", "E"],
        ["B", "C", "E", "D"],
        ["C", "B", "D", "E"],
        ["C", "B", "E", "D"]
    ]
    
    assert bfs_result in valid_bfs_results
    
    # Perform BFS from node "B"
    bfs_result = list(graph.bfs("B"))
    
    # Expected BFS result: ["D", "E"]
    assert bfs_result == ["D", "E"]
    
    # Perform BFS from node "C"
    bfs_result = list(graph.bfs("C"))
    
    # Expected BFS result: ["D", "E"]
    assert bfs_result == ["D", "E"]
