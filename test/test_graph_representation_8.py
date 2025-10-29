import subprocess
import re
import pytest

from student_code import TraversableDigraph
def test_dfs():
    graph = TraversableDigraph()
    
    # Adding nodes
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_node("D")
    graph.add_node("E")
    graph.add_node("F")
    
    # Adding edges
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    graph.add_edge("C", "D")
    graph.add_edge("D", "E")
    graph.add_edge("E", "F")
    graph.add_edge("B", "F")

    # Perform DFS from node "A"
    dfs_result = list(graph.dfs("A"))
    
    # Expected DFS orders: The order might vary but all nodes reachable from "A" should be included
    valid_dfs_orders = [
        ["B", "D", "E", "F", "C"],
        ["C", "D", "E", "F", "B"],
        ["B", "F", "C", "D", "E"],
        ["C", "D", "B", "F", "E"],
        ["B", "C", "D", "E", "F"],
        ["C", "B", "D", "E", "F"],
        ["B", "D", "C", "E", "F"],
        ["C", "E", "D", "B", "F"],
    ]
    
    assert dfs_result in valid_dfs_orders

    # Perform DFS from node "B"
    dfs_result = list(graph.dfs("B"))
    
    # Expected DFS orders from "B"
    valid_dfs_orders = [
        ["D", "E", "F"],
        ["F", "E", "D"],
        ["D", "F", "E"],
        ["F", "D", "E"]
    ]
    
    assert dfs_result in valid_dfs_orders
    
    # Perform DFS from node "C"
    dfs_result = list(graph.dfs("C"))
    
    # Expected DFS orders from "C"
    valid_dfs_orders = [
        ["D", "E", "F"],
        ["D", "B", "F", "E"],
        ["D", "F", "E"],
        ["D", "E", "B", "F"]
    ]
    
    assert dfs_result in valid_dfs_orders
    
    # Perform DFS from node "D"
    dfs_result = list(graph.dfs("D"))
    
    # Expected DFS orders from "D"
    valid_dfs_orders = [
        ["E", "F"]
    ]
    
    assert dfs_result in valid_dfs_orders
    
    # Perform DFS from node "E"
    dfs_result = list(graph.dfs("E"))
    
    # Expected DFS orders from "E"
    valid_dfs_orders = [
        ["F"]
    ]
    
    assert dfs_result in valid_dfs_orders
