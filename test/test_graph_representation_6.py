import subprocess
import re
import pytest

from student_code import DAG

def test_add_edge_cycle_detection():
    graph = DAG()
    # Adding nodes
    graph.add_node("A", 10)
    graph.add_node("B", 20)
    graph.add_node("C", 30)
    graph.add_node("D", 40)
    graph.add_node("E", 50)
    
    # Adding edges (no cycles)
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "D")
    graph.add_edge("D", "E")
    
    # Validate the current topological sort
    top_sort_result = graph.top_sort()
    assert top_sort_result in [
        ["A", "B", "C", "D", "E"],
        ["A", "B", "C", "D", "E"]  # Considering any valid topological sort
    ]
    
    # Adding more edges (still no cycles)
    graph.add_edge("A", "D")
    graph.add_edge("B", "E")
    
    # Validate the updated topological sort
    top_sort_result = graph.top_sort()
    assert top_sort_result in [
        ["A", "B", "C", "D", "E"],
        ["A", "B", "C", "D", "E"],
        ["A", "B", "D", "C", "E"],
        ["A", "D", "B", "C", "E"],  # Considering any valid topological sort
    ]
    
    # Adding edges that would create cycles and check for ValueError
    with pytest.raises(ValueError):
        graph.add_edge("E", "A")  # This should create a cycle
    with pytest.raises(ValueError):
        graph.add_edge("D", "A")  # This should create a cycle
    with pytest.raises(ValueError):
        graph.add_edge("C", "A")  # This should create a cycle
    
    # Further non-cyclic edge addition
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    
    # Validate the final topological sort
    top_sort_result = graph.top_sort()
    assert top_sort_result in [
        ["A", "B", "C", "D", "E"],
        ["A", "B", "D", "C", "E"],
        ["A", "D", "B", "C", "E"],  # Considering any valid topological sort
    ]
