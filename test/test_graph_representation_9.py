import subprocess
import re
import pytest

from student_code import DAG

def test_top_sort():
    graph = DAG()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_node("D")
    graph.add_edge("A", "C")
    graph.add_edge("B", "C")
    graph.add_edge("C", "D")
    top_sort_result = graph.top_sort()
    
    # Check that the result is a valid topological sort
    valid_orders = [
        ["A", "B", "C", "D"],
        ["B", "A", "C", "D"]
    ]
    assert top_sort_result in valid_orders
