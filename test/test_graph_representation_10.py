import subprocess
import re
import pytest

from student_code import DAG

def test_successors_predecessors():
    graph = DAG()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "C")
    assert graph.successors("A") == ["B", "C"]
    assert graph.predecessors("C") == ["A", "B"]
