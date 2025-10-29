import subprocess
import re
import pytest

from student_code import TraversableDigraph

def test_add_node():
    graph = TraversableDigraph()
    graph.add_node("A", 10)
    assert "A" in graph.get_nodes()
    assert graph.get_node_value("A") == 10
