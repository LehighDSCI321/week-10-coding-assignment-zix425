import subprocess
import re
import pytest

from student_code import DAG

def test_add_edge():
    graph = DAG()
    graph.add_node("A", 10)
    graph.add_node("B", 20)
    graph.add_edge("A", "B", edge_weight=5)
    assert graph.get_edge_weight("A", "B") == 5
