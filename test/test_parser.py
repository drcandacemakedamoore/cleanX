# -*- coding: utf-8 -*-

from io import StringIO

from cleanX.image_work import (
    create_pipeline,
    restore_pipeline,
    Acquire,
    Save,
    DirectorySource,
    Step,
    BlurEdges,
    Sharpie,
    PipelineError,
    Aggregate,
    GroupHistoHtWt,
    GroupHistoPorportion,
    Mean,
)
from cleanX.image_work.graph_parser import Parser


def test_parse_simple():
    pipeline = '''
    vertices: # inline comment
    1: Save("/foo/bar")
    2: Sharpie()
    3: BlurEdges()
    4: Acquire()
    arcs:
    # stand-alone comment
    # on two lines
    1 -> 2
    2 -> 3
    3 -> 4
    '''
    p = Parser(StringIO(pipeline))
    steps = tuple(p.parse())
    a, b = steps[0]
    c, d = steps[1]
    e, f = steps[2]
    assert b is c
    assert d is e
    assert isinstance(a, Save)
    assert isinstance(b, Sharpie)
    assert isinstance(d, BlurEdges)
    assert isinstance(f, Acquire)
