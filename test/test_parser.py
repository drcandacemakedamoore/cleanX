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
        pipeline(
            definitions(
                dir = cleanX.image_work:DirectorySource
                glob = cleanX.image_work:GlobSource
                acquire = cleanX.image_work:Acquire
                or = cleanX.image_work.steps:Or
                crop = cleanX.image_work:Crop
                save = cleanX.image_work:Save
            )
            steps(
                source1 = dir[path = "/foo/bar"]()
                source2 = glob[pattern = "/foo/*.jpg"]()

                out1 out2 out3 = acquire[arg1 = "foo" arg2 = 42](
                    source1 source2
                )
                out4 = or[arg1 = true](out1 out2)
                out5 = crop(out3 out4)
            )
            goal(
                save[path = "/foo/bar"](out5)
            )
        )
    '''
    p = Parser()
    result = p.parse(pipeline)
    expected_vars = set((
        'source1',
        'source2',
        'out1',
        'out2',
        'out3',
        'out4',
        'out5',
    ))
    assert expected_vars == set(result.steps.keys())
    assert result.goal.definition is Save
