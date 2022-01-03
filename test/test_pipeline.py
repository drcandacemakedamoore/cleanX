# -*- coding: utf-8 -*-

import os

from tempfile import TemporaryDirectory
from multiprocessing import Queue

import pytest

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
    BlackEdgeCrop,
    WhiteEdgeCrop,
    Mean,
    StepCall,
    PipelineDef,
)
from cleanX.image_work.steps import CleanRotate, FourierTransf, ProjectionHorizoVert


image_directory = os.path.join(os.path.dirname(__file__), 'directory')


class Fail(Step):

    def apply(self, image_data, image_name):
        return None, RuntimeError('Test error')


def test_copy_images():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        goal = Save(td)
        p = create_pipeline(steps=(
            Acquire(),
            goal,
        ))
        p.process(src, goal)
        src_files = set(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
        dst_files = set(os.listdir(td))
        assert src_files == dst_files


def test_alter_images():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        pipeline_def = PipelineDef(
            steps={
                'src': StepCall(
                    definition=DirectorySource,
                    options=(('directory', src_dir),),
                    variables=(),
                    serial=True,
                    splitter=None,
                    joiner=None,
                ),
                'acquire': StepCall(
                    definition=Acquire,
                    options=(),
                    variables=('src',),
                    serial=True,
                    splitter=None,
                    joiner=None,
                ),
                'blur': StepCall(
                    definition=BlurEdges,
                    options=(),
                    variables=('acquire',),
                    serial=True,
                    splitter=None,
                    joiner=None,
                ),
                'sharpie': StepCall(
                    definition=Sharpie,
                    options=(),
                    variables=('blur',),
                    serial=True,
                    splitter=None,
                    joiner=None,
                ),
            },
            goal=StepCall(
                definition=Save,
                options=(('path', td),),
                variables=('sharpie',),
                serial=True,
                splitter=None,
                joiner=None,
            )
        )

        create_pipeline(pipeline_def)
        src_files = set(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
        dst_files = set(os.listdir(td))
        assert src_files == dst_files


def test_crop_images():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        p = create_pipeline(steps=(
            Acquire(),
            BlackEdgeCrop(),
            WhiteEdgeCrop(),
            Save(td),
        ))
        p.process(src)
        src_files = set(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
        dst_files = set(os.listdir(td))
        assert src_files == dst_files

def test_fourier_transf():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        p = create_pipeline(steps=(
            Acquire(),
            FourierTransf(),
            Save(td),
        ))
        p.process(src)
        src_files = set(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
        dst_files = set(os.listdir(td))
        assert src_files == dst_files

def test_projectionhorizovert():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        p = create_pipeline(steps=(
            Acquire(),
            ProjectionHorizoVert(),
            Save(td),
        ))
        p.process(src)
        src_files = set(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
        dst_files = set(os.listdir(td))
        assert src_files == dst_files

# def test_cleanrotate():
#       # test cancelled as function due to rewrite for process
#       # pool library update          
#     src_dir = image_directory
#     with TemporaryDirectory() as td:
#         src = DirectorySource(src_dir)
#         p = create_pipeline(steps=(
#             Acquire(),
#             CleanRotate(),
#             Save(td),
#         ))
#         p.process(src)
#         src_files = set(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
#         dst_files = set(os.listdir(td))
#         assert src_files == dst_files

def test_aggregate():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        goal = Save(td)
        p = create_pipeline(steps=(
            Acquire(),
            Mean(),
            goal,
        ))
        p.process(src, goal)
        assert len(os.listdir(td)) == 1


def test_grouphistohtwt():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        goal = GroupHistoHtWt(td)
        p = create_pipeline(steps=(
            Acquire(),
            goal,
        ))
        p.process(src, goal)
        assert len(os.listdir(td)) == 1


def test_grouphistopor():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        goal = GroupHistoPorportion(td)
        p = create_pipeline(steps=(
            Acquire(),
            goal,
        ))
        p.process(src, goal)
        assert len(os.listdir(td)) == 1


def test_journaling_pipeline():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        goal = Save(td)
        p = create_pipeline(
            steps=(
                Acquire(),
                Fail(),
                goal,
            ),
            journal=True,
            keep_journal=True,
        )

        journal_dir = p.journal_dir

        with pytest.raises(PipelineError):
            p.process(src, goal)

        p = restore_pipeline(journal_dir, skip=1)
        p.process(src, goal)

        src_files = set(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
        dst_files = set(os.listdir(td))
        assert src_files == dst_files
