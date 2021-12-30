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
)
# from cleanX.image_work.steps import BlackEdgeCrop, WhiteEdgeCrop


image_directory = os.path.join(os.path.dirname(__file__), 'directory')


class Fail(Step):

    def apply(self, image_data, image_name):
        return None, RuntimeError('Test error')


def test_copy_images():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        p = create_pipeline(steps=(
            Acquire(),
            Save(td),
        ))
        p.process(src)
        src_files = set(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
        dst_files = set(os.listdir(td))
        assert src_files == dst_files


def test_alter_images():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        p = create_pipeline(steps=(
            Acquire(),
            BlurEdges(),
            Sharpie(),
            Save(td),
        ))
        p.process(src)
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
            # ROAR WhiteEdgeCrop(),
            Save(td),
        ))
        p.process(src)
        src_files = set(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
        dst_files = set(os.listdir(td))
        assert src_files == dst_files

def test_aggregate():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        p = create_pipeline(steps=(
            Acquire(),
            Mean(),
            Save(td),
        ))
        p.process(src)
        assert len(os.listdir(td)) == 1

def test_grouphistohtwt():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        p = create_pipeline(steps=(
            Acquire(),
            GroupHistoHtWt(td),
            #Save(td),
        ))
        p.process(src)
        assert len(os.listdir(td)) == 1

def test_grouphistopor():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        p = create_pipeline(steps=(
            Acquire(),
            GroupHistoPorportion(td),
            #Save(td),
        ))
        p.process(src)
        assert len(os.listdir(td)) == 1


def test_journaling_pipeline():
    src_dir = image_directory
    with TemporaryDirectory() as td:
        src = DirectorySource(src_dir)
        p = create_pipeline(
            steps=(
                Acquire(),
                Fail(),
                Save(td),
            ),
            journal=True,
            keep_journal=True,
        )

        journal_dir = p.journal_dir

        with pytest.raises(PipelineError):
            p.process(src)

        p = restore_pipeline(journal_dir, skip=1)
        p.process(src)

        src_files = set(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
        dst_files = set(os.listdir(td))
        assert src_files == dst_files
