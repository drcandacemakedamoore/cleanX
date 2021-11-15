# -*- coding: utf-8 -*-

import os

from tempfile import TemporaryDirectory

import pytest

from cleanX.image_work import (
    create_pipeline,
    restore_pipeline,
    Acquire,
    Save,
    DirectorySource,
    BlurEdges,
    Sharpie,
    PipelineError,
)


image_directory = os.path.join(os.path.dirname(__file__), 'directory')


class Fail(Step):

    def apply(self, image_data):
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
