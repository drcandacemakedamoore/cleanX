# -*- coding: utf-8 -*-

import os

from tempfile import TemporaryDirectory

import pytest

from cleanX.image_work.pipeline import (
    Pipeline,
    Acquire,
    Save,
    DirectorySource,
    Step,
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
        p = Pipeline(steps=(
            Acquire(),
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
        p = Pipeline(
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

        p = Pipeline.restore(journal_dir, skip=1)
        p.process(src)
        
        src_files = set(f for f in os.listdir(src_dir) if f.endswith('.jpg'))
        dst_files = set(os.listdir(td))
        assert src_files == dst_files
