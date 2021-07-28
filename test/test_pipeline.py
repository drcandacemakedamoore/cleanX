# -*- coding: utf-8 -*-

import os

from tempfile import TemporaryDirectory

from cleanX.image_work.pipeline import (
    Pipeline,
    Acquire,
    Save,
    DirectorySource,
)


image_directory = os.path.join(os.path.dirname(__file__), 'directory')


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
