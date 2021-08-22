# -*- coding: utf-8 -*-

import sys
import os
import subprocess

from tempfile import TemporaryDirectory
from glob import glob

import pandas as pd
import pytest

from cleanX.image_work import (
    Acquire,
    Save,
)

from harness import skip_if_missing


@skip_if_missing('no pydicom available', 'pydicom', 'click')
def test_cli_pydicom_report():
    dicomfile_directory1 = os.path.join(
        os.path.dirname(__file__),
        'dicom_example_folder',
    )
    with TemporaryDirectory() as td:
        result = subprocess.call(
            [
                sys.executable, '-m', 'cleanX',
                'dicom', 'report',
                '-i', 'dir', dicomfile_directory1,
                '-o', td,
            ]
        )
        assert not result
        df = pd.read_csv(os.path.join(td, 'report.csv'))
        assert 'source' in df.columns
        assert len(os.listdir(dicomfile_directory1)) == len(df)


@skip_if_missing('no pydicom available', 'pydicom', 'click')
def test_cli_pydicom_extract():
    dicomfile_directory1 = os.path.join(
        os.path.dirname(__file__),
        'dicom_example_folder',
    )
    with TemporaryDirectory() as td:
        result = subprocess.call(
            [
                sys.executable, '-m', 'cleanX',
                'dicom', 'extract',
                '-i', 'dir', dicomfile_directory1,
                '-o', td,
            ]
        )
        assert not result

        jpegs = sorted(glob(os.path.join(td, '*.jpg')))
        dicoms = sorted(os.listdir(dicomfile_directory1))
        assert len(jpegs) == len(dicoms)

        mismatches = []
        for jpg, dcm in zip(jpegs, dicoms):
            jpg = os.path.basename(jpg)
            jpg = os.path.splitext(jpg)[0]

            dcm = os.path.basename(dcm)
            dcm = os.path.splitext(dcm)[0]

            if jpg != dcm:
                mismatches.append((jpg, dcm))
        assert not mismatches


@skip_if_missing('no pydicom available', 'pydicom', 'click')
def test_cli_datasets():
    resources = os.path.dirname(__file__)
    with TemporaryDirectory() as td:
        result = subprocess.check_output(
            [
                sys.executable, '-m', 'cleanX',
                'dataset', 'report',
                '-r', os.path.join(resources, 'test_sample_df.csv'),
                '-t', os.path.join(resources, 'train_sample_df.csv'),
                # These two sets don't appear to have common columns
                '--no-report-leakage',
                '--no-report-bias',
            ]
        )
        result = result.decode()
        assert 'Duplicates' in result
        assert 'Knowledge' in result
        assert 'Value Counts on Sensitive Categories' not in result


@skip_if_missing('no click available', 'click')
def test_cli_create_pipeline():
    resources = os.path.dirname(__file__)
    with TemporaryDirectory() as td:
        result = subprocess.call(
            [
                sys.executable, '-m', 'cleanX',
                'images', 'run-pipeline',
                '-s', 'Acquire',
                '-s', 'Save(target={!r})'.format(td),
                '-j',
                '-r', os.path.join(resources, '*.jpg'),
            ]
        )
        assert not result
        src_files = set(f for f in os.listdir(resources) if f.endswith('.jpg'))
        dst_files = set(os.listdir(td))
        assert src_files == dst_files
