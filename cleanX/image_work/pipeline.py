# -*- coding: utf-8 -*-

import os
import shutil
import logging
import multiprocessing

from concurrent.futures import ProcessPoolExecutor
from tempfile import TemporaryDirectory
from glob import glob

import numpy as np
import cv2


class DirectorySource:

    def __init__(self, directory, extension='jpg'):
        self.directory = directory
        self.extension = extension

    def __iter__(self):
        exp = os.path.join(self.directory, '*.' + self.extension)
        for f in glob(exp):
            yield f


class PipelineError(RuntimeError):
    pass


class Step:

    def __init__(self):
        self.cache_dir = None
        self.position = None

    def apply(self, image_data):
        return image_data, None

    def read(self, path):
        try:
            res = np.load(path)
            return res, None
        except Exception as e:
            logging.exception(e)
            return None, e

    def write(self, image_data, path):
        try:
            assert image_data is not None, (
                'Image data should exist in {} at {}'.format(
                    type(self).__name__,
                    path,
                ))
            path = os.path.splitext(path)[0]
            np.save(path, image_data)
            return None
        except Exception as e:
            logging.exception(e)
            return e


class Acquire(Step):

    def read(self, path):
        try:
            res = cv2.imread(path)
            return np.array(res), None
        except Exception as e:
            logging.exception(e)
            return None, e


class Save(Step):

    def __init__(self, target, extension='jpg'):
        super().__init__()
        self.target = target
        self.extension = extension

    def write(self, image_data, path):
        err = super().write(image_data, path)
        if err:
            return err
        name = '{}.{}'.format(
            os.path.splitext(os.path.basename(path))[0],
            self.extension,
        )
        try:
            cv2.imwrite(
                os.path.join(self.target, name),
                image_data,
            )
        except Exception as e:
            logging.exception(e)
            return e


class Pipeline:

    def __init__(self, steps=None, batch_size=None):
        self.steps = steps or []
        try:
            self.batch_size = batch_size or len(os.sched_getaffinity(0))
        except AttributeError:
            self.batch_size = batch_size or os.cpu_count() or 1

    def process(self, source):
        with TemporaryDirectory() as td:
            previous_step = None
            processed_step = None
            counter = 0
            for step in self.steps:
                step.cache_dir = os.path.join(td, str(counter))
                os.mkdir(step.cache_dir)
                counter += 1

                if previous_step is None:
                    batch = []
                    for i, s in enumerate(source):
                        step.position = i
                        # TODO(wvxvw): Move reading into multiple
                        # processes
                        sdata, err = step.read(s)
                        if err is not None:
                            raise err
                        n = os.path.join(step.cache_dir, os.path.basename(s))
                        batch.append((n, sdata))
                        if (i + 1) % self.batch_size == 0:
                            self.process_batch(batch, step)
                            batch = []
                    if batch:
                        self.process_batch(batch, step)
                else:
                    batch = []
                    for i, f in enumerate(os.listdir(previous_step.cache_dir)):
                        step.position = i
                        sdata, err = step.read(
                            os.path.join(previous_step.cache_dir, f),
                        )
                        if err is not None:
                            raise err
                        s = os.path.join(step.cache_dir, f)
                        batch.append((s, sdata))
                        if (i + 1) % self.batch_size == 0:
                            self.process_batch(batch, step)
                            batch = []
                    if batch:
                        self.process_batch(batch, step)

                processed_step = previous_step
                previous_step = step

                if processed_step is not None:
                    if os.path.isdir(processed_step.cache_dir):
                        shutil.rmtree(processed_step.cache_dir)

    def process_batch(self, batch, step):
        # Forking only works on Linux.  The garbage that Python
        # multiprocessing is requires a lot of workarounds...
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(mp_context=ctx) as ex:
            results, errors = [], []
            batch_names, batch_data = zip(*batch)
            for res, err in ex.map(step.apply, batch_data):
                if err:
                    errors.append(err)
                elif res is not None:
                    results.append(res)
                else:
                    errors.append(PipelineError(
                        'step.apply returned neither error nor result',
                    ))
            if errors:
                raise PipelineError(
                    'Step {} had errors:\n'.format(type(step).__name__) +
                    '\n  '.join(str(err) for err in errors),
                )

            for err in ex.map(step.write, results, batch_names):
                if err:
                    errors.append(err)
            if errors:
                raise PipelineError(
                    'Step {} couldn\'t save intermediate results:\n'.format(
                        type(step).__name__,
                    ) + '\n  '.join(str(err) for err in errors),
                )
