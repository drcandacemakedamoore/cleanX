# -*- coding: utf-8 -*-

import os
import shutil
import logging
import multiprocessing

from concurrent.futures import ProcessPoolExecutor
from tempfile import TemporaryDirectory
from glob import glob


class MultiSource:
    """
    A class to append multiple sources such as :class:`~.GlobSource` or
    :class:`~.DirectorySource`.  All source classes implement iterator
    interface.
    """

    def __init__(self, sources):
        """
        Initializes this iterator with multiple sources in a way similar
        to :code:`itertools.chain()`

        :param sources: A sequence of sources, such as :class:`~.GlobSource`
                        or :class:`~.DirectorySource`.
        :type sources: :code:`Sequence`
        """
        self.sources = tuple(sources)

    def __iter__(self):
        """
        Iterator implementation.
        """
        for src in self.sources:
            for s in src:
                yield s


class GlobSource:
    """
    A class that creates an iterator to list all files matching
    glob pattern.
    """

    def __init__(self, expression, recursive=False):
        """
        Initializes this iterator with the arguments to be passed to
        :code:`glob.glob()`.

        :param expression: Expression to be passed to :code:`glob.glob()`
        :type expression: :code:`Union[str, bytes]`
        :param recursive: Controls the interpretation of :code:`**`
                          pattern.  If :code:`True`, will interpret it
                          to mean any number of path fragments.
        """
        self.expression = expression
        self.recursive = recursive

    def __iter__(self):
        """
        Iterator implementation.
        """
        for s in glob(self.expression, recursive=self.recursive):
            yield s


class DirectorySource:
    """
    A class that creates an iterator to look at files in the given
    directory.
    """

    def __init__(self, directory, extension='jpg'):
        """
        Initializes this iterator.

        :param directory: The directory in which to look for images.
        :type directory: Must be valid for :code:`os.path.join()`
        :param extension: A glob pattern for fle extension.  Whether
                          it is case-sensitive depends on the
                          filesystem being used.
        """
        self.directory = directory
        self.extension = extension

    def __iter__(self):
        """
        Iterator implementation.
        """
        exp = os.path.join(self.directory, '*.' + self.extension)
        for f in glob(exp):
            yield f


class PipelineError(RuntimeError):
    """
    These errors are reported when pipeline encounters errors with
    reading or writing images.
    """
    pass


class Pipeline:
    """
    This class is the builder for the image processing pipeline.

    This class executes a sequence of :class:`~.steps.Step`.  It attemts to
    execute as many steps as possible in parallel.  However, in order
    to avoid running out of memory, it saves the intermediate results
    to the disk.  You can control the number of images processed at
    once by specifying :code:`batch_size` parameter.
    """

    def __init__(self, steps=None, batch_size=None):
        """
        Initializes this pipeline, but doesn't start its execution.

        :param steps: A sequence of :class:`~.steps.Step` this pipeline should
                      execute.
        :type steps: Sequence[Step]
        :param batch_size: The number of images that will be processed
                           in parallel.
        :type batch_size: int
        """
        self.steps = tuple(steps) if steps else ()
        try:
            self.batch_size = batch_size or len(os.sched_getaffinity(0))
        except AttributeError:
            self.batch_size = batch_size or os.cpu_count() or 1

        self.counter = 0

        self.process_lock = multiprocessing.Lock()

    def workspace(self):
        """
        :meta private:
        """
        return TemporaryDirectory()

    def update_counter(self):
        """
        :meta private:
        """
        pass

    def begin_transaction(self, step):
        """
        :meta private:
        """
        pass

    def commit_transaction(self, step):
        """
        :meta private:
        """
        pass

    def find_previous_step(self):
        """
        :meta private:
        """
        return None

    def process(self, source):
        """
        Starts this pipeline.

        :param source: This must be an iterable that yields file names
                       for the images to be processed.
        :type source: Iterable
        """
        with self.process_lock:
            with self.workspace() as td:
                previous_step = self.find_previous_step()
                processed_step = None
                for step in self.steps:
                    step.cache_dir = os.path.join(td, str(self.counter))
                    os.mkdir(step.cache_dir)
                    self.counter += 1
                    self.begin_transaction(step)

                    if previous_step is None:
                        batch = []
                        for i, s in enumerate(source):
                            step.position = i
                            # TODO(wvxvw): Move reading into multiple
                            # processes
                            sdata, err = step.read(s)
                            if err is not None:
                                raise err
                            n = os.path.join(
                                step.cache_dir,
                                os.path.basename(s),
                            )
                            batch.append((n, sdata))
                            if (i + 1) % self.batch_size == 0:
                                self.process_batch(batch, step)
                                batch = []
                        if batch:
                            self.process_batch(batch, step)
                    else:
                        batch = []
                        files = os.listdir(previous_step.cache_dir)
                        for i, f in enumerate(files):
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
                    self.commit_transaction(step)

                self.counter = 0
                self.update_counter()

    def process_batch(self, batch, step):
        """
        :meta private:
        """
        # Forking only works on Linux.  The garbage that Python
        # multiprocessing is it requires a lot of workarounds...
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
