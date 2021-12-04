# -*- coding: utf-8 -*-

import os
import shutil
import logging
import multiprocessing

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tempfile import TemporaryDirectory
from glob import glob

from .steps import Aggregate, Step


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


class NoPathToGoal(LookupError):
    pass


class PSG:

    @classmethod
    def create(cls, source):
        if not source:
            return cls(())
        if isinstance(source[0], Step):
            pairs = tuple(zip(source[1:], source))
        else:
            pairs = source
        return cls(pairs)

    def __init__(self, pairs):
        # pairs:
        #   A -> B
        #   B -> C
        #   A -> C
        #   D -> A
        #   B -> E
        #
        # goal: D
        #
        #     [D]
        #      |
        #      v
        #     [A]     [F]
        #    /   \     |
        #   v     v    v
        #  [B] -> [C] [G]
        #   |
        #   v
        #  [E]
        #
        # In order to compute D, we need to compute A.
        # In order to compute A, we need to compute B and C.
        # In order to compute B, we need to compute C and E.
        # C and E have no dependencies.
        # F and G don't contribute to computing D, therefore
        # should be removed.
        #
        # Steps may be iterated in this order:
        #   E, C, B, A, D
        # or;
        #   C, E, B, A, D
        mapping = defaultdict(list)
        nodes = {}
        for a, b in pairs:
            mapping[a.id()].append(b)
            nodes[a.id()] = a
            nodes[b.id()] = b
        for a, b in pairs:
            if b.id() not in mapping:
                mapping[b.id()] = []
        self._deps = dict(mapping)
        self._nodes = nodes

    def as_dict(self):
        return self._nodes.items()

    def depends(self, step, others):
        ids = set(o.id() for o in others)
        deps = set(s.id() for s in self._deps[step.id()])
        return ids.intersection(deps)

    def dependencies(self, goal):
        pairs = dict(self._deps)
        tier = pairs[goal.id()]
        del pairs[goal.id()]

        yield goal

        while tier:
            next_tier = []
            for t in tier:
                next_tier += pairs[t.id()]
                del pairs[t.id()]
                yield t
            tier = next_tier


class Pipeline:
    """
    This class is the builder for the image processing pipeline.

    This class executes a sequence of :class:`~.steps.Step`.  It attempts to
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
        self.steps = PSG.create(steps)
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
        step.begin_transaction()

    def commit_transaction(self, step):
        """
        :meta private:
        """
        step.commit_transaction()

    def find_previous_step(self):
        """
        :meta private:
        """
        return None

    def cleanup(self, dirty, pending):
        new = []
        for d in dirty:
            if d is None:
                continue
            if self.steps.depends(d, pending):
                new.append(d)
                continue
            if os.path.isdir(d.cache_dir):
                shutil.rmtree(d.cache_dir)
        return new

    def process(self, source, goal):
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
                deps = tuple(reversed(tuple(self.steps.dependencies(goal))))
                dirty = []
                for n, step in enumerate(deps):
                    step.cache_dir = os.path.join(td, str(self.counter))
                    os.mkdir(step.cache_dir)
                    self.counter += 1
                    self.begin_transaction(step)

                    if previous_step is None:
                        srciter = enumerate(source)
                    else:
                        files = os.listdir(previous_step.cache_dir)
                        srciter = (
                            (i, os.path.join(previous_step.cache_dir, f))
                            for i, f in enumerate(files)
                        )
                    self.process_step(step, srciter)

                    processed_step = previous_step
                    previous_step = step

                    dirty.append(processed_step)
                    dirty = self.cleanup(dirty, deps[n:])
                    self.commit_transaction(step)

                self.counter = 0
                self.update_counter()

    def process_step(self, step, srciter):
        batch = []
        for i, s in srciter:
            logging.info('processing image: %s', s)
            step.position = i
            # TODO(wvxvw): Move reading into multiple processes.
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

    def process_batch(self, batch, step):
        """
        :meta private:
        """
        logging.info('applying step: %s', step)
        if isinstance(step, Aggregate):
            self.process_batch_agg(batch, step)
        else:
            self.process_batch_parallel(batch, step)

    def process_batch_agg(self, batch, step):
        # TODO(olegs): In principle, this can also be done in
        # parallel, but we'll implement the parallel reduction some
        # time later.
        accum, errors = step.accumulator, []
        # TODO(olegs): Use consistent order of image and name
        for name, img in batch:
            naccum, err = step.aggregate(accum, (img, name))
            if err is not None:
                errors.append(err)
            else:
                accum = naccum
        if errors:
            raise PipelineError(
                'Step {} had errors:\n  '.format(type(step).__name__) +
                '\n  '.join(str(err) for err in errors),
            )

    def process_batch_parallel(self, batch, step):
        # Forking only works on Linux.  The garbage that Python
        # multiprocessing is it requires a lot of workarounds...
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(mp_context=ctx) as ex:
            results, errors = [], []
            batch_names, batch_data = zip(*batch)
            for res, err in ex.map(step.apply, batch_data, batch_names):
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
                    'Step {} had errors:\n  '.format(type(step).__name__) +
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
