# -*- coding: utf-8 -*-

import os
import shutil
import logging
import multiprocessing
import time

from collections import defaultdict
from multiprocessing import Queue, Process
from multiprocessing.pool import ThreadPool
from tempfile import TemporaryDirectory
from glob import glob
from queue import Empty

from .steps import Aggregate, Step, Serial, Parallel

import numpy as np


class Loop(ValueError):
    pass


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

    def __init__(self, pipeline_def):
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
        self._def = pipeline_def
        self._ready = {}
        self._in_flight = set(())

    def goal(self):
        return self._def.goal

    def deps(self, goal):
        # linearized dependencies of the goal
        found = set(())
        unresolved = set(goal.variables)
        while unresolved:
            next_gen = set(())
            next_unresolved = set(())
            for u in unresolved:
                s = self._def.steps[u]
                if s not in found:
                    next_gen.add(s)
            for s in next_gen:
                yield s
                for v in s.variables:
                    next_unresolved.add(v)
            unresolved = next_unresolved

    def available(self, deps):
        # selects dependencies whose preconditions have been met
        for d in deps:
            if d in self._in_flight:
                continue
            if not d.variables:
                yield d
            elif all(v in self._ready for v in d.variables):
                yield d

    def resolve(self, dep, deps):
        # mark dependency as resolved and return the unresolved ones
        for v in self.out(dep):
            self._ready[v] = dep
        self._in_flight.pop(dep)
        for d in deps:
            if d != dep:
                yield d

    def out(self, dep):
        for v, s in self._def.steps.items():
            if dep == s:
                yield v

    def put(self, dep):
        # remember this step as in-flight, so we don't mark it as
        # available later.
        self._in_flight.add(dep)

    def pending(self, dep):
        # True, if dep is either in-flight, or unresolved
        for v in self._ready.values():
            if v == dep:
                return False
        return True

    def inflight(self):
        yield from self._in_flight


class Worker(Process):

    def __init__(self, inbound, outbound):
        super().__init__()
        self.inbound = inbound
        self.outbound = outbound

    def src(self, variables, td):
        if not variables:
            return ()

        svs = {
            v: sorted(os.listdir(os.path.join(td, v)))
            for v in variables
        }
        result = {v: [] for v in variables}

        pos = 0
        while any(svs):
            files = {v: os.path.basename(s[0]) for v, s in svs.items()}
            minf = min(files.values())
            for v, f in files.items():
                if f == minf:
                    result[v].append(f)
                else:
                    result[v].append(None)

        for i in range(len(result[variables[0]])):
            table.append({v: result[v][i] for v in variables})

        return tuple(
            {v: result[v][i] for v in variables}
            for i in range(len(result[variables[0]]))
        )

    def run(self):
        print('Starting:', os.getpid())
        while True:
            try:
                out, rec, td = self.inbound.get_nowait()
                if rec is None:
                    print('Terminating (1):', os.getpid())
                    break
            except (ValueError, AssertionError):
                # queue was closed
                print('Terminating: (2)', os.getpid())
                break
            except Empty:
                time.sleep(1)
                continue
            try:
                print('Received req:', rec)
                step = rec.definition(**dict(rec.options))
                src = self.src(rec.variables, td)
                step.apply(out, src)
                self.outbound.put((rec, None))
            except Exception as e:
                print('Failed req:', e)
                self.outbound.put((rec, e))


class Pipeline:
    """
    This class is the builder for the image processing pipeline.

    This class executes a sequence of :class:`~.steps.Step`.  It attempts to
    execute as many steps as possible in parallel.  However, in order
    to avoid running out of memory, it saves the intermediate results
    to the disk.  You can control the number of images processed at
    once by specifying :code:`batch_size` parameter.
    """

    def __init__(self, pipeline_def):
        """
        Initializes this pipeline, but doesn't start its execution.

        :param steps: A sequence of :class:`~.steps.Step` this pipeline should
                      execute.
        :type steps: Sequence[Step]
        """
        self.psg = PSG(pipeline_def)
        self.counter = 0
        self.process()

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

    def dep_dir(self, dep, td):
        return os.path.join(td, dep)

    def find_previous_step(self):
        """
        :meta private:
        """
        return None

    def cleanup(self, processed, todo, td):
        new = []
        for dep in processed:
            if not self.psg.pending(dep):
                dep_dir = self.dep_dir(dep, td)
                if os.path.isdir(dep_dir):
                    shutil.rmtree(dep_dir)
            else:
                new.append(dep)
        return new

    def process(self):
        """
        Starts this pipeline.
        """
        goal = self.psg.goal()
        deps = tuple(self.psg.deps(goal))

        with self.workspace() as td:
            ctx = multiprocessing.get_context('spawn')
            processes = os.cpu_count() or 1
            inbound = Queue(processes)
            outbound = Queue()
            pool = tuple(
                Worker(inbound, outbound) for _ in range(processes)
            )
            for w in pool:
                w.start()

            available = set(())
            processed = set(())
            err = None

            while deps and (err is None):
                available |= set(self.psg.available(deps))
                if (not available) and (not self.psg.inflight()):
                    err = Loop('Steps have a dependency loop')
                    break
                while (not inbound.full()) and available:
                    dep = available.pop()
                    out = tuple(self.psg.out(dep))
                    print('Sending:', (out, dep, td))
                    self.psg.put(dep)
                    inbound.put((out, dep, td))
                while not outbound.empty():
                    print('Receiving:')
                    dep, err = outbound.get()
                    print('Received:', dep, err)
                    if err:
                        break
                    processed.append(dep)
                    deps = tuple(self.psg.resolve(dep, deps))
                # TODO(wvxvw): schedule cleanups asynchronously
                processed = self.cleanup(processed, deps, td)

            if err is None:
                inbound.put(((), goal, td))
                _, err = outbound.get()

            for w in pool:
                inbound.put((None, None, None))
            inbound.close()
            inbound.join_thread()
            outbound.close()
            outbound.join_thread()
            for w in pool:
                w.terminate()
            for w in pool:
                w.join()

            if err:
                # TODO(wvxvw): Improve error reporting by making sure
                # we have proper stack trace.
                raise err

    def process_step(self, step, srciter, source):
        """
        :meta private:
        """
        logging.info('applying step: %s', step)
        if isinstance(step, Aggregate):
            self.process_batch_agg(scriter, step, source)
        elif isinstance(step, Serial):
            self.process_batch_serial(srciter, step, source)
        else:
            self.process_batch_parallel(srciter, step, source)

    def process_batch_agg(self, srciter, step, source):
        # TODO(olegs): In principle, this can also be done in
        # parallel, but we'll implement the parallel reduction some
        # time later.
        iaccum, maccum = step.initial()
        for name in srciter:
            img, err = step.read(name)
            if err is not None:
                errors.append(err)
                continue
            niaccum, nmaccum, err = step.aggregate(
                iaccum,
                maccum,
                img,
                name,
                source,
            )
            if err is not None:
                errors.append(err)
            else:
                iaccum = niaccum
                maccum = nmaccum

        if errors:
            raise PipelineError(
                'Step {} had errors:\n  '.format(type(step).__name__) +
                '\n  '.join(str(err) for err in errors),
            )

    def process_batch_parallel(self, srciter, step, source):
        pass

    def process_image(self, args):
        step, name, source = args
        print('process_image', name)
        img, err = step.read(name)
        if err:
            return None, err
        res, err = step.apply_one(img, name, source)
        if err:
            return None, err
        if res:
            return step.write(res, os.path.basename(name))

    def process_batch_serial(self, srciter, step, source):
        # Forking only works on Linux.  The garbage that Python
        # multiprocessing is it requires a lot of workarounds...
        ctx = multiprocessing.get_context('spawn')
        with Pool(context=ctx) as ex:
            errors = []
            metas = tuple(srciter)
            steps = tuple(step for _ in metas)
            sources = tuple(source for _ in metas)
            driver = ex.map(self.process_image, zip(steps, metas, sources))
            errors = tuple(e for e in driver if e)
            if errors:
                raise PipelineError(
                    'Step {} had errors:\n  '.format(type(step).__name__) +
                    '\n  '.join(str(err) for err in errors),
                )
