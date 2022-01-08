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

from .steps import Step

import numpy as np


class Loop(ValueError):
    pass


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
        self._in_flight.remove(dep)
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
        return dep not in set(self._ready.values())

    def inflight(self):
        yield from self._in_flight

    def used(self, var):
        for s in self._def.steps.values():
            if var not in s.variables:
                continue
            if self.pending(s):
                return True
            if s in self._in_flight:
                return True
        return False


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
        while any(svs.values()):
            files = {v: os.path.basename(s[0]) for v, s in svs.items()}
            minf = min(files.values())
            for v, f in files.items():
                if f == minf:
                    result[v].append(f)
                    svs[v].pop(0)
                else:
                    result[v].append(None)

        # return tuple(
        #     {v: result[v][i] for v in variables}
        #     for i in range(len(result[variables[0]]))
        # )
        return result

    def run(self):
        while True:
            try:
                out, rec, td = self.inbound.get_nowait()
                if rec is None:
                    break
            except (ValueError, AssertionError):
                # queue was closed, but doesn't really seem to work...
                break
            except Empty:
                time.sleep(1)
                continue
            try:
                print('scheduling step:', rec.definition, rec.options)
                step = rec.definition(**dict(rec.options))
                print('created step:', rec.definition)
                step.init_step(rec.serial, rec.splitter, rec.joiner, td)
                print('initialized step:', rec.definition)
                src = self.src(rec.variables, td)
                print('running step:', step)
                step.apply_split_join(out, src)
                self.outbound.put((rec, None))
            except Exception as e:
                import traceback
                traceback.print_exc()
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

    def find_previous_step(self):
        """
        :meta private:
        """
        return None

    def cleanup(self, processed, todo, td):
        new = []
        for dep in processed:
            if not self.psg.pending(dep):
                for v in dep.variables:
                    if not self.psg.used(v):
                        dep_dir = os.path.join(td, v)
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
