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
    """
    A class that creates an iterator to look at files in the given
    directory.
    """

    def __init__(self, directory, extension='jpg'):
        """
        Initializes this iterator.

        :param directory: The directory in which to look for images.
        :type directory: Must be valid for :code:`os.path.join()`.
        :param extension: A glob pattern for fle extension.  Whether
                          it is case-sensitive depends on the
                          filesystem being used.
        """
        self.directory = directory
        self.extension = extension

    def __iter__(self):
        exp = os.path.join(self.directory, '*.' + self.extension)
        for f in glob(exp):
            yield f


class PipelineError(RuntimeError):
    """
    These errors are reported when pipeline encounters errors with
    reading or writing images.
    """
    pass


class Step:
    """
    This class has default implementations for methods all steps are
    expected to implement.

    Use this as the base class if you intend to add custom steps.
    """

    def __init__(self):
        """
        If you extend this class, you need to call its :code:`__init__`.
        """
        self.cache_dir = None
        self.position = None

    def apply(self, image_data):
        """
        This is the method that will be called to do the actual image
        transformation.  This function must not raise exceptions, as it
        is used in the :code:`multiprocessing` context.

        :param image_data: Will be the data obtained when calling
                           :code:`read()` method of this class.
        :type image_data: Unless this class overrides the defaults, this
                          will be :code:`numpy.ndarray`.

        :return: This method should return two values.  First is the
                 processed image data.  This should be suitable for the
                 :code:`write()` method of this class to write.  Second
                 is the error, if procesing wasn't possible.  Only one
                 element of the tuple should be :code:`not None`.
        :rtype: Tuple[numpy.ndarray, Exception]
        """
        return image_data, None

    def read(self, path):
        """
        Read the image saved in the previvous step.  This function must not
        raise exceptions as it is used in :code:`multiprocessing` context.

        :param path: The path to the image to read.  Unless the
                     :code:`write()` method of the previous step
                     was modified to do it differently, the format
                     of the data in the file is the serialized NumPy array

        :return: This method should return two values.  First is the
                 image data read from :code:`path`.  It should be in the
                 format suitable for :code:`apply()`.  Second is the
                 :code:`Exception` if the read was not successful.  Only
                 one element in the tuple may be :code:`not None`.
        :rtype: Tuple[numpy.ndarray, Exception]
        """
        try:
            res = np.load(path)
            return res, None
        except Exception as e:
            logging.exception(e)
            return None, e

    def write(self, image_data, path):
        """
        This method should write the image data to make it available for
        the next step.  Default implementation use NumPy's persistence
        format.  This method is used in :code:`multiprocessing` context,
        therefore it must not raise exceptions.

        :param image_data: This is the result from calling :code:`apply()`
                           method of this class.
        :type image_data: Default implementation uses :code:`numpy.ndarray`.
        """
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
    """This class reads in images (to an array) from a path"""

    def read(self, path):
        try:
            res = cv2.imread(path)
            return np.array(res), None
        except Exception as e:
            logging.exception(e)
            return None, e


class Save(Step):
    """This class writes the images somewhere"""

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


class Crop(Step):
    """This class crops image arrays of black frames"""

    def apply(self, image_data):

        try:
            nonzero = np.nonzero(image_data)
            y_nonzero = nonzero[0]
            x_nonzero = nonzero[1]
    # , x_nonzero, _ = np.nonzero(image)
            return image_data[
                np.min(y_nonzero):np.max(y_nonzero),
                np.min(x_nonzero):np.max(x_nonzero)
            ], None
        except Exception as e:
            logging.exception(e)
            return None, e


class Normalize(Step):
    """This class makes a simple normalizing to get values 0 to 255."""

    def apply(self, image_data):
        try:
            new_max_value = 255

            max_value = np.amax(image_data)
            min_value = np.amin(image_data)

            img_py = image_data - min_value
            multiplier_ratio = new_max_value/max_value
            img_py = img_py*multiplier_ratio
            return img_py, None
        except Exception as e:
            logging.exception(e)
            return None, e


class HistogramNormalize(Step):
    """This class allows normalization by throwing off exxtreme values on
    image histogram. """

    def __init__(self, tail_cut_percent=5):
        super().__init__()
        self.tail_cut_percent = tail_cut_percent

    def apply(self, image_data):
        try:
            new_max_value = 255
            img_py = np.array((image_data), dtype='int64')
            # num_total = img_py.shape[0]*img_py.shape[1]
            # list_from_array = img_py.tolist()
            gray_hist = np.histogram(img_py, bins=256)[0]
            area = gray_hist.sum()
            cutoff = area * (self.tail_cut_percent/100)
            dark_cutoff = 0
            bright_cutoff = 255
            area_so_far = 0
            for i, b in enumerate(gray_hist):
                area_so_far += b
                if area_so_far >= cutoff:
                    dark_cutoff = max(0, i - 1)
                    break
            area_so_far = 0
            for i, b in enumerate(reversed(gray_hist)):
                area_so_far += b
                if area_so_far >= cutoff:
                    bright_cutoff = min(255, 255 - i)
                    break

            img_py = img_py - dark_cutoff
            img_py[img_py < 0] = 0
            max_value2 = np.amax(img_py)
            # min_value2 = np.amin(img_py)
            multiplier_ratio = new_max_value/max_value2
            img_py = img_py*multiplier_ratio

            return img_py, None
        except Exception as e:
            logging.exception(e)
            return None, e


class Pipeline:
    """
    This class is the builder for the image processing pipeline.

    This class executes a sequence of :code:`Steps`.  It attemts to
    execute as many steps as possible in parallel.  However, in order
    to avoid running out of memory, it saves the intermediate results
    to the disk.  You can control the number of images processed at
    once by specifying :code:`batch_size` parameter.
    """

    def __init__(self, steps=None, batch_size=None):
        """
        Initializes this pipeline, but doesn't start its execution.

        :param steps: A sequence of :code:`Steps` this pipeline should
                      execute.
        :type steps: List[Step]
        :param batch_size: The number of images that will be processed
                           in parallel.
        :type batch_size: int
        """
        self.steps = steps or []
        try:
            self.batch_size = batch_size or len(os.sched_getaffinity(0))
        except AttributeError:
            self.batch_size = batch_size or os.cpu_count() or 1

    def process(self, source):
        """
        Starts this pipeline.

        :param source: This must be an iterable that yields file names
                       for the images to be processed.
        :type source: Iterable
        """
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
