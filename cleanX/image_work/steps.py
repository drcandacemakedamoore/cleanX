# -*- coding: utf-8 -*-

import logging
import os
import json
import inspect
import matplotlib.pyplot as plt
import pandas as pd
import math
from multiprocessing import Queue
from queue import Empty

import numpy as np
import cv2

from .image_functions import rotated_with_max_clean_area


_known_steps = {}


def get_known_steps():
    return dict(_known_steps)


class RegisteredStep(type):
    def __init__(cls, name, bases, clsdict):
        if len(cls.mro()) > 2:
            _known_steps[cls.__name__] = cls
        super().__init__(name, bases, clsdict)


class Step(metaclass=RegisteredStep):
    """
    This class has default implementations for methods all steps are
    expected to implement.

    Use this as the base class if you intend to add custom steps.
    """

    def __init__(self, cache_dir=None):
        """
        If you extend this class, you need to call its :code:`__init__`.
        """
        self.cache_dir = cache_dir
        self.position = None
        self.transaction_started = False

    def apply(self, image_data, image_name):
        """
        This is the method that will be called to do the actual image
        transformation.  This function must not raise exceptions, as it
        is used in the :mod:`multiprocessing` context.

        :param image_data: Will be the data obtained when calling
                           :meth:`~.Step.read()` method of this class.
        :type image_data: Unless this class overrides the defaults, this
                          will be :class:`~numpy.ndarray`.

        :return: This method should return two values.  First is the
                 processed image data.  This should be suitable for the
                 :meth:`~.Step.write()` method of this class to write.
                 Second is the error, if procesing wasn't possible.  Only
                 one element of the tuple should be :code:`not None`.
        :rtype: Tuple[numpy.ndarray, Exception]
        """
        return image_data, None

    def read(self, path):
        """
        Read the image saved in the previous step.  This function must not
        raise exceptions as it is used in :code:`multiprocessing` context.

        :param path: The path to the image to read.  Unless the
                     :meth:`~.Step.write()` method of the previous step
                     was modified to do it differently, the format
                     of the data in the file is the serialized NumPy array

        :return: This method should return two values.  First is the
                 image data read from :code:`path`.  It should be in the
                 format suitable for :meth:`~.Step.apply()`.  Second is the
                 :class:`Exception` if the read was not successful.  Only
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
        format.  This method is used in :mod:`multiprocessing` context,
        therefore it must not raise exceptions.

        :param image_data: This is the result from calling
                           :meth:`~.Step.apply()` method of this class.
        :type image_data: Default implementation uses :class:`~numpy.ndarray`.
        :return: Exception if it was raised during the execution, or
                 :code:`None`.
        :rtype: Exception
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

    def begin_transaction(self):
        self.transaction_started = True

    def commit_transaction(self):
        self.transaction_started = False

    def __reduce__(self):
        return self.__class__, (self.cache_dir,)

    def to_json(self):
        result = {
            '__name__': type(self).__name__,
            '__module__': type(self).__module__,
        }
        names = inspect.getfullargspec(self.__init__)[0]
        values = self.__reduce__()[1]
        for k, v in zip(names, values):
            result[k] = v
        return json.dumps(result)

    @classmethod
    def from_cmd_args(cls, cmd_args):
        evaled_args = eval('dict({})'.format(cmd_args))
        return cls(**evaled_args)


class Aggregate(Step):
    """
    This class has default implementations for methods all aggregate steps are
    expected to implement. These types of steps combine and accumalte data on
    the individual images processed in a step.
    """

    def __init__(self, cache_dir=None):
        super().__init__(cache_dir=cache_dir)
        self.accumulator = None

    def begin_transaction(self):
        super().begin_transaction()
        self.accumulator = self.pre()

    def commit_transaction(self):
        super().commit_transaction()
        super().write(*self.post(self.accumulator))

    def pre(self):
        return None, None

    def post(self, accumulator):
        return accumulator

    def agg(self, acc_data, acc_name, image_data, image_name):
        raise NotImplementedError("Subclasses must implement this")

    def aggregate(self, accum, new):
        try:
            self.accumulator = self.agg(accum[0], accum[1], new[0], new[1])
        except Exception as e:
            return None, e
        return self.accumulator, None

    def __reduce__(self):
        return self.__class__, (
            self.pre,
            self.agg,
            self.post,
            self.cache_dir,
        )


class Mean(Aggregate):
    """
    This class builds an averaged (by mean) image.
    """

    def agg(self, acc_data, acc_name, image_data, image_name):
        if acc_data is None:
            return image_data, [image_name]
        acc_name.append(image_name)
        accw, acch = acc_data.shape[:2]
        iw, ih = image_data.shape[:2]
        mw = max(accw, iw)
        mh = max(acch, ih)
        acc_data = np.float32(cv2.resize(acc_data, (mw, mh)))
        image_data = np.float32(cv2.resize(image_data, (mw, mh)))
        return acc_data + image_data, acc_name

    def post(self, acc):
        return np.uint8(acc[0] / len(acc[1])), acc[1][-1]

    def __reduce__(self):
        return self.__class__, (self.cache_dir,)


class GroupHistoHtWt(Aggregate):
    """
    This class builds a histogram of individual image heights and widths.
    """
    def __init__(self, histo_dir, cache_dir=None):
        super().__init__(cache_dir=cache_dir)
        self.histo_dir = histo_dir

    def agg(self, acc_data, acc_name, image_data, image_name):
        iw, ih = image_data.shape[:2]
        if acc_data is None:
            return [(iw, ih)], None
        return acc_data + [(iw, ih)], None

    def post(self, acc):
        tuple_like = acc[0]
        list_like = []
        for element in tuple_like:
            list_like.append(list(element))
        height = [el[0] for el in list_like]
        width = [el[1] for el in list_like]
        new_datafrme = pd.DataFrame({
            'height':  height,
            'width': width
        })
        fig, ax = plt.subplots(1, 1)

        # Add axis labels
        ax.set_xlabel('dimension size')
        ax.set_ylabel('count')

        # Generate the histogram
        histo_ht_wt = ax.hist(
            (new_datafrme.height, new_datafrme.width),
            bins=10
        )

        # Add a legend
        ax.legend(('height', 'width'), loc='upper right')

        fig.savefig(os.path.join(self.histo_dir, 'example4.jpg'))
        return np.zeros([8, 8, 3]), 'example4.jpg'

    def __reduce__(self):
        return self.__class__, (self.histo_dir, self.cache_dir,)


class GroupHistoPorportion(Aggregate):

    """
    This class makes a histogram of all the image's proportions.
    """
    def __init__(self, histo_dir, cache_dir=None):
        super().__init__(cache_dir=cache_dir)
        self.histo_dir = histo_dir

    def agg(self, acc_data, acc_name, image_data, image_name):
        iw, ih = image_data.shape[:2]
        if acc_data is None:
            return [(iw, ih)], None
        return acc_data + [(iw, ih)], None

    def post(self, acc):
        tuple_like = acc[0]
        list_like = []
        for element in tuple_like:
            list_like.append(list(element))
        height = [el[0] for el in list_like]
        width = [el[1] for el in list_like]
        porpor = [el[0]/el[1] for el in list_like]
        fig, ax = plt.subplots(1, 1)
        # Add axis labels
        ax.set_xlabel('h/w')
        ax.set_ylabel('count')
        # Generate the histogram
        histo_ht_wt = ax.hist(
            (porpor),
            bins=10
        )
        # Add a legend
        ax.legend(('height/width'), loc='upper right')

        fig.savefig(os.path.join(self.histo_dir, 'example5.jpg'))
        return np.zeros([2, 2, 3]), 'example5.jpg'

    def __reduce__(self):
        return self.__class__, (self.histo_dir, self.cache_dir,)


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

    def __init__(self, target, extension='jpg', cache_dir=None):
        super().__init__(cache_dir)
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
            print("exporting", path)
            cv2.imwrite(
                os.path.join(self.target, name),
                image_data,
            )
        except Exception as e:
            logging.exception(e)
            return e

    def __reduce__(self):
        return self.__class__, (
            self.target,
            self.extension,
            self.cache_dir,
        )


class FourierTransf(Step):
    """This class makes a fourier transformation of the image"""

    def apply(self, image_data, image_name):

        try:
            if len(image_data.shape) > 2:
                img = image_data[:, :, 0]
            else:
                img = image_data
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            transformed = np.log(np.abs(fshift))
            return transformed, None
        except Exception as e:
            logging.exception(e)
            return None, e


class ContourImage(Step):
    """This class makes a transformation into a many line
    contour based image from the original image"""

    def apply(self, image_data, image_name):

        try:
            major_cv2 = int(cv2.__version__.split('.')[0])
            edges = cv2.Canny(image_data, 0, 12)
            # # get threshold image (older function that failed with cv2 dif)
            # threshy = image_data.max()/2
            # ret, thresh_img = cv2.threshold(edges,
            # threshy, 255, cv2.THRESH_BINARY)
            #  # find contours
            if major_cv2 > 3:
                contours, hierarchy = cv2.findContours(
                                            edges,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE,
                                            )
            else:
                ret2, contours, hierarchy = cv2.findContours(
                                                    edges,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE,
                                                    )
                # create an empty image for contours
            img_cont = np.zeros(image_data.shape)
            drawing = cv2.drawContours(img_cont, contours, -1, (0, 255, 0), 3)
            return drawing, None
        except Exception as e:
            logging.exception(e)
            return None, e


class ProjectionHorizoVert(Step):
    """This class makes a transformation into a projection of the image.
    The projections of horizontal and vertical are superimposed. These
    one dimensional projection images can be used for image registration
    algorithms, quality control or other purposes"""

    def apply(self, image_data, image_name):

        try:
            sumpix0 = np.sum(image_data, 0)
            sumpix1 = np.sum(image_data, 1)
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
            plt.plot(sumpix0)
            plt.plot(sumpix1)

            axes.axis('off')
            fig.tight_layout(pad=0)
            axes.margins(0)
            fig.canvas.draw()
            iplot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            iplot = iplot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return iplot, None
        except Exception as e:
            logging.exception(e)
            return None, e


class BlackEdgeCrop(Step):
    """This class crops image arrays of black frames"""

    def apply(self, image_data, image_name):

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


class WhiteEdgeCrop(Step):
    """This class crops image arrays of white frames"""

    def apply(self, image_data, image_name):

        try:
            image_array = image_data
            if len(image_array.shape) > 2:
                image_array = image_array[:, :, 0]

            ht, wt = image_array.shape
            r, c, j, k = 0, 0, wt - 1, ht - 1
            row1 = image_array[:, r]
            column1 = image_array[c, :]
            row_last = image_array[:, j]
            column_last = image_array[k, :]
            while math.ceil(row1.mean()) == 255:
                row1 = image_array[:, r]
                r += 1

            while math.ceil(column1.mean()) == 255:
                column1 = image_array[c, :]
                c += 1

            while math.ceil(row_last.mean()) == 255:
                row_last = image_array[:, j]
                j -= 1

            while math.ceil(column_last.mean()) == 255:
                column_last = image_array[k, :]
                k -= 1

            cropped_image_array = image_array[c:k, r:j]
            return cropped_image_array, None
        except Exception as e:
            logging.exception(e)
            return None, e

# class Tee(Step):
#     """This step makes duplicate images, then opens two parallel pipelines"""
        # Unfinished
#     def apply(self, image_data):


# class Salt(Step):
#     """This class is currently canceled. It was to take the image and apply
#      the salting function (augments with noise). In present version there
#     seems to be a problem with mutex and cv2- UNDER investigation.
#     In future versions should be be run after a Tee step"""

    # def __init__(
    #     self,
    #     kernel=(5, 5),
    #     erosion_iterations=90,
    #     dilation_iterations=10,
    #     cache_dir=None,
    # ):
    #     super().__init__(cache_dir)
    #     self.kernel = kernel
    #     self.erosion_iterations = erosion_iterations
    #     self.dilation_iterations = dilation_iterations

#     def apply(self, image_data):
#         erosion = cv2.erode(
#             image_data,
#             self.kernel,
#             iterations=self.erosion_iterations,
#         )
#         dilation = cv2.dilate(
#             erosion,
#             self.kernel,
#             iterations=self.dilation_iterations,
#         )
#         salty_noised = (image_data + (erosion - dilation))
#         return salty_noised, None

    # def __reduce__(self):
    #     return self.__class__, (
    #         self.kernel,
    #         self.erosion_interations,
    #         self.dilation_iterations,
    #         self.cache_dir,
    #     )


class Sharpie(Step):
    """This class takes the image and applies a variant of the subtle_sharpie
    function, but with control over the degree. In present version,
    it is recommended to run on copies.
    In future versions can be run after a Tee step. For a subtle sharpening
    a ksize of (2,2) is recommended, and a run of normalization afterwards is
    highly recommended (or you may get vals over 255 for whites)"""

    def __init__(
        self,
        ksize=(2, 2),
        cache_dir=None,
    ):
        super().__init__(cache_dir)
        self.ksize = ksize

    def apply(self, image_data, image_name):
        blur_mask = cv2.blur(image_data, ksize=self.ksize)
        new_image_array = 2 * image_data - blur_mask
        return new_image_array, None

    def __reduce__(self):
        return self.__class__, (
            self.ksize,
            self.cache_dir,
        )


class BlurEdges(Step):
    """This class takes the image and applies a variant of the blur out
    edges  function, which does what it sounds like (returns an image with
    edges blurred out called edge_image). For a good effect a ksize of
    (600,600) is recommended. In present version,it is recommended to run on
    copies.In future versions can be run after a Tee step. """

    def __init__(
        self,
        ksize=(600, 600),
        cache_dir=None,
    ):
        super().__init__(cache_dir)
        self.ksize = ksize

    def apply(self, image_data, image_name):
        msk = np.zeros(image_data.shape)
        center_coordinates = (
            image_data.shape[1] // 2,
            image_data.shape[0] // 2,
        )
        radius = int(
            (min(image_data.shape) // 100) * (min(image_data.shape)/40)
        )
        color = 255
        thickness = -1
        msk = cv2.circle(msk, center_coordinates, radius, color, thickness)
        ksize = self.ksize
        msk = cv2.blur(msk, ksize)
        filtered = cv2.blur(image_data, ksize)
        edge_image = image_data * (msk / 255) + filtered * ((255 - msk) / 255)
        return edge_image, None

    def __reduce__(self):
        return self.__class__, (
            self.ksize,
            self.cache_dir,
        )


class CleanRotate(Step):
    """This class takes the image and applies a rotation  function,
    with control over the degree. It crops off black added (makes smaller
    borders, but will not crop existing borders) In present version, it is
    recommended to run on copies. In future versions can be run after a Tee
    step. """

    def __init__(
        self,
        angle=2,
        cache_dir=None,
    ):
        super().__init__(cache_dir)
        self.angle = int(angle)

    def apply(self, image_data, image_name):
        try:
            rotated = rotated_with_max_clean_area(image_data, self.angle)
            return rotated, None
        except Exception as e:
            logging.exception(e)
            return None, e


class Normalize(Step):
    """This class makes a simple normalizing to get values 0 to 255."""

    def apply(self, image_data, image_name):
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
    """This class allows normalization by throwing off extreme values on
    image histogram. """

    def __init__(self, tail_cut_percent=5, cache_dir=None):
        super().__init__(cache_dir)
        self.tail_cut_percent = tail_cut_percent

    def apply(self, image_data, image_name):
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

    def __reduce__(self):
        return self.__class__, (self.tail_cut_percent, self.cache_dir)


class OtsuBinarize(Step):
    """This class makes binarized images with values of only 0 to 255, based on
    an Otsu segmentation with more or less blurring. ksize parameter should
    be between 1 and 99, and odd"""
    def __init__(
        self,
        ksize=5,
        cache_dir=None,
    ):
        super().__init__(cache_dir)
        self.ksize = ksize

    def apply(self, image_data, image_name):
        try:
            # find it's np.histogram
            bins_num = 256
            hist, bin_edges = np.histogram(image_data, bins=bins_num)

            # find the threshold for Otsu segmentation
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
            wght = np.cumsum(hist)
            wght2 = np.cumsum(hist[::-1])[::-1]
            mean1 = np.cumsum(hist * bin_mids) / wght
            mean2 = (np.cumsum((hist * bin_mids)[::-1]) / wght2[::-1])[::-1]

            # compute interclass variance
            inter_class_vari = (
                wght[:-1] * wght2[1:] * (mean1[:-1] - mean2[1:]) ** 2
            )
            index_of_max_val = np.argmax(inter_class_vari)
            thresh = bin_mids[:-1][index_of_max_val]

            # this gave the thresh based on Otsu, now apply it
            output_image = np.uint8((image_data < thresh) * 255)
            if self.ksize % 2 == 0:
                self.ksize = self.ksize + 1

            return cv2.medianBlur(output_image, self.ksize), None

        except Exception as e:
            logging.exception(e)
            return None, e

    def __reduce__(self):
        return self.__class__, (self.ksize, self.cache_dir)


class OtsuLines(Step):
    """This class makes an outline around a segemnted image segmented based on
    the an Otsu segmentation with more or less blurring. ksize parameter should
    be between 1 and 99, and odd"""
    def __init__(
        self,
        ksize=5,
        cache_dir=None,
    ):
        super().__init__(cache_dir)
        self.ksize = ksize

    def apply(self, image_data, image_name):
        try:
            # find it's np.histogram
            bins_num = 256
            hist, bin_edges = np.histogram(image_data, bins=bins_num)

            # find the threshold for Otsu segmentation
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
            wght = np.cumsum(hist)
            wght2 = np.cumsum(hist[::-1])[::-1]
            mean1 = np.cumsum(hist * bin_mids) / wght
            mean2 = (np.cumsum((hist * bin_mids)[::-1]) / wght2[::-1])[::-1]

            # compute interclass variance
            inter_class_vari = (
                wght[:-1] * wght2[1:] * (mean1[:-1] - mean2[1:]) ** 2
            )
            index_of_max_val = np.argmax(inter_class_vari)
            thresh = bin_mids[:-1][index_of_max_val]

            # this gave the thresh based on Otsu, now apply it
            output_image = np.uint8((image_data < thresh) * 255)
            if self.ksize % 2 != 0:
                # blur
                output_image = cv2.medianBlur(output_image, self.ksize)
            else:
                output_image = cv2.medianBlur(output_image, (self.ksize + 1))
            # now use canny to get edges
            edges_image = cv2.Canny(output_image, 50, 230)
            return edges_image, None
        except Exception as e:
            logging.exception(e)
            return None, e

    def __reduce__(self):
        return self.__class__, (self.ksize, self.cache_dir)


class Projection(Step):
    """This class makes two one dimensional projections on the same graph,
    one for horizontal, and one for vertical. Projections can be used in many
    processes including registration, and analysis."""

    def apply(self, image_data, image_name):
        try:
            # find it's np.histogram
            sumpix0 = np.sum(image_data, 0)
            sumpix1 = np.sum(image_data, 1)
            plt.plot(sumpix0)
            plt.plot(sumpix1)
            plt.title('Sum pixels along (vertical and horizontal) columns')
            plt.xlabel('Column number')
            plt.ylabel('Sum of column')
            plt.savefig('example_columns.jpg')
            new_graph_img = cv2.imread('example_columns.jpg')

            return new_graph_img, None

        except Exception as e:
            logging.exception(e)
            return None, e
