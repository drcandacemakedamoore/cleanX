# -*- coding: utf-8 -*-
"""
Library for cleaning radiological data used in machine learning
applications
"""

import subprocess
import logging

# imported libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


try:
    def __fix_tesserocr_locale():
        output = subprocess.check_output(
            ['ldconfig', '-v'],
            stderr=subprocess.DEVNULL,
        )
        for line in output.decode().split('\n'):
            if line.lstrip().startswith('libtesseract'):
                alias, soname = line.strip().split(' -> ')
                _, _, maj, min, patch = soname.split('.')
                maj, min, patch = int(maj), int(min), int(patch)
                if (maj, min, patch) < (4, 0, 1):
                    import locale
                    locale.setlocale(locale.LC_ALL, 'C')
                    logging.warning(
                        'Setting locale to C, otherwise tesseract segfaults',
                    )
    __fix_tesserocr_locale()
    del __fix_tesserocr_locale
except (FileNotFoundError, subprocess.CalledProcessError):
    logging.warning('Don\'t know how to find Tesseract library version')

from tesserocr import PyTessBaseAPI

import glob
import filecmp
import math
import os
import re

from filecmp import cmp
from pathlib import Path


class Cv2Error(RuntimeError):
    pass


def cv2_imread(image, *args):
    # OpenCV doesn't always raise an error when it fails to read the
    # image.  This causes the errors to happen later and give
    # confusing messages.  We want to catch this error earliers.
    result = cv2.imread(image, *args)
    if result is None:
        raise Cv2Error('OpenCV had a problem reading: {}'.format(image))
    return result


def crop_np(image_array):
    """
    Crops black edges of an image array

    :param image_array: Image array.
    :type image_array: :class:`~numpy.ndarray`


    :return: NumPy array with the image data with the black margins cropped.
    :rtype: :class:`~numpy.ndarray`
    """
    nonzero = np.nonzero(image_array)
    y_nonzero = nonzero[0]
    x_nonzero = nonzero[1]

    return image_array[
        np.min(y_nonzero):np.max(y_nonzero),
        np.min(x_nonzero):np.max(x_nonzero),
    ]


def crop_np_white(image_array):
    """
    Crops white edges of an image array
    :param image_array: Image array.
    :type image_array: :class:`~numpy.ndarray`
    :return: NumPy array with the image data with the white margins cropped.
    :rtype: :class:`~numpy.ndarray`
    """
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
    return cropped_image_array


def find_outliers_sum_of_pixels_across_set(directory, percent_to_examine):
    """
    This function finds images that are outliers in terms of having a large
    or small total pixel sum, which in most cases, once images are normalized
    will correlate with under or overexposure OR pathology if the percent
    parameter is set to a low number - 3% (will be written as 3) is recommended

    :param directory: Directory of images.
    :type directory: string


    :return: top, bottom (dataframes of highest and lowest total)
    :rtype: :class: tuple
    """
    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    names = []
    pix_list = []
    for pic in suspects:
        name = pic
        img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        pix = img.sum()/(h*w)
        names.append(name)
        pix_list.append(pix)
    frame = pd.DataFrame({
                'pixel_total': pix_list,
                'images': names,
            })
    frame = frame.sort_values('pixel_total')
    number = int((percent_to_examine/100) * len(frame)/2)
    print(number)
    print(len(frame))
    top = frame.head(number)
    bottom = frame.tail(number)
    return top, bottom


def hist_sum_of_pixels_across_set(directory):
    """
    This function finds the sum of pixels per image in a set of images, then
    turns these values into a histogram. This is useful to compare exposure
    across normalized groups of images.

    :param directory: directory of images.
    :type directory: : string


    :return: NumPy array shown as histogram.
    :rtype: :class:`~numpy.ndarray`
    """
    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2

    names = []
    pix_list = []
    for pic in suspects:
        name = pic
        img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        pix = img.sum()/(h*w)
        names.append(name)
        pix_list.append(pix)
    column_names = ['pix_list', 'names']
    frame = pd.DataFrame({
                'pixel_total': pix_list,
                'images': names})

    frame = frame.sort_values('pixel_total')
    histogram = frame.hist()

    return histogram


def crop(image):
    """
    Crops an image of a black or white frame: made for Numpy arrays only now.
    Previous version handled PIL images. Next version handles
    all colors of borders i.e. not only black or white frames

    :param image: Image
    :type image: This must be a NumPy array holding image data,

    :return: NumPy array with the image data with the black margins cropped.
    :rtype: :class:`~numpy.ndarray`.
    """

    if isinstance(image, np.ndarray):
        image_g1 = crop_np(image)
        image_g2 = crop_np_white(image_g1)
        return image_g2


def subtle_sharpie_enhance(image):
    """
    Makes a new image that is very subtly sharper to the human eye,
    but has new values in most of the pixels (besides the background). This
    is an augmentation, that has not been tested for how well the outputs match
    X-rays from new machines used well, but is within a reasonable range
    by human eye.

    :param image: String for image name
    :type image: str

    :return: new_image_array, a nearly imperceptibly sharpened image for humans
    :rtype: :class:`~numpy.ndarray`
    """
    ksize = (2, 2)
    image_body = cv2.imread(image)
    blur_mask = cv2.blur(image_body, ksize)
    new_image_array = 2 * image_body - blur_mask
    return new_image_array


def harsh_sharpie_enhance(image):
    """
    Makes a new image that is very sharper to the human eye,
    and has new values in most of the pixels (besides the background). This
    augmentation may allow humans to understand certain elements of an image,
    but should be used with care to make augmented data.

    :param image: String for image name
    :type image: str

    :return: new_image_array, a sharpened image for humans
    :rtype: :class:`~numpy.ndarray`
    """
    ksize = (20, 20)
    image_body = cv2.imread(image)
    blur_mask = cv2.blur(image_body, ksize)
    new_image_array = (2*image_body) - (blur_mask)
    return new_image_array


def salting(img):
    """
    This function adds some noise to an image. The noise is synthetic. It has
    not been tested for similarity to older machines, which also add noise.

    :param img_name: String for image name
    :type img_name: str

    :return: new_image_array, with noise
    :rtype: :class:`~numpy.ndarray`
    """
    kernel = (5, 5)
    img = cv2.imread(img)
    erosion = cv2.erode(img, kernel, iterations=90)
    dilation = cv2.dilate(erosion, kernel, iterations=10)
    salty_noised = (img + (erosion - dilation))
    return salty_noised


def simple_rotate_no_pil(image, angle, center=None, scale=1.0):
    """
    This function works without the PIL library. It takes one picture
    and rotates is by a number of degrees in the parameter angle.
    This function can be used with the
    augment_and_move function as follows (example):
    .. code-block:: python
       augment_and_move(
           'D:/my_academia/dataset/random_within_domain',
           'D:/my_academia/elo',
           [partial(simple_rotate_no_pil, 5)],
       )
    :param image: Image.
    :type image: Image (JPEG)
    :return: rotated image
    :rtype: numpy.ndarray
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    h, w = image.shape[0:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def blur_out_edges(image):
    """
    For an individual image, blurs out the edges as an augmentation. This
    augmentation is not like any real-world X-ray, but can make images
    which helps force attention in a neural net away from the edges of the
    images.

    :param image: Image
    :type image: Image (JPEG)

    :return: blurred_edge_image an array of an image blurred around the edges
    :rtype: :class:`~numpy.ndarray`
    """
    example = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    msk = np.zeros(example.shape)
    center_coordinates = (example.shape[1] // 2, example.shape[0] // 2)
    radius = int((min(example.shape) // 100) * (min(example.shape)/40))
    color = 255
    thickness = -1
    msk = cv2.circle(msk, center_coordinates, radius, color, thickness)
    ksize = (600, 600)
    msk = cv2.blur(msk, ksize)
    filtered = cv2.blur(example, ksize)
    blurred_edge_image = example * (msk / 255) + filtered * ((255 - msk) / 255)
    return blurred_edge_image


def multi_rotation_augmentation_no_pill(angle1, angle2, number_slices, image):
    """
    Works on a single image, and returns a list of augmented images which are
    based on twisting the angle from angle1 to angle2 with 'number_slices' as
    the number of augmented images to be made from the original. It is not
    realistic or desirable to augment images of most X-rays by flipping them.
    In abdominal or chest X-rays that would create an augmentation that could
    imply specific pathologies e.g. situs inversus. We suggest augmenting
    between angles -5 to 5.
    :param angle1: angle1 is the angle from the original to the first augmented
    :type angle1: float
    :param angle2: angle2 is the angle from the original to the last augmented
    :type angle2: float
    :param number_slices: number of images to be produced
    :type number_slices: int
    :param image: image
    :type image: string (string where image is located)
    :return: list of image arrays
    :rtype: list
    """
    increment = abs(angle1-angle2)/number_slices
    angle1f = float(angle1)
    angle2f = float(angle2)
    number_slicesf = float(number_slices)
    increment = abs(angle1f-angle2f)/number_slicesf
    num_list = np.arange(angle1f, angle2f,  increment)
    image4R = cv2.imread(image)
    augmentos = []
    for i in num_list:
        augmentos.append(simple_rotate_no_pil(image4R, i))

    return augmentos


def show_major_lines_on_image(pic_name):
    """
    A function that takes individual images and shows suspect lines i.e.
    lines more likely to be non-biological.

    :param pic_name: String of image full name e.g. "C:/folder/image.jpg"
    :type pic_name: str

    :return: shows image but technically returns a matplotlib plotted image
    :rtype: matplotlib.image.AxesImage
    """
    img = cv2.imread(pic_name, cv2.IMREAD_GRAYSCALE)
    max_slider = 6
    # Find the edges in the image using canny detector
    edges = cv2.Canny(img, 3, 100)
    # Detect points that form a line
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        max_slider,
        minLineLength=30,
        maxLineGap=20
    )
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # if abs(math.sqrt((x2 - x1)**2 + (y2 - y1)**2)) > 0:
        imag_lined = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return plt.imshow(imag_lined)


def find_big_lines(directory, line_length):
    """
    Finds number of lines in images at or over the length of
    :code:`line_length`, gives back a :code:`DataFrame` with this information.
    Note lines can fold back on themselves, and every pixel is counted if they
    are all contiguous

    :param directory: Directory with set_of_images (should include final '/').
    :type directory: Suitable for :func:`os.path.join()`
    :param line_length: Minimal length of lines for the function to count.
    :type line_length: int

    :return: :code:`DataFrame` with column for line count at or above
             :code:`line_length`
    :rtype: :class:`~pandas.DataFrame`
    """
    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    # suspects = glob.glob(os.path.join(directory, '*.jpg'))
    pic_to_nlines = {}
    for pic in suspects:
        img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, 3, 100)
        minLineLength = 50
        maxLineGap = 2
        max_slider = 6
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            max_slider,
            minLineLength=1,
            maxLineGap=25
        )
        nlines = 0
        for line in lines:
            # why line[0]- want tocheck all lines in a picture
            if np.linalg.norm(line[0][:2] - line[0][2:]) > line_length:
                nlines += 1
        pic_to_nlines[pic] = nlines

    return pd.DataFrame.from_dict(
                pic_to_nlines,
                columns=['nlines'],
                orient='index'
            )


def separate_image_averager(set_of_images, s=5):
    """
    This function runs on a list of images  to make a prototype tiny X-ray that
    is an average image of them. The images should be given as the strings that
    are the location for the image file.

    :param set_of_images: Set_of_images
    :type set_of_images: Collection of elements suitable for
                         :func:`os.path.join()`
    :param s: length of sides in the image made
    :type s: int

    :return: image
    :rtype: class: `numpy.ndarray`
    """
    canvas = np.zeros((s, s))
    for pic in set_of_images:
        example = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        example_small = cv2.resize(example, (s, s))
        canvas += np.array(example_small)
    return canvas / len(set_of_images)


def dimensions_to_df(image_directory):
    """
    Finds dimensions on images in a folder, and makes a dataframe for
    exploratory data analysis.

    :param folder_name: Adress of folder with images (should include final '/')
    :type folder_name: :func:`os.path.join()`

    :return: image height, width and proportion height/width as a new
             :code:`DataFrame`
    :rtype: :class:`~pandas.DataFrame`
    """
    non_suspects1 = glob.glob(os.path.join(image_directory, '*.[Jj][Pp][Gg]'))
    non_suspects2 = glob.glob(
        os.path.join(image_directory, '*.[Jj][Pp][Ee][Gg]'),
    )
    non_suspects = non_suspects1 + non_suspects2
    picy_list, list_ht, list_wt = [], [], []

    for picy in non_suspects:
        picy_list.append(picy)
        image = cv2.imread(picy, cv2.IMREAD_GRAYSCALE)
        ht, wt = image.shape
        list_ht.append(ht)
        list_wt.append(wt)
    new_datafrm = pd.DataFrame({
            'images': picy_list,
            'height': list_ht,
            'width': list_wt
        })
    new_datafrm['proportion'] = new_datafrm['height']/new_datafrm['width']
    return new_datafrm


def dimensions_to_histo(image_directory, bins_count=10):
    """
    Looks in the directory given, and produces a histogram of various widths
    and heights. Important information as many neural nets take images all the
    same size. Classically most chest-X-rays are :math:`2500 \\times 2000` or
    :math:`2500 \\times 2048`; however the dataset may be different and/or
    varied

    :param folder_name: Folder_name, directory name.(should include final '/')
    :type folder_name: str

    :param bins_count: bins_count, number of bins desired (defaults to 10)
    :type bins_count: int

    :return: histo_ht_wt, a labeled histogram
    :rtype: tuple
    """
    non_suspects1 = glob.glob(os.path.join(image_directory, '*.[Jj][Pp][Gg]'))
    non_suspects2 = glob.glob(
        os.path.join(image_directory, '*.[Jj][Pp][Ee][Gg]'),
    )
    non_suspects = non_suspects1 + non_suspects2

    picy_list, list_ht, list_wt = [], [], []

    for picy in non_suspects:
        picy_list.append(picy)
        image = cv2.imread(picy, cv2.IMREAD_GRAYSCALE)
        ht, wt = image.shape
        list_ht.append(ht)
        list_wt.append(wt)
    new_datafrme = pd.DataFrame({
            'images': picy_list,
            'height': list_ht,
            'width': list_wt
        })
    new_datafrme['proportion'] = new_datafrme['height']/new_datafrme['width']
    fig, ax = plt.subplots(1, 1)

    # Add axis labels
    ax.set_xlabel('dimension size')
    ax.set_ylabel('count')

    # Generate the histogram
    histo_ht_wt = ax.hist(
        (new_datafrme.height, new_datafrme.width),
        bins=bins_count
    )

    # Add a legend
    ax.legend(('height', 'width'), loc='upper right')
    return histo_ht_wt


def proportions_ht_wt_to_histo(folder_name, bins_count=10):
    """
    Looks in the directory given, produces a histogram of various proportions
    of the images by dividing their heights by widths.
    Important information as many neural nets take images all the
    same size. Classically most chest X-rays are :math:`2500 \\times 2000` or
    :math:`2500 \\times 2048`; however the dataset may be different and/or
    varied

    :param folder_name: Folder_name, directory name. (should include final '/')
    :type folder_name: str
    :param bins_count: bins_count, number of bins desired (defaults to 10)
    :type bins_count: int
    :return: histo_ht_wt_p, a labeled histogram
    :rtype: tuple
    """
    non_suspects1 = glob.glob(os.path.join(folder_name, '*.[Jj][Pp][Gg]'))
    non_suspects2 = glob.glob(os.path.join(folder_name, '*.[Jj][Pp][Ee][Gg]'))
    non_suspects = non_suspects1 + non_suspects2
    # non_suspects = glob.glob(os.path.join(folder_name, '*.jpg'))
    picy_list, list_ht, list_wt = [], [], []

    for picy in non_suspects:
        picy_list.append(picy)
        image = cv2.imread(picy, cv2.IMREAD_GRAYSCALE)
        ht, wt = image.shape
        list_ht.append(ht)
        list_wt.append(wt)
        new_datafrme = pd.DataFrame({
            'images': picy_list,
            'height': list_ht,
            'width': list_wt
        })
    new_datafrme['proportion'] = new_datafrme['height']/new_datafrme['width']
    fig, ax = plt.subplots(1, 1)

    # Add axis labels
    ax.set_xlabel('proportion height/width')
    ax.set_ylabel('count')

    # Generate the histogram
    histo_ht_wt_p = ax.hist(
        (new_datafrme.proportion),
        bins=bins_count
    )

    return histo_ht_wt_p


def find_very_hazy(directory):
    """
    Finds pictures that are really "hazy" i.e. there is no real straight line
    because they are blurred. Usually, at least the left or right tag
    should give straight lines, thus this function finds image of a certain
    questionable technique level.

    :param directory: The folder the images are in (should include final '/')
    :type directory: :func:`os.path.join()`

    :return: :code:`DataFrame` with images sorted as hazy or regular under
             :code:`label_for_haze`
    :rtype: :class:`~pandas.DataFrame`
    """
    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    hazy, lined = [], []
    for pic in suspects:
        image2 = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image2, 400, 200)
        z = edges[edges[:] > 0]
        asumy = z.sum()
        f = image2.shape[0]*image2.shape[1]
        sumdivf = z.sum()/f
        if sumdivf == 0:
            hazy.append(pic)

        else:
            lined.append(pic)

    dfr = pd.DataFrame(hazy)
    dfr['label_for_haze'] = 'hazy'
    dfh = pd.DataFrame(lined)
    dfh['label_for_haze'] = 'lined'
    df = pd.concat([dfr, dfh])

    return df


def find_by_sample_upper(
    source_directory,
    percent_height_of_sample,
    value_for_line
):
    """
    This function takes an average (mean) of upper pixels,
    and can show outliers defined by a percentage, i.e. the function shows
    images with an average of upper pixels in top x % where x is the percent
    height of the sample. Note: images with high averages in the upper pixels
    are likely to be inverted, upside down or otherwise different from more
    typical X-rays.

    :param source_directory: folder the images are in(should include final '/')
    :type source_directory: :func:`os.path.join()`
    :param percent_height_of_sample: From where on image to call upper
    :type percent_height_of_sample: int
    :param value_for_line: From where in pixel values to call averaged
        values abnormal
    :type value_for_line: int

    :return: :code:`DataFrame` with images labeled
    :rtype: :class:`~pandas.DataFrame`
    """
    suspects1 = glob.glob(os.path.join(source_directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(source_directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    estimates, piclist = [], []
    for pic in suspects:
        piclist.append(pic)
        example = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        height = example.shape[0]
        height_of_sample = int((percent_height_of_sample / 100) * height)
        estimates.append(np.mean(example[0:height_of_sample, :]))
    lovereturn = pd.DataFrame({
        'images': piclist,
        'estimates_b_find_by_sample_upper': estimates,
    })
    lovereturn['where'] = 'less'
    lovereturn.loc[
        lovereturn.estimates_b_find_by_sample_upper >= value_for_line,
        'where'
    ] = 'same or more'

    return lovereturn


def find_sample_upper_greater_than_lower(
    source_directory,
    percent_height_of_sample
):
    """
    Takes average of upper pixels, average of lower pixels (you define what
    percent of picture should be considered upper and lower) and compares.
    In a CXR if lower average is greater than upper it may be upside down
    or otherwise bizarre, as the neck is smaller than the abdomen.

    :param source_directory: folder the images are in(should include final '/')
    :type source_directory: :func:`os.path.join()`
    :param percent_height_of_sample: From where on image to call upper or lower
    :type percent_height_of_sample: int

    :return: :code:`DataFrame` with images labeled
    :rtype: :class:`~pandas.DataFrame`
    """
    estup, estdown, piclist = [], [], []
    suspects1 = glob.glob(os.path.join(source_directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(source_directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    # suspects = glob.glob(os.path.join(source_directory, '*.jpg'))
    for pic in suspects:
        piclist.append(pic)
        example = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        height = example.shape[0]
        height_of_sample = int((percent_height_of_sample / 100) * height)
        estup.append(np.mean(example[0:height_of_sample, :]))
        estdown.append(np.mean(example[(height - height_of_sample):height, :]))
    love = pd. DataFrame({
        'images': piclist,
        'estup': estup,
        'estdown': estdown,
    })
    love['which_greater'] = 'upper less'
    love.loc[love.estup > love.estdown, 'which_greater'] = 'upper more'
    return love


def find_outliers_by_total_mean(source_directory, percentage_to_say_outliers):
    """
    Takes the average of all pixels in an image, returns a :code:`DataFrame`
    with those images that are outliers by mean. This function can catch some
    inverted or otherwise problematic images

    :param source_directory: folder images in (include final /)
    :type source_directory: :func:`os.path.join()`
    :param percentage_to_say_outliers: Percentage to capture
    :type percentage_to_say_outliers: int

    :return: :code:`DataFrame` made up of outliers only
    :rtype: :class:`~pandas.DataFrame`
    """
    suspects1 = glob.glob(os.path.join(source_directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(source_directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    # suspects = glob.glob(os.path.join(source_directory, '*.jpg'))
    images, means = [], []
    for pic in suspects:
        example = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        mean = np.mean(example)
        images.append(pic)
        means.append(mean)
    df = pd.DataFrame({'images': images, 'means': means})
    df.sort_values(by='means', inplace=True)
    percentile = int((len(df) / 100) * percentage_to_say_outliers)
    lows = df.head(percentile)
    highs = df.tail(percentile)
    return (lows, highs)


def find_outliers_by_mean_to_df(source_directory, percentage_to_say_outliers):
    """
    Takes the average of all pixels in an image, returns a :code:`DataFrame`
    with those images classified. This function can catch some
    inverted or otherwise problematic images
    Important note: approximate, and the function can by chance cut the groups
    so images with the same mean are in and out of normal range,
    if the knife so falls

    :param source_directory: The folder in which the images are
    :type source_directory: :func:`os.path.join()`
    :param percentage_to_say_outliers: Percentage to capture
    :type percentage_to_say_outliers: int

    :return: :code:`DataFrame` all images, marked as high, low or within range
    :rtype: :class:`~pandas.DataFrame`
    """
    suspects1 = glob.glob(os.path.join(source_directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(source_directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    images, means = [], []
    for pic in suspects:
        example = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        mean = np.mean(example)
        images.append(pic)
        means.append(mean)
    df = pd.DataFrame({'images': images, 'means': means})
    df.sort_values(by='means', inplace=True)
    df.reset_index(inplace=True, drop=True)
    percentile = int((len(df) / 100) * percentage_to_say_outliers)
    df['results'] = 'within range'
    df.loc[:percentile, 'results'] = 'low'
    df.loc[len(df) - percentile:, 'results'] = 'high'
    return(df)


def create_matrix(width, height, default_element):
    # In python Sequence * Number = Sequence repeated Number of times
    """
    Takes width, height then creates a matrix populated by the default
    element. Super handy for advanced image manipulation. Note you can not
    create matrices bigger than your computer memory can handle making.
    Therefore the function will work on matrices with dimensions up to
    maybe 500*500 depending

    :param width: Width of the matrix to be created
    :type width: int
    :param height: The height of matrix to be created
    :type height: int
    :param default_element: Element to populate the matrix with
    :type default_element: Union[float, int, str]

    :return: 2D matrix populated
    :rtype: list
    """
    result = [0] * width

    for i in range(width):
        result[i] = [default_element] * height

    return result


def find_tiny_image_differences(directory, s=5, percentile=8):
    """
    Finds differences between a manufactured tiny image, and all your images at
    that size. If you return the outliers they are inverted,
    or dramatically different in some way. Note: percentile returned
    is approximate, may be a tad more

    :param directory: Directory with source images.
    :type directory: Suitable for :func:`os.path.join()`
    :param s: length to make the sides of the tiny image for comparison
    :type s: int
    :param percentile: percentile to mark as abnormal
    :type percentile: int

    :return: :code:`DataFrame` with a column that notes mismatches
             and within range images
    :rtype: :class:`~pandas.DataFrame`
    """
    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    avg_image = separate_image_averager(suspects, s)
    images, sums = [], []
    for pic in suspects:
        example = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        example_clipped = crop_np(example)
        example_small = cv2.resize(example_clipped, (s, s))
        experiment_a = (example_small - avg_image) ** 2
        experiment_sum = experiment_a.sum()
        images.append(pic)
        sums.append(experiment_sum)
    df = pd.DataFrame({'images': images, 'sums': sums})
    df.sort_values('sums', inplace=True, ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    df['results'] = 'within range'
    df.loc[int((len(df) / 100) * (100 - percentile)):, 'results'] = 'mismatch'
    return df


def tesseract_specific(directory):
    # this function runs tessseract ocr for text detection over images
    # in a directory, and gives a DataFrame with what it found
    """
    Finds images with text on them. Multi-lingual including English.

    :param directory: Directory with source images.
    :type directory: Suitable for :func:`os.path.join()`

    :return: :code:`DataFrame` with a column of text found
    :rtype: :class:`~pandas.DataFrame`
    """
    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    # suspects = glob.glob(os.path.join(directory, '*.jpg'))
    texts, clean_texts, confidences = [], [], []

    with PyTessBaseAPI() as api:
        for pic in suspects:
            api.SetImageFile(pic)
            texts.append(api.GetUTF8Text())
            ctext = api.GetUTF8Text().strip()
            dctext = ctext.strip()
            clean_texts.append(dctext)
            # confidences.append(api.AllWordConfidences())
    df = pd.DataFrame({
        'images': suspects,
        'text': texts,
        'clean_text': clean_texts,
        # 'confidence': confidences,
    })
    return df


def find_suspect_text(directory, label_word):
    # this function looks for one single string in texts (multilingual!)
    # on images
    """
    Finds images with a specific text you ask for on them.
    Multi-lingual including English. Accuracy is very high, but not perfect.

    :param directory: Directory with source images.
    :type directory: Suitable for :func:`os.path.join()`
    :param label_word: Label word
    :type label_word: str

    :return: :code:`DataFrame` with a column of text found over the length
    :rtype: :class:`~pandas.DataFrame`
    """

    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    # suspects = glob.glob(os.path.join(directory, '*.jpg'))
    images, texts, clean_texts = [], [], []

    with PyTessBaseAPI() as api:
        for pic in suspects:
            api.SetImageFile(pic)
            texts.append(api.GetUTF8Text())

            ctext = api.GetUTF8Text().strip()

            clean_texts.append(ctext)
            images.append(pic)
    df = pd.DataFrame({'images': images, 'text': clean_texts})
    df = df[df['text'].str.contains(label_word)]

    return df


def find_suspect_text_by_length(directory, length):
    # this function finds all texts above a specified length
    # (length is number of characters)
    """
    Finds images with text over a specific length (of letters, digits,
    and spaces), specified by you the user.
    Useful if you know you do not care about R and L or SUP.
    Multi-lingual including English. Accuracy is very high, but not perfect.

    :param directory: Directory with source images.
    :type directory: Suitable for :func:`os.path.join()`
    :param length: Length to find above, inclusive
    :type length: int

    :return: :code:`DataFrame` with a column of text found
    :rtype: :class:`~pandas.DataFrame`
    """
    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    # suspects = glob.glob(os.path.join(directory, '*.jpg'))
    images, texts, clean_texts = [], [], []
    with PyTessBaseAPI() as api:
        for pic in suspects:
            api.SetImageFile(pic)
            texts.append(api.GetUTF8Text())

            ctext = api.GetUTF8Text().strip()

            clean_texts.append(ctext)
            images.append(pic)
    # n = len(label_word_list)#return grabbed, pic
    df = pd.DataFrame({'images': images, 'text': clean_texts})
    yes = df[df['text'].str.len() >= length]
    no = df[df['text'].str.len() < length]
    no['text'] = 'shorter'
    df = pd.concat([yes, no])
    # df.reset_index(drop=True, inplace=True)
    return df


def histogram_difference_for_inverts(directory):
    # this function looks for images by a spike on the end of pixel
    # value histogram to find inverted images
    """
    This function looks for images by a spike on the end of their pixel
    value histogram to find inverted images. Note we assume classical
    X-rays, not inverted fluoroscopy images.

    :param directory: Directory with source images.
    :type directory: Suitable for :func:`os.path.join()`

    :return: a list of images suspected to be inverted
    :rtype: list
    """

    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    # suspects = glob.glob(os.path.join(directory, '*.jpg'))
    regulars, inverts, unclear = [], [], []
    for pic in suspects:
        example = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        lows = np.count_nonzero(example < (example.min()+11))
        highs = np.count_nonzero(example > (example.max()-11))
        if lows > highs:
            regulars.append(pic)
        elif highs > lows:
            inverts.append(pic)
        else:
            unclear.append(pic)

    return inverts


def inverts_by_sum_compare(directory):
    """
    This function looks for images and compares them to their inverts. In the
    case of inverted typical CXR images the sum of all pixels in the image will
    be higher than the sum of pixels in the un-inverted (or inverted*2) image

    :param directory: Directory with source images.
    :type directory: Suitable for :func:`os.path.join()`

    return: a :code:`DataFrame` with images categorized
    :rtype: :class:`~pandas.DataFrame`
    """

    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    regulars, inverts = [], []
    for pic in suspects:
        example = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        example_inv = cv2.bitwise_not(example)
        if example.sum() > example_inv.sum():
            inverts.append(pic)
        else:
            regulars.append(pic)
    dfr = pd.DataFrame(regulars)
    dfr['label'] = 'regular'
    dfh = pd.DataFrame(inverts)
    dfh['label'] = 'inverts'
    df = pd.concat([dfr, dfh])
    df = df.rename(columns={0: 'image', 'label': 'label'})
    return df


def histogram_difference_for_inverts_todf(directory):
    # looks for inverted and returns a DataFrame
    """
    This function looks for images by a spike on the end of their pixel
    value histogram to find inverted images, then puts what it found into
    a :code:`DataFrame`. Images are listed as regulars, inverts of unclear (the
    unclear have equal spikes on both ends). #histo

    :param directory: Directory with source images.
    :type directory: Suitable for :func:`os.path.join()`

    :return: a :code:`DataFrame` with images categorized
    :rtype: :class:`~pandas.DataFrame`
    """
    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    # suspects = glob.glob(os.path.join(directory, '*.jpg'))
    regulars, inverts, unclear = [], [], []
    for pic in suspects:
        example = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        lows = np.count_nonzero(example < (example.min()+11))
        highs = np.count_nonzero(example > (example.max()-11))
        if lows > highs:
            regulars.append(pic)
        elif highs > lows:
            inverts.append(pic)
        else:
            unclear.append(pic)

    dfr = pd.DataFrame(regulars)
    dfr['label'] = 'regular'
    dfh = pd.DataFrame(inverts)
    dfh['label'] = 'inverts'
    dfl = pd.DataFrame(unclear)
    dfl['label'] = 'unclear'
    df = pd.concat([dfr, dfh, dfl])
    df = df.rename(columns={0: 'image', 'label': 'label'})
    return df


def find_duplicated_images(directory):
    # this function finds duplicated images and return a list
    """
    Finds duplicated images and returns a list of them.

    :param directory: Directory with source images.
    :type directory: Suitable for :func:`os.path.join()`

    :return: a list of duplicated images
    :rtype: list
    """
    picture_directory = Path(directory)
    files = sorted(os.listdir(picture_directory))
    duplicates = []
    class_list = []
    # comparison of the files
    for file in files:

        is_duplicate = False

        for class_ in duplicates:
            is_duplicate = filecmp.cmp(
                picture_directory / file,
                picture_directory / class_[0],
                shallow=False
            )
            if is_duplicate:
                class_list.append(file)
                break

        if not is_duplicate:
            duplicates.append([file])

    return class_list


def find_duplicated_images_todf(directory):
    # looks for duplicated images, returns DataFrame
    """
    Finds duplicated images and returns a :code:`DataFrame` of them.

    :param directory: Directory with source images.
    :type directory: Suitable for :func:`os.path.join()`

    :return: a :code:`DataFrame` of duplicated images
    :rtype: :class:`~pandas.DataFrame`
    """
    picture_directory = Path(directory)
    files = sorted(os.listdir(picture_directory))
    duplicates, class_list = [], []

    # comparison of the files
    for file in files:

        is_duplicate = False

        for class_ in duplicates:
            is_duplicate = filecmp.cmp(
                picture_directory / file,
                picture_directory / class_[0],
                shallow=False
            )
            if is_duplicate:
                class_list.append(file)
                break

        if not is_duplicate:
            duplicates.append([file])
    df = pd.DataFrame(duplicates)  # , columns = 'image_name' )
    df['status'] = 'not duped'
    df = df.rename(columns={0: 'images'})
    pdf = pd.DataFrame(class_list)
    pdf['status'] = 'duplicated'
    pdf = pdf.rename(columns={0: 'images'})
    final_df = pd.concat([df, pdf])
    final_df.reset_index(drop=True, inplace=True)

    return final_df


def show_images_in_df(iter_ob, length_name):
    """
    Shows images by taking them off a :code:`DataFrame` column, and displays
    them but in smaller versions, so they can be compared quickly

    :param iter_ob: List, chould be a :code:`DataFrame` column, use .to_list()
    :type iter_ob: list
    :param length_name: Size of image name going from end
    :type length_name: int

    :return: technically no return but makes a plot of images with names
    :rtype: none
    """

    width = int(math.sqrt(len(iter_ob)))
    height = int(math.ceil(len(iter_ob) / width))
    f, axarr = plt.subplots(width, height, figsize=(14, 14))
    if width > 1:
        for x in range(width):
            for y in range(min(len(iter_ob) - x * height, height)):
                element = iter_ob[x * height + y]
                fname = os.path.splitext(element)[0]
                title = fname[-length_name:]
                exop = cv2.imread(element, cv2.IMREAD_GRAYSCALE)
                axarr[x, y].set_title(title)
                axarr[x, y].imshow(exop, cmap=plt.cm.gray)

    else:
        for y in range(height):
            element = iter_ob[y]
            exop = cv2.imread(element, cv2.IMREAD_GRAYSCALE)
            fname = os.path.splitext(element)[0]
            title = fname[-length_name:]
            axarr[y].set_title(title)
            axarr[y].imshow(exop, cmap=plt.cm.gray)
            # plt.title('Outlier images')
    plt.show()


def dataframe_up_my_pics(directory, diagnosis_string):
    """
    Takes images in a directory (should all be with same label), and puts the
    name (with path) and label into a :code:`DataFrame`

    :param directory: Directory with source images.
    :type directory: Suitable for :func:`os.path.join()`
    :param diagnosis_string: Usually a label, may be any string
    :type diagnosis_string: str

    :return: :code:`DataFrame` of pictures and label
    :rtype: :class:`~pandas.DataFrame`
    """
    picture_directory = Path(directory)
    files = sorted(os.listdir(picture_directory))
    dupli = []
    for file in files:
        dupli.append(file)
    df = pd.DataFrame(dupli)
    df['diagnosis'] = diagnosis_string
    df = df.rename(columns={0: 'identifier_pic_name'})
    return df


class Rotator:
    """Class for rotating OpenCV images. """

    class RotationIterator:
        """
        Class RotationIterator iterator implementation of a range of
        rotated images
        """

        def __init__(self, rotator, start, end, step):
            """Creates an instance of RotationIterator.

            :param rotator: The Rotator object for which this is an iterator.
            :type rotator: Rotator
            :param start: The initial angle (in degrees).
            :type start: numeric
            :param end: The final angle (in degrees).
            :type end: numeric
            :param step: Increment (in degrees).
            :type step: numeric
            """
            self.rotator = rotator
            self.seq = np.arange(start, end, step)
            self.pos = 0

        def __next__(self):
            """Necessary iterator implementation."""
            if self.pos >= len(self.seq):
                raise StopIteration()
            result = self.rotator[self.seq[self.pos]]
            self.pos += 1
            return result

        def __iter__(self):
            """Implementation of iteratble protocol"""
            return self

    def __init__(self, image, center=None, scale=1.0):
        """
        Creates a wrapper object that allows creation of ranges of
        rotation of the given image.

        :param image: OpenCV image
        :type image: :code:`cv2.Image`
        :param center: Coordinate of the center of rotation
                       (defaults to the middle of the image).
        :param scale: Scale ratio of the resulting image
                      (after rotation, defaults to 1.0).
        """
        self.image = image
        self.center = center
        self.scale = scale
        self.h, self.w = self.image.shape[:2]
        if self.center is None:
            self.center = (self.w / 2, self.h / 2)

    def __getitem__(self, angle):
        """
        Generates an image rotated by :code:`angle` degrees.
        credits to: https://stackoverflow.com/a/32929315/5691066
        for this approach to cv2 rotation.
        """
        matrix = cv2.getRotationMatrix2D(self.center, angle, self.scale)
        rotated = cv2.warpAffine(self.image, matrix, (self.w, self.h))
        return rotated

    def iter(self, start=0, end=360, step=1):
        """
        Class method :code:`iter` returns a generator group of images that
        are on angles from :code:`start` to :code:`stop` with increment of
        :code:`step`.
        Usage example:

        .. code-block:: python

           image = cv2.imread('normal-frontal-chest-x-ray.jpg')
           rotator = Rotator(image)
           for rotated in rotator.iter(0, 360, 10):
                # shows the np arrays for the 36 (step=10) images
               print(rotated)
        """
        return self.RotationIterator(self, start, end, step)


def simple_spinning_template(
    picy,
    greys_template,
    angle_start,
    angle_stop,
    slices,
    threshold4=.7,
):
    """
    This function creates an image compared to a rotated template as an image.

    :param picy: String for image name of base image
    :type picy: str
    :param greys_template: The image array of the template,
    :type greys_template: :class:`~numpy.ndarray`
    :param angle_start: angle to spin template to, it would normally start at
                        zero if picking up exact template itself is desired
    :type angle_start: float
    :param angle_stop: last angle to spin template to,
    :type angle_stop: float
    :param slices: number of different templates to make between angles
    :type slices: float
    :param threshold4: A number between zero and one which sets the precision
                       of matching. NB: .999 is stringent, .1 will pick up
                       too much
    :type threshold4: float

    :return: copy_image, a copy of base image with the template areas caught
        outlined in blue rectangles
    :rtype: :class:`~numpy.ndarray`
    """

    pic = picy
    img_rgb = cv2.imread(pic)
    copy_image = cv2.imread(pic)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # rotator_generator = generator
    rotator = Rotator(greys_template)
    for element in rotator.iter(angle_start, angle_stop, slices):
        template = element
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = threshold4
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(
                copy_image,
                pt,
                (pt[0] + w, pt[1] + h),
                (0, 0, 255),
                2,
            )
    return copy_image


def make_contour_image(im):

    """
    Makes an image into a contour image
    :param im: image name
    :type im: str

    :return: drawing, the contour image
    :rtype: :class:`~numpy.ndarray`
    """
    imgL = cv2.imread(im)
    cv2_vers = cv2.__version__
    major_cv2 = int(cv2.__version__.split('.')[0])
    edges = cv2.Canny(imgL, 0, 12)
    thresh = 128
    # get threshold image
    ret, thresh_img = cv2.threshold(edges, thresh, 255, cv2.THRESH_BINARY)
    # find contours
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
    img_contours = np.zeros(imgL.shape)
    drawing = cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
    return drawing


def avg_image_maker(set_of_images):
    """
    This function shows you an average sized image that has been made with the
    average per pixel place (in a normalized matrix) of all images averaged
    from the :code:`set_of_images` group.

    :param set_of_images: A set of images, can be read in with
                          :func:`glob.glob()` on a folder of jpgs.
    :type set_of_images: list

    :return: final_avg, an image that is the average image of images in the set
    :rtype: :class:`~numpy.ndarray`
    """
    list_h = []
    list_w = []

    for example in set_of_images:
        example = cv2.imread(example, cv2.IMREAD_GRAYSCALE)
        ht = example.shape[0]
        wt = example.shape[1]
        list_h.append(ht)
        list_w.append(wt)

    h = int(sum(list_h)/len(list_h))
    w = int(sum(list_w)/len(list_w))
    canvas = np.zeros((h, w))
    for example in set_of_images:
        example = cv2.imread(example, cv2.IMREAD_GRAYSCALE)
        example_small = cv2.resize(example, (w, h))
        canvas += np.array(example_small)
    final_avg = canvas / len(set_of_images)
    return final_avg


def set_image_variability(set_of_images):
    """
    This function shows you an average sized image created to show variability
    per pixel if all images were averaged (in terms of size) and compared.
    Here you will see where the variability- and therefore in some cases
    pathologies like pneumonia can be typically located, as well as patient-
    air interface (not all subjects same size) and other obviously variable
    aspects of your image set.

    :param set_of_images: A set of images, can be read in with
                          :func:`glob.glob()` on a folder of jpgs.
    :type set_of_images: list

    :return: Final_diff, an image that is the average virability per pixel
             of the image in images in the set.
    :rtype: :class:`~numpy.ndarray`
    """
    final_avg = avg_image_maker(set_of_images)

    h = final_avg.shape[0]
    w = final_avg.shape[1]
    diff = np.zeros((h, w))
    print("diff image", diff.shape)
    for example in set_of_images:
        example = cv2.imread(example, cv2.IMREAD_GRAYSCALE)
        example_small = cv2.resize(example, (w, h))
        print("example_small image", example_small.shape)
        diff += (example_small - final_avg)**2
    final_diff = diff / len(set_of_images)
    return final_diff


def avg_image_maker_by_label(
    master_df,
    dataframe_image_column,
    dataframe_label_column,
    image_folder,
):

    """
    This function sorts images by labels and makes an average image per label.
    If images are all the same size subtracting one from the other should
    reveal salient differences. N.B. blending different views e.g. PA and
    lateral is not suggested.
    :param master_df: Dataframe with image location and labels
    (must be in image folder)
    :type master_df: :class:`~pandas.DataFrame`
    :param dataframe_image_column: name of dataframe column with image location
    string
    :type dataframe_image_column: str
    :param dataframe_label_column: name of dataframe column with label string
    :type dataframe_label_column: str
    :param image_folder: name of folder where images are
    :type image_folder: str

    :return: list of titled average images per label
    :rtype: list
    """
    final_img_list = []
    final_name_list = []
    set_of_labels = master_df[dataframe_label_column].unique()
    for name in set_of_labels:
        sets_of_images = []

        list_h = []
        list_w = []

        by_label = master_df[dataframe_image_column][
            master_df[dataframe_label_column] == name
        ]
        for example in by_label:
            example = cv2_imread(
                os.path.join(image_folder, example),
                cv2.IMREAD_GRAYSCALE,
            )
            ht = example.shape[0]
            wt = example.shape[1]
            list_h.append(ht)
            list_w.append(wt)

        h = int(sum(list_h) / len(list_h))
        w = int(sum(list_w) / len(list_w))
        canvas = np.zeros((h, w))
        for example in master_df[dataframe_image_column]:
            example = cv2.imread(
                (os.path.join(image_folder, example)),
                cv2.IMREAD_GRAYSCALE,
            )
            example_small = cv2.resize(example, (w, h))
            canvas += np.array(example_small)
        final_avg = canvas / len(dataframe_image_column)
        final_img_list.append(final_avg)
        final_name_list.append(name)
    stick = pd.DataFrame(final_name_list)
    stick_data = {'name': final_name_list, 'images': final_img_list}

    df = pd.DataFrame(data=stick_data)

    stick['images'] = final_img_list
    return df


def zero_to_twofivefive_simplest_norming(img_pys):
    """
    This function takes an image and makes the highest pixel value 255,
    and the lowest zero. Note that this will not give anything like
    a true normalization, but will put all images
    into 0 to 255 values

    :param img_pys: Image name.
    :type img_pys: str

    :return: OpenCV image.
    :rtype: :code:`cv2.Image`
    """
    img_py = cv2.imread(img_pys, cv2.IMREAD_GRAYSCALE)

    new_max_value = 255

    max_value = np.amax(img_py)
    min_value = np.amin(img_py)

    img_py = img_py - min_value
    multiplier_ratio = new_max_value/max_value
    img_py = img_py*multiplier_ratio

    return img_py


def rescale_range_from_histogram_low_end(img, tail_cut_percent):
    """
    This function takes an image and makes the highest pixel value 255,
    and the lowest zero. It also normalizes based on the histogram
    distribution of values, such that the lowest percent
    (specified by tail_cut_percent) all become zero.
    The new histogram will be more sparse, but resamples
    should fix the problem (presumably you will have to sample down
    in size for a neural net anyways)

    :param img_pys: NumPy array with image data.
    :type img: :class:`~numpy.ndarray`
    :param tail_cut_percent: Percent of histogram to be discarded from low end
    :type tail_cut_percent: int


    :return: New NumPy array with image data.
    :rtype: :class:`~numpy.ndarray`
    """
    # set arbitrary variables
    new_max_value = 255
    # new_min_value = 0

    img_py = np.array((img), dtype='int64')

    num_total = img_py.shape[0]*img_py.shape[1]

    list_from_array = img_py.tolist()
    gray_hist = np.histogram(img_py, bins=256)[0]
    area = gray_hist.sum()
    cutoff = area * (tail_cut_percent/100)
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
    min_value2 = np.amin(img_py)
    multiplier_ratio = new_max_value/max_value2
    img_py = img_py*multiplier_ratio

    return img_py


def make_histo_scaled_folder(imgs_folder, tail_cut_percent, target_folder):
    """
    This function takes each image inside a folder and normalizes them by the
    histogram. It then puts the new normalized images in to a folder
    which is called the target folder (to be made by user)

    :param imgs_folder: Foulder with source images.
    :type imgs_folder: str
    :param tail_cut_percent: Percent of histogram to be discarded from low end
    :type tail_cut_percent: int

    :return: Target_name, but your images go into target folder with
             target_name.
    :rtype: str
    """

    suspects1 = glob.glob(os.path.join(imgs_folder, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(imgs_folder, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects2 + suspects1

    for individual_pic in suspects:
        individual_img = cv2.imread(individual_pic, cv2.IMREAD_GRAYSCALE)
        results = rescale_range_from_histogram_low_end(
            individual_img, tail_cut_percent,
        )
        basename = os.path.basename(individual_pic)
        target_name = os.path.join(target_folder, basename)
        cv2.imwrite(target_name, results)

    return target_name


def give_size_count_df(folder):
    """
    This function returns a dataframe of the unique sizes of the images ,and
    how many images have such a size.

    :param folder: folder with jpgs
    :type folder: string

    :return: df
    :rtype: pandas.core.frame.DataFrame
    """
    to_be_sorted = glob.glob(os.path.join(folder, '*.jpg'))
    pic_list = []
    heights = []
    widths = []
    dimension_groups = []
    for picy in to_be_sorted:
        example = cv2.imread(picy, cv2.IMREAD_GRAYSCALE)
        height = example.shape[0]
        width = example.shape[1]
        height_width = 'h' + str(height) + '_w' + str(width)
        heights.append(height)
        widths.append(width)
        pic_list.append(picy)
        dimension_groups.append(height_width)
        d = {
            'pics': pic_list,
            'height': heights,
            'width': widths,
            'height_width': dimension_groups,
        }
        data = pd.DataFrame(d)
        data = data.sort_values('height_width')
        compuniquesizes = data.height_width.unique()
        len_list = []
    size_name_list = []
    sizesdict = {elem: pd.DataFrame() for elem in compuniquesizes}
    for key in sizesdict.keys():
        sizesdict[key] = data[:][data.height_width == key]
    for sized in compuniquesizes:
        lener = len(sizesdict[sized])
        len_list.append(lener)
        size_name_list.append(sized)
    sized_data = {'size': size_name_list, 'count': len_list}
    df = pd.DataFrame(sized_data)
    return df


def give_size_counted_dfs(folder):
    """
    This function returns dataframes of uniquely sized images in a list

    :param folder: folder with jpgs
    :type folder: string

    :return: big_sizer
    :rtype: list
    """
    to_be_sorted = glob.glob(os.path.join(folder, '*.jpg'))
    pic_list = []
    heights = []
    widths = []
    dimension_groups = []
    for picy in to_be_sorted:
        example = cv2.imread(picy, cv2.IMREAD_GRAYSCALE)
        height = example.shape[0]
        width = example.shape[1]
        height_width = 'h' + str(height) + '_w' + str(width)
        heights.append(height)
        widths.append(width)
        pic_list.append(picy)
        dimension_groups.append(height_width)
        d = {
            'pics': pic_list,
            'height': heights,
            'width': widths,
            'height_width': dimension_groups
        }
        data = pd.DataFrame(d)
        data = data.sort_values('height_width')
        compuniquesizes = data.height_width.unique()
        len_list = []
    size_name_list = []
    sizesdict = {elem: pd.DataFrame() for elem in compuniquesizes}
    for key in sizesdict.keys():
        sizesdict[key] = data[:][data.height_width == key]
    big_sizer = []
    for nami in compuniquesizes:
        frames = sizesdict[nami]
        big_sizer.append(frames)
    return big_sizer


def image_quality_by_size(specific_image):
    """
    This function returns the size of an image which can indicate one aspect of
    quality (can be used as a helper function)

    :param specific_image: the jpg image
    :type specific_imager: string

    :return: q
    :rtype: int
    """
    q = os.stat(specific_image).st_size
    return q


def find_close_images(folder, compression_level, ref_mse):
    """
    This function finds potentially duplicated images by
    comparing compressed versions of the images.

    :param folder: folder with jpgs
    :type folder: str
    :param compression_level: size to compress down to
    :type compression_level: float
    :param ref_mse: mse is a mean squared error
    :type ref_mse: float

    :return: near_dupers
    :rtype: :class:`~pandas.DataFrame`
    """
    compression = compression_level
    # lists of the found duplicate/similar images, images, and err
    duplicates_A = []
    duplicates_B = []
    images_A = []
    images_B = []
    image_files = []
    err_list = []
    # list of all files in directory
    suspects1 = glob.glob(os.path.join(folder, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(folder, '*.[Jj][Pp][Ee][Gg]'))
    folder_files = suspects1 + suspects2

    # create images array
    is_first = True
    for filename in folder_files:
        img = cv2.imread(filename)
        if type(img) is np.ndarray:
            img = img[..., 0:3]
            # resize the image based to compression level value
            img = cv2.resize(
                img,
                dsize=(compression, compression),
                interpolation=cv2.INTER_CUBIC,
            )
            if is_first:
                imgs_array = img
                is_first = False
            else:
                imgs_array = np.concatenate((imgs_array, img))

            image_files.append(filename)
    # cook it
    img_area = float(compression ** 2)
    total_images = len(image_files)

    for i in range(total_images):
        for j in range(i + 1, total_images):
            srow_A = i * compression
            erow_A = (i + 1) * compression
            srow_B = j * compression
            erow_B = (j + 1) * compression
            img_name_A = image_files[i]
            img_name_B = image_files[j]
            # select two images from imgs_matrix
            imgA = imgs_array[srow_A:erow_A]
            imgB = imgs_array[srow_B:erow_B]
            # compare the images
            err = np.sum(
                (imgA.astype("float") - imgB.astype("float")) ** 2
            ) / img_area
            if err < ref_mse:
                duplicates_A.append(img_name_A)
                duplicates_B.append(img_name_B)
                err_list.append(err)

    dupers = {
        'twinA?': duplicates_A,
        'twinB?': duplicates_B,
        'mse': err_list,
    }
    near_dupers = pd.DataFrame(dupers)

    print("\n***\n Output: ", str(len(duplicates_A)),
          " potential duplicate image pairs in ", str(len(image_files)),
          " total images.\n",
          "At compression level", compression, "and mse", ref_mse)

    return near_dupers


def show_close_images(folder, compression_level, ref_mse, plot_limit=20):
    """
    This function shows potentially duplicated images by comparing compressed
    versions of the images, then displays them for inspection.

    :param folder: folder with jpgs
    :type folder: str
    :param compression_level: size to compress down to
    :type compression_level: float
    :param ref_mse: mse is a mean squared error
    :type ref_mse: float
    :param plot_limit: How many images to plot when showing duplicates.
                       Negative values mean to show all images.
    :type plot_limit: int
    """
    df = find_close_images(folder, compression_level, ref_mse)
    ndisplayed = min(len(df), plot_limit) if plot_limit > 0 else len(df)
    df_displayed = df.head(ndisplayed)
    f, axarr = plt.subplots(ndisplayed, 2, figsize=(14, ndisplayed * 4))

    for i, row in df_displayed.iterrows():
        imgA = cv2.imread(row['twinA?'], cv2.IMREAD_GRAYSCALE)
        imgB = cv2.imread(row['twinB?'], cv2.IMREAD_GRAYSCALE)
        nameA = os.path.basename(row['twinA?'])
        nameB = os.path.basename(row['twinB?'])
        mse = row['mse']
        if ndisplayed > 1:
            axarr[i, 0].set_title('{}, mse: {}'.format(nameA, mse))
            axarr[i, 0].imshow(imgA, cmap=plt.cm.gray)
            axarr[i, 0].axis('off')
            axarr[i, 1].set_title('{}, mse: {}'.format(nameB, mse))
            axarr[i, 1].imshow(imgB, cmap=plt.cm.gray)
            axarr[i, 1].axis('off')
        else:
            axarr[0].set_title('{}, mse: {}'.format(nameA, mse))
            axarr[0].imshow(imgA, cmap=plt.cm.gray)
            axarr[0].axis('off')
            axarr[1].set_title('{}, mse: {}'.format(nameB, mse))
            axarr[1].imshow(imgB, cmap=plt.cm.gray)
            axarr[1].axis('off')
    plt.show()


def image_to_histo(image):
    """
    This is a small helper function that makes returns the arrray of an
    image histogram
    :param image: the image as an array (not filename)
    :type image: array

    :return: histogram
    :rtype: float
    """
    histogram = np.histogram(image, bins=256, range=(0, 255))
    return histogram[0]


def black_end_ratio(image_array):
    """
    This is a function to assess a specific parameter of image quality.
    The parameter checked is whether there are enough very dark/black pixels.
    In a normal chest X-ray we would expect black
    around the neck, and therefore have a lot of those low values.
    If the image was shot without the neck, we will assume poor technique (note
    in some theoretical cases this technique might have been requested,
    but it is not standard at ALL)
    If the ratio is below 0.3, you have a chestX-ray that is unusual in value
    distributions, and in 9.9/10 cases one shot with poor technique.
    The images MUST be cropped of any frames and normalized to 0-255.

    :param image_array: the image as an array
    :type image_array: array

    :return: ratio
    :rtype: float
    """
    low_end = image_to_histo(image_array)[0:50].sum()
    mid_1 = image_to_histo(image_array)[51:100].sum()
    mid_2 = image_to_histo(image_array)[101:150].sum()
    mid_3 = image_to_histo(image_array)[151:200].sum()
    ender = image_to_histo(image_array)[200:255].sum()
    ratio = low_end/np.mean([mid_1, mid_2, mid_3, ender])
    return ratio


def outline_segment_by_otsu(image_to_transform, blur_k_size=1):
    """
    This is a function to turn an Xray into an outline
    with a specific method that involves an implementation
    of Otsu's algorithm, and cv2 version of Canny
    the result is line images that can be very useful
    in and of themselves to run a neural net on
    or can be used for segmentation in some cases
    blur_k_size used in a blur to make our lines less detailed
    if set to a higher value, 0 < values < 100, and odd

    :param image_to_transform: the image name
    :type image_to_transform: string
    :param blur_k_size: must be odd and value <100, kernel to blur
    to make ourlines less detailed
    :type blur_k_size: int

    :return: edges (an image with lines)
    :rtype: numpy.ndarray
    """
    # read in  image
    image_to_transform = cv2.imread(image_to_transform, cv2.IMREAD_GRAYSCALE)
    # find it's np.histogram
    bins_num = 256

    hist, bin_edges = np.histogram(image_to_transform, bins=bins_num)

    # find the threshold for Otsu segmentation
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    wght = np.cumsum(hist)
    wght2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_mids) / wght
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / wght2[::-1])[::-1]
    # compute interclass variance
    inter_class_vari = wght[:-1] * wght2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    index_of_max_val = np.argmax(inter_class_vari)

    thresh = bin_mids[:-1][index_of_max_val]
    # width, height = image_to_transform.shape[0], image_to_transform.shape[1]
    # this gave the thresh based on Otsu, now apply it
    output_image = image_to_transform
    # for x in range(width):
    #     for y in range(height):
    #         # for the given pixel at w,h, check value against the threshold
    #         if output_image[x, y] < thresh:
    #             # lets set this to zero
    #             output_image[x, y] = 0
    mask = output_image < thresh
    output_image[mask] = 0
    # output_image[~mask] = 255
    if blur_k_size % 2 != 0:
        # blur
        output_image = cv2.medianBlur(output_image, blur_k_size)
    else:
        output_image = cv2.medianBlur(output_image, (blur_k_size + 1))

    # now use canny to get edges
    edges = cv2.Canny(output_image, 50, 230)
    return edges


def binarize_by_otsu(image_to_transform, blur_k_size):
    """
    This is a function to turn an Xray into an binarized image
    with a specific method that involves an implementation
    of Otsu's algorithm,
    the result is line images that can be very useful
    in and of themselves to run a neural net on
    or can be used for segmentation in some cases
    blur_k_size used in a blur to make our lines less detailed
    if set to a higher value, 0 < values < 100, and odd

    :param image_to_transform: the image name
    :type image_to_transform: string
    :param blur_k_size: must be odd and value <100, kernel to blur
    to make ourlines less detailed
    :type blur_k_size: int

    :return: output_image (an image binarized to 0s or 255s)
    :rtype: numpy.ndarray
    """
    # read in  image
    image_to_transform = cv2.imread(image_to_transform, cv2.IMREAD_GRAYSCALE)
    # find it's np.histogram
    bins_num = 256

    hist, bin_edges = np.histogram(image_to_transform, bins=bins_num)

    # find the threshold for Otsu segmentation
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    wght = np.cumsum(hist)
    wght2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_mids) / wght
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / wght2[::-1])[::-1]
    # compute interclass variance
    inter_class_vari = wght[:-1] * wght2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    index_of_max_val = np.argmax(inter_class_vari)

    thresh = bin_mids[:-1][index_of_max_val]
    # this gave the thresh based on Otsu, now apply it
    output_image = image_to_transform
    mask = output_image < thresh
    output_image[mask] = 0
    output_image[~mask] = 255
    # now use canny to get edges
    output_image = cv2.medianBlur(output_image, blur_k_size)
    return output_image


def column_sum_folder(directory):
    """
    Takes images in directory and makes a graph for each image
    of sums along horizontal or vertical lines
    this is saved as an accompanying image.
    Returns a dataframe with this information for each image,
    but also deposits new images into a new folder
    because each run will include the newly made images.
    NB: This is a home-made projection algorithm.
    Projection algorithms can be used in image registration,
    and future versions of cleanX will have more efficient projection
    algorithms. Also note the df will be enormous...

    :param directory: Directory with set_of_images.
    :type directory: string

    :return: sumpix_df (df with info from new images of column sums)
    :rtype: pandas.core.frame.DataFrame

    """
    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2
    names_list = []
    sumpix_list0 = []
    sumpix_list1 = []
    for pic in suspects:
        pic_name = pic.split('\\')[1]
        # check if directory exists or not yet
        new_directory = os.path.join(directory, 'column_pics')
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        if os.path.exists(new_directory):
            file_path = os.path.join(new_directory, pic_name)

        sumpix0 = 0
        sumpix1 = 0
        img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        sumpix0 = np.sum(img, 0)
        sumpix1 = np.sum(img, 1)
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes.clear()
        plt.plot(sumpix0)
        plt.plot(sumpix1)
        plt.title('Sum pixels along (vertical and horizontal) columns')
        plt.xlabel('Column number')
        plt.ylabel('Sum of column')
        fig.savefig(file_path)

        sumpix_list0.append(sumpix0)
        sumpix_list1.append(sumpix1)
        names_list.append(pic)
    data = list(zip(sumpix_list0, sumpix_list1))
    sumpix_df = pd.DataFrame(data, index=names_list)
    return sumpix_df


def blind_quality_matrix(directory):
    """
    Creates a dataframe of image quality charecteristics
    including: laplacian variance (somewhat correlated to blurriness/
    resolution),total pixel sum (somewhat correlated to exposure),
    and a fast Fourier transform variance measure
    (correlated to resolution and contrast),
    contrast by two different measures (standard deviation, and Michaelson),
    bit depth (with an eye to a future when there may well be higher bit depths
    , although probably not on your screen since at some point these
    distinctions go beyond human eye ability)
    and filesize divided by image area
    The data frame is colored with a diverging color scheme (purple low,
    green high) map so that groups of images can be compared intuitively
    NB: images should be roughly comparable in dimension size for results
    to be meaningful.

    :param directory: Directory with set_of_images.
    :type directory: string

    :return: frame (dataframe)
    :rtype: class 'pandas.io.formats.style.Styler'

    """

    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2

    names = []
    laplacian_var = []
    pix_list = []
    fft_list = []
    q_list = []
    contrast_list1 = []
    contrast_list_michaelson = []
    bit_depth_list = []

    for pic in suspects:
        name = pic
        img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        full_img = cv2.imread(pic)
        sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
        q1 = os.stat(name).st_size
        h, w = full_img[:, :, 0].shape
        pix = img.sum()/(h*w)
        sbd = str(img.dtype)
        list_sbd = list(sbd)
        bd = [i for i in list_sbd if i.isdigit()]
        delim = ''
        bd = int(delim.join(bd))
        q = q1/(h*w)
        if w > h:
            size = int(h/2)
        else:
            size = int(w/2)
        cntrX, cntrY = (int(w / 2.0), int(h / 2.0))
        fft = np.fft.fft2(full_img[:, :, 0])
        fftShift = np.fft.fftshift(fft)
        fftShift[cntrY - size:cntrY + size, cntrX - size:cntrX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)
        magnitude = 20 * np.log(np.abs(recon))
        mean_high_fft = np.mean(magnitude)
        contrast_std = img.std()
        cmin = np.min(img)
        cmax = np.max(img)
        contrast_michaelson = (cmax-cmin)/(cmax+cmin)

        names.append(name)
        pix_list.append(pix)
        laplacian_var.append(sharpness)
        fft_list.append(mean_high_fft)
        q_list.append(q)
        contrast_list1.append(contrast_std)
        contrast_list_michaelson.append(contrast_michaelson)
        bit_depth_list.append(bd)

    dict = {'name_image': names,
            'pixel_sum_over_area': pix_list,
            'laplacian_variance': laplacian_var,
            'fastforiertransform_crispness': fft_list,
            'file_size_over_area': q_list,
            'contrast_std': contrast_list1,
            'michaelson_contrast': contrast_list_michaelson,
            'bit_depth': bit_depth_list,
            }
    frame = pd.DataFrame(dict)
    frame = frame.style.background_gradient(cmap='PiYG')

    return frame


def fourier_transf(image):
    """
    A fourier transformed image from an X-ray can actually provide information
    on everything from aliasing (moire pattern) and other noise patterns
    to image orientation and potential registration in the right hands.
    This creates Fourier transformed images out of all in a directory.
    This function is simply the appropriate numpy fast Fourier transforms made
    into a single code line/ "wrapper".

    :param image: original image (3 or single channel)
    :type image: numpy.ndarray

    :return: transformed
    :rtype: numpy.ndarray

    """
    if len(image.shape) > 2:
        img = image[:, :, 0]
    else:
        img = image
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    transformed = np.log(np.abs(fshift))
    return transformed


def pad_to_size(img, ht, wt):
    """
    This function applies a padding with value 0 around the image symmetrically
    until it is the ht and wt parameters specificed. Note if ht or wt below
    the existing ones are chosen, the image will be returned unpadded with
    a message.
    Note: this is not suggested as a pre-convolution padding. A preconvolution
    padding can be done easily in opencv with copyMakeBorder function. This
    function is a helper function, but can be used alone.

    :param img: original image (3 or single channel)
    :type img: numpy.ndarray
    :param ht: desired image height
    :type ht: int
    :param wt: desired image width
    :type wt: wt

    :return: image
    :rtype: numpy.ndarray
    """
    if len(img.shape) > 2:
        nowht, nowwt = img[:, :, 0].shape
    else:
        nowht, nowwt = img.shape
    addinght = int((ht - nowht)/2)
    addingwt = int((wt - nowwt)/2)
    if nowht <= ht and nowwt <= wt:
        image = cv2.copyMakeBorder(
            img,
            addinght,
            addinght,
            addingwt,
            addingwt,
            cv2.BORDER_CONSTANT,
            None,
            value=0,
            )
    else:
        print("image dimensions are incorrect for  this function")
        print("choose different ht and/or wt")
        image = img

    return image


def cut_to_size(img, ht, wt):
    """
    This function applies a crop around the image symmetrically
    until it is the ht and wt parameters specified. Note if ht or wt above
    the existing ones are chosen, the original image will be returned uncut,
    and a message will be printed.

    :param img: original image (3 or single channel)
    :type img: numpy.ndarray
    :param ht: desired image height
    :type ht: int
    :param wt: desired image width
    :type wt: wt

    :return: image
    :rtype: numpy.ndarray
    """
    if len(img.shape) > 2:
        nowht, nowwt = img[:, :, 0].shape
    else:
        nowht, nowwt = img.shape
    subht = int((nowht - ht)/2)
    subwt = int((nowwt - wt)/2)
    if nowht <= ht or nowwt <= wt:
        print("Choose smaller ht or wt, image not cuttable")
        image = img
    else:
        image = img[subht:nowht-subht, subwt:nowwt-subwt]

    return image


def cut_or_pad(img, ht, wt):
    """
    This function applies a cropping or a padding around the image
    symmetrically until it is the ht and wt parameters specified.
    Please note: what is usually appropriate for neural nets is to crop off
    frames, then resize all the images, then pad them all, so they are all as
    unform as possible.

    :param img: original image (3 or single channel)
    :type img: numpy.ndarray
    :param ht: desired image height
    :type ht: int
    :param wt: desired image width
    :type wt: wt

    :return: image
    :rtype: numpy.ndarray
    """
    if len(img.shape) > 2:
        nowht, nowwt = img[:, :, 0].shape
    else:
        nowht, nowwt = img.shape

    if nowht <= ht and nowwt <= wt:
        image = pad_to_size(img, ht, wt)
    elif nowht > ht and nowwt > wt:
        image = cut_to_size(img, ht, wt)
    elif nowht <= ht and nowwt > wt:
        subwt = int((nowwt - wt)/2)
        image = img[:, subwt:nowwt-subwt]
        image = pad_to_size(image, ht, wt)
    else:
        subht = int((nowht - ht)/2)
        image = img[subht:nowht-subht, :]
        image = pad_to_size(image, ht, wt)

    return image


def rotated_with_max_clean_area(image, angle):
    """
    Given an image, will rotate the image and crop off the blank triangle edges
    Note: if image is given with a triangle edge (previously rotated?), or
    border these existing edges and borders will not be cropped.

    :param img: original image (3 or single channel)
    :type img: numpy.ndarray
    :param angle: desired angle for rotation
    :type angle: int


    :return: image
    :rtype: numpy.ndarray
    """
    radians = (math.pi / 180) * angle
    h, w = image.shape[0:2]
    if w <= 0 or h <= 0:
        wr, hr = 0, 0
    else:
        width_longer = w >= h
        side_long, side_short = (w, h) if width_longer else (h, w)
        sin_a, cos_a = abs(math.sin(radians)), abs(math.cos(radians))
        if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
            x = 0.5*side_short
            wr, hr = (x/sin_a, x/cos_a) if width_longer else (x/cos_a, x/sin_a)
        else:
            cos_2a = cos_a*cos_a - sin_a*sin_a
            wr, hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
    image1 = simple_rotate_no_pil(image, angle, center=None, scale=1.0)
    image = cut_to_size(image1, wr, hr)
    return image


def noise_sum_cv(image):
    """
    Given an image, will try to sum up the noise, then divide by the area of
    the image. The noise summation here is based on an opencv2 algorithm for
    noise called fastNlMeansDenoising which is an implementation of non-local
    means denoising.

    :param img: original image (3 or single channel)
    :type img: numpy.ndarray

    :return: final_sum
    :rtype: float
    """
    un_noise = cv2.fastNlMeansDenoising(image)
    difference = abs(un_noise - image)
    sumed = difference.sum()
    ht, wt = image.shape[0:2]
    area = ht*wt
    final_sum = sumed/area
    return final_sum


def noise_sum_median_blur(image):
    """
    Given an image, will try to sum up the noise, then divide by the area of
    the image. The noise summation here is based on a median filter denoising

    :param img: original image (3 or single channel)
    :type img: numpy.ndarray

    :return: final_sum
    :rtype: float
    """
    kernel_size = 3
    un_noise = cv2.medianBlur(image, kernel_size)
    difference = abs(un_noise - image)
    sumed = difference.sum()
    ht, wt = image.shape[0:2]
    area = ht*wt
    final_sum = sumed/area
    return final_sum


def noise_sum_gaussian(image):
    """
    Given an image, will try to sum up the noise, then divide by the area of
    the image. The noise summation here is based on a gaussian filter denoising

    :param img: original image (3 or single channel)
    :type img: numpy.ndarray

    :return: final_sum
    :rtype: float
    """
    un_noise = cv2.GaussianBlur(image, (3, 3), 0)
    difference = abs(un_noise - image)
    sumed = difference.sum()
    ht, wt = image.shape[0:2]
    area = ht*wt
    final_sum = sumed/area
    return final_sum


def noise_sum_bilateral(image):
    """
    Given an image, will try to sum up the noise, then divide by the area of
    the image. The noise summation here is based on a bilatera filter denoising
    given a fairly large area (15 pixels)

    :param img: original image (3 or single channel)
    :type img: numpy.ndarray

    :return: final_sum
    :rtype: float
    """

    un_noise = cv2.bilateralFilter(image, 15, 75, 75)
    difference = abs(un_noise - image)
    sumed = difference.sum()
    ht, wt = image.shape[0:2]
    area = ht*wt
    final_sum = sumed/area
    return final_sum


def noise_sum_bilateralLO(image):
    """
    Given an image, will try to sum up the noise, then divide by the area of
    the image. The noise summation here is based on a bilatera filter denoising
    given a fairly large area (15 pixels)

    :param img: original image (3 or single channel)
    :type img: numpy.ndarray

    :return: final_sum
    :rtype: float
    """

    un_noise = cv2.bilateralFilter(image, 5, 5, 5)
    difference = abs(un_noise - image)
    sumed = difference.sum()
    ht, wt = image.shape[0:2]
    area = ht*wt
    final_sum = sumed/area
    return final_sum


def noise_sum_5k(image):
    """
    Given an image, will try to sum up the noise, then divide by the area of
    the image. The noise summation here is based on a median filter denoising
    using a 5*5 kernel. This kernel is reccomended for picking up moire
    patterns and other repetitive noise that may be missed by a smaller kernel.

    :param img: original image (3 or single channel)
    :type img: numpy.ndarray

    :return: final_sum
    :rtype: float
    """
    kernel_size = 5
    un_noise = cv2.medianBlur(image, kernel_size)
    difference = abs(un_noise - image)
    sumed = difference.sum()
    ht, wt = image.shape[0:2]
    area = ht*wt
    final_sum = sumed/area
    return final_sum


def noise_sum_7k(image):
    """
    Given an image, will try to sum up the noise, then divide by the area of
    the image. The noise summation here is based on a median filter denoising
    using a 7*7 kernel. This kernel is reccomended for picking up moire
    patterns and other repetitive noise that may be missed by a smaller kernel.

    :param img: original image (3 or single channel)
    :type img: numpy.ndarray

    :return: final_sum
    :rtype: float
    """
    kernel_size = 7
    un_noise = cv2.medianBlur(image, kernel_size)
    difference = abs(un_noise - image)
    sumed = difference.sum()
    ht, wt = image.shape[0:2]
    area = ht*wt
    final_sum = sumed/area
    return final_sum


def blind_noise_matrix(directory):
    """
    Creates a dataframe of image noise approximations by different algorithms
    here run over the whole image.
    The data frame is colored with a diverging color scheme (purple low,
    green high) map so that groups of images can be compared intuitively
    NB: images should be roughly comparable in dimension size for results
    to be meaningful.

    :param directory: Directory with set_of_images.
    :type directory: string

    :return: frame (dataframe)
    :rtype: class 'pandas.io.formats.style.Styler'

    """

    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2

    names = []
    noise_median_list = []
    noise_cv_list = []
    noise_5k_list = []
    noise_7k_list = []
    noise_gaussian_list = []
    noise_bilateral_list = []
    noise_bilateralLO_list = []

    for pic in suspects:
        name = pic
        img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        medi = noise_sum_median_blur(img)
        medi5 = noise_sum_5k(img)
        medi7 = noise_sum_7k(img)
        noise_cv = noise_sum_cv(img)
        gauss = noise_sum_gaussian(img)
        bilateral = noise_sum_bilateral(img)
        bilateralLO = noise_sum_bilateralLO(img)
        names.append(name)
        noise_median_list.append(medi)
        noise_cv_list.append(noise_cv)
        noise_7k_list.append(medi7)
        noise_gaussian_list.append(gauss)
        noise_5k_list.append(medi5)
        noise_bilateral_list.append(bilateral)
        noise_bilateralLO_list.append(bilateralLO)

    dict = {'name_image': names,
            'noise_non_local_mean': noise_cv_list,
            'noise_gaussian': noise_gaussian_list,
            'noise_3_k_median': noise_median_list,
            'noise_5_k_median': noise_5k_list,
            'noise_7_k_median': noise_7k_list,
            'noise_bilat_large': noise_bilateral_list,
            'noise_bilat_small': noise_bilateralLO_list,

            }
    frame = pd.DataFrame(dict)
    frame = frame.style.background_gradient(cmap='PiYG')

    return frame


def segmented_blind_noise_matrix(directory):
    """
    Creates a dataframe of image noise approximations by different algorithms
    but only on the very dark areas. Essentially this is a segmentation
    to the background, and a judgement of noise there.
    The data frame is colored with a diverging color scheme (purple low,
    green high) map so that groups of images can be compared intuitively
    NB: images should be roughly comparable in dimension size for results
    to be meaningful.

    :param directory: Directory with set_of_images.
    :type directory: string

    :return: frame (dataframe)
    :rtype: class 'pandas.io.formats.style.Styler'

    """

    suspects1 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Gg]'))
    suspects2 = glob.glob(os.path.join(directory, '*.[Jj][Pp][Ee][Gg]'))
    suspects = suspects1 + suspects2

    names = []
    noise_median_list = []
    noise_cv_list = []
    noise_5k_list = []
    noise_7k_list = []
    noise_gaussian_list = []
    noise_bilateral_list = []
    noise_bilateralLO_list = []

    for pic in suspects:
        name = pic
        img = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        thresh = 20
        output_image = img
        blur_image = cv2.medianBlur(img, 7)
        mask = blur_image < thresh
        output_image[~mask] = 0
        whole_image_ht, whole_image_wt = img.shape[0:2]
        whole_image_area = whole_image_ht*whole_image_wt
        area = whole_image_area - mask.sum()
        medi = (noise_sum_median_blur(img) * whole_image_area)/area
        medi5 = (noise_sum_5k(img) * whole_image_area)/area
        medi7 = (noise_sum_7k(img) * whole_image_area)/area
        noise_cv = (noise_sum_cv(img) * whole_image_area)/area
        gauss = (noise_sum_gaussian(img) * whole_image_area)/area
        bilateral = (noise_sum_bilateral(img) * whole_image_area)/area
        bilateralLO = (noise_sum_bilateralLO(img) * whole_image_area)/area
        names.append(name)
        noise_median_list.append(medi)
        noise_cv_list.append(noise_cv)
        noise_7k_list.append(medi7)
        noise_gaussian_list.append(gauss)
        noise_5k_list.append(medi5)
        noise_bilateral_list.append(bilateral)
        noise_bilateralLO_list.append(bilateralLO)

    dict = {'name_image': names,
            'noise_non_local_mean': noise_cv_list,
            'noise_gaussian': noise_gaussian_list,
            'noise_3_k_median': noise_median_list,
            'noise_5_k_median': noise_5k_list,
            'noise_7_k_median': noise_7k_list,
            'noise_bilat_large': noise_bilateral_list,
            'noise_bilat_small': noise_bilateralLO_list,

            }
    frame = pd.DataFrame(dict)
    frame = frame.style.background_gradient(cmap='PiYG')

    return frame
