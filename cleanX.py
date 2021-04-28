# -*- coding: utf-8 -*-
"""
Library for cleaning radiological data used in machine learning
applications
"""

# imported libraries
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
from PIL import Image, ImageOps
import math
import filecmp
import tesserocr
from tesserocr import PyTessBaseAPI
from filecmp import cmp
from pathlib import Path
import re
from functools import partial


# to run on dataframes
def check_paths_for_group_leakage(train_df, test_df, unique_id):
    """
    Finds train samples that have been accidentally leaked into test
    samples

    :param train_df: Pandas dataframe containing information about
                     train assets.
    :type train_df: DataFrame
    :param test_df: Pandas dataframe containing information about
                    train assets.
    :type test_df: DataFrame

    :return: duplications of any image into both sets as a new dataframe
    :rtype: DataFrame
    """
    pics_in_both_groups = train_df.merge(test_df, on=unique_id, how='inner')
    return pics_in_both_groups


def see_part_potential_bias(df, label, sensitive_column_list):
    """
    This function gives you a tabulated dataframe of sensitive columns e.g.
    gender, race, or whichever you think are relevant,
    in terms of a labels (put in the label column name).
    You may discover all your pathologically labeled sample are of one ethnic
    group, gender or other category in your dataframe. Remeber some early
    neural nets for chest-Xrays were less accurate in women and the fact that
    there were fewer Xrays of women in the datasets they built on did not help

    :param df: Dataframe including sample IDs, labels, and sensitive columns
    :type df: Dataframe
    :param label: The name of the column with the labels
    :type label: string
    :param sensitive_column_list: List names sensitive columns on dataframe
    :type sensitive_column_list: list

    :return: tab_fight_bias2, a neatly sorted dataframe
    :rtype: Dataframe
    """

    label_and_sensitive = [label]+sensitive_column_list
    tab_fight_bias = pd.DataFrame(
        df[label_and_sensitive].value_counts()
    )
    tab_fight_bias2 = tab_fight_bias.groupby(label_and_sensitive).sum()
    tab_fight_bias2 = tab_fight_bias2.rename(columns={0: 'sums'})
    return tab_fight_bias2

# to run on single images, one at a time


def simpler_crop(image):
    """
    Crops an image of a black frame

    :param image: Image
    :type image: Image (JPEG)

    :return: image cropped of black edges
    :rtype: image[
        np.min(y_nonzero):np.max(y_nonzero),
        np.min(x_nonzero):np.max(x_nonzero))
    ]
    """
    nonzero = np.nonzero(image)
    y_nonzero = nonzero[0]
    x_nonzero = nonzero[1]
    # , x_nonzero, _ = np.nonzero(image)
    return image[
        np.min(y_nonzero):np.max(y_nonzero),
        np.min(x_nonzero):np.max(x_nonzero)
    ]


def crop_np(image_array):
    """
    Crops black edges of an image array

    :param image_array: Image array.
    :type image_array: array


    :return: image_array[
        np.min(y_nonzero):np.max(y_nonzero),
        np.min(x_nonzero):np.max(x_nonzero),
    ]
    :rtype: ndarray
    """
    nonzero = np.nonzero(image_array)
    y_nonzero = nonzero[0]
    x_nonzero = nonzero[1]

    return image_array[
        np.min(y_nonzero):np.max(y_nonzero),
        np.min(x_nonzero):np.max(x_nonzero),
    ]


def crop_pil(image):
    """
    Crops black edges of an Pil/Pillow image

    :param image_array: Image.
    :type image_array: image


    :return: image_array
    :rtype: array
    """
    mode = image.mode
    return Image.fromarray(
        crop_np(np.array(image)),
        mode=mode,
    )


def crop(image):
    """
    Crops an image of a black frame: does both PIL and opencv2 images

    :param image: Image
    :type image: Image (JPEG)

    :return: image cropped of black edges
    :rtype: image[
        np.min(y_nonzero):np.max(y_nonzero),
        np.min(x_nonzero):np.max(x_nonzero))
    ]
    """
    if isinstance(image, Image.Image):
        return crop_pil(image)
    if isinstance(image, np.ndarray):
        return crop_np(image)


def subtle_sharpie_enhance(image):
    """
    Makes a new image that is very subtly sharper to the human eye,
    but has new values in most of the pixels(besides the background)

    :param image: String for image name
    :type image: string

    :return: new_image_array, a nearly imperceptibly sharpened image for humans
    :rtype: nd.array

    """
    ksize = (2, 2)
    image_body = cv2.imread(image)
    blur_mask = cv2.blur(image_body, ksize)
    new_image_array = (2*image_body) - (blur_mask)
    return new_image_array


def harsh_sharpie_enhance(image):

    """
    Makes a new image that is very sharper to the human eye,
    and has new values in most of the pixels(besides the background)

    :param image: String for image name
    :type image: string

    :return: new_image_array, a sharpened image for humans
    :rtype: nd.array
    """
    ksize = (20, 20)
    image_body = cv2.imread(image)
    blur_mask = cv2.blur(image_body, ksize)
    new_image_array = (2*image_body) - (blur_mask)
    return new_image_array


def salting(img):
    """
    This function adds some noise to an image.

    :param img_name: String for image name
    :type img_name: string

    :return: new_image_array, with noise
    :rtype: nd.array

    """
    kernel = (5, 5)
    img = cv2.imread(img)
    erosion = cv2.erode(img, kernel, iterations=90)
    dilation = cv2.dilate(erosion, kernel, iterations=10)
    salty_noised = (img + (erosion - dilation))
    return salty_noised


def simple_rotation_augmentation(angle_list1, image):
    """
    This function takes one picture and rotates is by a number
    of degress is angle_list1.This function can be used with the
    augment_and_move function as follows (example):
    augment_and_move(
        'D:/my_academia/dataset/random_within_domain',
        'D:/my_academia/elo',
        [partial(simple_rotation_augmentation,5)],
    )

    :param image: Image.
    :type image: Image (JPEG)

    :return: rotated image
    :rtype: PIL image

    """
    if isinstance(image, str):
        image4R = Image.open(image)
    else:
        image4R = image
    rotated1 = image4R.rotate(angle_list1)
    return rotated1


def blur_out_edges(image):
    """
    For an individual image, blurs out the edges as an augmentation.

    :param image: Image
    :type image: Image (JPEG)

    :return: blurred_edge_image an array of an image blurred around the edges
    :rtype: ndarray
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


def reasonable_rotation_augmentation(angle1, angle2, number_slices, image):
    """
    Works on a single image, and returns a list of augmented images which are
    based on twisting the angle from angle1 to angle2 with 'number_slices' as
    the number of augmented images to be made from the original. It is not
    realistic or desirable to augment images of most Xrays by flipping them. In
    abdominal or chest X-rays that would create an augmentation that could
    imply specific pathologies e.g. situs inversus. We suggest augmenting
    between angles -5 to 5.

    :param angle1: angle1 is the angle from the original to the first augmented
    :type angle1: float
    :param angle2: angle2 is the angle from the original to the last augmented
    :type angle2: float
    :param number_slices: number of images to be produced
    :type number_slices: int
    :param image: Image
    :type image: Image (JPEG)

    :return: list of PIL images
    :rtype: list

    """
    increment = abs(angle1-angle2)/number_slices
    angle1f = float(angle1)
    angle2f = float(angle2)
    number_slicesf = float(number_slices)
    increment = abs(angle1f-angle2f)/number_slicesf
    num_list = np.arange(angle1f, angle2f,  increment)
    image4R = Image.open(image)
    augmentos = []
    for i in num_list:
        augmentos.append(image4R.rotate(i))

    return augmentos


def show_major_lines_on_image(pic_name):
    """
    A function that takes individual images and shows suspect lines i.e.
    lines more likely to be non-biological.

    :param pic_name: String of image full name e.g. "C:/folder/image.jpg"
    :type pic_name: string

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
# to run on a list to make a prototype tiny Xray


def seperate_image_averger(set_of_images, s=5):
    """
    To run on a list to make a prototype tiny Xray that is an averages image

    :param set_of_images: Set_of_images
    :type set_of_images:
    :param s: length of sides in the image made
    :type s: integer

    :return: image
    :rtype: image
    """
    canvas = np.zeros((s, s))
    for pic in set_of_images:
        example = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        example_small = cv2.resize(example, (s, s))
        canvas += np.array(example_small)
    return canvas / len(set_of_images)

# to run on files which are inside a folder


def augment_and_move(origin_folder, target_folder, transformations):

    """
    Takes images and applies the same list of augmentations, which can include
    the cleanX function crop, to all of them

    :param origin_folder: The folder in which the images are
    :type origin_folder: directory
    :param target_folder: The folder where augmented images will be sent
    :type target_folder: directory
    :param transformations: A list of augmentation functions to apply
    :type transformations: list

    :return: technically a nonreturning function, but new images will be made
    :rtype: none

    """
    non_suspects = glob.glob(os.path.join(origin_folder, '*.jpg'))
    for picy in non_suspects:
        example = Image.open(picy)
        if example.mode == 'RGBA':
            example = example.convert('RGB')
        novo = os.path.basename(picy)
        for transformation in transformations:
            example = transformation(example)
        example.save(os.path.join(target_folder, novo + ".jpg"))


def dimensions_to_df(folder_name):
    """
    Finds dimensions on images in a folder, and makes a df for exploratory
    data analysis

    :param folder_name: Adress of folder with images.
    :type folder_name: Folder/directory


    :return: image height, width and proportion height/width as a new dataframe
    :rtype: DataFrame
    """
    non_suspects = glob.glob(os.path.join(folder_name, '*.jpg'))
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


def dimensions_to_histo(folder_name, bins_count=10):
    """
    Looks in the directory given, and produces a histogram of various widths
    and heights. Important information as many neural nets take images all the
    same size. Classically most chest-X-rays are 2500*2000 or 2500 *2048;
    however the dataset may be different and/or varied

    :param folder_name: Folder_name, directory name.
    :type folder_name: string

    :param bins_count: bins_count, number of bins desired (defaults to 10)
    :type bins_count: int

    :return: histo_ht_wt, a labeled histogram
    :rtype: tuple

    """
    non_suspects = glob.glob(os.path.join(folder_name, '*.jpg'))
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
    of the images by dividing their height by widths.
    Important information as many neural nets take images all the
    same size. Clasically most chestXrays are 2500*2000 or 2500 *2048;
    however the dataset may be different and/or varied

    :param folder_name: Folder_name, directory name.
    :type folder_name: string
    :param bins_count: bins_count, number of bins desired (defaults to 10)
    :type bins_count: int
    :return: histo_ht_wt_p, a labeled histogram
    :rtype: tuple
    """
    non_suspects = glob.glob(os.path.join(folder_name, '*.jpg'))
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


def crop_them_all(origin_folder, target_folder):
    """
    Crops all images and moves them to a target folder

    :param origin_folder: The folder in which the images are
    :type origin_folder: directory
    :param target_folder: The folder where augmented images will be sent
    :type target_folder: directory


    :return: technically nothing returned, but new images will be made
    :rtype: none
    """
    # crops and moves to a new folder for a set inside origin folder
    augment_and_move(
        origin_folder,
        target_folder,
        [crop],
    )


def find_very_hazy(directory):
    """
    Finds pictures that are really "hazy" i.e. there is no real straight line
    because they are blurred. Usually, at least the left or right tag
    should give straight lines, thus this function finds image of a certain
    questionable technique level.

    :param directory: The folder where the images are
    :type directory: directory


    :return: dataframe with imgs sorted as hazy or regular under label_for_haze
    :rtype: Dataframe
    """
    suspects = glob.glob(os.path.join(directory, '*.jpg'))
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
        Takes average of upper pixels, and can show you outliers defined by a
        percentage, e.g. shows images with an average of top pixels in top x %
        where x is the percent height of the sample.

        :param source_directory: The folder in which the images are
        :type source_directory: directory
        :param percent_height_of_sample: From where on image to call upper
        :type source_directory: integer
        :param value_for_line: From where in pixel values to call averaged
            values abnormal
        :type value_for_line: integer


        :return: Dataframe with images labeled
        :rtype: Dataframe
         """

    suspects = glob.glob(os.path.join(source_directory, '*.jpg'))
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
        Takes average of upper pixels, average of lower (you define what
        percent of picture should be considered upper and lower) and compares.
        In a CXR if lower average is greater than upper it may be upside down
        or otherwise bizzare, as the neck is smaller than the abdomen.
        """
    estup, estdown, piclist = [], [], []
    suspects = glob.glob(os.path.join(source_directory, '*.jpg'))
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
        Takes the average of all pixels in an image, returns a dataframe with
        those images that are outliers by mean...should catch some inverted or
        problem images

        :param source_directory: The folder in which the images are
        :type source_directory: directory
        :param percentage_to_say_outliers: Percentage to capture
        :type percentage_to_say_outliers: integer


        :return: Dataframe made up of outliers only
        :rtype: Dataframe

        """
    suspects = glob.glob(os.path.join(source_directory, '*.jpg'))
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
        Important note: approximate, and it can by chance cut the group
        so images with
        the same mean are in and out of normal range if the knife so falls

        :param source_directory: The folder in which the images are
        :type source_directory: directory
        :param percentage_to_say_outliers: Percentage to capture
        :type percentage_to_say_outliers: integer


        :return: Dataframe all images, marked as high, low or within range
        :rtype: Dataframe
        """
    suspects = glob.glob(os.path.join(source_directory, '*.jpg'))
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
    # lows = df.head(percentile)
    # highs = df.tail(percentile)
    df['results'] = 'within range'
    df.loc[:percentile, 'results'] = 'low'
    df.loc[len(df) - percentile:, 'results'] = 'high'
    # whole = [lows, highs]
    # new_df = pd.concat(whole)
    return(df)


def understand_df(df):
    """
        Takes a dataframe (if you have a dataframe for images) and prints
        information including length, data types, nulls and number of
        duplicated rows
        """

    print("The dataframe has", len(df.columns), "columns, named", df.columns)
    print("")
    print("The dataframe has", len(df), "rows")
    print("")
    print("The types of data:\n", df.dtypes)
    print("")
    print("In terms of nulls, the dataframe has: \n", df[df.isnull()].count())
    print("")
    print(
        "Number of duplicated rows in the data is ",
        df.duplicated().sum(),
        ".",
    )
    print("")
    print("Numeric qualities of numeric data: \n", df.describe())


def show_duplicates(df):
    """
        Takes a dataframe (if you have a dataframe for images) and prints
        duplicated rows
        """
    if df.duplicated().any():
        print(
            "This dataframe table has",
            df.duplicated().sum(),
            " duplicated rows"
        )
        print("They are: \n", df[df.duplicated()])
    else:
        print("There are no duplicated rows")


def create_matrix(width, height, default_element):
    # In python Sequence * Number = Sequence repeated Number of times
    """
        Takes width, height then creates a matrix populated by the default
        element. Super handy for advanced image manipulation. Note you can not
        create matrices bigger than your computer memory can handle making.
        Therefore the function will work on matrices with dimensions up to
        maybe 500*500 depending

        :param width: Width of the matrix to be created
        :type width: integer
        :param height: The height of matrix to be created
        :type height: integer
        :param default_element: Element to populate the matrix with
        :type default_element: float or integer or string

        :return: 2D matrix populated
        :rtype: matrix

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

    :param directory: Directory (folder).
    :type directory: Directory
    :param s: legnth to make the sides of the tiny image for comparison
    :type s: integer
    :param percentile: percentile to mark as abnormal
    :type percentile: integer

    :return: Dataframe with a column that notes mismatches
    and within range images
    :rtype: Dataframe

    """
    suspects = glob.glob(os.path.join(directory, '*.jpg'))
    avg_image = seperate_image_averger(suspects, s)  # np.zeros((5, 5)) + 128
    images, sums = [], []
    for pic in suspects:
        example = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
        example_clipped = simpler_crop(example)
        example_small = cv2.resize(example_clipped, (s, s))
        experiment_a = (example_small - avg_image) ** 2
        experiment_sum = experiment_a.sum()
        images.append(pic)
        sums.append(experiment_sum)
    df = pd.DataFrame({'images': images, 'sums': sums})
    df.sort_values('sums', inplace=True, ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    # return df.tail(int((len(df) / 100) * percentile))
    # df.loc((int((len(df) / 100) * percentile): ) = True
    df['results'] = 'within range'
    df.loc[int((len(df) / 100) * (100 - percentile)):, 'results'] = 'mismatch'
    # df.loc[len(df) - percentile:, 'results'] = 'high'
    return df


def tesseract_specific(directory):
    # this function runs tessseract ocr for text detection over images
    # in a directory, and gives a dataframe with what it found
    """
    Finds images with text on them. Multi-lingual including English.

    :param directory: Directory (folder).
    :type directory: Directory

    :return: Dataframe with a column of text found
    :rtype: Dataframe

    """
    suspects = glob.glob(os.path.join(directory, '*.jpg'))
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

    :param directory: Directory (folder).
    :type directory: Directory
    :param label_word: Label word
    :type label_word: string

    :return: Dataframe with a column of text found over the legnth
    :rtype: Dataframe

    """

    suspects = glob.glob(os.path.join(directory, '*.jpg'))
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


def find_suspect_text_by_legnth(directory, legnth):
    # this function finds all texts above a specified legnth
    # (legnth is number of charecters)
    """
    Finds images with text over a specific legnth you ask for.
    Useful if you know you do not care about R and L or SUP.
    Multi-lingual including English. Accuracy is very high, but not perfect.

    :param directory: Directory (folder).
    :type directory: Directory
    :param legnth: Legnth to find above, inclusive
    :type legnth: integer

    :return: Dataframe with a column of text found
    :rtype: Dataframe


    """
    suspects = glob.glob(os.path.join(directory, '*.jpg'))
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
#     if df['text'].str.len() < legnth:
#         df['text'] = 'shorter'
    yes = df[df['text'].str.len() >= legnth]
    no = df[df['text'].str.len() < legnth]
    no['text'] = 'shorter'
    df = pd.concat([yes, no])
    # df.reset_index(drop=True, inplace=True)
    return df


def histogram_difference_for_inverts(directory):
    # this function looks for images by a spike on the end of pixel
    # value histogram to find inverted images
    """
        This function looks for images by a spike on the end of thier pixel
        value histogram to find inverted images. Note we assume classical
        X-rays, not inverted floroscopy images.

        :param directory: Directory (folder).
        :type directory: Directory

        :return: a list of images suspected to be inverted
        :rtype: list
        """

    suspects = glob.glob(os.path.join(directory, '*.jpg'))
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


def histogram_difference_for_inverts_todf(directory):
    # looks for inverted and returns a dataframe
    """
        This function looks for images by a spike on the end of thier pixel
        value histogram to find inverted images, then puts what it found into
        a dataframe. Images are listed as regulars, inverts of unclear (the
        unclear have equal spikes on both ends)

        :param directory: Directory (folder).
        :type directory: Directory

        :return: a dataframe with images categorized
        :rtype: DataFrame
        """
    suspects = glob.glob(os.path.join(directory, '*.jpg'))
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

    :param directory: Directory (folder).
    :type directory: Directory

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
    # looks for duplicated images, returns dataframe
    """
    Finds duplicated images and returns a dataframe of them.

    :param directory: Directory (folder).
    :type directory: Directory

    :return: a dataframe of duplicated images
    :rtype: Dataframe

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

# takes a dataframe


def show_images_in_df(iter_ob, legnth_name):
    """
    Shows images by taking them off a dataframe column, and puts them up
    but smaller, so they can be compared quickly

    :param inter_ob: List, should be a dataframe column
    :type iter_ob: list
    :param legnth_name: Size of image name going from end
    :type legnth_name: integer

    :return: technically no return but makes a plot of images with names
    :rtype: none

        """

    width = int(math.sqrt(len(iter_ob)))
    height = int(math.ceil(len(iter_ob) / width))
    f, axarr = plt.subplots(width, height, figsize=(14, 14))
    if width > 1:
        for x in range(width):
            for y in range(height):
                element = iter_ob[x * width + y]
                fname = os.path.splitext(element)[0]
                title = fname[-legnth_name:]
                exop = cv2.imread(element, cv2.IMREAD_GRAYSCALE)
                axarr[x, y].set_title(title)
                axarr[x, y].imshow(exop, cmap='gray')

    else:
        for y in range(height):
            element = iter_ob[y]
            exop = cv2.imread(element, cv2.IMREAD_GRAYSCALE)
            fname = os.path.splitext(element)[0]
            title = fname[-legnth_name:]
            axarr[y].set_title(title)
            axarr[y].imshow(exop, cmap='gray')
            # plt.title('Outlier images')
    plt.show()


def dataframe_up_my_pics(directory, diagnosis_string):
    """
    Takes images in a directory (should all be with same label), and puts the
    name (with path) and label into a dataframe

    :param directory: Directory (folder).
    :type directory: Directory
    :param diagnosis_string: Usually a label, may be any string
    :type diagnosis_string: string

    :return: Dataframe of pictures and label
    :rtype: Dataframe

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
