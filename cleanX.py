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
    return train_df.merge(test_df, on=unique_id, how='inner')

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
    :rtype: array
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


def blur_out_edges(image):
    """
    For an individual image, blurs out the edges as an augmentation.

    :param image: Image
    :type image: Image (JPEG)

    :return: blurred_edge_image an array of an image blurred around the edges
    :rtype: image array
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

# to run on a list to make a prototype tiny Xray


def seperate_image_averger(set_of_images, s=5):
    """
    To run on a list to make a prototype tiny Xray that is an averages image

    :param set_of_images: Set_of_images
    :type set_of_images:
    :param s: legnth of sides in image made
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
    Takes images, and applies the same list of augmentations to all of them

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

    :param folder_name: adress of folder with images
    :type folder_name: folder/directory


    :return: image height, width and proportion heigth/width as a new dataframe
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


def find_by_sample_upper(
    source_directory,
    percent_height_of_sample,
    value_for_line
):
    """
        Takes average of upper pixels, and can show you outliers defined by a
        percentage e.g. shows images with averge of top pixels in top x % where
        x is the percent height of the sample.

        :param source_directory: The folder in which the images are
        :type source_directory: directory
        :param percent_height_of_sample: from where on image to call upper
        :type source_directory: integer
        :param value_for_line: from where in pixel values to call averaged
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
        Takes the average of all pixels in image, returns a dataframe with
        those images that are outliers by mean...should catch some inverted or
        or problem images

        :param source_directory: The folder in which the images are
        :type source_directory: directory
        :param percentage_to_say_outliers: percentage to capture
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
        :param percentage_to_say_outliers: percentage to capture
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
        information including legnth, data types, nulls and number of
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
        element. Super handy for advanced image manipulation.

        :param width: width of matrix to be created
        :type width: integer
        :param height: height of matrix to be created
        :type height: integer
        :param default_element: element to populate the matrix with
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
    :param percentile: percentil to mark as abnormal
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
    :param label_word: label word
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
    :param legnth: legnth to find above, inclusive
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
    Shows images by taking them off a dataframe column, and puths them up
    but smaller, so they can be compared quickly

    :param inter_ob: list, should be a dataframe column
    :type iter_ob: list
    :param legnth_name: size of image name going from end
    :type legnth_name: integer

    :return: techically no return but makes a plot of images with names
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
    :param diagnosis_string: usually a label, may be any string
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
