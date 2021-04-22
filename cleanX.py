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
    nonzero = np.nonzero(image_array)
    y_nonzero = nonzero[0]
    x_nonzero = nonzero[1]

    return image_array[
        np.min(y_nonzero):np.max(y_nonzero),
        np.min(x_nonzero):np.max(x_nonzero),
    ]


def crop_pil(image):
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

    Args:
        origin_folder: folder with 'virgin' images
        target_folder: folder to drop
        images after transformations
        transformations : example tranformations = [
            ImageOps.mirror, ImageOps.flip
        ]
        ...some function to transform the image

    Returns:
        technical nonreturn
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
        Args:
        source_directory: directory with image files (should be more than 20)
        percentage_to_say_outliers: a number which will be the percentage of
            images contained in the high mean and low mean sets

        Returns:
        lows,highs: images with low mean, images with high mean
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

        Args:
        source_directory: directory with image files (should be more than 20)
        percentage_to_say_outliers: a number which will be the percentage of
        images contained in the high mean OR low mean sets- note if you set to
        50, then all images will be high or low

        Returns:
        lows,highs: images with low mean, images with high mean into a
        dataframe
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
    result = [0] * width

    for i in range(width):
        result[i] = [default_element] * height

    return result


def find_tiny_image_differences(directory, s=5, percentile=8):
    """
    Note: percentile returned is approximate, may be a tad more
    Args:
        directory: directory of all the images you want to compare
        s: size of image sizes to compare
        percentile: what percentage you want to return
    Returns:
    difference_outliers: outliers in terms of difference from an average image
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
    Args:
        iter_ob: should be list(df.column)
        legnth_name: size of image name going from end
    Returns: plot of images with names
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
    picture_directory = Path(directory)
    files = sorted(os.listdir(picture_directory))
    dupli = []
    for file in files:
        dupli.append(file)
    df = pd.DataFrame(dupli)
    df['diagnosis'] = diagnosis_string
    df = df.rename(columns={0: 'identifier_pic_name'})
    return df
