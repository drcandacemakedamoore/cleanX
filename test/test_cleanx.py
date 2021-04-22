# testing for cleanX

import os
import pandas as pd
import matplotlib as plt
import cleanX
import cv2
from PIL import Image, ImageOps
import numpy as np
import pytest

image_directory = os.path.join(os.path.dirname(__file__), 'directory')
target_directory = os.path.join(os.path.dirname(__file__), 'target')


def test_crop():
    example = cv2.imread(os.path.join(image_directory, 'testtocrop.jpg'), cv2.IMREAD_GRAYSCALE)
    cropped_example = cleanX.crop(example)
    assert cropped_example.shape < example.shape

def test_simpler_crop():
    example = cv2.imread(os.path.join(image_directory, 'testtocrop.jpg'), cv2.IMREAD_GRAYSCALE)
    cropped_example = cleanX.simpler_crop(example)
    assert cropped_example.shape < example.shape

def test_check_paths_for_group_leakage():
    train_dfE = (os.path.join(image_directory,'train_sample_df.csv'))
    test_dfE = (os.path.join(image_directory,'test_sample_df.csv'))
    train_df= pd.read_csv(train_dfE)
    test_df = pd.read_csv(test_dfE)
    uniqueIDE = 'image_path'
    checked_example = cleanX.check_paths_for_group_leakage(train_df, test_df, uniqueIDE)
    assert len(checked_example) > 1 

def test_seperate_image_averager():
    test_dfE = (os.path.join(image_directory,'test_sample_df.csv'))
    test_df = pd.read_csv(test_dfE)
    images = image_directory + '/' + test_df.image_path.dropna()
    blended =  cleanX.seperate_image_averger(images, s=5)
    assert type(blended) is np.ndarray

def test_dimensions_to_df():
    deflep = cleanX.dimensions_to_df(image_directory)
    assert len(deflep) > 1    


# def test_show_images_in_df():

#     # this testing remains undone... the function returns basically the line bwlo, and testing will be difficult     
#     plt.show()


def test_tesseract_specific():
    lettered = cleanX.tesseract_specific(image_directory)
    assert len(lettered) > 1 

def test_find_suspect_text():
    letters_spec = cleanX.find_suspect_text(image_directory, 'SUPINE')
    assert len(letters_spec) >= 1     

def test_find_suspect_text_by_legnth():   
    jobs = cleanX.find_suspect_text_by_legnth(image_directory, 3)
    assert len(jobs) > 1    

    
def test_augment_and_move():
    try:
        os.makedirs(target_directory)
    except FileExistsError:
        pass
    cleanX.augment_and_move(image_directory, target_directory, [ImageOps.mirror, ImageOps.flip])
    vovo = os.path.join(target_directory, 'testtocrop.jpg.jpg')
    assert os.path.isfile(vovo) 

def test_crop_them_all():
    try:
        os.makedirs(target_directory)
    except FileExistsError:
        pass
    cleanX.crop_them_all(image_directory, target_directory)
    vovo = os.path.join(target_directory, 'testtocrop.jpg.jpg')
    assert os.path.isfile(vovo) 

def test_find_by_sample_upper():
    lovereturned = cleanX.find_by_sample_upper(image_directory, 10, 10)
    assert len(lovereturned) >= 1

def test_find_sample_upper_greater_than_lower():
    lovereturnee = cleanX.find_sample_upper_greater_than_lower(image_directory, 10)
    assert len(lovereturnee) >= 1

def test_find_duplicated_images():
    found = cleanX.find_duplicated_images(image_directory)
    assert len(found) > 0     
    
def test_find_duplicated_images_todf():
    found = cleanX.find_duplicated_images_todf(image_directory)
    assert len(found) > 0       

def test_histogram_difference_for_inverts():
    histosy = cleanX.histogram_difference_for_inverts(image_directory)
    assert len(histosy) > 0

def test_histogram_difference_for_inverts_todf():
    histos = cleanX.histogram_difference_for_inverts_todf(image_directory)
    assert len(histos) >0

def test_dataframe_up_my_pics():
    dfy = cleanX.dataframe_up_my_pics(image_directory, 'diagnosis_string')
    assert len(dfy) > 0
    

        