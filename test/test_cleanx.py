# testing for cleanX

import os
import pandas as pd
import matplotlib as plt
import cleanX
import cv2
from PIL import Image, ImageOps
import numpy as np
import pytest
from functools import partial

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

def test_blur_out_edges():
    image = os.path.join(image_directory, 'testtocrop.jpg')
    defblur = cleanX.blur_out_edges(image)
    assert type(defblur) == np.ndarray 

def test_subtle_sharpie_enhance():
    image = os.path.join(image_directory, 'testtocrop.jpg')
    lo = cleanX.subtle_sharpie_enhance(image)
    assert lo.shape[0] >1
    
def harsh_sharpie_enhance():
    image = os.path.join(image_directory, 'testtocrop.jpg')
    ho = cleanX.harsh_sharpie_enhance(image)
    assert ho.shape[0] >1

def test_salting():
    lindo_image = os.path.join(image_directory, 'testtocrop.jpg')
    salt = cleanX.salting(lindo_image)
    assert salt.shape[0] > 1

def test_simple_rotation_augmentation():
    lindo_image = os.path.join(image_directory, 'testtocrop.jpg')
    lindo_rotated = cleanX.simple_rotation_augmentation(6, lindo_image)
    assert np.array(lindo_rotated).shape[0] > 1


#
def test_check_paths_for_group_leakage():
    train_dfE = (os.path.join(image_directory,'train_sample_df.csv'))
    test_dfE = (os.path.join(image_directory,'test_sample_df.csv'))
    train_df= pd.read_csv(train_dfE)
    test_df = pd.read_csv(test_dfE)
    uniqueIDE = 'image_path'
    checked_example = cleanX.check_paths_for_group_leakage(train_df, test_df, uniqueIDE)
    assert len(checked_example) > 1 

def test_separate_image_averager():
    test_dfE = (os.path.join(image_directory,'test_sample_df.csv'))
    test_df = pd.read_csv(test_dfE)
    images = image_directory + '/' + test_df.image_path.dropna()
    blended =  cleanX.separate_image_averager(images, s=5)
    assert type(blended) is np.ndarray
#
def test_dimensions_to_df():
    deflep = cleanX.dimensions_to_df(image_directory)
    assert len(deflep) > 1    
#
def test_see_part_potential_bias():
    e2 = (os.path.join(image_directory,'example_for_bias.csv'))
    e3 = pd.read_csv(e2)
    donwa = cleanX.see_part_potential_bias(e3,"Label", ["Gender", "Race"])
    assert len(donwa) > 1

# def test_show_images_in_df():

#     # this testing remains undone... the function returns basically the line bwlo, and testing will be difficult     
#     plt.show()
#
def test_dimensions_to_histo():
    output = cleanX.dimensions_to_histo(image_directory, 10)
    assert len(output) > 1
#
def test_find_very_hazy():
    found = cleanX.find_very_hazy(image_directory)
    assert len(found) > 0    

def test_show_major_lines_on_image():
    pic_name1 = os.path.join(image_directory, 'testtocrop.jpg')
    deflop = cleanX.show_major_lines_on_image(pic_name1)
    assert deflop # needs a much better test

def test_find_big_lines():
    lined = cleanX.find_big_lines(image_directory, 2)
    assert len(lined) > 0    
#
def proportions_ht_wt_to_histo():
    output =  cleanX.proportions_ht_wt_to_histo(image_directory, 10)
    assert len(output) > 1
#
def test_tesseract_specific():
    lettered = cleanX.tesseract_specific(image_directory)
    assert len(lettered) > 1 

def test_find_suspect_text():
    letters_spec = cleanX.find_suspect_text(image_directory, 'SUPINE')
    assert len(letters_spec) >= 1     

def test_find_suspect_text_by_length():   
    jobs = cleanX.find_suspect_text_by_length(image_directory, 3)
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

def test_reasonable_rotation_augmentation():
    imageE = (os.path.join(image_directory,'testtocrop.jpg'))
    dman = cleanX.reasonable_rotation_augmentation(0, 12, 3, imageE)
    assert len(dman) > 1
    
#
def test_find_by_sample_upper():
    lovereturned = cleanX.find_by_sample_upper(image_directory, 10, 10)
    assert len(lovereturned) >= 1
#
def test_find_sample_upper_greater_than_lower():
    lovereturnee = cleanX.find_sample_upper_greater_than_lower(image_directory, 10)
    assert len(lovereturnee) >= 1

def test_find_duplicated_images():
    found = cleanX.find_duplicated_images(image_directory)
    assert len(found) > 0     
#    
def test_find_duplicated_images_todf():
    found = cleanX.find_duplicated_images_todf(image_directory)
    assert len(found) > 0       

def test_histogram_difference_for_inverts():
    histosy = cleanX.histogram_difference_for_inverts(image_directory)
    assert len(histosy) > 0

def test_histogram_difference_for_inverts_todf():
    histos = cleanX.histogram_difference_for_inverts_todf(image_directory)
    assert len(histos) >0
#
def test_dataframe_up_my_pics():
    dfy = cleanX.dataframe_up_my_pics(image_directory, 'diagnosis_string')
    assert len(dfy) > 0
    
def test_simple_spinning_template():
    vovo = os.path.join(image_directory, 'testtocrop.jpg')
    picy1 = vovo
    img = cv2.imread(vovo, cv2.IMREAD_GRAYSCALE)
    greys_template1 = img[90:110, 200:350]
    angle_start1 = 0
    angle_stop1 = 30
    slices1 = 3
    lanter = cleanX.simple_spinning_template(
        picy1,
        greys_template1,
        angle_start1,
        angle_stop1,
        slices1,
    )
    assert len(lanter) > 0

def test_def_make_contour_image():
    vovo = os.path.join(image_directory, 'testtocrop.jpg')
    picy1 = vovo
    defMkcont = cleanX.make_contour_image(picy1)
    assert len(defMkcont) > 0

def test_avg_image_maker():
    #set_of_images = glob.glob(*.jpg)
    test_dfE = (os.path.join(image_directory,'test_sample_df.csv'))
    test_df = pd.read_csv(test_dfE)
    set_of_images = image_directory + '/' + test_df.image_path.dropna()
    tab = cleanX.avg_image_maker(set_of_images)  
    assert tab.shape[0] > 2  


def test_set_image_variability():
    #set_of_images = glob.glob(*.jpg)
    test_dfE = (os.path.join(image_directory,'test_sample_df.csv'))
    test_df = pd.read_csv(test_dfE)
    set_of_images = image_directory + '/' + test_df.image_path.dropna()
    tab = cleanX.set_image_variability(set_of_images)  
    assert tab.shape[0] > 2      

def test_avg_image_maker_by_label():
    test_dfE = (os.path.join(image_directory,'alt_test_labeled.csv'))
    test_df = pd.read_csv(test_dfE)
    #set_of_images = image_directory + '/' + test_df.image_path.dropna()
    lotus = cleanX.avg_image_maker_by_label(test_df,'imageID','path_label',image_directory)
    assert len(lotus) > 0       

def test_find_tiny_image_differences():
    defleper = cleanX.find_tiny_image_differences(image_directory)
    assert len(defleper) > 0
