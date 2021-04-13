# testing for cleanX

import os
import pandas as pd
import cleanX
import cv2


image_directory = os.path.join(os.path.dirname(__file__), 'directory')


def test_crop():
    example = cv2.imread(os.path.join(image_directory, 'testtocrop.jpg'), cv2.IMREAD_GRAYSCALE)
    cropped_example = cleanX.crop(example)
    assert cropped_example.shape < example.shape

def test_check_paths_for_group_leakage():
    train_dfE = (os.path.join(image_directory,'train_sample_df.csv'))
    test_dfE = (os.path.join(image_directory,'test_sample_df.csv'))
    train_df= pd.read_csv(train_dfE)
    test_df = pd.read_csv(test_dfE)
    uniqueIDE = 'image_path'
    checked_example = cleanX.check_paths_for_group_leakage(train_df, test_df, uniqueIDE)
    assert len(checked_example) > 1 

# def test_seperate_image_averager():


# # seperate_image_averger(set_of_images, s=5 ):

# # """
# # Args:
    
# #     set_of_images: a list 
# #     s: number of pixels for height and wifth

# # Returns:
# #     canvas/len(set_of_images): an average tiny image (can feed another function which compares to this mini)
# # """
