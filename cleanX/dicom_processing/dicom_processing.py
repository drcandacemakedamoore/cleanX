# -*- coding: utf-8 -*-
"""
Library for cleaning radiological data used in machine learning
applications
"""

# imported libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from PIL import Image, ImageOps
from tesserocr import PyTessBaseAPI

import glob
import filecmp
import math
import os
import re

from filecmp import cmp
from pathlib import Path


def rip_out_jpgs_sitk(dicomfile_directory, output_directory):
    """
    This function is for users with simpleITK library only.
    If you do not have the library it will throw an error.
    The funuction function jpeg files out of a dicom file directory,
    one by one, each of them (not just the first series as), and puts them in
    an out put directory. It also returns the images for inspection (as arrays)
    , which you can look at the [0] layer with matplotlib

    :param dicomfile_directory: dicomfile_directory, directory with dicom/.dcm
    :type dicomfile_directory: string
    :param output_directory: output_directory, where they should be placed
    :type output_directory: string

    :return: saved_images
    :rtype: list
    """
    import SimpleITK as sitk
    # include final slash in output directory
    dicom_files = glob(dicomfile_directory + '/*')
    reader = sitk.ImageFileReader()
    saved_images = []
    for i in range(0, len(dicom_files)):
        # give the reader a filename
        reader.SetFileName(dicom_files[i])
        # use the reader to read the image
        image = reader.Execute()
        image_np = sitk.GetArrayFromImage(image)
        saved_images.append(image_np)
        target_base = output_directory + os.path.basename(dicom_files[i])
        target_name = target_base + ".jpg"
        cv2.imwrite(target_name, image_np[0])
    return saved_images


def get_jpg_with_pydicom(dicom_folder_path, jpeg_folder_path):
    """
    This function is for users with pydicom library only.
    If you do not have the library it will throw an error.
    The funuction function jpeg files out of a dicom file directory,
    one by one, each of them (not just the first series as), and puts them in
    an out put directory.

    :param dicom_folder_path: dicomfile_directory, directory with dicom/.dcm
    :type dicom_folder_path: string
    :param jpeg_folder_path: output_directory, where they should be placed
    :type jpeg_folder_path: string

    :return: love (will put your images in the new folder but not return them)
    :rtype: bool
    """
    import pydicom
    images_path = os.listdir(dicom_folder_path)
    for n, image in enumerate(images_path):
        ds = dicom.dcmread(os.path.join(dicom_folder_path, image))
        pixel_array_numpy = ds.pixel_array
        image = image.replace('.dcm', '.jpg')
        love = cv2.imwrite(
            os.path.join(jpg_folder_path, image),
            pixel_array_numpy,
        )

    print('{} image converted'.format(n))
    return love
