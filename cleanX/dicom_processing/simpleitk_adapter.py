# -*- coding: utf-8 -*-

import os

from glob import glob

import SimpleITK as sitk
import cv2


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
    # include final slash in output directory
    dicom_files = glob(dicomfile_directory + '/*')
    reader = sitk.ImageFileReader()
    saved_images = []
    for i in range(len(dicom_files)):
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
