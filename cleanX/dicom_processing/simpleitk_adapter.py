# -*- coding: utf-8 -*-

import os

from glob import glob
import pandas as pd
from datetime import datetime, date
import SimpleITK as sitk
import cv2


class MakedaReader:
    def __init__(self, reader):
        self.reader = reader

    def fetch_metadata(self, dicom_file):

        self.reader.SetFileName(dicom_file)

        self.reader.LoadPrivateTagsOn()

        self.reader.ReadImageInformation()
        parsed = {}
        for k in self.reader.GetMetaDataKeys():
            parsed[k] = self.reader.GetMetaData(k)

        return parsed


class SimpleITKDicomReader:
    """Class for reading DICOM metadata with SimpleITK."""

    # exclude_field_types = (Sequence, MultiValue, bytes)
    date_fields = ('ContentDate', 'SeriesDate', 'ContentDate', 'StudyDate')
    time_fields = ('ContentTime', 'StudyTime')
    exclude_fields = ()

    def __init__(
            self,
            # exclude_field_types=None,
            date_fields=None,
            time_fields=None,
            exclude_fields=None,
    ):
        #         if exclude_field_types:
        #             self.exclude_field_types = exclude_field_types
        if date_fields:
            self.date_fields = date_fields
        if time_fields:
            self.time_fields = time_fields
        if exclude_fields:
            self.exclude_fields = exclude_fields

    def dicom_date_to_date(self, source):
        year = int(source[:4])
        month = int(source[4:6])
        day = int(source[6:])
        return date(year=year, month=month, day=day)

    def dicom_time_to_time(self, source):
        #     seconds, milis = source.split('.')
        # TODO: We don't know how to conver this yet
        return source

    def read(self, source):
        reader = sitk.ImageFileReader()
        m_reader = MakedaReader(reader)
        tag = source.get_tag()
        columns = {tag: []}
        for entry, parsed in source.items(m_reader.fetch_metadata):
            columns[tag].append(entry)
            for k, v in parsed.items():
                col = columns.get(k, [])
                col.append(v)
                columns[k] = col
        return pd.DataFrame(columns)


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
