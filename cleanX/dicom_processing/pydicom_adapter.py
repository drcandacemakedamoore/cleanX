# -*- coding: utf-8 -*-

import os

import pydicom as dicom
import pandas as pd
import cv2

from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from datetime import datetime, date


class PydicomDicomReader:
    """Class for reading DICOM metadata with pydicom."""

    exclude_field_types = (Sequence, MultiValue, bytes)
    date_fields = ('ContentDate', 'SeriesDate', 'ContentDate', 'StudyDate')
    time_fields = ('ContentTime', 'StudyTime')
    exclude_fields = ()

    def __init__(
            self,
            exclude_field_types=None,
            date_fields=None,
            time_fields=None,
            exclude_fields=None,
    ):
        if exclude_field_types:
            self.exclude_field_types = exclude_field_types
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
        tag = source.get_tag()
        columns = {tag: []}
        colnames = set([])
        excluded_columns = set([])
        for key, parsed in source.items(dicom.dcmread):
            for field in parsed.dir():
                colnames.add(field)
                val = parsed[field].value
                if isinstance(val, self.exclude_field_types):
                    excluded_columns.add(field)
        colnames -= excluded_columns
        colnames -= set(self.exclude_fields)
        for key, parsed in source.items(dicom.dcmread, os.path.basename):
            columns[tag].append(key)
            for field in colnames:
                val = parsed[field].value
                col = columns.get(field, [])
                if field in self.date_fields:
                    val = self.dicom_date_to_date(val)
                elif field in self.time_fields:
                    val = self.dicom_time_to_time(val)
                elif isinstance(val, int):
                    val = int(val)
                elif isinstance(val, float):
                    val = float(val)
                elif isinstance(val, str):
                    val = str(val)
                col.append(val)
                columns[field] = col
        return pd.DataFrame(columns)


# TODO(wvxvw): Redo this in a way similar to the reader
def get_jpg_with_pydicom(dicom_folder_path, jpg_folder_path):
    """
    This function is for users with pydicom library only.
    If you do not have the library it will throw an error.
    The funuction function jpeg files out of a dicom file directory,
    one by one, each of them (not just the first series as), and puts them in
    an out put directory.

    :param dicom_folder_path: dicomfile_directory, directory with dicom/.dcm
    :type dicom_folder_path: string
    :param jpg_folder_path: output_directory, where they should be placed
    :type jpg_folder_path: string

    :return: love (will put your images in the new folder but not return them)
    :rtype: bool
    """
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
