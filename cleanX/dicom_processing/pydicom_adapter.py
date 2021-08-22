# -*- coding: utf-8 -*-

import os

from datetime import datetime, date

import pydicom as dicom
import pandas as pd
import cv2

from pydicom.multival import MultiValue
from pydicom.sequence import Sequence

from .source import rename_file


class PydicomDicomReader:
    """Class for reading DICOM metadata with pydicom."""

    exclude_field_types = (Sequence, MultiValue, bytes)
    """
    Default types of fields not to be included in the dataframe
    produced from parsed DICOM files.
    """

    date_fields = ('ContentDate', 'SeriesDate', 'ContentDate', 'StudyDate')
    """
    Default DICOM tags that should be interpreted as containing date
    information.
    """

    time_fields = ('ContentTime', 'StudyTime')
    """
    Default DICOM tags that should be interpreted as containing
    datetime information.
    """

    exclude_fields = ()
    """
    Default tags to be excluded from genrated :code:`DataFrame` for any
    other reason.
    """

    def __init__(
            self,
            exclude_field_types=None,
            date_fields=None,
            time_fields=None,
            exclude_fields=None,
    ):
        """
        Initializes the reader with some filtering options.

        :param exclude_field_types: Some DICOM types have internal structure
                                    difficult to represent in a dataframe.
                                    These are filtered by default:

                                    * :class:`~pydicom.sequence.Sequence`
                                    * :class:`~pydicom.multival.MultiValue`
                                    * :class:`bytes` (this is usually the
                                      image data)
        :type exclude_field_types: Sequence[type]
        :param date_fields: Fields that should be interpreted as having
                            date information in them.
        :type date_fields: Sequence[str]
        :param time_fields: Fields that should be interpreted as having
                            time information in them.
        :type time_fields: Sequence[str]
        :param exclude_fields: Fields to exclude (in addition to those selected
                               by :code:`exclude_field_types`
        :type exclude_fields: Sequence[str]
        """
        if exclude_field_types:
            self.exclude_field_types = exclude_field_types
        if date_fields:
            self.date_fields = date_fields
        if time_fields:
            self.time_fields = time_fields
        if exclude_fields:
            self.exclude_fields = exclude_fields

    def dicom_date_to_date(self, source):
        """
        Utility method to help translate DICOM dates to :class:`~datetime.date`

        :param source: Date stored as a string in DICOM file.
        :type source: str
        :return: Python date object.
        :rtype: :class:`~datetime.date`
        """
        year = int(source[:4])
        month = int(source[4:6])
        day = int(source[6:])
        return date(year=year, month=month, day=day)

    def dicom_time_to_time(self, source):
        """
        Utility method to help translate DICOM date and time objects to python
        :class:`~datetime.datetime`.

        .. warning::

            This isn't implemented yet.  Needs research on DICOM time
            representation.

        :param source: Date and time stored in DICOM as a string.
        :type source: str
        :return: Python's datetime object.
        :rtype: :class:`~datetime.datetime`
        """
        #     seconds, milis = source.split('.')
        # TODO: We don't know how to conver this yet
        return source

    def rip_out_jpgs(self, source, destination):
        """
        Extract image data from DICOM files and save it as JPG in
        :code:`destination`.

        :param source: A source generator.  For extended explanation see
                       :class:`~cleanX.dicom_processing.Source`.
        :type source: :class:`~cleanX.dicom_processing.Source`
        :param destination: The name of the directory where JPG files
                            should be stored.
        :type destination: Compatible with :func:`os.path.join`
        """
        for key, parsed in source.items(dicom.dcmread):
            cv2.imwrite(
                rename_file(key, destination, 'jpg'),
                parsed.pixel_array,
            )

    def read(self, source):
        """
        This function allows reading of metadata in what source gives.

        :param source: A source generator.  For extended explanation see
                       :class:`~cleanX.dicom_processing.Source`.
        :type source: :class:`~cleanX.dicom_processing.Source`

        :return: dataframe with metadata from dicoms
        :rtype: :class:`~pandas.DataFrame`
        """

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
    :type dicom_folder_path: str
    :param jpg_folder_path: output_directory, where they should be placed
    :type jpg_folder_path: str

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
