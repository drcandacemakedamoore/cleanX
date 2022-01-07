# -*- coding: utf-8 -*-

import os

from glob import glob
from datetime import datetime, date

import pandas as pd
import SimpleITK as sitk
import cv2

from .source import rename_file


tag_dictionary = {   # 'key' , 'datapoint_name'
    '0002|0000': 'File Meta Information Group Length',
    '0002|0001': 'File Meta Information Version',
    '0002|0002': 'Media Storage SOP Class UID',
    '0002|0003': 'Media Storage SOP Instance UID',
    '0002|0010': 'Transfer Syntax UID',
    '0002|0012': 'Implementation Class UID',
    '0002|0013': 'Implementation Version Name',
    # -------- above is extra meta ?!
    '0008|0005': 'Specific Character Set',
    '0008|0008': 'Image Type',
    '0008|0016': 'SOP Class UID',
    '0008|0018': 'SOP Instance UID',
    '0008|0020': 'Study Date',
    '0008|0021': 'Series Date',
    '0008|0022': 'Acquisition Date',
    '0008|0023': 'Content Date',
    '0008|0030': 'Study Time',
    '0008|0031': 'Series Time',
    '0008|0032': 'Acquisition Time',
    '0008|0033': 'Content Time',
    '0008|0050': 'Accession Number',
    '0008|0060': 'Modality',
    '0008|0070': 'Manufacturer',
    '0008|0090': 'Referring Physician Name',
    '0008|1010': 'Station Name',
    '0008|1030': 'Study Description',
    '0008|103e': 'Series Description',
    '0008|1090': 'Manufacturers Model Name',
    '0008|2218': 'Anatomic Region Sequence',
    '0008|0100': 'Code Value',
    '0008|0102': 'Coding Scheme Designator',
    '0008|0104': 'Code Meaning ',
    '0010|0010': 'Patients Name',
    '0010|0020': 'Patient ID',
    '0010|0030': 'Patients Birth Date',
    '0010|0040': 'Patients Sex',
    '0010|1010': 'Patients Age',
    '0010|1020': 'Patients Size',
    '0010|1030': 'Patients Weight',
    '0018|0026': 'Intervention Drug Information Sequence',
    '0018|0028': 'Intervention Drug Dose',
    '0018|0035': 'Intervention Drug Start Time',
    '0018|0070': 'Counts Accumulated',
    '0018|0071': 'Acquisition Termination Condition',
    '0018|1020': 'Software Versions',
    '0018|1030': 'Protocol Name',
    '0018|1200': 'Date of Last Calibration',
    '0018|1201': 'Time of Last Calibration',
    '0018|1242': 'Actual Frame Duration ',
    '0019|0010': 'Private Creator',
    '0019|100f': '[Siemens ICON Data Type]',
    '0020|000d': 'Study Instance UID ',
    '0020|000e': 'Series Instance UID ',
    '0020|0010': 'Study ID ',
    '0020|0011': 'Series Number',
    '0020|0013': 'Instance Number',
    '0020|1002': 'Images in Acquisition',
    '0028|0002': 'Samples per Pixel ',
    '0028|0004': 'Photometric Interpretation',
    '0028|0008': 'Number of Frames',
    '0028|0009': 'Frame Increment Pointer',
    '0028|0010': 'Rows',
    '0028|0011': 'Columns',
    '0028|0030': 'Pixel Spacing',
    '0028|0051': 'Corrected Image',
    '0028|0100': 'Bits Allocated ',
    '0028|0101': 'Bits Stored ',
    '0028|0102': 'High Bit',
    '0028|0103': 'Pixel Representation',
    '0028|0106': 'Smallest Image Pixel Value',
    '0028|0107': 'Largest Image Pixel Value',
    '0029|0010': 'Private Creator',
    '0029|0011': 'Private Creator',
    '0029|1008': '[CSA Image Header Type]',
    '0029|1009': '[CSA Image Header Version]',
    '0029|1010': '[CSA Image Header Info] ',
    '0029|1120': '[MedCom History Information]',
    '0029|1131': '[PMTF Information 1]',
    '0029|1132': '[PMTF Information 2]',
    '0029|1133': '[PMTF Information 3]',
    '0029|1134': '[PMTF Information 4]',
    '0033|0010': 'Private Creator',
    '0033|1029': '[Crystal thickness]',
    '0033|1031': '[Camera config angle]',
    '0033|1032': '[Crystal type Startburst or not]',
    '0033|1037': '[Starburst flags]',
    '0035|0010': 'Private Creator ',
    '0035|1001': '[Energy window type]',
    '0054|0010': 'Energy Window Vector',
    '0054|0011': 'Number of Energy Windows',
    '0054|0012': ' Energy Window Information Sequence',
    '0054|0013': ' Energy Window Range Sequence',
    '0054|0014': 'Energy Window Lower Limit',
    '0054|0015': 'Energy Window Upper Limit',
    '0054|0018': 'Energy Window Name',
    '0054|0016': ' Radiopharmaceutical Information Sequence',
    '0018|1074': 'Radionuclide Total Dose',
    '0054|0300': ' Radionuclide Code Sequence',
    '0054|0020': 'Detector Vector',
    '0054|0021': 'Number of Detectors',
    '0054|0022': 'Detector Information Sequence',
    '0018|1142': 'Radial Position',
    '0018|1145': 'Center of Rotation Offset',
    '0018|1147': 'Field of View Shape',
    '0018|1149': 'Field of View Dimension(s)',
    '0018|1180': 'Collimator/grid Name',
    '0018|1181': 'Collimator Type',
    '0018|1182': 'Focal Distance',
    '0018|1183': 'X Focus Center',
    '0018|1184': 'Y Focus Center',
    '0028|0031': 'Zoom Factor',
    '0054|0220': 'View Code Sequence',
    '0054|0410': ' Patient Orientation Code Sequence',
    '0054|0412': ' Patient Orientation Modifier Code Sequence',
    '0054|0414': 'Patient Gantry Relationship Code Sequence',
    '0055|0010': 'Private Creator',
    '0055|107e': '[Collimator thickness]',
    '0055|107f': '[Collimator angular resolution]',
    '0055|10c0': '[Unknown]',
    '0088|0140': 'Storage Media File-set UID',
    '7fe0|0010': 'Pixel Data',
}


class MetadataHelper:
    """Class for getting DICOM metadata with SimpleITK."""

    def __init__(self, reader):
        """
        Initializes this helper with the instance of
        :sitk:`ImageFileReader`

        :param reader: The SimpleITK redader used to read DICOM to
                       extract metadata.
        :type reader: :sitk:`ImageFileReader`
        """
        self.reader = reader

    def fetch_metadata(self, dicom_file):
        """
        Reads enough of the DICOM file to fetch its metadata.

        :param dicom_file: The file for which to read the metadata.
        :type dicom_file: str

        :return: Parsed metadata.
        :rtype: dict
        """
        self.reader.SetFileName(dicom_file)

        self.reader.LoadPrivateTagsOn()

        self.reader.ReadImageInformation()
        parsed = {}
        for k in self.reader.GetMetaDataKeys():
            parsed[k] = self.reader.GetMetaData(k)

        return parsed

    def fetch_image(self, dicom_file):
        """
        Read the pixel data of the DICOM image.

        :param dicom_file: The file to extract image data from.
        :type dicom_file: str

        :return: Pixel array of the extracted image.
        :rtype: :class:`~numpy.ndarray`
        """
        # TODO(wvxvw): Deal with the case when dicom_file is a stream,
        # not a file name.
        self.reader.SetFileName(dicom_file)
        dcm = self.reader.Execute()
        return sitk.GetArrayFromImage(image)


class SimpleITKDicomReader:
    """Class for reading DICOM metadata with SimpleITK."""

    date_fields = set((
        'Content Date',
        'Series Date',
        'Content Date',
        'Study Date',
    ))
    """
    Default fields in the parsed DICOM file to be interpreted as date.
    """

    time_fields = set(('Content Time', 'Study Time'))
    """
    Default fields in the parsed DICOM file to be interpreted as datetime.
    """

    exclude_fields = set([])
    """
    Default fields to exclude from dataframe generated form DICOM files.
    """

    def __init__(
            self,
            date_fields=None,
            time_fields=None,
            exclude_fields=None,
    ):
        """
        Initializes this reader with flags.

        :param date_fields: Overrides the default values from
                            :attr:`~.SimpleITKDicomReader.date_fields`.
        :type date_fields: Iterable
        :param time_fields: Overrides the default values from
                           :attr:`~.SimpleITKDicomReader.time_fields`.
        :type date_fields: Iterable
        :param exclude_fields: Overrides the default values from
                               :attr:`~.SimpleITKDicomReader.exclude_fields`.
        :type exclude_fields: Iterable
        """
        if date_fields:
            self.date_fields = set(date_fields)
        if time_fields:
            self.time_fields = set(time_fields)
        if exclude_fields:
            self.exclude_fields = set(exclude_fields)

    def dicom_date_to_date(self, source):
        """
        Utility method to parse DICOM dates to Python's
        :class:`~datetime.date`.

        :param source: DICOM date given as string.
        :type source: str
        :return: Pyton date object.
        :rtype: :class:`~datetime.date`.
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
        # TODO: We don't know how to convert this yet
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
        reader = sitk.ImageFileReader()
        m_reader = MetadataHelper(reader)

        for entry, parsed in source.items(m_reader.fetch_metadata):
            cv2.imwrite(rename_file(key, destination, 'jpg'), parsed[0])

    def read(self, source):
        """
        Read DICOM files, parse their metadata, generate a :code:`DataFrame`
        based on that metadata.

        :param source: A source generator.  For extended explanation see
                       :class:`~cleanX.dicom_processing.Source`.
        :type source: :class:`~cleanX.dicom_processing.Source`

        :return: dataframe with metadata from dicoms
        :rtype: :class:`~pandas.DataFrame`
        """
        reader = sitk.ImageFileReader()
        m_reader = MetadataHelper(reader)
        tag = source.get_tag()
        columns = {tag: []}
        known_names = set([])

        for entry, parsed in source.items(m_reader.fetch_metadata):
            record_names = set(parsed.keys())
            new_columns = record_names - known_names
            for col in new_columns:
                columns[col] = [None] * len(columns[tag])
            columns[tag].append(entry)
            known_names |= record_names
            for k in known_names - self.exclude_fields:
                v = parsed.get(k)
                col = columns.get(k, [])
                col.append(v)
                columns[k] = col
        return pd.DataFrame(columns).rename(
            columns=tag_dictionary,
        ).drop(columns=self.exclude_fields, errors='ignore')


def rip_out_jpgs_sitk(dicomfile_directory, output_directory):
    """
    This function is for users with simpleITK library only.  If you do
    not have the library it will throw an error.  The function
    function jpeg files out of a dicom file directory, one by one,
    each of them (not just the first series as), and puts them in an
    out put directory. It also returns the images for inspection (as
    arrays), which you can look at the [0] layer with :mod:`matplotlib`

    :param dicomfile_directory: dicomfile_directory, directory with dicom/.dcm
    :type dicomfile_directory: str
    :param output_directory: output_directory, where they should be placed
    :type output_directory: str

    :return: List of images represented as NumPy arrays.
    :rtype: List[numpy.ndarray]

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
