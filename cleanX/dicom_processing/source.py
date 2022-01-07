# -*- coding: utf-8 -*-

import os

from glob import glob
from abc import ABC, abstractmethod


class Source(ABC):
    """
    This class is provided as a helper for those who want to implement
    their own sources.

    It is not necessary to extend this class.  cleanX code doesn't do
    type checking, but if you want to ensure type checking on your side,
    you may inherit from this class.
    """

    @abstractmethod
    def get_tag(self):
        """
        The value returned from this function should be suitable for
        pandas to name a column.
        """
        raise NotImplementedError()

    @abstractmethod
    def items(self, reader, transformer=None):
        """
        This function will be expected to produce file names or file-like
        objects of DICOM files.  The results will be then fed to either
        pydicom or SimpleITK libraries for metadata extraction.

        This function should return a generator yielding a tuple of two
        elements.  First element will be inserted into the source column
        (the one labeled by :code:`get_tag()` method), the second is the
        result of calling :code:`reader`.

        :param reader: A function that takes an individual source, either
                       a file path or a file-like object, and returns the
                       processed metadata.
        :param transformer: Optionally, the caller of this function will
                            supply a transformer function that needs to
                            be called on the value that will be stored
                            in the source column of the resulting DataFrame
        """
        while False:
            yield None

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Source:
            get_tag_found, items_found = False, False
            for sub in C.mro():
                for prop in sub.__dict__:
                    if prop == 'get_tag':
                        get_tag_found = True
                    elif prop == 'items':
                        items_found = True
                    if get_tag_found and items_found:
                        return True
        return NotImplemented


class DirectorySource:
    """Class to aid reading DICOMs, package agnostically"""

    def __init__(self, directory, tag):
        self.directory = directory
        self.tag = tag

    def get_tag(self):
        return self.tag

    def items(self, reader, transformer=None):
        for file in os.listdir(self.directory):
            full_path = os.path.join(self.directory, file)
            parsed = reader(full_path)
            if transformer is not None:
                full_path = transformer(full_path)
            yield full_path, parsed


class GlobSource:
    """Class to aid finding files from path (for later reading out DICOM)"""

    def __init__(self, exp, tag, recursive=True):
        self.exp = exp
        self.tag = tag
        self.recursive = recursive

    def get_tag(self):
        return self.tag

    def items(self, reader, transformer=None):
        for file in glob(self.exp, recursive=self.recursive):
            parsed = reader(file)
            if transformer is not None:
                full_path = transformer(file)
            yield file, parsed


class MultiSource:

    def __init__(self, tag, *sources):
        self.tag = tag
        self.sources = sources

    def get_tag(self):
        return self.tag

    def items(self, reader, transformer=None):
        for s in self.sources:
            for key, parsed in s.items(reader, transformer):
                yield key, parsed


def rename_file(original, target, ext):
    dst_file = os.path.basename(original)
    dst_file = os.path.splitext(dst_file)[0]
    return os.path.join(target, '{}.{}'.format(dst_file, ext))
