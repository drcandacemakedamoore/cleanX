# -*- coding: utf-8 -*-

import os

from glob import glob


class DirectorySource:

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

    def __init__(self, exp, tag):
        self.exp = exp
        self.tag = tag

    def get_tag(self):
        return self.tag

    def items(self, reader, transformer=None):
        for file in glob(self.exp):
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
