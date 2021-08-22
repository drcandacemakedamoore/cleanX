# -*- coding: utf-8 -*-

import os
import copy
import logging

import click


class Config:
    """
    Store for various settings necessary to run the functions
    from :code:`cleanX` from command-line.

    Recognized configuration variables:

        * :code:`PREFERRED_DICOM_PARSER` can be either :code:`"pydicom"`
          or :code:`"SimpleITK"`.  Controls which DICOM parser to use.
          Only makes sense if both modules are available.  The default
          is :code:`"pydicom"`.
        * :code:`GLOB_IS_RECURSIVE` can be :code:`True` or :code:`False`.
          controls how :func:`glob()` patterns are interpreted when
          they use :code:`**` command.  The default is :code:`False`.
        * :code:`JOURNAL_HOME` is a path to the directory where journals
          for journaling pipeline are stored.  This defaults to
          :code:`~/cleanx/journal/` if not specified.
    """

    defaults = {
        'PREFERRED_DICOM_PARSER': 'pydicom',
        'GLOB_IS_RECURSIVE': True,
        'JOURNAL_HOME': os.path.expanduser('~/cleanx/journal/'),
    }

    def __init__(self, source=None):
        """
        Initializes configuration with :code:`source`.  Source should
        be a JSON file with a dictionary, where keys will be interpreted
        as configuration variables and values as those variables values.

        :param source: Path to the configuration file.
        :type source: Suitable for :func:`open()`
        """
        self.source = source
        self.properties = {}
        if self.source is not None:
            self.parse()
        else:
            self.properties = copy.deepcopy(self.defaults)

    def parse(self):
        """
        Parse configuration file.
        """
        with open(self.source) as f:
            self.properties = self.merge(self.defaults, json.read(f))

    def merge(self, defaults, extras):
        """
        Merge default configuration with overrides from the configuration
        file.
        """
        # TODO(wvxvw): smarter merging
        copy = dict(defaults)
        copy.update(extras)
        return copy

    def add_setting(self, k, v):
        """
        Override existing setting with the given setting.

        :param k: The name of the setting to replace.
        :type k: str
        :param v: The new value of the setting.
        """
        # TODO(wvxvw): hierarchical property names
        self.properties = self.merge(self.properties, {k: v})

    def get_setting(self, k):
        """
        Read the configuration setting.

        :param k: The configuration variable whose value should be found.
        :type k: str

        :return: The current value of the configuration variable :code:`k`.
        """
        # TODO(wvxvw): hierarchical property names
        return self.properties[k]


@click.group()
@click.option(
    '-c', '--config',
    nargs=2,
    multiple=True,
    default=[],
    help='''
    Configuration value pairs. The values will be processed using JSON parser.
    For the list of possible values see :class:`.Config`.
    ''',
)
@click.option(
    '-f', '--config-file',
    default=None,
    help='''
    Similar to :code:`--config` it is possible to provide all the necessary
    configuraiton settings as a file.  The file needs to be in JSON format
    suitable for :meth:`.Config.parse()`.
    ''',
)
@click.option(
    '-v',
    '--verbosity',
    default='warning',
    type=click.Choice([
        'critical',
        'debug',
        'error',
        'fatal',
        'info',
        'warning',
        'warn',
        'notset',
    ], case_sensitive=False),
    help='''
    Controls verbosity level set for :mod:`logging`.  The default is
    :code:`logging.WARNING`.
    '''
)
@click.pass_context
def main(ctx, config, config_file, verbosity):
    if config and config_file:
        raise SystemExit(dedent(
            '''
            Must specify either configuration pairs, or configuration file
            '''
        ))
    logging.basicConfig(level=getattr(logging, verbosity.upper()))
    ctx.obj = Config(config_file)
    for k, v in config:
        ctx.obj.add_setting(k, v)
