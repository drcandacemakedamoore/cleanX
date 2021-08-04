# -*- coding: utf-8 -*-
"""
The documentation for CLI is generated separately using
`sphinx_click <https://github.com/click-contrib/sphinx-click>`_. It should
be available `here <https://drcandacemakedamoore.github.io/cleanX/cli.html>`_.
"""

import json
import os
import copy
import logging

from importlib import import_module
from textwrap import dedent
from pydoc import locate
from uuid import uuid4

import click

from cleanX.dicom_processing import (
    DirectorySource,
    GlobSource,
    MultiSource,
)
from cleanX.dataset_processing import MLSetup
from cleanX.image_work import (
    MultiSource as ImageMultiSource,
    GlobSource,
    create_pipeline,
    restore_pipeline,
)
from cleanX.image_work.steps import get_known_steps


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


@main.group()
@click.pass_obj
def dicom(cfg):
    pass


@main.group()
@click.pass_obj
def dataset(cfg):
    pass


@main.group()
@click.pass_obj
def images(cfg):
    pass


unique_flag_value = str(uuid4())


def deserialize_step(record):
    """
    :meta private:
    """
    prefix_len = record.find('(')
    if prefix_len < 0:
        prefix_len = len(record)
    class_name = record[:prefix_len]
    step_class = get_known_steps()[class_name]
    cmd_args = '' if prefix_len == len(record) else record[prefix_len + 1:-1]
    return step_class.from_cmd_args(cmd_args)


def str_or_bool(s):
    """
    :meta private:
    """
    print(type(s))
    return s


@images.command()
@click.pass_obj
@click.option(
    '-s',
    '--step',
    default=[],
    multiple=True,
    help='''
    Step to be executed by the pipeline
    ''',
)
@click.option(
    '-b',
    '--batch-size',
    default=None,
    type=int,
    help='''
    How many images to process concurrently.
    ''',
)
@click.option(
    '-j',
    '--journal',
    flag_value=unique_flag_value,
    is_flag=False,
    default=False,
    type=str_or_bool,
    help='''
    Where to store the journal.  If not specified, the default
    journal location is used.  You can control the default location
    by modifying JOURNAL_HOME configuration setting.
    ''',
)
@click.option(
    '-k',
    '--keep-journal',
    default=False,
    is_flag=True,
    help='''
    Whether to keep journal after the pipeline finishes.
    ''',
)
@click.option(
    '-r',
    '--source',
    default=['./*.[jJ][pP][gG]'],
    multiple=True,
    help='''
    Glob-like expression to look for source images
    ''',
)
def run_pipeline(cfg, step, source, batch_size, journal, keep_journal):
    if journal == unique_flag_value:
        journal = os.path.join(cfg.get_setting('JOURNAL_HOME'), str(uuid4()))

    steps = [deserialize_step(s) for s in step]
    p = create_pipeline(
        steps,
        batch_size=batch_size,
        journal=journal,
        keep_journal=keep_journal,
    )
    p.process(ImageMultiSource(GlobSource(src) for src in source))


@images.command()
@click.pass_obj
@click.option(
    '-j',
    '--journal-dir',
    required=True,
    help='''
    Where is the journal stored
    ''',
)
@click.option(
    '-s',
    '--skip',
    default=0,
    help='''
    Number of steps to skip before resuming the pipeline
    ''',
)
@click.option(
    '-r',
    '--source',
    default=['./*.[jJ][pP][gG]'],
    multiple=True,
    help='''
    Glob-like expression to look for source images
    ''',
)
def restore_pipeline(cfg, journal_dir, skip, overrides, source):
    overrides = dict(overrides)
    p = restore_pipeline(journal_dir=journal_dir, skip=skip, **overrides)
    p.process(ImageMultiSource(GlobSource(src) for src in source))


@dataset.command()
@click.pass_obj
@click.option(
    '-r',
    '--train-source',
    help='''
    The source of the test data (usually, a file path).

    Supported source types are:

    \b
    * json
    * csv
    ''',
)
@click.option(
    '-t',
    '--test-source',
    help='''
    The source of the test data (usually, a file path).

    Supported source types are:

    \b
    * json
    * csv
    ''',
)
@click.option(
    '-i',
    '--unique_id',
    default=None,
    help='''
    The name of the column that uniquely selects cases in the dataset
    (typically, patient's id).  If not given, the first matching column
    in the test and train datasets is considered to be the unique id.
    ''',
)
@click.option(
    '-l',
    '--label-tag',
    default='Label',
    help='''
    The name of the column that typically has the diagnosis, the propery
    that is being learned in this machine learning task.  The default
    value is "Label".
    ''',
)
@click.option(
    '-s',
    '--sensitive-category',
    default=[],
    multiple=True,
    help='''
    Repeatable.  The name of the column that describes the property of the
    dataset that may potentially exhibit bias, eg. "gender", "ethnicity"
    etc.
    ''',
)
@click.option(
    '--report-duplicates/--no-report-duplicates',
    default=True,
    help='''
    Whether the report should contain information about ducplicates.
    ''',
)
@click.option(
    '--report-leakage/--no-report-leakage',
    default=True,
    help='''
    Whether the report should contain information about leakage.
    ''',
)
@click.option(
    '--report-bias/--no-report-bias',
    default=True,
    help='''
    Whether the report should contain information about leakage.
    ''',
)
@click.option(
    '--report-understand/--no-report-understand',
    default=True,
    help='''
    Whether the report should contain information about understanding.
    ''',
)
@click.option(
    '-o',
    '--output',
    default=None,
    help='''
    The file to output the report to.  If no file is given, the report will
    be printed on stdout.  Supported report formats are
    (inferred from file extension):

    \b
    * txt
    ''',
)
def generate_report(
        cfg,
        train_source,
        test_source,
        unique_id,
        label_tag,
        sensitive_category,
        output,
        report_duplicates,
        report_leakage,
        report_bias,
        report_understand,
):
    mlsetup = MLSetup(
        train_source,
        test_source,
        unique_id=unique_id,
        label_tag=label_tag,
        sensitive_list=sensitive_category,
    )
    report = mlsetup.generate_report(
        duplicates=report_duplicates,
        leakage=report_leakage,
        bias=report_bias,
        understand=report_understand,
    )
    if output is None:
        print(report.to_text())
    else:
        ext = os.path.splitext(output)[1]
        if ext.lower() == '.txt':
            with open(output, 'w') as out:
                out.write(report.to_text())
        else:
            raise ValueError('Unsupported report type: {}'.format(ext))


@dicom.command()
@click.pass_obj
@click.option(
    '-i',
    '--input',
    nargs=2,
    multiple=True,
    help='''
    Repeatable.  Takes two arguments.  First argument is a type of source,
    the second is the source description.

    Supported source types are:

    \b
    * dir
    * glob

    If source type is `dir', then the source description must be a path
    to a directory.

    If source type is `glob', then the source description must be a glob
    pattern as used by Python's builtin glob function.  Whether glob
    pattern will be interpreted as recursive is controlled by configuration
    setting GLOB_IS_RECURSIVE.
    ''',
)
@click.option(
    '-o',
    '--output',
    default='.',
    help='''
    The directory where the extracted images will be placed.
    ''',
)
@click.option(
    '-c',
    '--config-reader',
    nargs=2,
    multiple=True,
    help='''
    Options to pass to the DICOM reader at initialization time.

    These will depend on the chosen reader.
    '''
)
def extract_images(cfg, input, output, config_reader):
    preferred_parser = cfg.get_setting('PREFERRED_DICOM_PARSER')
    mod_name = 'cleanX.dicom_processing.'
    class_name = 'DicomReader'

    if preferred_parser == 'pydicom':
        mod_name += 'pydicom_adapter'
    else:
        mod_name += 'simpleitk_adapter'
    try:
        dicom_mod = import_module(mod_name)
    except ModuleNotFoundError:
        try:
            if preferred_parser == 'pydicom':
                mod_name = mod_name.replace('pydicom', 'simpleitk')
            else:
                mod_name = mod_name.replace('simpleitk', 'pydicom')
            dicom_mod = import_module(mod_name)
        except ModuleNotFoundError:
            raise SystemExit(dedent(
                '''
                Neither pydicom nor SimpleITK is available
                '''
            ))
    if 'pydicom' in mod_name:
        class_name = 'Pydicom' + class_name
    else:
        class_name = 'SimpleITK' + class_name
    reader_class = getattr(dicom_mod, class_name)
    reader_args = {
        k: parse_reader_arg(k, v, dicom_mod)
        for k, v in config_reader
    }
    reader = reader_class(**reader_args)
    df = reader.read(parse_sources(input, cfg))
    df.to_csv(os.path.join(output, 'report.csv'))


def pydicom_str_to_type(raw, module):
    """
    :meta private:
    """
    try:
        return getattr(module, raw)
    except Exception:
        return locate(raw)


pydicom_reader_args = {
    'exclude_field_types': [pydicom_str_to_type],
    'date_fields': [str],
    'time_fields': [str],
    'exclude_fields': [str],
}

simpleitk_reader_args = {}


def parse_reader_val_list(parser, value):
    """
    :meta private:
    """
    return [
        parse_reader_val(parser, v)
        for v in value
    ]


def parse_reader_val_dict(k, v, value):
    """
    :meta private:
    """
    return {
        parse_reader_val(k, vk): parse_reader_val(v, vv)
        for vk, vv in value.items()
    }


def parse_reader_val(parser, value):
    """
    :meta private:
    """
    if parser is list:
        return parse_reader_val_list(parser[0], value)
    if parser is dict:
        for k, v in parser.items():
            return parse_reader_val_dict(k, v, value)
    return parser(value)


def parse_reader_arg(name, value, module):
    """
    :meta private:
    """
    if module.__name__ == 'pydicom_adapter':
        args = pydicom_reader_args
    else:
        args = simpleitk_reader_args
    parser = args[name]
    return parse_reader_val(parser, json.loads(value))


def parse_sources(sources, cfg):
    """
    :meta private:
    """
    raw_result = []
    for st, sv in sources:
        if st == 'dir':
            raw_result.append(DirectorySource(sv, ''))
        elif st == 'glob':
            raw_result.append(GlobSource(
                sv,
                '',
                recursive=cfg.get_setting('GLOB_IS_RECURSIVE'),
            ))
        else:
            raise LookupError('Unsupported source type: {}'.format(st))
    return MultiSource('source', *raw_result)
