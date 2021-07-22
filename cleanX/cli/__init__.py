# -*- coding: utf-8 -*-

import json
import os
import copy

from importlib import import_module
from textwrap import dedent
from pydoc import locate

import click

from cleanX.dicom_processing import (
    DirectorySource,
    GlobSource,
    MultiSource,
)
from cleanX.dataset_processing import MLSetup


class Config:

    defaults = {
        'PREFERRED_DICOM_PARSER': 'pydicom',
        'GLOB_IS_RECURSIVE': True,
    }

    def __init__(self, source=None):
        self.source = source
        self.properties = {}
        if self.source is not None:
            self.parse()
        else:
            self.properties = copy.deepcopy(self.defaults)

    def parse(self):
        with open(self.source) as f:
            self.properties = self.merge(self.defaults, json.read(f))

    def merge(self, defaults, extras):
        # TODO(wvxvw): smarter merging
        copy = dict(defaults)
        copy.update(extras)
        return copy

    def add_setting(self, k, v):
        # TODO(wvxvw): hierarchical property names
        self.properties = self.merge(self.properties, {k: v})

    def get_setting(self, k):
        # TODO(wvxvw): hierarchical property names
        return self.properties[k]


@click.group()
@click.option('-c', '--config', nargs=2, multiple=True, default=[])
@click.option('-f', '--config-file', default=None)
@click.pass_context
def main(ctx, config, config_file):
    if config and config_file:
        raise SystemExit(dedent(
            '''
            Must specify either configuration pairs, or configuration file
            '''
        ))
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
    return [
        parse_reader_val(parser, v)
        for v in value
    ]


def parse_reader_val_dict(k, v, value):
    return {
        parse_reader_val(k, vk): parse_reader_val(v, vv)
        for vk, vv in value.items()
    }


def parse_reader_val(parser, value):
    if parser is list:
        return parse_reader_val_list(parser[0], value)
    if parser is dict:
        for k, v in parser.items():
            return parse_reader_val_dict(k, v, value)
    return parser(value)


def parse_reader_arg(name, value, module):
    if module.__name__ == 'pydicom_adapter':
        args = pydicom_reader_args
    else:
        args = simpleitk_reader_args
    parser = args[name]
    return parse_reader_val(parser, json.loads(value))


def parse_sources(sources, cfg):
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
