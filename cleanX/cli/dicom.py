# -*- coding: utf-8 -*-

import logging
import os

from importlib import import_module
from textwrap import dedent
from pydoc import locate

import click

from .main import main
from cleanX.dicom_processing import (
    DirectorySource,
    GlobSource,
    MultiSource,
)


@main.group()
@click.pass_obj
def dicom(cfg):
    pass


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
    pattern as used by Python's built-in glob function.  Whether glob
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
def report(cfg, input, output, config_reader):
    reader = create_reader(cfg, config_reader)
    df = reader.read(parse_sources(input, cfg))
    df.to_csv(os.path.join(output, 'report.csv'))


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
def extract(cfg, input, output, config_reader):
    reader = create_reader(cfg, config_reader)
    reader.rip_out_jpgs(parse_sources(input, cfg), output)


def create_reader(cfg, config_reader):
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
    return reader_class(**reader_args)


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
