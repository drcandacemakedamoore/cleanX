# -*- coding: utf-8 -*-

import os

from uuid import uuid4

import click

from .main import main
from cleanX.image_work import (
    MultiSource as ImageMultiSource,
    GlobSource,
    create_pipeline,
    restore_pipeline,
)
from cleanX.image_work.steps import get_known_steps


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
