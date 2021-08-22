# -*- coding: utf-8 -*-

import click

from .main import main
from cleanX.dataset_processing import MLSetup


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
def report(
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
