# -*- coding: utf-8 -*-
"""
Library for cleaning radiological data used in machine learning
applications.

Module dataframes: a module for processing of datasets related to images.
This module can be implemented by functions,
or can be implemented with classes.
"""

import os
import html
import logging
import re

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
import numpy as np
import textwrap as tw


class GuesserError(TypeError):
    """
    This error is raised when the loading code cannot figure
    out what kind of source it is dealing with.

    """
    pass


class ColumnsSource(ABC):
    """
    Formal superclass for all sources that should be used to
    produce :class:`~pandas.DataFrame`.
    """

    @abstractmethod
    def to_dataframe(self):
        """
        Descendants of this class must implement this method.

        :return: Dataframe produced from the source represented by this class.
        :rtype: :class:`~pandas.DataFrame`.
        """
        raise NotImplementedError()


class CSVSource(ColumnsSource):
    """Class that helps turn csv into a dataframe"""

    def __init__(self, csv, **pd_args):
        """
        Initializes this class with the path to the csv file from which
        to create the dataframe and the pass-through arguments to
        :func:`pandas.read_csv()`.

        :param csv: The path to CSV file or an open file handle.  Must be
                    suitable for :func:`pandas.read_csv()`.
        :param \\**pd_args: The pass-through arguments to
                            :func:`pandas.read_csv()`.
        """
        self.csv = csv
        self.pd_args = pd_args or {}

    def to_dataframe(self):
        """
        Necessary implementation of the abstractmethod.

        :return: Dataframe produced from the source represented by this class.
        :rtype: :class:`~pandas.DataFrame`.
        """
        return pd.read_csv(self.csv, **self.pd_args)


class JSONSource(ColumnsSource):
    """Class that helps turn json into a dataframe for later exploration"""

    def __init__(self, json, **pd_args):
        """
        Initializes this class with the path to the json file from which
        to create the dataframe and the pass-through arguments to
        :func:`pandas.read_json()`.

        :param json: The path to JSON file or an open file handle.  Must be
                    suitable for :func:`pandas.read_json()`.
        :param \\**pd_args: The pass-through arguments to
                            :func:`pandas.read_json()`.
        """
        self.json = json
        self.pd_args = pd_args or {}

    def to_dataframe(self):
        """
        Necessary implementation of the abstractmethod.

        :return: Dataframe produced from the source represented by this class.
        :rtype: :class:`~pandas.DataFrame`.
        """
        return pd.read_json(self.json, **self.pd_args)


class DFSource(ColumnsSource):
    """
    This class is a no-op source.  Use this when you already have a dataframe
    ready.
    """

    def __init__(self, df):
        """
        Initializes this class with the existing dataframe.

        :param df: Existing dataframe that should be used as the source.
        :type df: :class:`~pandas.DataFrame`.
        """
        self.df = df

    def to_dataframe(self):
        """
        Necessary implementation of the abstractmethod.

        :return: Dataframe produced from the source represented by this class.
        :rtype: :class:`~pandas.DataFrame`.
        """
        return self.df


class MultiSource(ColumnsSource):
    """
    This class allows aggregation of multiple sources.
    """

    def __init__(self, *sources):
        """
        Initializes this class with the number of sources that will
        be used in a way similar to :func:`itertools.chain()`.

        :param \\*sources: Sources to be concatenated together to
                           create a single dataframe.
        """
        self.sources = sources

    def to_dataframe(self):
        """
        Necessary implementation of the abstractmethod.

        :return: Dataframe produced from the source represented by this class.
        :rtype: :class:`~pandas.DataFrame`.
        """
        return pd.concat(s.to_dataframe() for s in self.sources)


def string_source(raw_src):
    """
    Helper function to select source based on file extension.

    :param raw_src: The path to the file to be interpreted as a source.
    :type raw_src: Suitable for :func:`os.path.splitext()`

    :return: Either :class:`~.CSVSource` or :class:`~.JSONSource`, depending
             on file extension.
    :rtype: Union[CSVSource, JSONSource]
    """
    ext = os.path.splitext(raw_src)[1]
    if isinstance(ext, bytes):
        ext = ext.decode()
    ext = ext.lower()
    if ext == '.csv':
        return CSVSource(raw_src)
    elif ext == '.json':
        return JSONSource(raw_src)

    raise GuesserError('Cannot guess source of {}'.format(raw_src))


class MLSetup:
    """
    This class allows configuration of the train and test datasets
    organized into a pandas dataframe to be checked for problems, and
    creates reports, which can be put in multiple output options.
    """

    known_sources = {
        str: string_source,
        bytes: string_source,
        Path: string_source,
        pd.DataFrame: lambda _: DFSource,
    }
    """
    Mapping of types of sources to the factory functions for creating
    source objects.
    """

    def __init__(
            self,
            train_src,
            test_src,
            unique_id=None,
            label_tag='Label',
            sensitive_list=None,
    ):
        """
        Initializes this class with various aspects of a typical
        machine-learning study.

        :param train_src: The source for training dataset.
        :type train_src: If this is a path-like object, it is interpreted
                         to be a path to a file that needs to be read into
                         a :class:`~pandas.DataFrame`.  If this is already
                         a dataframe, it's used as is.  If this is an iterable
                         it is interpreted as a sequence of different sources
                         and will be processed using :class:`~.MultiSource`.
        :param test_src: Similar to :code:`train_src`, but for testing dataset.
        :type test_src: Same as :code:`train_src`
        :param unique_id: The name of the column that uniquely identifies the
                          cases (typically, this is patient's id).
        :type unique_id: Suitable for accessing columns in :mod:`pandas`
                         dataframe.
        :param label_tag: Usually, the training dataset has an assesment column
                          that assings the case to the category of interest.
                          Typically, this is diagnosis, or finding, etc.
        :type label_tag: Suitable for accessing columns in :mod:`pandas`
                         dataframe.
        :param sensitive_list: The list of columns that you suspect might be
                               affecting the fairness of the study.  These
                               are typically gender, age, ethnicity etc.
        :type sensitive_list: A sequence of regular expression that will be
                              applied to the colum names (converted to strings
                              if necessary).
        """
        self.train_src = self.guess_source(train_src)
        self.test_src = self.guess_source(test_src)
        self.unique_id = unique_id
        self.label_tag = label_tag
        self.sensitive_list = sensitive_list

    def get_unique_id(self):
        """
        Tries to find the column in training and testing datasests that
        uniquely identifies the entries in both.  Typically, this is
        patient id of some sort.  If the setup was initialized with
        :code:`unique_id`, than that is used.  Otherwise, a rather simple
        heuristic is used.

        :return: The name of the column that should uniquely identify
                 the cases being studied.
        :rtype: The value from :attr:`~pandas.DataFrame.columns`.  Typically,
                this is a :class:`str`.
        """
        if self.unique_id:
            return self.unique_id
        c1, c2 = self.metadata()
        for a in c1:
            for b in c2:
                if a == b:
                    return a
        logging.warning('No common column names, cannot guess unique id')
        return c1[0]

    def get_sensitive_list(self):
        """
        Returns a list of regular expressions that will be applied to the
        list of columns to identify the sensitive categories (those that
        might bias the training towards overrepresented categories).
        """
        # TODO(wvxvw): Try to come up with names that might catch some
        # sensitive category. The names will be interpreted as regular
        # expressions to match against column names
        if self.sensitive_list:
            return self.sensitive_list
        return [
            re.compile(r'gender', re.IGNORECASE),
        ]

    def guess_source(self, raw_src):
        """
        Helper method to convert sources given by external factors to internal
        representation.

        :param raw_src: The externally supplied source.  This is typically
                        either a path to a file, or an existing dataframe
                        or a collection of such sources.

        :return: An internal representation of source.
        :rtype: :class:`~.ColumnSource`
        """
        guesser = self.known_sources.get(type(raw_src))
        if guesser:
            return guesser(raw_src)
        if isinstance(raw_src, Iterable):
            return MultiSource(self.guess_source(src) for src in raw_src)
        elif isinstance(raw_src, ColumnsSource):
            return raw_src

    def metadata(self):
        """
        Returns a tuple of column names of train and test datasets.

        :return: Column names of train and test datasets.
        """
        return (
            self.train_src.to_dataframe().columns,
            self.test_src.to_dataframe().columns,
        )

    def concat_dataframe(self):
        """
        Helper method to generate the dataset containing both training and
        test data.

        :returns: A combined dataframe.
        :rtype: :class:`~pandas.DataFrame`
        """
        return pd.concat((
            self.train_src.to_dataframe(),
            self.test_src.to_dataframe(),
        ))

    def duplicated(self):
        """
        Provides information on duplicates found in training and test data.

        :return: A dataframe with information about duplicates.
        :rtype: :class:`~pandas.DataFrame`
        """
        return self.concat_dataframe().duplicated()

    def duplicated_frame(self):
        """
        Provides more detailed information about the duplicates found in
        training and test data.

        :return: A tuple of two dataframes first listing duplicates in
                 training data, second listing duplicates in test data.
        """
        train_df = self.train_src.to_dataframe()
        test_df = self.test_src.to_dataframe()
        train_dupe_names = train_df[train_df.duplicated()]
        test_dupe_names = test_df[test_df.duplicated()]

        return (
            train_dupe_names,
            test_dupe_names,
        )

    def duplicates(self):
        """
        Calculates the number of duplicates in training and test datasets
        separately.

        :return: A tuple with number of duplicates in training and test
                 datasets.
        :rtype: Tuple[int, int]
        """
        return (
            self.train_src.to_dataframe().duplicated().sum(),
            self.test_src.to_dataframe().duplicated().sum(),
        )

    def pics_in_both_groups(self, unique_id):
        """
        Generates dataframe listing cases that appear in both training and
        testing datasets.

        :return: A dataframe with images found in both training and testing
                 datasets.
        :rtype: :class:`~pandas.DataFrame`
        """
        train_df = self.train_src.to_dataframe()
        test_df = self.test_src.to_dataframe()
        return train_df.merge(test_df, on=unique_id, how='inner')

    def generate_report(
        self,
        duplicates=True,
        leakage=True,
        bias=True,
        understand=True,
    ):
        """
        Generates report object summarizing the properties of this setup.
        This report can later be used to produce formatted output for
        inspection by human.

        :param duplicates: Whether the information on duplicates needs to
                           be included in the report.
        :type duplicates: bool
        :param leakage: Whether the information on data leakage (cases from
                        training set appearing in test set) should be
                        included in report.
        :type leakage: bool
        :param bias: Whether the information about distribution in sensitive
                     categories should be included in the report.
        :type bias: bool
        :param understand: Whether information about general properties of
                           the dataset should be included in the report.
        :type understand: bool
        """
        return Report(
            self,
            duplicates,
            leakage,
            bias,
            understand,
        )

    def leakage(self):
        """This method explores the data in terms of any instances found in
        both training and test sets."""
        uid = self.get_unique_id()
        train_df = self.train_src.to_dataframe()
        test_df = self.test_src.to_dataframe()
        return train_df.merge(test_df, on=uid, how='inner')

    def bias(self):
        """This method sorts the data instances by sensitive categories for
        each label e.g. if the ML is intended to diagnose pneumonia then cases
        of pneumonia and not pneumonia would get counts of gender or other
        specified sensitive categories."""
        sensitive_patterns = self.get_sensitive_list()
        df = self.train_src.to_dataframe()
        aggregate_cols = set(())
        for col in df.columns:
            col = str(col)
            for p in sensitive_patterns:
                if re.fullmatch(p, col):
                    aggregate_cols.add(col)
                    break
        aggregate_cols = [self.label_tag] + list(aggregate_cols)
        tab_fight_bias = pd.DataFrame(
            df[aggregate_cols].value_counts()
        )
        tab_fight_bias2 = tab_fight_bias.groupby(aggregate_cols).sum()
        tab_fight_bias2 = tab_fight_bias2.rename(columns={0: 'sums'})
        return tab_fight_bias2


class Report:
    """
    This class is for a report which can be produced about the data.
    """

    def __init__(
        self,
        mlsetup,
        duplicates=True,
        leakage=True,
        bias=True,
        understand=True,
    ):
        """
        Initializes report instance with flags indicating what parts to
        include in the report.

        :param mlsetup: The setup this report is about.
        :type mlsetup: :class:`~.MLSetup`
        :param duplicates: Whether information about duplicates is to be
                           reported.
        :type duplicates: bool
        :param leakage: Whether information about leakage is to be
                        reported.
        :type leakage: bool
        :param bias: Whether information about bias is to be reported.
        :type bias: bool
        :param understand: Whether general information about the given
                           setup is to be reported.
        :type understand: bool
        """
        self.mlsetup = mlsetup
        self.sections = {}
        if duplicates:
            self.report_duplicates()
        if leakage:
            self.report_leakage()
        if bias:
            self.report_bias()
        if understand:
            self.report_understand()

    def report_duplicates(self):
        """
        This method extracts information on duplicates in the datasets,
        once make into dataframes.  The information can be reported.
        """
        train_dupes, test_dupes = self.mlsetup.duplicates()
        dupe_names = self.mlsetup.duplicated_frame()
        train_dupe_names, test_dupe_names = self.mlsetup.duplicated_frame()
        self.sections['Duplicates'] = {
            'Train Duplicates Count': train_dupes,
            'Duplicated train names': train_dupe_names,
            'Test Duplicates Count': test_dupes,
            'Duplicated test names': test_dupe_names,
        }

    def report_leakage(self):
        """
        Adds a report section on data leakage (training results found in
        testing samples).
        """
        self.sections['Leakage'] = {
           'Leaked entries': self.mlsetup.leakage(),
        }

    def report_bias(self):
        """
        Adds a report section on distribution in sensitive categories.
        """
        self.sections['Value Counts on Sensitive Categories'] = {
           'Value counts of categorty 1': self.mlsetup.bias(),
        }

    def report_understand(self):
        """
        This method extracts information on the datasets,
        once make into dataframes.  The information can be reported.
        """
        # TODO(wvxvw): The calculation part needs to go into the
        # MLSetup, only the functionality relevant to reporting needs
        # to be here.
        train_df = self.mlsetup.train_src.to_dataframe()
        test_df = self.mlsetup.test_src.to_dataframe()
        train_columns = train_df.columns
        test_columns = test_df.columns
        columns = train_columns + test_columns
        train_rows = len(train_df)
        test_rows = len(test_df)
        rows = train_rows + test_rows
        train_nulls = train_df.isna().sum().sum()
        test_nulls = test_df.isna().sum().sum()
        nulls = train_nulls + test_nulls
        train_dtypes = list(train_df.dtypes)
        test_dtypes = list(test_df.dtypes)
        datatypes = train_dtypes + test_dtypes
        train_description = train_df.describe()
        test_description = test_df.describe()
        description = train_description + test_description
        self.sections['Knowledge'] = {
           'Columns': columns,
           'Train columns': train_columns,
           'Test columns': test_columns,
           'Rows': rows,
           'Train rows': train_rows,
           'Test rows': test_rows,
           'Nulls': nulls,
           'Train nulls': train_nulls,
           'Test nulls': test_nulls,
           'Datatypes': datatypes,
           'Train datatypes': train_dtypes,
           'Test datatypes': test_dtypes,
           'Descriptions': description,
           'Train description': train_description,
           'Test description': test_description,
        }

    def subsection_html(self, data, level=2):
        """
        Utility method to recursively generate subsections
        for HTML report.

        :param data: The data to be reported.
        :type data: Various data structures constituting the report
        :param level: How deeply this section is indented.
        :type level: int
        :return: A list containing HTML markup elements.
        :rtype: List[str]
        """
        elements = ['<ul>']
        for k, v in data.items():
            if type(v) is dict:
                elements.append(
                    '<li><h{}>{}</h{}></li>'.format(
                        level,
                        html.escape(k),
                        level,
                    ))
                elements += self.subsection_html(v, level + 1)
            elif isinstance(v, pd.DataFrame):
                elements += ['<li>', v._repr_html_(), '</li>']
            else:
                elements.append(
                    '<li><strong>{}: </strong>{}</li>'.format(
                        html.escape(str(k)),
                        html.escape(str(v))
                    ))
        elements.append('</ul>')
        return elements

    def subsection_text(self, data, level=2):
        """
        Utility method to recursively generate subsections
        for text report.

        :param data: The data to be reported.
        :type data: Various datastructures constituting the report
        :param level: How deeply this section is indented.
        :type level: int
        :return: A strings containing the text of subsection.  Only the
                 subsections of the returned subsection are indented.
                 I.e. you need to indent the result according to
                 nesting level.
        :rtype: str
        """
        elements = []
        prefix = '    '
        for k, v in data.items():
            if type(v) is dict:
                elements.append(str(k))
                elements.append('-' * len(str(k)))
                elements.append(
                    tw.indent(
                        self.subsection_text(v, level + 1),
                        prefix,
                        ))
                elements.append('')
            elif isinstance(v, pd.DataFrame):
                elements.append(str(k))
                elements.append('-' * len(str(k)))
                elements.append(tw.indent(str(v), prefix))
                elements.append('')
            elif isinstance(v, (Sequence, pd.Index)):
                elements.append(str(k))
                elements.append('-' * len(str(k)))
                for i, item in enumerate(v):
                    elements.append('  {}. {}'.format(i, item))
                elements.append('')
            else:
                elements.append('{}: {}'.format(k, v))
        return '\n'.join(elements)

    def to_ipwidget(self):
        """
        Generates an :class:`~IPython.display.HTML` widget.  This is
        mostly usable when running in Jupyter notebook context.

        .. warning::

            This will try to import the widget class which is *not*
            installed as the dependency of :code:`cleanX`.  It relies
            on it being available as part of Jupyter installation.

        :return: An HTML widget with the formatted report.
        :rtype: :class:`~IPython.display.HTML`
        """
        from IPython.display import HTML

        elements = ['<ul>']
        for k, v in self.sections.items():
            if type(v) is dict:
                elements.append('<li><h1>{}</h1></li>'.format(html.escape(k)))
                elements += self.subsection_html(v)
            elif isinstance(v, pd.DataFrame):
                elements += ['<li>', v._repr_html_(), '</li>']
            else:
                elements.append(
                    '<li><strong>{}: </strong>{}</li>'.format(
                        html.escape(str(k)),
                        html.escape(str(v))
                    ))
        elements.append('</ul>')

        return HTML(''.join(elements))

    def to_text(self):
        """
        Generates plain text representation of this report.

        :return: A string suitable for either printing to the
                 screen or saving in a text file.
        :rtype: str
        """
        elements = []
        for k, v in self.sections.items():
            if type(v) is dict:
                elements.append(str(k))
                elements.append('=' * len(str(k)))
                elements.append(tw.indent(self.subsection_text(v), '    '))
                elements.append('')
            else:
                elements.append('{}: {}'.format(k, str(v)))
        return '\n'.join(elements)


def check_paths_for_group_leakage(train_df, test_df, unique_id):
    """
    Finds train samples that have been accidentally leaked into test
    samples

    :param train_df: Pandas :code:`DataFrame` containing information about
                     train assets.
    :type train_df: :class:`~pandas.DataFrame`
    :param test_df: Pandas :code:`DataFrame` containing information about
                    train assets.
    :type test_df: :class:`~pandas.DataFrame`
    :return: duplications of any image into both sets as a new
             :code:`DataFrame`
    :rtype: :class:`~pandas.DataFrame`
    """
    pics_in_both_groups = train_df.merge(test_df, on=unique_id, how='inner')
    return pics_in_both_groups


def see_part_potential_bias(df, label, sensitive_column_list):
    """
    This function gives you a tabulated :code:`DataFrame` of sensitive columns
    e.g. gender, race, or whichever you think are relevant,
    in terms of a labels (put in the label column name).
    You may discover all your pathologically labeled sample are of one ethnic
    group, gender or other category in your :code:`DataFrame`. Remember some
    early neural nets for chest X-rays were less accurate in women and the
    fact that there were fewer X-rays of women in the datasets they built on
    did not help

    :param df: :code:`DataFrame` including sample IDs, labels, and sensitive
               columns
    :type df: :class:`~pandas.DataFrame`
    :param label: The name of the column with the labels
    :type label: str
    :param sensitive_column_list: List names sensitive columns on
                                  :code:`DataFrame`
    :type sensitive_column_list: list
    :return: tab_fight_bias2, a neatly sorted :code:`DataFrame`
    :rtype: :class:`~pandas.DataFrame`
    """

    label_and_sensitive = [label]+sensitive_column_list
    tab_fight_bias = pd.DataFrame(
        df[label_and_sensitive].value_counts()
    )
    tab_fight_bias2 = tab_fight_bias.groupby(label_and_sensitive).sum()
    tab_fight_bias2 = tab_fight_bias2.rename(columns={0: 'sums'})
    return tab_fight_bias2


def understand_df(df):
    """
    Takes a :code:`DataFrame` (if you have a :code:`DataFrame` for images)
    and prints information including length, data types, nulls and number
    of duplicated rows

    :param df: :code:`DataFrame` you are interested in getting features of.
    :type df: :class:`~pandas.DataFrame`
    :return: Prints out information on :code:`DataFrame`.
    """
    print("The DataFrame has", len(df.columns), "columns, named", df.columns)
    print("")
    print("The DataFrame has", len(df), "rows")
    print("")
    print("The types of data:\n", df.dtypes)
    print("")
    print("In terms of NaNs, the DataFrame has: \n", df.isna().sum().sum())
    print("")
    print(
        "Number of duplicated rows in the data is ",
        df.duplicated().sum(),
        ".",
    )
    print("")
    print("Numeric qualities of numeric data: \n", df.describe())


def show_duplicates(df):
    """
    Takes a :code:`DataFrame` (if you have a :code:`DataFrame` for images)
    and prints duplicated rows

    :param df: Dataframe that needs to be searched for ducplicates.
    :type df: :class:`~pandas.DataFrame`
    """
    if df.duplicated().any():
        print(
            "This DataFrame table has",
            df.duplicated().sum(),
            " duplicated rows"
        )
        print("They are: \n", df[df.duplicated()])
    else:
        print("There are no duplicated rows")
