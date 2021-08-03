# -*- coding: utf-8 -*-

import os
import shutil
import sqlite3
import pickle
import logging

from uuid import uuid4

from .pipeline import Pipeline


class JournalingPipeline(Pipeline):
    """
    This class extends :class:`~cleanX.image_work.pipeline.Pipeline` with
    the ability to store the progress and the state in a database.
    """

    class JournalDirectory:
        """
        :meta private:
        """

        def __init__(self, journal_dir, keep=False):
            """
            :meta private:
            """
            self.journal_dir = journal_dir
            self.keep = keep

        def __enter__(self):
            return self.journal_dir

        def __exit__(self, x, y, z):
            if not self.keep:
                logging.info(
                    'Removing jouranl from {}'.format(self.journal_dir),
                )
                shutil.rmtree(self.journal_dir)

    def __init__(
            self,
            steps=None,
            batch_size=None,
            journal=True,
            keep_journal=False,
    ):
        """
        Initializes pipeline with two additional arguments controlling
        the behavior of presistent storage.  See
        :class:`~cleanX.image_work.pipeline.Pipeline` for remaining arguments.

        :param journal: If :code:`True` is passed, the pipeline code will use a
                        preconfigured directory to store the journal.
                        Otherwise, this must be the path to the directory to
                        store the journal database.
        :type journal: Union[bool, str]
        :param keep_journal: Controls whether the journal is kept after
                             successful completion of the pipeline.
        :type keep_journal: bool
        """
        super().__init__(steps, batch_size)

        self.journal_dir = None
        self.keep_journal = keep_journal
        self.db_file = None
        self.connection = None

        self.initialize_journal(journal)

    @classmethod
    def restore(cls, journal_dir, skip=0, **overrides):
        """
        Restore the previously created journaling pipeline from the last
        executed step.

        :param journal_dir: The directory containing journal database to
                            restore from.
        :type journal_dir: Suitable for :code:`os.path.join()`
        :param skip: Skip this many steps before attempting to resume the
                     pipeline.  This is useful if you know that the step that
                     failed will fail again, but you want to execute the rest
                     of the steps in the pipeline.
        :param \\**overrides: Arguments to pass to the created pipeline
                              instance that will override those restored from
                              the journal.

        :return: Fresh :class:`JournalingPipeline` object fast-forwarded to the
                       last executed step + :code:`skip`.
        :rtype: :code:`JournalingPipeline`
        """
        result = cls(**overrides)
        result.journal_dir = journal_dir
        result.db_file = os.path.join(journal_dir, 'journal.db')
        result.connection = sqlite3.connect(result.db_file)
        result.connection.isolation_level = None
        result.cursor = result.connection.cursor()
        select = 'select id, step from history where processed = 0'
        lastrowid = 'select max(id) from history where processed = 1'
        props = 'select property, contents from pipeline'
        steps = []
        for id, step in result.cursor.execute(select).fetchall():
            steps.append(pickle.loads(step))
        result.steps = tuple(steps[skip:])
        processed = result.cursor.execute(lastrowid).fetchone()
        result.lastrowid = processed[0] if processed else 0
        result.lastrowid += 1
        for k, v in result.cursor.execute(props).fetchall():
            setattr(result, k, pickle.loads(v))
        result.counter += skip
        return result

    def initialize_journal(self, journal):
        """
        :meta private:
        """
        if journal is True:
            journal = os.path.expanduser(
                '~/cleanx/journal/{}'.format(uuid4()),
            )
        try:
            os.makedirs(journal)
        except FileExistsError:
            logging.warning(
                'Creating journal in existing directory: {}'.forma(journal),
            )
            pass
        serialized = [(pickle.dumps(s),) for s in self.steps]
        self.journal_dir = journal
        self.db_file = os.path.join(self.journal_dir, 'journal.db')

        logging.info('Creating journal database in {}'.format(self.db_file))

        self.connection = sqlite3.connect(self.db_file)
        self.connection.isolation_level = None
        self.cursor = self.connection.cursor()
        self.cursor.execute(
            '''
            create table history(
                id integer primary key,
                step blob not null,
                processed integer not null default 0
            );
            '''
        )
        self.cursor.executemany(
            'insert into history(step) values(?)',
            serialized,
        )
        self.cursor.execute(
            '''
            create table pipeline(
                property text primary key,
                contents blob
            );
            '''
        )
        self.cursor.executemany(
            'insert into pipeline(property, contents) values(?, ?)',
            self.serializable_properties(),
        )
        self.connection.commit()

        logging.info('Created journal database in {}'.format(self.db_file))

        self.lastrowid = 1

    def serializable_properties(self):
        """
        :meta private:
        """
        return (
            ('keep_journal', pickle.dumps(self.keep_journal)),
            ('counter', pickle.dumps(self.counter)),
            ('batch_size', pickle.dumps(self.batch_size)),
        )

    def workspace(self):
        """
        :meta private:
        """
        return self.JournalDirectory(self.journal_dir, self.keep_journal)

    def update_counter(self):
        """
        :meta private:
        """
        self.cursor.execute(
            'update pipeline set contents = ? where property = "counter"',
            (pickle.dumps(self.counter),),
        )

    def begin_transaction(self, step):
        """
        :meta private:
        """
        self.cursor.execute('begin')
        self.cursor.execute(
            '''
            update history
            set
              processed = 1,
              step = ?
            where id = ?
            ''',
            (pickle.dumps(step), self.lastrowid),
        )
        self.update_counter()

    def commit_transaction(self, step):
        """
        :meta private:
        """
        self.cursor.execute('commit')
        self.lastrowid += 1

    def find_previous_step(self):
        """
        :meta private:
        """
        last_exp = 'select step from history where processed = 1 order by id'
        last = self.cursor.execute(last_exp).fetchone()
        if not last:
            return None
        return pickle.loads(last[0])
