=================
Developer's Guide
=================

Sending Your Work
=================

We accept pull requests made through GitHub.  Alternatively, you can
send patch files in an email (please find developers' emails on their
profile pages).  As is usual, we request that the changes be rebased
on the branch they are to be integrated into.  We also request that

.. code-block:: bash

   python ./setup.py test
   python ./setup.py lint

succeed with your changes applied.  We'll try our best to attribute
your work to you, however, you need to release your work under
compatible license for us to be able to use it.

.. warning::

   We don't use `git-merge` command, and if your submission has merge
   commits, we'll have to remove them.  This means that in such case
   commit hashes will be different from those in your original
   submission.

Setting Up Development Environment
==================================

.. warning::

   You may find this setup unusual.  Please read the `Rationale`_
   section to satisfy your curiosity.

There isn't *a* development environment.  You may need to create an
environment for every combination of supported versions of operating
system, Python distribution and Python version.  If you believe that
your code is portable accross versions and platforms, then, of course,
you don't need all the possible combinations.  Our CI should be able
to pick up the most obvious incompatibilities between those.

The Way We Do It
^^^^^^^^^^^^^^^^

Suppose you want to check your changes against Python 3.7 from
python.org on Linux, then you would check out our sources using `git`
from our `GitHub repo`_, and:

.. code-block:: bash

   python3.7 -m venv .venv37
   . ./.venv3.7/bin/activate
   python ./setup.py install_dev

What this will do is quite a bit different from what
`setuptools.install` command would normally do: It will create an Egg
with `cleanX` sources, install it, then it will look for development
dependencies, and install them too.  This is likely to take a lot longer
than what a typical project setup would do.  Please, be patient.

Similarly, if you want to develop using Anaconda Python, you would:

.. code-block:: bash

   conda create -n cleanx-37
   conda activate -n cleanx-37
   conda run -n cleanx-37 python ./setup.py install_dev

Similar to above, this will create a virtual environment, build
`conda` package, install it and then add development dependencies to
what was installed.  This will likely take a *very* long time.
Please, be patient.

The Traditional Way
^^^^^^^^^^^^^^^^^^^

Regradless of the downsides of this approach, we try to support more
common ways to work with Python projects.  It's a common practice to
"install" a project during development by either using `pip install
--editable` command, or by using `conda` environment files.

We provide limited support for this approach.  For instance, if you
want to work on the project using `pip`, you could try:

.. code-block:: bash

   python3.7 -m venv .venv37
   . ./.venv/bin/activate
   pip install -e .[dev]

If you encounter problems with this approach related to runnig tests
or generating documentation, please let us know.  Unfortunately,
supporting this worklfow is a low-priority task for us, as internally
we don't rely on this to work.

Similarly, if you want to develop using `conda` environment files, we
provide limited support for that.

The environment files are generated using:

.. code-block:: bash

   conda run -n cleanx-build python setup.py anaconda_gen_env

However, this means you already have to have all dependencies
installed.  We run this script in CI and archive the environment files
it produces.  To find such file suitable for your platform you will
need to navigate to `GitHub Actions dashboard`_, and look in the
artifacts section of the selected job for artifacts named
`cleanx-env-$python_version-$os`.  Select the one that applies to your
Python version and operating system, download and unzip it in the
directory where you checked out the `cleanX` sources.  Afterwards,
you can:

.. code-block:: bash

   conda env create -f ./cleanx-env-*.yml

You don't have to do this for every supported platform to be able to
work on the project in some capacity, however, this is the way to
reproduce problems that either happen in our CI or are reported on
platforms different from the one you develop on.

Rationale
^^^^^^^^^

There are several problems with traditional way Python programmers are
taught to organize their development environment.  The way a typical
Python project is developed, it is designed to support a single
version of Python, rarely multiple Python distributions or operating
systems.  This, of course, makes development process easy, and it is
completely justifed if the goal of the project is to be deployed in a
highly controlled environment, such as a rented or private server, or
a system maintained by the IT department of the organziation for which
the product is designed.

In the situation above, it's typical to rely on `requirements.txt` or
`environment.yml` to manage development dependencies.  These formats,
however, are inappropriate if the project needs to support multiple
platforms with dependencies that vary based on the platform.  We
didn't find, nor did we invent a better format for describing
cross-platform dependencies.  We don't have the capacity for such
things, and the task itself would, probably be at least as difficult
as working on the source code of the project.

Since producing distributable packages is an essential goal for the
project, we decided to base our development setup on our ability to
produce those.  It is already cross-platform and supports at least the
platform we intend to support.

Another traditional aspect of Python development process is the use of
`pip`.  We found that in all essential aspects of our project's
lifecycle `pip` defers to `setuptools`.  `setuptools` are also a
convenient middle ground between `conda` and `pip`.  Therefore we try
to avoid the use of `pip` everywhere in our development process.
`setuptools` is complex and unreliable as it is, adding another layer
of unreliability, especially in the situation where multiple platforms
are to be supported seems like a bad choice.

We may, eventually, support development workflows that incorporate
`pip`, but this is a low priority task aimed at developers outside of
the core of our project.

Testing
=======

The project must be installed in order to run the tests.  It doesn't
have to be instaleld with development dependencies though.  Once that
is done, you may run:

.. code-block:: bash

  python ./setup.py test

Under the hood, this runs `pytest` command on the tests found in
`tests` directory.  The `test` command has an option to pass `pytest`
arguments down to `pytest`:

.. code-block:: bash

   python ./setup.py --pytest-args='-s -k dicom'

For example, if you want to only run tests related to DICOM integration.

Style Guide for Python Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python ./setup.py lint

This is vanilla PEP8 linter, just do what PEP8 tells you to do and you
should be fine.

Continuous Integration
^^^^^^^^^^^^^^^^^^^^^^

This project has extensive CI setup that uses GitHub Actions platform.
There's an experimental branch with Jenkins setup, but it's far from
being ready yet, and we currently don't work on it.

.. warning::

   Please note that CI has automated release build and publishing: if
   a tag is pushed with a name that starts with `v`, (eg. `v1.2.3`),
   CI will interpret this as asking to create a release for that
   version.

.. _GitHub repo: https://github.com/drcandacemakedamoore/cleanX.git
.. _GitHub Actions dashboard: https://github.com/drcandacemakedamoore/cleanX/actions/runs
