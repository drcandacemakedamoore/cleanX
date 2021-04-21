import sys
import shlex
import os

from glob import glob

from setuptools import setup
from setuptools.command.test import test as TestCommand
from setuptools import Command


project_dir = os.path.dirname(os.path.realpath(__file__))
# This will exclude the project directory from sys.path so that Sphinx
# doesn't get confused about where to load the sources.
# _Note_ you _must_ install the project
# before you generate documentation, otherwise it will not work.
sys.path = [x for x in sys.path if not x == project_dir]


class PyTest(TestCommand):

    user_options = [('pytest-args=', 'a', "Arguments to pass into py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


class Pep8(TestCommand):

    def run_tests(self):
        from pycodestyle import StyleGuide

        package_dir = os.path.dirname(os.path.abspath(__file__))
        sources = [os.path.join(package_dir, 'cleanX.py')]
        style_guide = StyleGuide(paths=sources)
        options = style_guide.options

        report = style_guide.check_files()
        report.print_statistics()

        if report.total_errors:
            if options.count:
                sys.stderr.write(str(report.total_errors) + '\n')
            sys.exit(1)


class SphinxApiDoc(Command):

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from sphinx.ext.apidoc import main
        sys.exit(main([
            '-o', os.path.join(project_dir, 'source'),
            '-f',
            project_dir,
            '.*',
            'setup.py',
        ]))


with open('README.md', 'r') as f:
    readme = f.read()


name = 'cleanX'
version = '0.0.4'

setup(
    name=name,
    version=version,
    description="Python library for cleaning data in large datasets of Xrays",
    long_description=readme,
    long_description_content_type='text/markdown',
    author='doctormakeda@gmail.com',
    author_email='doctormakeda@gmail.com',
    maintainer='doctormakeda@gmail.com',
    maintainer_email= 'doctormakeda@gmail.com',
    url="https://github.com/drcandacemakedamoore/cleanX",
    license="MIT",
    py_modules=["cleanX"],
    cmdclass={
        'test': PyTest,
        'lint': Pep8,
        'apidoc': SphinxApiDoc,
    },
    tests_require=['pytest', 'pycodestyle'],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'source_dir': ('setup.py', './source'),
            'config_dir': ('setup.py', './source'),
        },
    },
    setup_requires = ['sphinx'],
    install_requires=[
        "pandas",
        'numpy',
        "matplotlib",
        "pillow",
        "tesserocr",
        "opencv-python",
    ],
)
