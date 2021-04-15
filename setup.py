import sys
import shlex

from setuptools import setup
from setuptools.command.test import test as TestCommand


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



with open('README.md', 'r') as f:
    readme = f.read()


setup(
    name="cleanX",
    version='0.0.2',
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
    cmdclass={'test': PyTest},
    tests_require=['pytest'],
    install_requires=[
        "pandas",
        'numpy',
        "matplotlib",
        "pillow",
        "tesserocr",
        "cv2",
    ],
)