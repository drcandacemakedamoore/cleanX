#!/usr/bin/env python

import sys
import shlex
import os
import subprocess
import site
import shutil

from glob import glob
from distutils.dir_util import copy_tree

from setuptools import setup
from setuptools.command.install import install as InstallCommand
from setuptools.command.easy_install import easy_install as EZInstallCommand
from setuptools.dist import Distribution
from setuptools import Command


project_dir = os.path.dirname(os.path.realpath(__file__))
# This will exclude the project directory from sys.path so that Sphinx
# doesn't get confused about where to load the sources.
# _Note_ you _must_ install the project
# before you generate documentation, otherwise it will not work.
sys.path = [x for x in sys.path if not x == project_dir]


with open('README.md', 'r') as f:
    readme = f.read()

name = 'cleanX'
try:
    tag = subprocess.check_output([
        'git',
        '--no-pager',
        'describe',
        '--abbrev=0',
        '--tags',
    ]).strip().decode()
except subprocess.CalledProcessError as e:
    print(e.output)
    tag = 'v0.0.0'

version = tag[1:]


class TestCommand(Command):

    user_options = [('pytest-args=', 'a', "Arguments to pass into py.test")]

    def initialize_options(self):
        self.pytest_args = ''

    def finalize_options(self):
        self.test_args = []
        self.test_suite = True

    def run(self):
        recs = self.distribution.tests_require

        if os.environ.get('CONDA_DEFAULT_ENV'):
            if recs:
                result = subprocess.call([
                    'conda',
                    'install',
                    '-y',
                    '-c', 'conda-forge',
                    ] + recs
                )
                if result:
                    raise RuntimeError('Cannot install test requirements')
        else:
            test_dist = Distribution()
            test_dist.install_requires = recs
            ezcmd = EZInstallCommand(test_dist)
            ezcmd.initialize_options()
            ezcmd.args = recs
            ezcmd.finalize_options()
            ezcmd.run()
            site.main()

        self.run_tests()


class PyTest(TestCommand):

    def run_tests(self):
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


class Pep8(TestCommand):

    def run_tests(self):
        from pycodestyle import StyleGuide

        package_dir = os.path.dirname(os.path.abspath(__file__))
        sources = glob(
            os.path.join(package_dir, 'cleanX', '**/*.py'),
            recursive=True,
        )
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
            os.path.join(project_dir, 'cleanX'),
        ]))


class GenerateCondaYaml(Command):

    user_options = [(
        'target-python=',
        't',
        'Python version to build the package for',
    )]

    def initialize_options(self):
        self.target_python = None

    def finalize_options(self):
        if self.target_python is None:
            maj, min, patch = sys.version.split(maxsplit=1)[0].split('.')
            
            self.target_python = '{}.{}'.format(maj, min)

    def run(self):
        from string import Template

        tpls = glob(os.path.join(project_dir, 'conda-pkg/*.in'))
        try:
            egg = glob(os.path.join(project_dir, 'dist/*.egg'))[0]
        except IndexError as e:
            raise RuntimeError(
                'You need to run `setup.py bdist_egg\' first',
            ) from e

        for tpl_path in tpls:
            with open(tpl_path) as f:
                tpl = Template(f.read())

            dst_path = tpl_path[:-3]

            with open(dst_path, 'w') as f:
                f.write(tpl.substitute(
                    version=version,
                    tag=tag,
                    python_version=self.target_python,
                    egg=egg,
                ))


class FindEgg(Command):

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(glob('./dist/*.egg')[0])


class Install(InstallCommand):

    def run(self):
        if os.environ.get('CONDA_DEFAULT_ENV'):
            packages = subprocess.check_output(['conda', 'list', '--export'])
            cmd = ['conda', 'install', '-y', 'conda-build', 'conda-verify']
            for line in packages.split(b'\n'):
                if line.startswith(b'conda-build='):
                    break
            else:
                if subprocess.call(cmd):
                    raise RuntimeError('Cannot install conda-build')
            shutil.rmtree(
                os.path.join(project_dir, 'dist'),
                ignore_errors=True,
            )
            shutil.rmtree(
                os.path.join(project_dir, 'build'),
                ignore_errors=True,
            )

            cmd = [
                'conda',
                'build',
                '-c', 'conda-forge',
                os.path.join(project_dir, 'conda-pkg'),
            ]
            if subprocess.call(cmd):
                raise RuntimeError('Couldn\'t build {} package'.format(name))
            cmd = [
                'conda',
                'install',
                '-c', 'conda-forge',
                '--use-local',
                '--update-deps',
                '--force-reinstall',
                '-y',
                'cleanx',
            ]
            if subprocess.call(cmd):
                raise RuntimeError('Couldn\'t install {} package'.format(name))
        else:
            if subprocess.call([sys.executable, __file__, 'bdist_egg']):
                raise RuntimeError('Couldn\'t build {} package'.format(name))
            egg = glob(os.path.join(project_dir, 'dist', '*.egg'))[0]
            if subprocess.call([sys.executable, __file__, 'easy_install', egg]):
                raise RuntimeError('Couldn\'t install {} package'.format(name))
            package_dir = os.path.dirname(os.path.abspath(__file__))
            egg_info = os.path.join(package_dir, 'cleanX.egg-info')
            make_pypa_happy = os.path.join(
                package_dir,
                self.root,
                'cleanX-{}.egg-info'.format(version),
            )
            copy_tree(egg_info, make_pypa_happy)


def install_requires():
    if os.environ.get('CONDA_DEFAULT_ENV'):
        return [
            'pandas',
            'numpy',
            'matplotlib',
            'pillow',
            'tesserocr',
            'opencv',
        ]
    return [
        'pandas',
        'numpy',
        'matplotlib',
        'pillow',
        'tesserocr',
        'opencv-python',
    ]


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
    url='https://github.com/drcandacemakedamoore/cleanX',
    license='MIT',
    packages=[
        'cleanX',
        'cleanX.csv_processing',
        'cleanX.dicom_processing',
        'cleanX.image_work',
    ],
    cmdclass={
        'test': PyTest,
        'lint': Pep8,
        'apidoc': SphinxApiDoc,
        'genconda': GenerateCondaYaml,
        'install': Install,
        'find_egg': FindEgg,
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
    setup_requires=['sphinx'],
    install_requires=install_requires(),
    zip_safe=False,
)
