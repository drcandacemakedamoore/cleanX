#!/usr/bin/env python

import sys
import shlex
import os
import subprocess
import site
import shutil
import platform
import posixpath
import zipfile
import email
import pkg_resources
import setuptools

from glob import glob
from distutils.dir_util import copy_tree
from io import BytesIO
from string import Template

from setuptools import setup
from setuptools.command.install import install as InstallCommand
from setuptools.command.easy_install import easy_install as EZInstallCommand
from setuptools.dist import Distribution
from setuptools.wheel import Wheel
from setuptools.command.egg_info import write_requirements
from setuptools import Command
from pkg_resources import parse_version


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


def _convert_metadata(_, zf, destination_eggdir, dist_info, egg_info):
    def get_metadata(name):
        with zf.open(posixpath.join(dist_info, name)) as fp:
            value = fp.read().decode('utf-8')
            return email.parser.Parser().parsestr(value)

    wheel_metadata = get_metadata('WHEEL')
    # Check wheel format version is supported.
    wheel_version = parse_version(wheel_metadata.get('Wheel-Version'))
    wheel_v1 = (
        parse_version('1.0') <= wheel_version < parse_version('2.0dev0')
    )
    if not wheel_v1:
        raise ValueError(
            'unsupported wheel format version: %s' % wheel_version)
    # Extract to target directory.

    # PATCH(wvxvw): This function is copied from wheel.py.
    # There are some issues. Specifically, this place will get conflicted with
    # itself because it's trying to create a directory that it already
    # created earlier...
    try:
        os.mkdir(destination_eggdir)
    except FileExistsError:
        pass
    zf.extractall(destination_eggdir)
    # Convert metadata.
    dist_info = os.path.join(destination_eggdir, dist_info)
    dist = pkg_resources.Distribution.from_location(
        destination_eggdir, dist_info,
        metadata=pkg_resources.PathMetadata(destination_eggdir, dist_info),
    )

    # Note: Evaluate and strip markers now,
    # as it's difficult to convert back from the syntax:
    # foobar; "linux" in sys_platform and extra == 'test'
    def raw_req(req):
        req.marker = None
        return str(req)
    install_requires = list(sorted(map(raw_req, dist.requires())))
    extras_require = {
        extra: sorted(
            req
            for req in map(raw_req, dist.requires((extra,)))
            if req not in install_requires
        )
        for extra in dist.extras
    }
    try:
        # This used to be outside try-except, but, as this function
        # ultimately doesn't care if whatever it does succeeds... why
        # not put this here too, as it fails anyways because one of
        # the directories had some files in them.

        # Eventually, we just need to create wheels ourselves, not
        # relying on `pip` and its friends, and life will be a
        # lot easier.
        os.rename(dist_info, egg_info)
        os.rename(
            os.path.join(egg_info, 'METADATA'),
            os.path.join(egg_info, 'PKG-INFO'),
        )
        setup_dist = setuptools.Distribution(
            attrs=dict(
                install_requires=install_requires,
                extras_require=extras_require,
            ),
        )
        write_requirements(
            setup_dist.get_command_obj('egg_info'),
            None,
            os.path.join(egg_info, 'requires.txt'),
        )
    except Exception as e:
        # The original function didn't care about exceptions here
        # either.  Turns out, all this work it did to store the
        # metatada, and then: whatever, who cares if it was stored,
        # right?
        print(e)


Wheel._convert_metadata = _convert_metadata


class TestCommand(Command):

    user_options = [
        ('pytest-args=', 'a', 'Arguments to pass into py.test'),
        ('fast', 'f', (
            'Don\'t install dependencies, test in the current environment'
            )
        ),
    ]

    def initialize_options(self):
        self.pytest_args = ''
        self.fast = False

    def finalize_options(self):
        self.test_args = []
        self.test_suite = True

    def prepare(self):
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
            ezcmd.always_copy = True
            ezcmd.finalize_options()
            ezcmd.run()
            site.main()

    def run(self):
        if not self.fast:
            self.prepare()
        self.run_tests()


class PyTest(TestCommand):

    description = 'run unit tests'

    def run_tests(self):
        import pytest

        if self.fast:
            here = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, here)
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


class Pep8(TestCommand):

    description = 'validate sources against PEP8'

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

    description = 'run apidoc to generate documentation'

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from sphinx.ext.apidoc import main

        src = os.path.join(project_dir, 'docs')
        special = (
            'index.rst',
            'cli.rst',
            'developers.rst',
            'medical-professionals.rst',
        )

        for f in glob(os.path.join(src, '*.rst')):
            for end in special:
                if f.endswith(end):
                    os.utime(f, None)
                    break
            else:
                os.unlink(f)

        sys.exit(main([
            '-o', src,
            '-f',
            os.path.join(project_dir, 'cleanX'),
            '--separate',
        ]))


class AnacondaUpload(Command):

    description = 'upload packages for Anaconda'

    user_options = [
        ('token=', 't', 'Anaconda token'),
        ('package=', 'p', 'Package to upload'),
    ]

    def initialize_options(self):
        self.token = None
        self.package = None

    def finalize_options(self):
        if (self.token is None) or (self.package is None):
            sys.stderr.write('Token and package are required\n')
            raise SystemExit(2)

    def run(self):
        env = dict(os.environ)
        env['ANACONDA_API_TOKEN'] = self.token
        upload = glob(self.package)[0]
        sys.stderr.write('Uploading: {}\n'.format(upload))
        args = ['upload', '--force', '--label', 'main', upload]
        try:
            proc = subprocess.Popen(
                ['anaconda'] + args,
                env=env,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            for elt in os.environ.get('PATH', '').split(os.pathsep):
                found = False
                sys.stderr.write('Searching for anaconda: {!r}\n'.format(elt))
                base = os.path.basename(elt)
                if base == 'condabin':
                    # My guess is conda is adding path to shell
                    # profile with backslashes.  Wouldn't be the first
                    # time they do something like this...
                    sub = os.path.join(os.path.dirname(elt), 'conda', 'bin')
                    sys.stderr.write(
                        'Anacondas hiding place: {}\n'.format(sub),
                    )
                    sys.stderr.write(
                        '    {}: {}\n'.format(elt, os.path.isdir(elt)),
                    )
                    sys.stderr.write(
                        '    {}: {}\n'.format(sub, os.path.isdir(sub)),
                    )
                    if os.path.isdir(sub):
                        elt = sub
                    executable = os.path.join(elt, 'anaconda')
                    exists = os.path.isfile(executable)
                    sys.stderr.write(
                        '    {}: {}\n'.format(executable, exists),
                    )
                    sys.stderr.write('    Possible matches:\n')
                    for g in glob(os.path.join(elt, '*anaconda*')):
                        sys.stderr.write('        {}\n'.format(g))
                elif base == 'miniconda':
                    # Another thing that might happen is that whoever
                    # configured our environment forgot to add
                    # miniconda/bin messed up the directory nam somehow
                    minibin = os.path.join(elt, 'bin')
                    if os.path.isdir(minibin):
                        sys.stderr.write(
                            'Maybe anaconda is here:{}\n'.format(minibin),
                        )
                        elt = minibin
                for p in glob(os.path.join(elt, 'anaconda')):
                    sys.stderr.write('Found anaconda: {}'.format(p))
                    anaconda = p
                    found = True
                    break
                if found:
                    proc = subprocess.Popen(
                        [anaconda] + args,
                        env=env,
                        stderr=subprocess.PIPE,
                    )
                    break
            else:
                import traceback
                traceback.print_exc()
                raise

        _, err = proc.communicate()
        if proc.returncode:
            sys.stderr.write('Upload to Anaconda failed\n')
            sys.stderr.write('Stderr:\n')
            for line in err.decode().split('\n'):
                sys.stderr.write(line)
                sys.stderr.write('\n')
            raise SystemExit(1)


class GenerateCondaYaml(Command):

    description = 'generate metadata for conda package'

    user_options = [(
        'target-python=',
        't',
        'Python version to build the package for',
    )]

    user_options = [(
        'target-conda=',
        'c',
        'Conda version to build the package for',
    )]

    def initialize_options(self):
        self.target_python = None
        self.target_conda = None

    def finalize_options(self):
        if self.target_python is None:
            maj, min, patch = sys.version.split(maxsplit=1)[0].split('.')

            self.target_python = '{}.{}'.format(maj, min)
        if self.target_conda is None:
            conda_exe = os.environ.get('CONDA_EXE', 'conda')
            self.target_conda = subprocess.check_output(
                [conda_exe, '--version'],
            ).split()[-1].decode()

    def run(self):
        tpls = glob(os.path.join(project_dir, 'conda-pkg/*.in'))
        versions = bool(os.environ.get('STRICT_PACKAGE_VERSIONS'))
        rdeps = install_requires(versions, self.target_python)
        ddeps = dev_deps(versions, self.target_python)
        run_deps = '\n    - '.join(rdeps)
        sphinx_version = ''
        for d in ddeps:
            if d.startswith('sphinx'):
                sphinx_version = d[len('sphinx'):]

        for tpl_path in tpls:
            if tpl_path.endswith('env.yml.in'):
                continue
            with open(tpl_path) as f:
                tpl = Template(f.read())

            dst_path = tpl_path[:-3]

            with open(dst_path, 'w') as f:
                f.write(tpl.substitute(
                    version=version,
                    tag=tag,
                    python_version=self.target_python,
                    conda_version=self.target_conda,
                    run_deps=run_deps,
                    sphinx_version=sphinx_version,
                ))


class FindEgg(Command):

    description = 'find Eggs built by this script'

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(glob('./dist/*.egg')[0])


class GenCondaEnv(Command):

    description = (
        'generate YAML file with requirements for conda environmnent'
    )

    user_options = [(
        'output=',
        'o',
        'File to save the environmnent description',
    )]

    def initialize_options(self):
        self.output = None

    def finalize_options(self):
        if self.output is None:
            self.output = 'cleanx-env-{}-py{}{}.yml'.format(
                platform.system().lower(),
                sys.version_info[0],
                sys.version_info[1],
            )

    def run(self):
        if os.environ.get('CONDA_DEFAULT_ENV') is None:
            raise RuntimeError(
                'This command can only run in conda environmnent',
            )
        env_tpl_path = os.path.join(
            os.path.dirname(__file__),
            'conda-pkg',
            'env.yml.in',
        )
        with open(env_tpl_path) as f:
            tpl = Template(f.read())
        conda_recs = []
        # TODO(wvxvw): Add flags to also include extras
        all_recs = (
            self.distribution.install_requires +
            self.distribution.tests_require +
            self.distribution.setup_requires
        )
        for rec in all_recs:
            result = subprocess.check_output(['conda', 'list', '-e', rec])
            for line in BytesIO(result):
                if not line.startswith(b'#'):
                    conda_recs.append(line.strip().decode())
                    break
            else:
                raise RuntimeError(
                    'Missing {}\n'.format(rec) +
                    'run "conda install -c conda-forge {}"'.format(rec),
                )
        output_contents = tpl.substitute(
            env_name=os.path.splitext(self.output)[0],
            conda_recs='\n  - '.join(conda_recs),
        )
        with open(self.output, 'w') as f:
            f.write(output_contents)


class Install(InstallCommand):

    def run(self):
        if os.environ.get('CONDA_DEFAULT_ENV'):
            # Apparently, we need to specify this. You'd think that a
            # sane package installer would leave your Python alone,
            # and yet...
            frozen = 'python={}.{}'.format(*sys.version_info[:2])
            conda = subprocess.check_output(
                ['conda', '--version'],
            ).decode().replace(' ', '=')
            packages = subprocess.check_output(['conda', 'list', '--export'])
            cmd = [
                'conda',
                'install', '-y',
                '--strict-channel-priority',
                '--override-channels',
                '-c', 'conda-forge',
                'conda-build',
                'conda-verify',
                frozen,
                conda,
            ]
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
                '--no-anaconda-upload',
                '--override-channels',
                '-c', 'conda-forge',
                os.path.join(project_dir, 'conda-pkg'),
            ]
            if subprocess.call(cmd):
                raise RuntimeError('Couldn\'t build {} package'.format(name))
            cmd = [
                'conda',
                'install',
                '--strict-channel-priority',
                '--override-channels',
                '-c', 'conda-forge',
                '--use-local',
                '--update-deps',
                '--force-reinstall',
                '-y',
                'cleanx',
                frozen,
                conda,
            ]
            if subprocess.call(cmd):
                raise RuntimeError('Couldn\'t install {} package'.format(name))
        else:
            # TODO(wvxvw): Find a way to avoid using subprocess to do
            # this
            if subprocess.call([sys.executable, __file__, 'bdist_egg']):
                raise RuntimeError('Couldn\'t build {} package'.format(name))
            egg = glob(os.path.join(project_dir, 'dist', '*.egg'))[0]
            # TODO(wvxvw): Use EZInstallCommand instead
            if subprocess.call([
                    sys.executable,
                    __file__,
                    'easy_install',
                    '--always-copy',
                    egg
            ]):
                raise RuntimeError('Couldn\'t install {} package'.format(name))
            package_dir = os.path.dirname(os.path.abspath(__file__))
            egg_info = os.path.join(package_dir, 'cleanX.egg-info')

            # Apparently, this is only set if we are in bdist_xxx
            if self.root:
                # PyPA members run setup.py install inside setup.py
                # bdist_wheel.  Because we don't do what a typical
                # install command would, and they rely on a bunch of
                # side effects of a typical install command, we need
                # to pretend that install happened in a way that they
                # expect.
                egg_info_cmd = self.distribution.get_command_obj(
                    'install_egg_info',
                )
                egg_info_cmd.ensure_finalized()
                make_pypa_happy = egg_info_cmd.target
                package_contents = os.path.join(
                    package_dir,
                    'build',
                    'lib',
                )
                copy_tree(egg_info, make_pypa_happy)
                copy_tree(package_contents, self.root)


class InstallDev(Install):

    def run(self):
        super().run()
        if os.environ.get('CONDA_DEFAULT_ENV'):
            frozen = 'python={}.{}'.format(*sys.version_info[:2])
            cmd = [
                'conda',
                'install',
                '-c', 'conda-forge',
                '-y',
                frozen,
            ] + self.distribution.extras_require['dev']
            if subprocess.call(cmd):
                raise RuntimeError('Couldn\'t install {} package'.format(name))
        else:
            extras_dist = Distribution()
            extras_dist.install_requires = self.distribution.extras_require['dev']
            ezcmd = EZInstallCommand(extras_dist)
            ezcmd.initialize_options()
            ezcmd.args = self.distribution.extras_require['dev']
            ezcmd.always_copy = True
            ezcmd.finalize_options()
            ezcmd.run()


def install_requires(versions=False, python=None):
    if os.environ.get('CONDA_DEFAULT_ENV'):
        if python is None:
            python = '{}.{}'.format(*sys.version_info[:2])
        location = os.path.join(
            os.path.dirname(__file__),
            '.conda-versions',
            python
        )
        deps = [d.strip() for d in open(location).readlines()]
        if not versions:
            deps = [d.split('=')[0] for d in deps]
        return deps
    return [
        'pandas',
        'numpy',
        'matplotlib',
        'tesserocr',
        'opencv-contrib-python',
        'pytz',
    ]


def dev_deps(versions=False, python=None):
    if os.environ.get('CONDA_DEFAULT_ENV'):
        if python is None:
            python = '{}.{}'.format(*sys.version_info[:2])
        location = os.path.join(
            os.path.dirname(__file__),
            '.conda-versions',
            python + '.dev'
        )
        deps = [d.strip() for d in open(location).readlines()]
        if not versions:
            deps = [d.split('=')[0] for d in deps]
        return deps
    return [
        'wheel',
        'sphinx',
        'pytest',
        'codestyle',
        'click',
        'pydicom',
        'SimpleITK',
    ]


# If we don't do this, we cannot run tests that involve
# multiprocessing
if __name__ == '__main__':
    versions = bool(os.environ.get('STRICT_PACKAGE_VERSIONS'))
    setup(
        name=name,
        version=version,
        description='Python library for cleaning data in large datasets of Xrays',
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
            'cleanX.dataset_processing',
            'cleanX.dicom_processing',
            'cleanX.image_work',
            'cleanX.cli',
        ],
        cmdclass={
            'test': PyTest,
            'lint': Pep8,
            'apidoc': SphinxApiDoc,
            'genconda': GenerateCondaYaml,
            'install': Install,
            # TODO(wvxvw): make sure we have wheel
            'install_dev': InstallDev,
            'find_egg': FindEgg,
            'anaconda_upload': AnacondaUpload,
            'anaconda_gen_env': GenCondaEnv,
        },
        tests_require=['pytest', 'pycodestyle'],
        command_options={
            'build_sphinx': {
                'project': ('setup.py', name),
                'version': ('setup.py', version),
                'source_dir': ('setup.py', './docs'),
                'config_dir': ('setup.py', './docs'),
            },
        },
        setup_requires=['sphinx'],
        install_requires=install_requires(versions),
        extras_require={
            'cli': ['click'],
            'pydicom': ['pydicom'],
            'simpleitk': ['SimpleITK'],
            'dev': dev_deps(versions),
        },
        zip_safe=False,
    )
