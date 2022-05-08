<p align="center">
    <img style="width: 30%; height: 30%" src="cleanx-logo.svg">
</p>

# CleanX

[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4725904.svg)](https://doi.org/10.5281/zenodo.4725904)
[![License GPL-3](https://img.shields.io/github/license/drcandacemakedamoore/cleanX.svg)](https://raw.githubusercontent.com/drcandacemakedamoore/cleanX/main/LICENSE)
[![Anaconda-Server Badge](https://anaconda.org/doctormakeda/cleanx/badges/license.svg)](https://anaconda.org/doctormakeda/cleanx)
[![JOSS Publication](https://joss.theoj.org/papers/47ee52ff835dcd67c1f0b4c9cb74225a/status.svg)](https://joss.theoj.org/papers/47ee52ff835dcd67c1f0b4c9cb74225a)
[![Anaconda-Server Badge](https://anaconda.org/doctormakeda/cleanx/badges/platforms.svg)](https://anaconda.org/doctormakeda/cleanx)
[![PYPI Version](https://img.shields.io/pypi/v/cleanX.svg)](https://pypi.org/project/cleanX)
[![Anaconda-Server Badge](https://anaconda.org/doctormakeda/cleanx/badges/version.svg)](https://anaconda.org/doctormakeda/cleanx)
[![Sanity](https://github.com/drcandacemakedamoore/cleanX/actions/workflows/on-commit.yml/badge.svg)](https://github.com/drcandacemakedamoore/cleanX/actions/workflows/on-commit.yml)
[![Sanity](https://github.com/drcandacemakedamoore/cleanX/actions/workflows/on-tag.yml/badge.svg)](https://github.com/drcandacemakedamoore/cleanX/actions/workflows/on-tag.yml)
[![Documentation](https://img.shields.io/static/v1?label=docs&message=release&color=green)](https://drcandacemakedamoore.github.io/cleanX)
[![GitHub issues](https://img.shields.io/github/issues/drcandacemakedamoore/cleanX)](https://github.com/drcandacemakedamoore/cleanX/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/drcandacemakedamoore/cleanX)](https://github.com/drcandacemakedamoore/cleanX/discussions)



CleanX is an open-source python library for exploring, cleaning and augmenting
large datasets of X-rays, or certain other types of radiological images.  The images
can be extracted from [DICOM](https://www.dicomstandard.org/) files or used 
directly. The primary authors are Candace Makeda H. Moore, Oleg Sivokon, and Andrew Murphy.


## Documentation

Online documentation is at https://drcandacemakedamoore.github.io/cleanX/.
You can also build up-to-date documentation, which will be generated in ./build/sphinx/html directory, by command as follows:


    python setup.py apidoc
    python setup.py build_sphinx


Special additional documentation for medical professionals with limited programming ability is available [here](https://github.com/drcandacemakedamoore/cleanX/wiki/Legacy-medical-professional-documentation) on the project wiki. To get a high level overview of some of the functionality of the program you can look at the Jupyter notebooks inside [workflow_demo folder](https://github.com/drcandacemakedamoore/cleanX/tree/main/workflow_demo).

## Requirements

- [Python](https://www.python.org/downloads/) 3.7, 3.8, 3.9.  Python
  3.10 has not been tested yet.
- Ability to create virtual environments (recommended, not absolutely
  necessary)
- [`tesserocr`](https://github.com/sirfz/tesserocr),
  [`matplotlib`](https://matplotlib.org/),
  [`pandas`](https://pandas.pydata.org/), and
  [`opencv`](https://opencv.org/)
- Optional recommendation of [`SimpleITK`](https://simpleitk.org/) or
  [`pydicom`](https://github.com/pydicom/pydicom) for DICOM/dcm to JPG
  conversion
- Anaconda is now supported, but not technically necessary

### Supported Platforms

CleanX is a pure Python package, but it has many dependencies on
native libraries.  We try to test it on as many platforms as we can to
see if dependencies can be installed there.  Below is the list of
platforms that will potentially work. Please note that where
python.org Python or Anaconda Python stated as supported, it means
that versions 3.7, 3.8 and 3.9 (but not 3.10) are supported.

#### AMD64 (x86)

|                             | Linux     | Win       | OSX       |
|:---------------------------:|:---------:|:---------:|:---------:|
| ![p](etc/python-logo.png)   | Supported | Unknown   | Unknown   |
| ![a](etc/anaconda-logo.png) | Supported | Supported | Supported |


#### ARM64

Unsupported at the moment on both Linux and OSX, but it's likely that
support will be added in the future.

#### 32-bit Intell and ARM

We don't know if either one of these is supported. There's a good
chance that 32-bit Intell will work.  There's a good chance that ARM
won't. It's unlikely that the support for ARM will be added in the
future.

## Installation
- setting up a virtual environment is desirable, but not absolutely
  necessary

- activate  the environment


### Anaconda Installation

- use command for conda as below

``` sh
conda install -c doctormakeda -c conda-forge cleanx
```

You need to specify both channels because there are some cleanX
dependencies that exist in both Anaconda main channel and in
conda-forge


### pip installation

- use pip as below

``` sh
pip install cleanX
```

You can install some optional dependencies.

To have CLI functionality:

``` sh
pip install cleanX[cli]
```

To have PyDicom installed and used to process DICOM files:

``` sh
pip install cleanX[pydicom]
```

Similarly, if you want SimpleITK used to process DICOM files:

``` sh
pip install cleanX[simpleitk]
```

The `tesserocr` package deserves a special mention.  It is not
possible to install `tesseract` library from PyPI server.  The
`tesserocr` is simply a binding to the library.  You will need to
install the library yourself.  For example, on Debian flavor Linux,
this might work:

``` sh
sudo apt-get install libleptonica-dev \
    tesseract-ocr-all \
    libtesseract-dev
```

We've heard that

``` sh
brew install tesseract
```

works on Mac.


## Getting Started

We will imagine a very simple scenario, where we need to automate
normalization of the images we have.  We stored the images in
directory `/images/to/clean/` and they all have a `jpg` extension.  We
want the cleaned images to be saved in the `cleaned` directory.

Normalization here means ensuring that the lowest pixel value (the
darkest part of the image) is as dark as possible and that the
lightest part of the image is as light as possible.

### Docker

Docker images are available from Dockerhub.  You should be able to run
them using:

``` sh
docker run --rm -v "$(pwd)":/cleanx drcandacemakedamoore/cleanx --help
```

The `/cleanx` directory in the image is intentionaly left to be used
as a mount point.  The image, by default, runs as root, but doesn't
require root privileged.  In the future, it's possible that the image
will come with a non-root user and will default to running as a
non-root user.

Additionally, there is a Docker image with several examples in a form
of Jupyter notebooks.  To run this image:

``` sh
docker run --rm -ti -p 8888:8888 --network=host \
    drcandacemakedamoore/cleanx-jupyter-examples
```

This will generate output similar to:

``` sh
[I 12:59:52.383 NotebookApp] Writing notebook server cookie secret  \
to /home/jupyter/.local/share/jupyter/runtime/notebook_cookie_secret
[I 12:59:52.704 NotebookApp] Serving notebooks from local directory:\
/home/jupyter
[I 12:59:52.704 NotebookApp] Jupyter Notebook 6.4.11 is running at:
[I 12:59:52.705 NotebookApp] http://localhost:8888/?token=...
[I 12:59:52.705 NotebookApp]  or http://127.0.0.1:8888/?token=...
[I 12:59:52.705 NotebookApp] Use Control-C to stop this server and \
shut down all kernels (twice to skip confirmation).
[W 12:59:52.709 NotebookApp] No web browser found: could not locate\
runnable browser.
[C 12:59:52.709 NotebookApp] 

    To access the notebook, open this file in a browser:
        file:///.../nbserver-1-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=...
     or http://127.0.0.1:8888/?token=...
```

Copy the text that starts with `http://127.0.0.1:8888` (including the
token) and paste it into your browser's address bar.  The demos should
be fully operational (you may interact with them, re-evaluate them,
change available parameters etc.)

### CLI Example

The problem above doesn't require writing any new Python code.  We can
accomplish our task by calling the `cleanX` command like this:

``` sh
mkdir cleaned

python -m cleanX images run-pipeline \
    -s Acquire \
    -s Normalize \
    -s "Save(target='cleaned')" \
    -j \
    -r "/images/to/clean/*.jpg"
```

Let's look at the command's options and arguments:

* `python -m cleanX` is the Python's command-line option for loading
  the `cleanX` package.  All command-line arguments that follow this
  part are interpreted by `cleanX`.
* `images` sub-command is used for processing of images.
* `run-pipeline` sub-command is used to start a `Pipeline` to process
  the images.
* `-s` (repeatable) option specifies `Pipeline` `Step`.  Steps map to
  their class names as found in the `cleanX.image_work.steps` module.
  If the `__init__` function of a step doesn't take any arguments, only
  the class name is necessary.  If, however, it takes arguments, they
  must be given using Python's literals, using Python's named arguments
  syntax.
* `-j` option instructs to create *journaling* pipeline.  Journaling
  pipelines can be restarted from the point where they failed, or had
  been interrupted.
* `-r` allows to specify source for the pipeline.  While, normally, we
  will want to start with `Acquire` step, if the pipeline was
  interrupted, we need to tell it where to look for the initial
  sources.
  
Once the command finishes, we should see the `cleaned` directory filled
with images with the same names they had in the source directory.


Let's consider another simple task: batch-extraction of images from
DICOM files:

---

``` sh
mkdir extracted

python -m cleanX dicom extract \
    -i dir /path/to/dicoms/
    -o extracted
```

This calls `cleanX` CLI in the way similar to the example above, however,
it calls the `dicom` sub-command with `extract-images` subcommand.

* `-i` tells `cleanX` to look for directory named `/path/to/dicoms`
* `-o` tells `cleanX` to save extracted JPGs in `extracted` directory.

If you have any problems with this check
[#40](https://github.com/drcandacemakedamoore/cleanX/issues/40) and add
issues or discussions.


### Coding Example

Below is the equivalent code in Python:

``` python
import os

from cleanX.image_work import (
    Acquire,
    Save,
    GlobSource,
    Normalize,
    create_pipeline,
)

dst = 'cleaned'

# This is just an illustration, this code isn't sufficient in most
# cases to remove a directory.  It's up to you to come up with a
# reasonable code to remove this directory if it already exist.
try:
    os.rmdir(dst)
except FileNotFoundError:
    pass

os.mkdir(dst)

src = GlobSource('/images/to/clean/*.jpg')
p = create_pipeline(
    steps=(
        Acquire(),
        Normalize(),
        Save(dst),
    ),
    journal=True,
)

p.process(src)
```

Let's look at what's going on here.  As before, we've created a
pipeline using `create_pipeline` with three steps: `Acquire`,
`Normalize` and `Save`.  There are several kinds of sources available
for pipelines.  We'll use the `GlobSource` to match our CLI example.
We'll specify `journal=True` to match the `-j` flag in our CLI
example.

---

And for the DICOM extraction we might use similar code:

``` python
import os

from cleanX.dicom_processing import DicomReader, DirectorySource

dst = 'extracted'
os.mkdir(dst)

reader = DicomReader()
reader.rip_out_jpgs(DirectorySource('/path/to/dicoms/', 'file'), dst)
```

This will look for the files with `dcm` extension in
`/path/to/dicoms/` and try to extract images found in those files,
saving them in `extracted` directory.

## Developer's Guide

Please refer to [Developer's Guide](https://drcandacemakedamoore.github.io/cleanX/developers.html)
for more detailed explanation.

### Developing Using Anaconda's Python

Use Git to check out the project's source, then, in the source
directory run:

```sh
conda create -n cleanx
conda activate cleanx
python ./setup.py genconda
python ./setup.py install_dev
```

Note that the last command may result in errors related to `conda-build`
being unable to delete Microsoft's C++ runtime DLL.  This is typical
behavior of `conda-build` as can be seen here: 
https://github.com/conda/conda/issues/7682

The workaround is to add:

```sh
conda config --set always_copy true
```

And re-run the last step (this will make virtual environment created with
`conda` noticeably bigger).

You may have to do this for Python 3.7, Python 3.8 and Python 3.9 if
you need to check that your changes will work in all supported
versions.

The `genconda` command needs to run only once per checkout and version
of Python used.  At the moment, it's not possible to have multiple
`conda` package configurations generated at the same time.  So, if you
are switching Python versions, you will need to rerun this command.

Also note that the build will package only the changes committed to
the Git repository.  This means that if you are building with
uncommitted changes, they will not make it into the built package.
The decision to do this was motivated by the presence of symbolic
links in the working directory, which makes it impossible to build
without superuser permissions on MS Windows.  It is possible that in
the future we will add a flag to `setup.py install` to allow "dirty"
builds.

To run unit test and linter you may use:

```sh
python setup.py lint
```

and

```sh
python setup.py test
```

respectively.  Note that by default, these commands will try to install
`cleanX` and its dependencies before doing any work.  This may take a
very long time, especially on MS Windows.  There is a way to skip the
installation part by running:

```sh
python setup.py lint --fast
```

and

```sh
python setup.py test --fast
```


### Developing Using python.org's Python

Use Git to check out the project's source, then in the source
directory run:

```sh
python -m venv .venv
. ./.venv/bin/activate
python ./setup.py install_dev
```

Similar to `conda` based setup, you may have to use Python versions
3.7, 3.8 and 3.9 to create three different environments to recreate
our CI process.

### Build up-to-date documentation

Documentation can be [generated by command](#documentation). The documentation 
will be generated in a `./build/sphinx/html` directory. Documentation is generated
automatically as new functions are added.


## About using this library

If you use the library, please cite the package.
CleanX is free ONLY when used according to license.

You can get in touch with me by starting a [discussion](https://github.com/drcandacemakedamoore/cleanX/discussions/37) if you
have a legitimate reason to use my library without open-sourcing your
code base, or following other conditions, and I can make you,
specifically, a different license.

We are adding new functions and classes all the time. Many unit tests
are available in the test folder. Test coverage is currently
partial. Some newly added functions allow for rapid automated data
augmentation (in ways that are realistic for radiological data) and some
preliminary image quality checks. Some other classes and functions are for
cleaning datasets including ones that:

* Get image and metadata out of dcm (DICOM) files into jpeg and csv
  files
* Process datasets from csv or json or other formats to generate
  reports
* Run on dataframes to make sure there is no image leakage
* Run on a dataframe to look for demographic or other biases in
  patients
* Crop off excessive black frames (run this on single images) one at a
  time
* Run on a list to make a prototype tiny Xray others can be compared
  to
* Run on image files which are inside a folder to check if they are
  "clean"
* Take a dataframe with image names and return plotted(visualized)
  images
* Run to make a dataframe of pics in a folder (assuming they all have
  the same 'label'/diagnosis)
* Normalize images in terms of pixel values (multiple methods)

All important functions are documented in the online documentation for
programmers. You can also check out one of our videos by clicking the
linked picture below:

### cleanX: video demonstration

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/jaX5tXmiWrQ/0.jpg)](https://www.youtube.com/watch?v=jaX5tXmiWrQ)
