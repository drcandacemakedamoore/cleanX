<p align="center">
<img style="width: 30%; height: 30%" src="https://github.com/drcandacemakedamoore/cleanX/blob/main/test/cleanXpic.png">
</p>

# cleanX

 <a href="https://doi.org/10.5281/zenodo.4725904"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4725904.svg" alt="(DOI)"></a> <a href="https://github.com/drcandacemakedamoore/cleanX/blob/master/LICENSE"><img alt="License: GPL-3" src="https://img.shields.io/github/license/drcandacemakedamoore/cleanX"></a>[![Anaconda-Server Badge](https://anaconda.org/doctormakeda/cleanx/badges/license.svg)](https://anaconda.org/doctormakeda/cleanx)  <a href="https://joss.theoj.org/papers/47ee52ff835dcd67c1f0b4c9cb74225a"><img src="https://joss.theoj.org/papers/47ee52ff835dcd67c1f0b4c9cb74225a/status.svg"></a>[![Anaconda-Server Badge](https://anaconda.org/doctormakeda/cleanx/badges/platforms.svg)](https://anaconda.org/doctormakeda/cleanx) <a href="https://pypi.org/project/cleanX/"><img alt="PyPI" src="https://img.shields.io/pypi/v/cleanX"></a> [![Anaconda-Server Badge](https://anaconda.org/doctormakeda/cleanx/badges/version.svg)](https://anaconda.org/doctormakeda/cleanx) [![Sanity](https://github.com/drcandacemakedamoore/cleanX/actions/workflows/on-commit.yml/badge.svg)](https://github.com/drcandacemakedamoore/cleanX/actions/workflows/on-commit.yml) [![Sanity](https://github.com/drcandacemakedamoore/cleanX/actions/workflows/on-tag.yml/badge.svg)](https://github.com/drcandacemakedamoore/cleanX/actions/workflows/on-tag.yml)


CleanX <a href="https://doi.org/10.5281/zenodo.4725904"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4725904.svg" alt="(DOI)"></a> <a href="https://github.com/drcandacemakedamoore/cleanX/blob/master/LICENSE"><img alt="License: GPL-3" src="https://img.shields.io/github/license/drcandacemakedamoore/cleanX"></a>[![Anaconda-Server Badge](https://anaconda.org/doctormakeda/cleanx/badges/license.svg)](https://anaconda.org/doctormakeda/cleanx)
is an open source  python library
for exploring, cleaning and augmenting large datasets of X-rays, or certain other types of radiological images.
JPEG files can be extracted from [DICOM](https://www.dicomstandard.org/) files or used directly.
[![Anaconda-Server Badge](https://anaconda.org/doctormakeda/cleanx/badges/platforms.svg)](https://anaconda.org/doctormakeda/cleanx)


### The latest official release:

<a href="https://pypi.org/project/cleanX/"><img alt="PyPI" src="https://img.shields.io/pypi/v/cleanX"></a>
[![Anaconda-Server Badge](https://anaconda.org/doctormakeda/cleanx/badges/version.svg)](https://anaconda.org/doctormakeda/cleanx)


primary author: Candace Makeda H. Moore

other authors + contributors: Oleg Sivokon, Andrew Murphy

## Continous Integration (CI) status

[![Sanity](https://github.com/drcandacemakedamoore/cleanX/actions/workflows/on-commit.yml/badge.svg)](https://github.com/drcandacemakedamoore/cleanX/actions/workflows/on-commit.yml)
[![Sanity](https://github.com/drcandacemakedamoore/cleanX/actions/workflows/on-tag.yml/badge.svg)](https://github.com/drcandacemakedamoore/cleanX/actions/workflows/on-tag.yml)


## Requirements

- a [python](https://www.python.org/downloads/) installation (3.7, 3.8
  or 3.9)
- ability to create virtual environments (recommended, not absolutely
  necessary)
- [`tesserocr`](https://github.com/sirfz/tesserocr),
  [`matplotlib`](https://matplotlib.org/),
  [`pandas`](https://pandas.pydata.org/),
  [`pillow`](https://python-pillow.org/) and
  [`opencv`](https://opencv.org/)
- optional recommendation of [`SimpleITK`](https://simpleitk.org/) or
  [`pydicom`](https://github.com/pydicom/pydicom) for DICOM/dcm to JPG
  conversion
- Anaconda is now supported, but not technically necessary


### Supported Platforms

`cleanX` package is a pure Python package, but it has many
dependencies on native libraries.  We try to test it on as many
platforms as we can to see if dependencies can be installed there.
Below is the list of platforms that will potentially work.

Whether python.org Python or Anaconda Python are supported, it means
that version 3.7, 3.8 and 3.9 are supported.  We know for certain that
3.6 is not supported, and there will be no support in the future.


#### 32-bit Intell and ARM

We don't know if either one of these is supported.  There's a good
chance that 32-bit Intell will work.  There's a good chance that ARM
won't.

It's unlikely that the support will be added in the future.


#### AMD64 (x86)

|                             | Linux     | Win       | OSX       |
|:---------------------------:|:---------:|:---------:|:---------:|
| ![p](etc/python-logo.png)   | Supported | Unknown   | Unknown   |
| ![a](etc/anaconda-logo.png) | Supported | Supported | Supported |


#### ARM64

Seems to be unsupported at the moment on both Linux and OSX, but it's
likely that support will be added in the future.


## Documentation

Online documentation at https://drcandacemakedamoore.github.io/cleanX/

You can also build up-to-date documentation by command.

Documentation can be generated by command:

``` sh
python setup.py apidoc
python setup.py build_sphinx
```

The documentation will be generated in `./build/sphinx/html`
directory. Documentation is generated automatically as new functions
are added.

Special additional documentation for medical professionals with
limited programming ability is available on the wiki
(https://github.com/drcandacemakedamoore/cleanX/wiki/Medical-professional-documentation).

To get a high level overview of some of the functionality of the
program you can look at the Jupyter notebooks inside workflow_demo.


# Installation
- setting up a virtual environment is desirable, but not absolutely
  necessary

- activate  the environment


## Anaconda Installation

- use command for conda as below

``` sh
conda install -c doctormakeda -c conda-forge cleanx
```

You need to specify both channels because there are some cleanX
dependencies that exist in both Anaconda main channel and in
conda-forge


## pip installation

- use pip as below

``` sh
pip install cleanX
```

# Getting Started

We will imagine a very simple scenario, where we need to automate
normalization of the images we have.  We stored the images in
directory `/images/to/clean/` and they all have `jpg` extension.  We
want the cleaned images to be saved in the `cleaned` directory.

Normalization here means ensuring that the lowest pixel value (the
darkest part of the image) is as dark as possible and that the
lightest part of the image is as light as possible.

## CLI Example

The problem above doesn't require writing any new Python code.  We can
accomplish our task by calling the `cleanX` command like this:

``` sh
mkdir cleaned

python -m cleanX images run-pipeline \
    -s Acqure \
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


## Coding Example

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
imort os

from cleanX.dicom_processing import DicomReader, DirectorySource

dst = 'extracted'
os.mkdir(dst)

reader = DicomReader()
reader.rip_out_jpgs(DirectorySource('/path/to/dicoms/', 'file'), dst)
```

This will look for the files with `dcm` extension in
`/path/to/dicoms/` and try to extract images found in those files,
saving them in `extracted` directory.

# About using this library

If you use the library, please credit me and my collaborators.  You
are only free to use this library according to license. We hope that
if you use the library you will open source your entire code base, and
send us modifications.  You can get in touch with me by starting a
discussion
(https://github.com/drcandacemakedamoore/cleanX/discussions/37) if you
have a legitimate reason to use my library without open-sourcing your
code base, or following other conditions, and I can make you
specifically a different license.

We are adding new functions and classes all the time. Many unit tests
are available in the test folder. Test coverage is currently
partial. Some newly added functions allow for rapid automated data
augmentation (in ways that are realistic for radiological data). Some
other classes and functions are for cleaning datasets including ones
that:

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

[![Video](https://raw.githubusercontent.com/drcandacemakedamoore/cleanX/main/test/cleanXpic.png)](https://youtu.be/jaX5tXmiWrQ)
