==================================
Medical Professional Documentation
==================================

This is the present documentation home **for medical
professionals** who do not spend most of their time coding.  From here
you can link to our `Jupyter notebook`_ specifically for medical
professionals (one of a series of notebooks), `a brief introductory
video`_ or a special `video demo`_ which shows some basic
functionality for people who bring expertise outside of coding.

Introduction
============

`cleanX` is a package meant to ease the preparation of data for
machine learning or other algorithms created with some types of
radiology imaging data.  The package was developed specifically for
chest X-rays; however, some of Cleanx's functionality can be used with
other types of data.  For example, the data_processing module can be
used with tabular data for radiology or other areas.

Programmers have a saying, "garbage in, garbage out". That is to say,
if you feed a machine learning algorithm incorrect and/or mislabeled data, 
you should expect incorrect outputs (the algorithm will not work as desired).  
Therefore the work of radiologists and allied professionals such as radiographers
is critical  in creating machine learning algorithms for the field.

At present, state of the art datasets often include over 100,000
images.  The task of reading this many images can be aided by using a
tool such as `cleanX`.  It will automatically reveal some
questionable images, whilst revealing images that are problematic for
computer vision programs e.g. inverted or upside-down images that a
human reader can read easily.

`cleanX` can also be used by anyone (even programmers) to eliminate images
that obviously should not be in a dataset. There are public
datasets of chest X-rays that contain occasional coronal CT slices, or
other types of images.  Such images should ideally be thrown out of a
dataset before reading commences.  `cleanX` can help anyone with good
vision accomplish this task.

`cleanX` is also desinged to extract and clean the metadata on 
images. Imaging professionals that are not proficient
in coding, but still want to make a substantial contribution to a
machine learning project, can add value via data curation, one of the most
important tasks in the project.  We have a video, `our video for
non-coders demo`_, that shows how `cleanX` can help with some of these
tasks.

Workflow
========

`cleanX` workflow is made of three modules:

- Dicom processing
- Data processing
- Image work

To see some functionality of `cleanX`, it is suggested to run **both**
notebooks in the `workflow demo`_ folder.  If you do not have, or know
how to comfortably operate Jupyter notebooks, you can check out our
videos `video demo of several classes and functions`_ and `video for
non-coders demo`_

.. image:: https://raw.githubusercontent.com/drcandacemakedamoore/cleanX/main/test/cleanXpic.png

What makes `cleanX` different and/or important?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Much research about biomedical imaging relies on a combination of
proprietary technology and in-house code that is not shared across
institutions.  This reality stymies cooperation, and locks out individuals
and institutions without great amounts of capital.  It is often poorer
institutions e.g. hospitals in nations with very few radiologists,
that would benefit the most from AI.  Unfortunately, research
algorithms made on imaging data from brand new, expensive,
top-of-the-line machines may or may not easily translate if applied to
images made on lower-quality machines for purely technical
reasons.

Additionally, no matter how similar our machines, to even investigate
how well an algorithm made on imaging data from a place like Utah or
Holland translates the chest X-rays made in Malawi (or any very
different setting from the one the data comes from), we would need a
dataset of images from Malawi (or whichever country we are trying out
the algorithm in) or we actually risk worsening health inequity by
simply applying the exact same algorithm everywhere (both present
research, and practical field experience indicate this is a likely
outcome).  `cleanX` is an open-source solution designed to be used
freely everywhere.  Some algorithms in machine learning are
essentially solved problems, but what is not solved is how a broader
group of people and institutions can get or create appropriate
datasets to power these algorithms.  `cleanX` is for everyone, free
and open source.

Special note for traditionalists: You may wonder why we chose to
do most of our output work with JPG and CSV files when most imaging professionals
are more familiar with DICOMs and Excel files i.e. xlsx.  Among the
reasons are the following:

        
- by using JPG files instead of whole DICOMs you can avoid touching a
  lot of metadata which can be a problem in terms of anonymization

- big data is not handled all that well by Microsoft Excel, for
  example, if you have over 1,048,576 rows (`according to Microsoft`_)
  you have a problem.  For a reference point, if you took certain
  chest X-ray datasets on Kaggle and added 3 augmentations to each
  image (with a new row in an Excel file for each), you would have a
  run out of rows...

- by using CSV we are being inclusive of people who do not have access
  to Microsoft products and still providing a file that can be opened
  in Excel

.. _Jupyter notebook: https://github.com/drcandacemakedamoore/cleanX/blob/main/workflow_demo/for_medical_people.ipynb
.. _a brief introductory video: https://youtu.be/FRqb932u5bc
.. _video demo: https://youtu.be/EYnX4NgQqYw
.. _our video for non-coders demo: https://youtu.be/EYnX4NgQqYw
.. _workflow demo: https://github.com/drcandacemakedamoore/cleanX/blob/main/workflow_demo
.. _video demo of several classes and functions: https://youtu.be/jaX5tXmiWrQ
.. _video for non-coders demo: https://youtu.be/EYnX4NgQqYw
.. _according to Microsoft: https://support.microsoft.com/en-us/office/excel-specifications-and-limits-1672b34d-7043-467e-8e27-269d656771c3
