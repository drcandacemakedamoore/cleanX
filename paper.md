---
title: 'CleanX: A Python library for data cleaning of large sets of X-rays'
tags:
  - Python
  - radiology
  - imaging
  - medical imaging

authors:
  - name: Candace Makeda Moore, MD
    orcid: 0000-0003-1672-7565
    affiliation: 1,3

  - name: Andrew Murphy, MMIS BMedImagingSc RT(R)
    orcid: 0000-0002-7710-6095
    affiliation: 2    

  - name: Oleg Sivokon
    affiliation: 3  
     

affiliations:
 - name: MaxCor lab, Sapir College, Israel
   index: 1

 - name: Department of Medical Imaging, Princess Alexandra Hospital, Brisbane, QLD, Australia
   index: 2   

 - name: CMHM/cmhm.info, Israel
   index: 3   

date: 18 April 2021
bibliography: paper.bib
---

# Summary


Radiological images of various anatomy are part of the diagnostic work-up for millions of patients for indications as diverse as trauma, infection, and malignancies, to name a few.  A considerable amount of time and resources have gone into developing an automated diagnostic interpretation of these images. The purpose of this library is to help scientists, medical professionals, and programmers create better datasets to create algorithms related to X-rays, MRIs or CTs.
CleanX is a Python package for data cleaning that was developed for radiology AI. 


# Statement of need

`CleanX` is a Python package for data exploration, cleaning, and augmentation that was originally developed for radiology AI. Python is a widely used language on a global level. Data preparation for building quality machine learning algorithms is known to be a time-consuming task (1). Of the tasks involved, 'data cleaning' alone usually takes the majority of time spent on analysis for clinical research projects (2). The task of data cleaning is a necessary step even in the case of relatively high-quality data to avoid the known problem of 'garbage in, garbage out' (3).
In contemporary research, many approaches to data cleaning for radiology datasets overlook the content of the images themselves. The quality of data, especially the image data, is often context-specific to a specific AI model.
Algorithms that rely on shape detection may be accomplished with contrast and positional invariance, but many neural networks or radiomics algorithms can and should not be insensitive contrast or position. Thus scales like MIDaR (4) are necessary but not sufficient to describe data. Despite the specific nature of quality issues for each model, some potential data contamination problems should be cleaned out of imaging datasets for most algorithms.
In the case of radiological datasets, the task of data cleaning involves checking the accuracy of labelling and/or the quality of the images themselves. Potential problems inside the images themselves in large datasets include the inclusion of 'out of domain data' and 'label leakage'. Some types of 'out of domain data' may not be apparent to non-radiologists and have been a particular problem in datasets web-scraped together by non-radiologists (5).
'Label leakage' depends on the desired labels for a dataset but can happen in many ways. More subtle forms of label leakage may occur when certain machines are more likely to be used on certain patients. Depending upon the goals of a model, there may be other types of 'out of domain data' that are easy to see, such as inverted or flipped images. Even this can cost tremendous amounts of time to remove from a dataset with hundreds of thousands of images.
While data cleaning can not be fully automated at present, it is unrealistic for many data science practitioners and researchers to afford the hours of an imaging specialist for every data cleaning task. This package speeds up data cleaning, and gives researchers some basic insights into datasets of images. It also has functions for augmenting X-ray images so that the resultant images are within domain data. 
Automated data cleaning can improve dataset quality on some parameters. This work includes open code originally built to help with automatic chest X-ray dataset exploratory data analysis and data cleaning. It was expanded to include functions for DICOM processing, and image data normalization and augmentations. Some of the functions can be used to clean up a dataset of any two dimensional images. Several algorithms for identifying out of domain data in a large dataset of chest-X rays facilitated by the functions in this code library.




# Acknowledgements

We acknowledge many contributions from Eliane Birba (delwende) and Oleg Sivokon (wvxvw) during the testing and documentation of the code related to this project. We did not receive any financial support for this project. 

# References

1. “A Study on the Importance of and Time Spent on Different Modeling Steps”. M.A. Munson. 2011, ACM SIGKDD Explorations, pp. Pages 65-71. https://doi.org/10.1145/2207243.2207253
2. Tidy data. Wickham, H. 2014, Journal of Statistical Software, Vol. 5910, pp. 1-23. http://www.jstatsoft.org/v59/i10/
3. Data cleaning: Problems and current approaches. E Rahm, HH Do. 2000, IEEE Data Eng. Bull., Vol. 23 (4), pp. 3-13. http://sites.computer.org/debull/A00dec/A00DEC-CD.pdf
4. Harvey H., Glocker B. (2019) A Standardised Approach for Preparing Imaging Data for Machine Learning Tasks in Radiology. In: Ranschaert E., Morozov S., Algra P. (eds) Artificial Intelligence in Medical Imaging. Springer, Cham. https://doi.org/10.1007/978-3-319-94878-2_6
5. Tizhoosh, H.R., Fratesi, J. COVID-19, AI enthusiasts, and toy datasets: radiology without radiologists. Eur Radiol 31, 3553–3554 (2021). https://doi.org/10.1007/s00330-020-07453-w


