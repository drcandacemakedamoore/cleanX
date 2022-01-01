# testing for cleanX

import os
import sys
import csv
import json

from functools import partial
from tempfile import TemporaryDirectory
from random import Random

import cv2
import pytest
import pandas as pd
import matplotlib as plt
import numpy as np


from cleanX import (
    dataset_processing as csvp,
    dicom_processing as dicomp,
    image_work as iwork,
)
from harness import skip_if_missing


image_directory = os.path.join(os.path.dirname(__file__), 'directory')
target_directory = os.path.join(os.path.dirname(__file__), 'target')


def gen_row(columns, rnd):
    result = list(range(columns))
    rnd.shuffle(result)
    return result


def gen_csv(directory, name, rows, columns, rnd):
    with open(os.path.join(directory, name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        colnames = ['c{}'.format(c) for c in gen_row(columns, rnd)]
        writer.writerow(colnames)
        for _ in range(rows):
            writer.writerow(gen_row(columns, rnd))


def gen_json(directory, name, rows, columns, rnd):
    with open(os.path.join(directory, name), 'w') as jsonfile:
        colnames = ['c{}'.format(c) for c in range(columns)]
        rnd.shuffle(colnames)
        data = {c: [] for c in colnames}
        for i in range(rows):
            for j, v in zip(data.keys(), gen_row(columns, rnd)):
                data[j].append(v)
        json.dump(data, jsonfile)

def test_crop():
    example = cv2.imread(os.path.join(image_directory, 'testtocrop.jpg'), cv2.IMREAD_GRAYSCALE)
    cropped_example = iwork.crop(example)
    assert cropped_example.shape < example.shape

def test_crop_np_white():
    example = cv2.imread(os.path.join(image_directory, 'testtocrop.jpg'), cv2.IMREAD_GRAYSCALE)
    cropped_example = iwork.crop_np_white(example)
    # our algorithm cuts off a row and column so... we can't call ==
    assert type(cropped_example) is np.ndarray

def test_blur_out_edges():
    image = os.path.join(image_directory, 'testtocrop.jpg')
    defblur = iwork.blur_out_edges(image)
    assert type(defblur) == np.ndarray 


def test_subtle_sharpie_enhance():
    image = os.path.join(image_directory, 'testtocrop.jpg')
    lo = iwork.subtle_sharpie_enhance(image)
    assert lo.shape[0] >1


def harsh_sharpie_enhance():
    image = os.path.join(image_directory, 'testtocrop.jpg')
    ho = iwork.harsh_sharpie_enhance(image)
    assert ho.shape[0] >1


def test_salting():
    lindo_image = os.path.join(image_directory, 'testtocrop.jpg')
    salt = iwork.salting(lindo_image)
    assert salt.shape[0] > 1

def test_check_paths_for_group_leakage():
    train_dfE = (os.path.join(image_directory,'train_sample_df.csv'))
    test_dfE = (os.path.join(image_directory,'test_sample_df.csv'))
    train_df= pd.read_csv(train_dfE)
    test_df = pd.read_csv(test_dfE)
    uniqueIDE = 'image_path'
    checked_example = csvp.check_paths_for_group_leakage(
        train_df,
        test_df,
        uniqueIDE,
    )
    assert len(checked_example) > 1 


def test_separate_image_averager():
    test_dfE = (os.path.join(image_directory,'test_sample_df.csv'))
    test_df = pd.read_csv(test_dfE)
    images = image_directory + '/' + test_df.image_path.dropna()
    blended =  iwork.separate_image_averager(images, s=5)
    assert type(blended) is np.ndarray


def test_dimensions_to_df():
    deflep = iwork.dimensions_to_df(image_directory)
    assert len(deflep) > 1    


def test_see_part_potential_bias():
    e2 = (os.path.join(image_directory,'example_for_bias.csv'))
    e3 = pd.read_csv(e2)
    donwa = csvp.see_part_potential_bias(e3,'Label', ['Gender', 'Race'])
    assert len(donwa) > 1

# def test_show_images_in_df():

#     # this testing remains undone... the function returns basically
#     the line bwlo, and testing will be difficult plt.show()
#
def test_dimensions_to_histo():
    output = iwork.dimensions_to_histo(image_directory, 10)
    assert len(output) > 1


def test_find_very_hazy():
    found = iwork.find_very_hazy(image_directory)
    assert len(found) > 0    


def test_show_major_lines_on_image():
    pic_name1 = os.path.join(image_directory, 'testtocrop.jpg')
    deflop = iwork.show_major_lines_on_image(pic_name1)
    assert deflop # needs a much better test


def test_find_big_lines():
    lined = iwork.find_big_lines(image_directory, 2)
    assert len(lined) > 0    


def proportions_ht_wt_to_histo():
    output =  iwork.proportions_ht_wt_to_histo(image_directory, 10)
    assert len(output) > 1


def test_tesseract_specific():
    lettered = iwork.tesseract_specific(image_directory)
    assert len(lettered) > 1 


def test_find_suspect_text():
    letters_spec = iwork.find_suspect_text(image_directory, 'SUPINE')
    assert len(letters_spec) >= 1     


def test_find_suspect_text_by_length():   
    jobs = iwork.find_suspect_text_by_length(image_directory, 3)
    assert len(jobs) > 1    


def test_find_by_sample_upper():
    lovereturned = iwork.find_by_sample_upper(image_directory, 10, 10)
    assert len(lovereturned) >= 1


def test_find_sample_upper_greater_than_lower():
    lovereturnee = iwork.find_sample_upper_greater_than_lower(image_directory, 10)
    assert len(lovereturnee) >= 1

def test_find_duplicated_images():
    found = iwork.find_duplicated_images(image_directory)
    assert len(found) > 0     


def test_find_duplicated_images_todf():
    found = iwork.find_duplicated_images_todf(image_directory)
    assert len(found) > 0       


def test_histogram_difference_for_inverts():
    histosy = iwork.histogram_difference_for_inverts(image_directory)
    assert len(histosy) > 0


def test_histogram_difference_for_inverts_todf():
    histos = iwork.histogram_difference_for_inverts_todf(image_directory)
    assert len(histos) >0


def test_dataframe_up_my_pics():
    dfy = iwork.dataframe_up_my_pics(image_directory, 'diagnosis_string')
    assert len(dfy) > 0
    

def test_simple_spinning_template():
    vovo = os.path.join(image_directory, 'testtocrop.jpg')
    picy1 = vovo
    img = cv2.imread(vovo, cv2.IMREAD_GRAYSCALE)
    greys_template1 = img[90:110, 200:350]
    angle_start1 = 0
    angle_stop1 = 30
    slices1 = 3
    lanter = iwork.simple_spinning_template(
        picy1,
        greys_template1,
        angle_start1,
        angle_stop1,
        slices1,
    )
    assert len(lanter) > 0


def test_def_make_contour_image():
    vovo = os.path.join(image_directory, 'testtocrop.jpg')
    picy1 = vovo
    defMkcont = iwork.make_contour_image(picy1)
    assert len(defMkcont) > 0


def test_avg_image_maker():
    #set_of_images = glob.glob(*.jpg)
    test_dfE = (os.path.join(image_directory,'test_sample_df.csv'))
    test_df = pd.read_csv(test_dfE)
    set_of_images = image_directory + '/' + test_df.image_path.dropna()
    tab = iwork.avg_image_maker(set_of_images)  
    assert tab.shape[0] > 2  


def test_set_image_variability():
    test_dfE = (os.path.join(image_directory,'test_sample_df.csv'))
    test_df = pd.read_csv(test_dfE)
    set_of_images = image_directory + '/' + test_df.image_path.dropna()
    tab = iwork.set_image_variability(set_of_images)  
    assert tab.shape[0] > 2      


def test_avg_image_maker_by_label():
    test_dfE = (os.path.join(image_directory,'alt_test_labeled.csv'))
    test_df = pd.read_csv(test_dfE)
    lotus = iwork.avg_image_maker_by_label(test_df,'imageID','path_label',image_directory)
    assert len(lotus) > 0       


def test_find_tiny_image_differences():
    defleper = iwork.find_tiny_image_differences(image_directory)
    assert len(defleper) > 0


def test_zero_to_twofivefive_simplest_norming():
    vovo = os.path.join(image_directory, 'testtocrop.jpg')
    test_norm1 = iwork.zero_to_twofivefive_simplest_norming(vovo)
    assert test_norm1.max() == 255


def test_rescale_range_from_histogram_low_end():
    image_path = os.path.join(image_directory, 'testtocrop.jpg')
    image_from_path = cv2.imread(image_path)
    defmax = iwork.rescale_range_from_histogram_low_end(image_from_path, 5)
    assert defmax.max() == 255


def test_make_histo_scaled_folder():
    A= image_directory
    targy = target_directory
    d = iwork.make_histo_scaled_folder(A, 5, targy)
    assert len(d) > 0


@skip_if_missing('no simpleITK available', 'SimpleITK')
def test_rip_out_jpgs_sitk():
    dicomfile_directory1 = os.path.join(
        os.path.dirname(__file__),
        'dicom_example_folder',
    )
    with TemporaryDirectory() as td:
        import cleanX.dicom_processing.simpleitk_adapter as sitk_adapter
        jpegs_made = sitk_adapter.rip_out_jpgs_sitk(
            dicomfile_directory1,
            td,
        )
    assert len(jpegs_made) > 0    


@skip_if_missing('no simpleITK available', 'SimpleITK')
def test_read_dicoms_with_sitk():
    dicomfile_directory1 = os.path.join(
        os.path.dirname(__file__),
        'dicom_example_folder',
    )
    source_column = 'file'
    import cleanX.dicom_processing.simpleitk_adapter as sitk_adapter
    reader = sitk_adapter.SimpleITKDicomReader(
        exclude_fields=('Patients Name',),
    )
    source = dicomp.DirectorySource(dicomfile_directory1, source_column)
    df = reader.read(source)

    assert source_column in df.columns
    assert 'Patients Name' not in df.columns
    assert 'Protocol Name' in df.columns


@skip_if_missing('no pydicom available', 'pydicom')
def test_get_jpg_with_pydicom():
    dicomfile_directory1 = os.path.join(
        os.path.dirname(__file__),
        'dicom_example_folder',
    )
    with TemporaryDirectory() as td:
        jpegs_made = dicomp.pydicom_adapter.get_jpg_with_pydicom(
            dicomfile_directory1,
            td,
        )
    assert jpegs_made


@skip_if_missing('no pydicom available', 'pydicom')
def test_read_dicoms_with_pydicom():
    dicomfile_directory1 = os.path.join(
        os.path.dirname(__file__),
        'dicom_example_folder',
    )
    tag = 'file'
    reader = dicomp.pydicom_adapter.PydicomDicomReader()
    source = dicomp.DirectorySource(dicomfile_directory1, tag)
    df = reader.read(source)
    assert tag in df.columns
    assert set(os.listdir(dicomfile_directory1)) == set(df[tag].to_list())


@skip_if_missing('no pydicom available', 'pydicom')
def test_read_dicoms_options_with_pydicom():
    dicomfile_directory1 = os.path.join(
        os.path.dirname(__file__),
        'dicom_example_folder',
    )
    source_column = 'file'
    reader = dicomp.pydicom_adapter.PydicomDicomReader(
        exclude_fields=('PatientName',),
    )
    source = dicomp.DirectorySource(dicomfile_directory1, source_column)
    df = reader.read(source)

    assert 'PatientName' not in df.columns
    assert source_column in df.columns
    assert (
        set(os.listdir(dicomfile_directory1)) ==
        set(df[source_column].to_list())
    )


def test_dataset_creation():
    rnd = Random()
    with TemporaryDirectory() as td:
        rows, columns = rnd.randint(10, 100), rnd.randint(10, 100)
        csvfile = 'file.csv'
        jsonfile = 'file.json'

        gen_csv(td, csvfile, rows, columns, rnd)
        gen_json(td, jsonfile, rows, columns // 2, rnd)
        setup = csvp.MLSetup(
            os.path.join(td, csvfile),
            os.path.join(td, jsonfile),
        )
        m1, m2 = setup.metadata()
        m1, m2 = set(m1), set(m2)
        common = len(m1.intersection(m2))
        assert len(m2) == common
        all_df = setup.concat_dataframe()
        assert len(m1) == len(all_df.columns)


def test_give_size_counted_dfs():
    df_by_size = iwork.give_size_counted_dfs(image_directory)
    assert len(df_by_size) > 0


def test_give_size_count_df():
    df_by_size = iwork.give_size_count_df(image_directory)
    assert len(df_by_size) > 0


def test_image_quality_by_size():
    vovo = os.path.join(image_directory, 'testtocrop.jpg')
    quality = iwork.image_quality_by_size(vovo)
    assert quality > 0


def test_find_close_images():
    output1 = iwork.find_close_images(image_directory, 2, 190)
    assert len(output1) > 0


def test_show_close_images(monkeypatch):
    # TODO(wvxvw): Find out if anything can be verified about the
    # result of plt.show()
    monkeypatch.setattr(iwork.image_functions.plt, 'show', lambda: None)
    iwork.show_close_images(image_directory, 2, 190)

def test_image_to_histo():
## currently blocked out for oleg to check why failing
    vovo = os.path.join(image_directory, 'testtocrop.jpg')
    outpic = cv2.imread(vovo)
    hist = iwork.image_to_histo(outpic)
    assert type(hist) is np.ndarray

def test_black_end_ratio():
    vovo = os.path.join(image_directory, 'testtocrop.jpg')
    outpic = cv2.imread(vovo)
    ratio = iwork.black_end_ratio(outpic)
    assert type(ratio) is np.float64

def test_outline_segment_by_otsu():
    vovo = os.path.join(image_directory, 'testtocrop.jpg')
    outlined = iwork.outline_segment_by_otsu(vovo,11)
    assert type(outlined) is np.ndarray 

def test_binarize_by_otsu():
    vovo = os.path.join(image_directory, 'testtocrop.jpg')
    outlined = iwork.binarize_by_otsu(vovo,11)
    assert type(outlined) is np.ndarray 

def test_find_outliers_sum_of_pixels_across_set():
    result = iwork.find_outliers_sum_of_pixels_across_set(image_directory, 100)
    assert len(result) > 1

def test_hist_sum_of_pixels_across_set():
    result_hist = iwork.hist_sum_of_pixels_across_set(image_directory)
    assert type(result_hist) is np.ndarray 

def test_blind_quality_matrix():
    result_df = iwork.blind_quality_matrix(image_directory)
    assert type(result_df) is pd.io.formats.style.Styler

def test_fourier_transf():
    sample_image = os.path.join(image_directory, 'testtocrop.jpg')
    sample_imager = cv2.imread(sample_image)
    transformed = iwork.fourier_transf(sample_imager)
    assert sample_imager.shape[0] == transformed.shape[0]

def test_pad_to_size():
    sample_image = os.path.join(image_directory, 'testtocrop.jpg')
    sample_imager = cv2.imread(sample_image)
    padded = iwork.pad_to_size(sample_imager, 5000, 5000)
    assert type(padded) is np.ndarray

def test_cut_to_size():
    sample_image = os.path.join(image_directory, 'testtocrop.jpg')
    sample_imager = cv2.imread(sample_image)
    padded = iwork.cut_to_size(sample_imager, 3, 2)
    assert type(padded) is np.ndarray 

def test_cut_or_pad():
    sample_image = os.path.join(image_directory, 'testtocrop.jpg')
    sample_imager = cv2.imread(sample_image)
    padded = iwork.cut_or_pad(sample_imager, 5000, 2)
    assert type(padded) is np.ndarray 