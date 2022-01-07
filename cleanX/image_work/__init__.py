# -*- coding: utf-8 -*-

from .image_functions import (
    crop_np,
    crop,
    subtle_sharpie_enhance,
    harsh_sharpie_enhance,
    salting,
    blur_out_edges,
    show_major_lines_on_image,
    find_big_lines,
    separate_image_averager,
    dimensions_to_df,
    dimensions_to_histo,
    proportions_ht_wt_to_histo,
    find_very_hazy,
    find_by_sample_upper,
    find_sample_upper_greater_than_lower,
    find_outliers_by_total_mean,
    find_outliers_by_mean_to_df,
    create_matrix,
    find_tiny_image_differences,
    tesseract_specific,
    find_suspect_text,
    find_suspect_text_by_length,
    histogram_difference_for_inverts,
    inverts_by_sum_compare,
    histogram_difference_for_inverts_todf,
    find_duplicated_images,
    find_duplicated_images_todf,
    show_images_in_df,
    dataframe_up_my_pics,
    simple_spinning_template,
    make_contour_image,
    hist_sum_of_pixels_across_set,
    crop_np_white,
    avg_image_maker,
    set_image_variability,
    find_outliers_sum_of_pixels_across_set,
    avg_image_maker_by_label,
    zero_to_twofivefive_simplest_norming,
    rescale_range_from_histogram_low_end,
    make_histo_scaled_folder,
    give_size_count_df,
    give_size_counted_dfs,
    image_quality_by_size,
    show_close_images,
    find_close_images,
    image_to_histo,
    black_end_ratio,
    outline_segment_by_otsu,
    binarize_by_otsu,
    column_sum_folder,
    blind_quality_matrix,
    fourier_transf,
    pad_to_size,
    cut_to_size,
    cut_or_pad,
    rotated_with_max_clean_area,
    noise_sum_median_blur,
    noise_sum_cv,
    blind_noise_matrix,
    segmented_blind_noise_matrix,


    Rotator,
)
from .pipeline import (
    DirectorySource,
    GlobSource,
    MultiSource,
    PipelineError,
    Pipeline,
)
from .journaling_pipeline import JournalingPipeline
from .steps import (
    Step,
    Acquire,
    Save,
    FourierTransf,
    BlackEdgeCrop,
    WhiteEdgeCrop,
    Normalize,
    HistogramNormalize,
    ProjectionHorizoVert,
    ContourImage,
    Sharpie,
    BlurEdges,
    Aggregate,
    GroupHistoHtWt,
    GroupHistoPorportion,
    Mean,
    OtsuBinarize,
    OtsuLines,
    Projection,
    CleanRotate,
)


def create_pipeline(steps, batch_size=None, journal=None, keep_journal=False):
    """
    Create a pipeline that will execute the :code:`steps`.  If
    :code:`journal` is not false, create a journaling pipeline, that can
    be pick up from the failed step.

    :param steps: A sequence of :class:`Step` to be executed in this pipeline.
    :type steps: Sequence[Step]
    :param batch_size: Controls how many steps are processed concurrently.
    :type batch_size: int
    :param journal: If :code:`True` is passed, the pipeline code will use a
                    preconfigured directory to store the journal.  Otherwise,
                    this must be the path to the directory to store the journal
                    database.
    :type journal: Union[bool, str]
    :param keep_journal: Controls whether the journal is kept after successful
                         completion of the pipeline.
    :type keep_journal: bool

    :return: a :class:`~.pipeline.Pipeline` object or one of its descendants.
    :rtype: :class:`~.pipeline.Pipeline`
    """
    if journal:
        return JournalingPipeline(
            steps,
            batch_size=batch_size,
            journal=journal,
            keep_journal=keep_journal,
        )
    return Pipeline(steps, batch_size)


def restore_pipeline(journal_dir, skip=0, **overrides):
    """
    Restores previously interrupted pipeline.  The pipeline should have been
    created with :code:`journal` set.  If the creating code didn't specify
    the directory to keep the journal, it may be obtained in this way:

    .. code-block:: python

        p = create_pipeline(steps=(...), journal=True)
        journal_dir = p.journal_dir
        # After pipeline failed
        p = restore_pipeline(journal_dir)

    :param journal_dir: The directory containing journal database to restore
                        from.
    :type journal_dir: Suitable for :code:`os.path.join()`
    :param skip: Skip this many steps before attempting to resume the pipeline.
                 This is useful if you know that the step that failed will
                 fail again, but you want to execute the rest of the steps
                 in the pipeline.
    :param \\**overrides: Arguments to pass to the created pipeline instance
                          that will override those restored from the journal.

    :return: Fresh :class:`~.journaling_pipeline.JournalingPipeline` object
                   fast-forwarded to the last executed step + :code:`skip`.
    :rtype: :class:`~.journaling_pipeline.JournalingPipeline`
    """
    return JournalingPipeline.restore(journal_dir, skip=skip, **overrides)
