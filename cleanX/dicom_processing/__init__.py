# -*- coding: utf-8 -*-

import warnings
from textwrap import dedent

HAS_PYDICOM = False
HAS_SIMPLEITK = False

try:
    import pydicom
    from .pydicom_adapter import (
        PydicomDicomReader as DicomReader,
        get_jpg_with_pydicom as rip_out_jpgs,
    )
    HAS_PYDICOM = True
except ModuleNotFoundError:
    pass

if not HAS_PYDICOM:
    try:
        import SimpleITK
        from .simpleitk_adapter import (
            rip_out_jpgs_sitk as rip_out_jpgs,
        )
        HAS_SIMPLEITK = True
    except ModuleNotFoundError:
        pass

if not (HAS_SIMPLEITK or HAS_PYDICOM):
    warnings.warn(dedent(
        '''
        Neither SimpleITK nor PyDICOM are installed.

        Will not be able to extract information from DICOM files.
        ''',
        UserWarning,
    ))

from .source import (
    DirectorySource,
    GlobSource,
    MultiSource,
)
