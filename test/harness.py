# -*- coding: utf-8 -*-

from importlib import import_module

import pytest


def missing(*modules):
    try:
        for m in modules:
            import_module(m)
        return False
    except ModuleNotFoundError:
        return True


def skip_if_missing(reason, *modules):
    return pytest.mark.skipif(missing(*modules), reason=reason)
