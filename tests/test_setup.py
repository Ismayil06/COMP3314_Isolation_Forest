import sys


def test_python_version_is_new_enough():
    assert sys.version_info >= (3, 10)


def test_all_libraries_can_be_imported():
    import numpy
    import scipy
    import sklearn
    import pandas
    import matplotlib