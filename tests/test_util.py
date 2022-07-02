# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
from unittest import TestCase
import doctest

from tlab_analysis import util


class Test_find_start_point(TestCase):
    pass
    # TODO: Implement unit tests


def load_tests(loader, tests, _):  # type: ignore
    tests.addTests(doctest.DocTestSuite(util))
    return tests
