# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import abc
import functools

import pandas as pd


class AbstractData(abc.ABC):
    """Abstract class for Data."""

    @abc.abstractmethod
    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """A dataframe."""
