# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import typing as t

import pandas as pd


def find_start_point(
    xdata: t.Sequence[float],
    ydata: t.Sequence[float],
    window: int = 10,
    k: int = 2
) -> tuple[float, float]:
    """Return coordinates of a start point of a rising curve.

    Parameters
    ----------
    xdata : Sequence[float]
        Data for x axis.
    ydata : Sequence[float]
        Data for y axis.
    window : int
        The rolling size to determine a range before rising.
    k : int
        The multiplier to determine a range before rising.

    Returns
    -------
    tuple[float, float]
        Coordinates of a start point of rising curve: (x, y).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 1000)
    >>> np.random.seed(222)
    >>> noise = np.random.normal(scale=5, size=len(x))
    >>> x0, y0 = -2, 50
    >>> y = np.where(x - x0 > 0, np.exp(x - x0) - 1, 0) + y0 + noise
    >>> find_start_point(x, y)
    (0.1451451451451451, 53.79270065158227)
    """
    assert len(xdata) == len(ydata), "The length of `xdata` and `ydata` must be the same."
    df = pd.DataFrame(
        dict(
            x=xdata,
            y=ydata
        )
    ).sort_values(by="x", ignore_index=True)
    # Determine a range of the background signal
    rolling = df["y"].rolling(window)
    sup_noise = rolling.mean() + k * rolling.std()
    background = df["x"].le(df["x"][df["y"].gt(sup_noise).shift(-1, fill_value=False)].min())
    y_background = df["y"][background]
    y_baseline = y_background[
        y_background.between(
            y_background.quantile(0.05),
            y_background.quantile(0.95)
        )
    ].mean()
    # Determine (x, y) coordinates of a start point
    index = df.index[
        (df.index < df["y"].argmax())
        & df["y"].lt(y_baseline).shift(1, fill_value=False)
    ].max()
    return (float(df["x"][index]), float(df["y"][index]))
