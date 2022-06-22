# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
import os
import io
import dataclasses
import functools
import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize


FilePath = str | os.PathLike[str]
DEFAULT_HEADER = bytes.fromhex(
    "49 4d cd 01 80 02 e0 01 00 00 00 00 02 00 00 00"
    "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
    "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
    "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
)


@dataclasses.dataclass(frozen=True)
class Data:
    intensity: npt.NDArray[np.float32]
    wavelength: npt.NDArray[np.float32]
    time: npt.NDArray[np.float32]
    header: bytes = DEFAULT_HEADER
    metadata: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_raw_file(cls, filepath_or_buffer: FilePath | io.BufferedIOBase) -> "Data":
        if isinstance(filepath_or_buffer, (str, os.PathLike)):
            with open(filepath_or_buffer, "rb") as f:
                self = cls._from_raw_buffer(f)
        elif isinstance(filepath_or_buffer, io.BufferedIOBase):
            self = cls._from_raw_buffer(filepath_or_buffer)
        else:
            raise TypeError("The type of filepath_or_buffer must be FilePath or io.BufferedIOBase")
        return self

    @classmethod
    def _from_raw_buffer(cls, file: io.BufferedIOBase) -> "Data":
        sector_size = 1024
        wavelength_resolution = 640
        time_resolution = 480
        header = file.read(64)
        metadata = [file.readline() for _ in range(4)]
        intensity = np.frombuffer(file.read(sector_size*600), dtype=np.uint16)
        wavelength = np.frombuffer(file.read(sector_size*4), dtype=np.float32)[:wavelength_resolution]
        time = np.frombuffer(file.read(sector_size*4), dtype=np.float32)[:time_resolution]
        data: dict[str, t.Any] = dict()
        data["header"] = header
        data["metadata"] = [b.decode("UTF-8") for b in metadata]
        data["intensity"] = intensity.astype(np.float32)
        data["wavelength"] = wavelength.astype(np.float32)
        data["time"] = time.astype(np.float32)
        return cls(**data)

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            dict(
                time=np.repeat(self.time, len(self.wavelength)),        # [ns]
                wavelength=np.tile(self.wavelength, len(self.time)),    # [nm]
                intensity=self.intensity                                # [arb. units]
            )
        )
        return df

    @functools.cached_property
    def streak_image(self) -> npt.NDArray[np.float32]:
        return self.intensity.reshape(len(self.time), len(self.wavelength))

    def time_resolved(
        self,
        time_range: tuple[float, float] | None = None,
    ) -> "TimeResolved":
        assert "wavelength" in self.df.columns
        assert "intensity" in self.df.columns
        if time_range is None:
            time = self.time
            time_range = time.min(), time.max()
        return TimeResolved(self, time_range)

    def wavelength_resolved(
        self,
        wavelength_range: tuple[float, float] | None = None,
        time_offset: t.Literal["auto"] | float = "auto",
    ) -> "WavelengthResolved":
        assert "time" in self.df.columns
        assert "intensity" in self.df.columns
        if wavelength_range is None:
            wavelength = self.df["wavelength"]
            wavelength_range = wavelength.min(), wavelength.max()
        return WavelengthResolved(self, wavelength_range, time_offset)

    def to_raw_binary(self) -> bytes:
        data = self.header \
            + "".join(self.metadata).encode("UTF-8") \
            + self.intensity.astype(np.uint16).tobytes("C") \
            + self.wavelength.astype(np.float32).tobytes("C").ljust(4096, b"\x00") \
            + self.time.astype(np.float32).tobytes("C").ljust(4096, b"\x00")
        return data


@dataclasses.dataclass(frozen=True)
class TimeResolved:
    data: Data
    range: tuple[float, float]

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        assert "time" in self.data.df.columns
        assert "wavelength" in self.data.df.columns
        assert "intensity" in self.data.df.columns
        df = self.data.df[self.data.df["time"].between(*self.range)] \
            .groupby("wavelength") \
            .sum() \
            .drop("time", axis=1) \
            .reset_index()
        return df

    @property
    def peak_wavelength(self) -> float:
        assert "wavelength" in self.df.columns
        return float(self.df["wavelength"][self.smoothed_intensity().argmax()])

    @property
    def peak_intensity(self) -> float:
        return float(self.smoothed_intensity().max())

    @property
    def half_range(self) -> tuple[float, float]:
        assert "intensity" in self.df.columns
        assert "wavelength" in self.df.columns
        intensity = self.smoothed_intensity()
        wavelength = self.df["wavelength"]
        under_half = intensity < intensity.max() / 2
        left = wavelength[(wavelength < wavelength[intensity.argmax()]) & under_half].max()
        right = wavelength[(wavelength > wavelength[intensity.argmax()]) & under_half].min()
        return (
            float(left if left is not np.nan else wavelength.min()),
            float(right if right is not np.nan else wavelength.max())
        )

    @property
    def FWHM(self) -> float:
        left, right = self.half_range
        return abs(right - left)

    def smoothed_intensity(self, window: int = 5) -> pd.Series:
        assert "wavelength" in self.df.columns
        return self.df["intensity"].rolling(window, center=True).mean()


@dataclasses.dataclass(frozen=True)
class WavelengthResolved:
    data: Data
    range: tuple[float, float]
    time_offset: t.Literal["auto"] | float = "auto"

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        assert "time" in self.data.df.columns
        assert "wavelength" in self.data.df.columns
        assert "intensity" in self.data.df.columns
        df = self.data.df[self.data.df["wavelength"].between(*self.range)] \
            .groupby("time") \
            .sum() \
            .drop("wavelength", axis=1) \
            .reset_index()
        if self.time_offset == "auto":
            intensity = df["intensity"]
            window, k = 10, 2
            rolling = intensity.rolling(window)
            time_offset = float(
                df["time"][
                    (intensity > rolling.mean() + k * rolling.std()).shift(-1, fill_value=False)
                ].min()
            )
        else:
            time_offset = self.time_offset
        df["time"] -= time_offset
        return df

    def fit(
        self,
        func: t.Callable[[t.Any], t.Any],
        fitting_range: tuple[float, float] | None = None,
    ) -> tuple[t.Any, t.Any]:  # pragma: no cover
        assert "intensity" in self.df.columns
        assert "time" in self.df.columns
        if fitting_range is None:
            fitting_range = self._get_auto_fitting_range()
        df = self.df
        index = df.index[df["time"].between(*fitting_range)]
        max_intensity = df["intensity"].max()
        smoothed = df["intensity"].rolling(5, center=True).mean() / max_intensity
        params, cov = optimize.curve_fit(func, df["time"][index], smoothed[index])
        df["fit"] = np.nan
        df.loc[index, "fit"] = func(df["time"][index], *params) * max_intensity
        return params, cov

    def _get_auto_fitting_range(self) -> tuple[float, float]:  # pragma: no cover
        assert "intensity" in self.df.columns
        assert "time" in self.df.columns
        df = self.df
        left = df["time"][df["intensity"].shift(1).argmax()]
        noise = df["intensity"][df["time"] < 0]
        right = df["time"][
            (df.index > df["intensity"].argmax())
            & (df["intensity"] < noise.mean() + 2*noise.std())
        ].iloc[0]
        return float(left), float(right)
