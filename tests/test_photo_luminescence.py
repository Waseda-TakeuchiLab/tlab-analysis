# Copyright (c) 2022 Shuhei Nitta. All rights reserved.
from unittest import TestCase
import io
import os
import tempfile
import typing as t

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

from tlab_analysis import photo_luminescence as pl


HEADER = bytes.fromhex(
    "49 4d cd 01 80 02 e0 01 00 00 00 00 02 00 00 00"
    "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
    "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
    "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
)
METADATA = [
    b"HiPic,1.0,100,1.0,0,0,4,8,0,0,0,01-01-1970,00:00:00,"
    b"0,0,0,0,0, , , , ,0,0,0,0,0, , ,0,, , , ,0,0,, ,0,0,0,0,0,0,0,0,0,0,"
    b"2,1,nm,*0614925,2,1,ns,*0619021,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0.0,0,0,"
    b"StopCondition:PhotonCounting, Frame=10000, Time=673.1[sec], CountingRate=0.13[%]\n",
    b"Streak:Time=10 ns, Mode=Operate, Shutter=0, MCPGain=12, MCPSwitch=1, \n",
    b"Spectrograph:Wavelength=490.000[nm], Grating=2 : 150g/mm, SlitWidthIn=100[um], Mode=Spectrograph\n",
    b"Date:2022/06/03,14:09:55\n"
]
WAVELENGTH_RESOLUTION = 640
TIME_RESOLUTION = 480
STREAK_IMAGE = np.random.randint(0, 32, WAVELENGTH_RESOLUTION * TIME_RESOLUTION, dtype=np.uint16).tobytes("C")
WAVELENGTH = np.linspace(435, 535, WAVELENGTH_RESOLUTION, dtype=np.float32).tobytes("C").ljust(1024*4, b"\x00")
TIME = np.linspace(0, 10, TIME_RESOLUTION, dtype=np.float32).tobytes("C").ljust(1024*4, b"\x00")
RAW = HEADER + b"".join(METADATA) + STREAK_IMAGE + WAVELENGTH + TIME


def get_data() -> pl.Data:
    time = np.frombuffer(TIME, dtype=np.float32)[:TIME_RESOLUTION]
    wavelength = np.frombuffer(WAVELENGTH, dtype=np.float32)[:WAVELENGTH_RESOLUTION]
    intensity = np.frombuffer(STREAK_IMAGE, dtype=np.uint16).astype(np.float32)
    return pl.Data(
        header=HEADER,
        metadata=[b.decode("UTF-8") for b in METADATA],
        time=time,
        wavelength=wavelength,
        intensity=intensity
    )


class TestData_from_raw_file(TestCase):

    def setUp(self) -> None:
        self.data = get_data()

    def _test(self, Data: pl.Data) -> None:
        self.assertEqual(Data.header, self.data.header)
        self.assertListEqual(Data.metadata, self.data.metadata)
        npt.assert_array_equal(Data.time, self.data.time)
        npt.assert_array_equal(Data.wavelength, self.data.wavelength)
        npt.assert_array_equal(Data.intensity, self.data.intensity)

    def test_filepath_or_buffer(self) -> None:
        with self.subTest("Filepath"):
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "photo_luminescence_testcase.img")
                with open(filepath, "wb") as f:
                    f.write(RAW)
                Data = pl.Data.from_raw_file(filepath)
                self._test(Data)
        with self.subTest("Buffer"):
            with io.BytesIO(RAW) as f:
                Data = pl.Data.from_raw_file(f)
            self._test(Data)
        with self.subTest("Invalid Type"):
            with self.assertRaises(TypeError):
                pl.Data.from_raw_file(None)  # type: ignore


class TestData_property(TestCase):

    def setUp(self) -> None:
        self.data = get_data()

    def test_df(self) -> None:
        df = pd.DataFrame(dict(
            time=np.repeat(self.data.time, len(self.data.wavelength)),
            wavelength=np.tile(self.data.wavelength, len(self.data.time)),
            intensity=self.data.intensity,
        ))
        pdt.assert_frame_equal(self.data.df, df)

    def test_streak_image(self) -> None:
        streak_image = self.data.intensity.reshape(len(self.data.time), len(self.data.wavelength))
        npt.assert_array_equal(self.data.streak_image, streak_image)


class TestData_time_resolved(TestCase):

    def setUp(self) -> None:
        self.data = get_data()

    def _test(self, time_range: tuple[float, float] | None = None) -> None:
        tr = self.data.time_resolved(time_range)
        if time_range is None:
            time = self.data.time
            time_range = time.min(), time.max()
        self.assertEqual(tr, pl.TimeResolved(self.data, time_range))

    def test_time_range(self) -> None:
        time = self.data.time
        time_range_list = [
            None,
            (float(time.min()), float(time.max())),
            (float(time[5]), float(time[-5]))
        ]
        for time_range in time_range_list:
            with self.subTest(time_range=time_range):
                self._test(time_range=time_range)


class TestData_wavelength_resolved(TestCase):

    def setUp(self) -> None:
        self.data = get_data()

    def _test(
        self,
        wavelength_range: tuple[float, float] | None = None,
        time_offset: t.Literal["auto"] | float = "auto",
    ) -> None:
        wr = self.data.wavelength_resolved(wavelength_range, time_offset)
        if wavelength_range is None:
            wavelength = self.data.wavelength
            wavelength_range = wavelength.min(), wavelength.max()
        self.assertEqual(wr, pl.WavelengthResolved(self.data, wavelength_range, time_offset))

    def test_wavelength_range(self) -> None:
        wavelength = self.data.wavelength
        wavelength_range_list: list[tuple[float, float] | None] = [
            None,
            (float(wavelength.min()), float(wavelength.max())),
            (float(wavelength[5]), float(wavelength[-5]))
        ]
        for wavelength_range in wavelength_range_list:
            with self.subTest(wavelength_range=wavelength_range):
                self._test(wavelength_range=wavelength_range)

    def test_time_offset(self) -> None:
        time_offsets: list[t.Literal["auto"] | float] = ["auto", 0.0, 3.0]
        for time_offset in time_offsets:
            with self.subTest(time_offset=time_offset):
                self._test(time_offset=time_offset)


class TestData_to_raw_binary(TestCase):

    def setUp(self) -> None:
        self.data = get_data()

    def test_default(self) -> None:
        data = self.data.header \
            + "".join(self.data.metadata).encode("UTF-8") \
            + self.data.intensity.astype(np.uint16).tobytes("C") \
            + self.data.wavelength.tobytes("C").ljust(4096, b"\x00") \
            + self.data.time.tobytes("C").ljust(4096, b"\x00")
        self.assertEqual(self.data.to_raw_binary(), data)


class TestTimeResolved_property(TestCase):

    def setUp(self) -> None:
        self.trs = [
            pl.TimeResolved(get_data(), (0.0, 5.0)),
            pl.TimeResolved(get_data(), (5.0, 10.0)),
            pl.TimeResolved(get_data(), (-1.0, 1.0)),
        ]

    def test_df(self) -> None:
        for tr in self.trs:
            df = tr.data.df[tr.data.df["time"].between(*tr.range)] \
                .groupby("wavelength") \
                .sum() \
                .drop("time", axis=1) \
                .reset_index()
            with self.subTest(tr=tr):
                pdt.assert_frame_equal(tr.df, df)

    def test_peak_wavelength(self) -> None:
        for tr in self.trs:
            with self.subTest(tr=tr):
                self.assertEqual(
                    tr.peak_wavelength,
                    float(tr.df["wavelength"][tr.smoothed_intensity().argmax()])
                )

    def test_peak_intensity(self) -> None:
        for tr in self.trs:
            with self.subTest(tr=tr):
                self.assertEqual(
                    tr.peak_intensity,
                    float(tr.smoothed_intensity().max())
                )

    def test_half_range(self) -> None:
        for tr in self.trs:
            intensity = tr.smoothed_intensity()
            wavelength = tr.df["wavelength"]
            under_half = intensity < intensity.max() / 2
            left = wavelength[(wavelength < wavelength[intensity.argmax()]) & under_half].max()
            right = wavelength[(wavelength > wavelength[intensity.argmax()]) & under_half].min()
            half_range = (
                float(left if left is not np.nan else wavelength.min()),
                float(right if right is not np.nan else wavelength.max())
            )
            with self.subTest(tr=tr):
                self.assertEqual(tr.half_range, half_range)

    def test_FWHM(self) -> None:
        for tr in self.trs:
            left, right = tr.half_range
            FWHM = abs(right - left)
            with self.subTest(tr=tr):
                self.assertEqual(tr.FWHM, FWHM)


class TestTimeResolved_smoothed_intensity(TestCase):

    def setUp(self) -> None:
        self.tr = pl.TimeResolved(get_data(), (0.0, 7.0))

    def _test(self, window: int = 5) -> None:
        pdt.assert_series_equal(
            self.tr.smoothed_intensity(window),
            self.tr.df["intensity"].rolling(window, center=True).mean()
        )

    def test_window(self) -> None:
        windows = [0, 3, 5, 7, 9]
        for window in windows:
            with self.subTest(window=window):
                self._test(window)


class TestWavelengthResolved_property(TestCase):

    def setUp(self) -> None:
        self.wrs = [
            pl.WavelengthResolved(get_data(), (0.0, 5.0)),
            pl.WavelengthResolved(get_data(), (-1.0, 1.0)),
            pl.WavelengthResolved(get_data(), (0.0, 5.0), 0.0),
            pl.WavelengthResolved(get_data(), (0.0, 5.0), 5.0)
        ]

    def test_df(self) -> None:
        for wr in self.wrs:
            df = wr.data.df[wr.data.df["wavelength"].between(*wr.range)] \
                .groupby("time") \
                .sum() \
                .drop("wavelength", axis=1) \
                .reset_index()
            if wr.time_offset == "auto":
                intensity = df["intensity"]
                window, k = 10, 2
                rolling = intensity.rolling(window)
                time_offset = float(
                    df["time"][
                        (intensity > rolling.mean() + k * rolling.std()).shift(-1, fill_value=False)
                    ].min()
                )
            else:
                time_offset = wr.time_offset
            df["time"] -= time_offset
            with self.subTest(wr=wr):
                pdt.assert_frame_equal(wr.df, df)
