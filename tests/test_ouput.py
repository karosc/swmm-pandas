"""Tests for `swmm-pandas` package."""

import importlib
import os
from datetime import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from swmm.toolkit.shared_enum import SubcatchAttribute
from swmm.pandas import Output

_HERE = pathlib.Path(__file__).parent
example_out_path = str(_HERE / "data" / "Model.out")


@pytest.fixture(scope="module")
def outfile():
    out = Output(example_out_path)
    yield out
    out._close()


@pytest.fixture(scope="module")
def preloaded_outfile():
    out = Output(example_out_path, preload=True)
    yield out
    out._close()


def _sorted_export_df(path):
    df = pd.read_parquet(path)
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def _sorted_export_df_fs(path, filesystem):
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq

    arrow_fs = pafs.PyFileSystem(pafs.FSSpecHandler(filesystem))
    arrow_path = (
        f"/{path}"
        if getattr(filesystem, "protocol", None) == "memory"
        and not str(path).startswith("/")
        else path
    )
    df = pq.read_table(arrow_path, filesystem=arrow_fs).to_pandas()
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def _dataset_files(path):
    return sorted(file.relative_to(path).as_posix() for file in path.rglob("*.parquet"))


def test_open_warning(outfile):
    """Test outfile has a pollutant named rainfall. This should raise warning"""
    with pytest.warns(UserWarning):
        outfile._open()


def test_output_props(outfile):
    assert outfile.project_size == [3, 9, 8, 1, 3]
    assert isinstance(outfile._version, int)
    assert outfile.start == datetime(1900, 1, 1, 0, 0)
    assert outfile.end == datetime(1900, 1, 2, 0, 0)
    assert len(outfile.node_attributes) == 9
    assert len(outfile.subcatch_attributes) == 11
    assert len(outfile.link_attributes) == 8
    assert outfile.period == 288
    assert outfile.pollutants == ("groundwater", "pol_rainfall", "sewage")
    assert outfile.report == 300
    assert outfile._unit == (0, 0, 0, 0, 0)


# test series getters
# check values against those validated in EPA SWMM GUI Release 5.1.015
def test_subcatch_series(outfile):
    arr = outfile.subcatch_series(
        ["SUB1", "SUB2"], ["runoff_rate", "pol_rainfall"], asframe=False
    )
    assert type(arr) == np.ndarray
    df = outfile.subcatch_series(["SUB1", "SUB2"], ["runoff_rate", "pol_rainfall"])
    ts = pd.Timestamp("1/1/1900 01:05")
    assert type(df) == pd.DataFrame
    refarray = np.array([[0.005574, 100], [0.021814, 100]])
    assert np.allclose(
        df.loc[[(ts, "SUB1"), (ts, "SUB2")], :].to_numpy(), refarray, atol=0.000001
    )


def test_node_series(outfile):
    arr = outfile.node_series(
        ["JUNC3", "JUNC4"], ["invert_depth", "sewage"], asframe=False
    )
    assert type(arr) == np.ndarray
    df = outfile.node_series(["JUNC3", "JUNC4"], ["invert_depth", "sewage"])
    ts = pd.Timestamp("1/1/1900 01:05")
    assert type(df) == pd.DataFrame
    refarray = np.array([[0.632757, 82.557610], [0.840164, 82.403465]])
    assert np.allclose(
        df.loc[[(ts, "JUNC3"), (ts, "JUNC4")], :].to_numpy(), refarray, atol=0.000001
    )


def test_link_series(outfile):
    arr = outfile.link_series(
        ["COND4", "PUMP1"], ["flow_rate", "groundwater"], asframe=False
    )
    assert type(arr) == np.ndarray
    df = outfile.link_series("PUMP1", ["flow_rate", "groundwater"])
    ts = pd.Timestamp("1/1/1900 01:05")
    assert type(df) == pd.DataFrame
    refarray = np.array([1.03671658, 10.87113953])
    assert np.allclose(df.loc[ts, :].to_numpy(), refarray, atol=0.000001)


def test_system_series(outfile):
    arr = outfile.system_series(["gw_inflow", "flood_losses"], asframe=False)
    assert type(arr) == np.ndarray
    df = outfile.system_series(["gw_inflow", "flood_losses"])
    ts = pd.Timestamp("1/1/1900 13:30")
    assert type(df) == pd.DataFrame
    refarray = np.array([0.15494138, 3.97151256])
    assert np.allclose(df.loc[ts].to_numpy(), refarray, atol=0.000001)


# test attribute getters
# check values against those validated in EPA SWMM GUI Release 5.1.015
def test_subcatch_attribute(outfile):
    ts = pd.Timestamp("1/1/1900 13:30")
    arr = outfile.subcatch_attribute(ts, None, asframe=False)
    assert type(arr) == np.ndarray
    df = outfile.subcatch_attribute(ts, None)
    assert df.shape == (3, 11)
    refarray = np.array(
        [
            0.156000,
            0.000000,
            0.000000,
            0.324994,
            2.800647,
            0.115297,
            -3.141794,
            0.280193,
            0.000000,
            100.000000,
            0.000000,
        ]
    )
    assert np.allclose(df.loc["SUB3"].to_numpy(), refarray, atol=0.000001)


def test_node_attribute(outfile):
    ts = pd.Timestamp("1/1/1900 13:30")
    arr = outfile.node_attribute(ts, None, asframe=False)
    assert type(arr) == np.ndarray
    df = outfile.node_attribute(ts, None)
    assert df.shape == (9, 9)
    refarray = np.array(
        [
            13.39598274,
            9.92598152,
            9938.31542969,
            0.0,
            3.40221405,
            0.94061023,
            0.51279306,
            95.54773712,
            3.84922433,
        ]
    )
    assert np.allclose(df.loc["JUNC3"].to_numpy(), refarray, atol=0.000001)


def test_link_attribute(outfile):
    ts = pd.Timestamp("1/1/1900 13:30")
    arr = outfile.link_attribute(ts, None, asframe=False)
    assert type(arr) == np.ndarray
    df = outfile.link_attribute(ts, None)
    assert df.shape == (8, 8)
    refarray = np.array(
        [
            8.96449947,
            1.5,
            3.54724121,
            1851.10754395,
            1.0,
            0.81769407,
            93.12129211,
            5.93402863,
        ]
    )
    assert np.allclose(df.loc["COND4"].to_numpy(), refarray, atol=0.000001)


def test_system_attribute(outfile):
    ts = pd.Timestamp("1/1/1900 13:30")
    arr = outfile.system_attribute(ts, "rainfall", asframe=False)
    assert type(arr) == np.ndarray
    df = outfile.system_attribute(ts, "rainfall")
    assert df.shape == (1, 1)
    assert round(df.loc["rainfall", "result"], 2) == 0.16


# test result getters
# check values against those validated in EPA SWMM GUI Release 5.1.015
def test_subcatch_result(outfile):
    ts = pd.Timestamp("1/1/1900 13:30")
    arr = outfile.subcatch_result("SUB3", ts, asframe=False)
    assert type(arr) == np.ndarray
    df = outfile.subcatch_result("SUB3", ts)
    assert df.shape == (1, 11)
    refarray = refarray = np.array(
        [
            0.156000,
            0.000000,
            0.000000,
            0.324994,
            2.800647,
            0.115297,
            -3.141794,
            0.280193,
            0.000000,
            100.000000,
            0.000000,
        ]
    )
    assert np.allclose(df.to_numpy(), refarray, atol=0.000001)


def test_node_result(outfile):
    ts = pd.Timestamp("1/1/1900 13:30")
    ts2 = pd.Timestamp("1/1/1900 15:30")
    arr = outfile.node_result("JUNC3", [ts, ts2], asframe=False)
    assert type(arr) == np.ndarray
    df = outfile.node_result("JUNC3", [ts, ts2])
    assert df.shape == (2, 9)
    refarray = np.array(
        [
            [
                13.39598274,
                9.92598152,
                9938.31542969,
                0.0,
                3.40221405,
                0.94061023,
                0.51279306,
                95.54773712,
                3.84922433,
            ],
            [
                3.59898233,
                0.12898226,
                0.0,
                0.0,
                1.1946373,
                0.0,
                4.38057995,
                76.85120392,
                18.56258392,
            ],
        ]
    )
    assert np.allclose(df.to_numpy(), refarray, atol=0.000001)


def test_link_result(outfile):
    ts = 161
    arr = outfile.link_result(["COND4", "COND2"], ts, asframe=False)
    assert type(arr) == np.ndarray
    df = outfile.link_result(["COND4", "COND2"], ts)
    assert df.shape == (2, 8)
    refarray = np.array(
        [
            [
                8.96449947,
                1.5,
                3.54724121,
                1851.10754395,
                1.0,
                0.81769407,
                93.12129211,
                5.93402863,
            ],
            [
                -2.46341562,
                0.75,
                -3.20461726,
                460.85583496,
                1.0,
                0.4940097,
                95.69831848,
                3.72042918,
            ],
        ]
    )
    assert np.allclose(df.to_numpy(), refarray, atol=0.000001)


def test_system_result(outfile):
    ts = pd.Timestamp("1/1/1900 13:30")
    arr = outfile.system_result(ts, asframe=False)
    assert type(arr) == np.ndarray
    df = outfile.system_result(ts)
    assert df.shape == (15, 1)
    refarray = np.array(
        [
            70.0,
            0.15599999,
            0.0,
            0.2558046,
            3.69547844,
            1.00800002,
            0.15494138,
            0.0,
            0.0,
            4.8584199,
            3.97151256,
            9.42725468,
            56644.5625,
            0.0,
            0.0,
        ]
    )

    assert np.allclose(df.result.to_numpy(), refarray, atol=0.000001)


@pytest.mark.parametrize(
    "inputAttribute,inputValidAttribute,expectedAttribute,expectedIndex,out",
    [
        (
            "rainfall",
            SubcatchAttribute,
            ["rainfall"],
            [SubcatchAttribute["RAINFALL"]],
            "outfile",
        ),
        (
            ["rainfall", 3],
            SubcatchAttribute,
            ["rainfall", "infil_loss"],
            [SubcatchAttribute["RAINFALL"], SubcatchAttribute["INFIL_LOSS"]],
            "outfile",
        ),
        (
            ["rainfall", 3, SubcatchAttribute["SOIL_MOISTURE"]],
            SubcatchAttribute,
            ["rainfall", "infil_loss", "soil_moisture"],
            [
                SubcatchAttribute["RAINFALL"],
                SubcatchAttribute["INFIL_LOSS"],
                SubcatchAttribute["SOIL_MOISTURE"],
            ],
            "outfile",
        ),
    ],
)
def test_validateAttribute(
    inputAttribute,
    inputValidAttribute,
    expectedAttribute,
    expectedIndex,
    out,
    request,
):
    outfile = request.getfixturevalue(out)
    attributeArray, attributeIndexArray = outfile._validateAttribute(
        inputAttribute, inputValidAttribute
    )
    assert attributeArray == expectedAttribute
    assert attributeIndexArray == expectedIndex


@pytest.mark.parametrize(
    "inputElement,inputValidElements,expectedElement,expectedElementIndex,out",
    [
        (
            "COND1",
            ("COND1", "COND2", "COND3"),
            ["COND1"],
            [0],
            "outfile",
        ),
        (
            ["COND3", 0],
            ("COND1", "COND2", "COND3"),
            ["COND3", "COND1"],
            [2, 0],
            "outfile",
        ),
        (
            None,
            ("COND1", "COND2", "COND3"),
            ["COND1", "COND2", "COND3"],
            [0, 1, 2],
            "outfile",
        ),
    ],
)
def test_validateElement(
    inputElement,
    inputValidElements,
    expectedElement,
    expectedElementIndex,
    out,
    request,
):
    outfile = request.getfixturevalue(out)
    elementArray, elementIndexArray = outfile._validateElement(
        inputElement, inputValidElements
    )
    assert elementArray == expectedElement
    assert elementIndexArray == expectedElementIndex


def test_to_parquet_streamed_full_export(outfile, tmp_path):
    path = tmp_path / "streamed.parquet"
    returned_path = outfile.to_parquet(path)
    assert returned_path == str(path)

    df = _sorted_export_df(path)
    assert df.shape == (55584, 5)
    assert df.columns.tolist() == [
        "time",
        "element_type",
        "element_name",
        "attribute",
        "value",
    ]

    first_row = df.iloc[0]
    assert first_row["time"] == pd.Timestamp("1900-01-01 00:05:00")
    assert first_row["element_type"] == "link"
    assert first_row["element_name"] == "COND1"
    assert first_row["attribute"] == "capacity"


def test_to_parquet_preloaded_matches_streamed(outfile, preloaded_outfile, tmp_path):
    streamed_path = tmp_path / "streamed.parquet"
    preloaded_path = tmp_path / "preloaded.parquet"

    outfile.to_parquet(streamed_path, row_batch_size=17)
    preloaded_outfile.to_parquet(preloaded_path, row_batch_size=17)

    streamed = _sorted_export_df(streamed_path)
    preloaded = _sorted_export_df(preloaded_path)
    assert_frame_equal(streamed, preloaded)


def test_to_parquet_filtered_export(outfile, tmp_path):
    path = tmp_path / "filtered.parquet"
    outfile.to_parquet(
        path,
        link_attributes=["flow_rate", "flow_depth"],
        links=["COND4"],
        node_attributes=[],
        subcatchment_attributes=[],
        system_attributes=[],
    )

    df = _sorted_export_df(path)
    assert df["element_type"].unique().tolist() == ["link"]
    assert df["element_name"].unique().tolist() == ["COND4"]
    assert set(df["attribute"].unique()) == {"flow_rate", "flow_depth"}
    assert len(df) == 288 * 2


def test_to_parquet_empty_export(outfile, tmp_path):
    path = tmp_path / "empty.parquet"
    outfile.to_parquet(
        path,
        link_attributes=[],
        node_attributes=[],
        subcatchment_attributes=[],
        system_attributes=[],
    )

    df = pd.read_parquet(path)
    assert df.empty
    assert df.columns.tolist() == [
        "time",
        "element_type",
        "element_name",
        "attribute",
        "value",
    ]


def test_to_parquet_row_batch_size_one(outfile, tmp_path):
    path = tmp_path / "batch-one.parquet"
    outfile.to_parquet(
        path,
        node_attributes=["invert_depth"],
        nodes=["JUNC3"],
        link_attributes=[],
        subcatchment_attributes=[],
        system_attributes=[],
        row_batch_size=1,
    )

    df = _sorted_export_df(path)
    assert len(df) == 288
    assert df["attribute"].unique().tolist() == ["invert_depth"]
    assert df["element_name"].unique().tolist() == ["JUNC3"]


def test_to_parquet_with_fsspec_filesystem(outfile):
    fsspec = pytest.importorskip("fsspec")
    filesystem = fsspec.filesystem("memory")
    path = "container/streamed.parquet"

    returned_path = outfile.to_parquet(path, filesystem=filesystem)

    assert returned_path == path
    assert filesystem.exists(path)

    df = _sorted_export_df_fs(path, filesystem)
    assert df.shape == (55584, 5)
    assert df.columns.tolist() == [
        "time",
        "element_type",
        "element_name",
        "attribute",
        "value",
    ]


@pytest.mark.parametrize(
    "partition_freq,expected_columns",
    [
        ("D", ["year", "month", "day"]),
        ("MS", ["year", "month"]),
        ("H", ["year", "month", "day", "hour"]),
    ],
)
def test_to_parquet_partition_columns(outfile, tmp_path, partition_freq, expected_columns):
    path = tmp_path / f"partition-{partition_freq.lower()}"
    outfile.to_parquet(
        path,
        node_attributes=["invert_depth"],
        nodes=["JUNC3"],
        link_attributes=[],
        subcatchment_attributes=[],
        system_attributes=[],
        partition_freq=partition_freq,
    )

    df = _sorted_export_df(path)
    assert df.columns.tolist() == [
        "time",
        "element_type",
        "element_name",
        "attribute",
        "value",
        *expected_columns,
    ]

    files = _dataset_files(path)
    assert files
    assert all(file.split("/")[-1].count("_") == 1 for file in files)
    assert all(file.endswith(".parquet") for file in files)


def test_to_parquet_partitioned_with_fsspec_filesystem(outfile):
    fsspec = pytest.importorskip("fsspec")
    filesystem = fsspec.filesystem("memory")
    path = "container/partitioned"

    returned_path = outfile.to_parquet(
        path,
        node_attributes=["invert_depth"],
        nodes=["JUNC3"],
        link_attributes=[],
        subcatchment_attributes=[],
        system_attributes=[],
        partition_freq="D",
        row_batch_size=100,
        filesystem=filesystem,
    )

    assert returned_path == path
    files = sorted(
        file.lstrip("/")
        for file in filesystem.find(path)
        if file.endswith(".parquet")
    )
    assert files == [
        "container/partitioned/year=1900/month=1/day=1/19000101000500_19000101082500.parquet",
        "container/partitioned/year=1900/month=1/day=1/19000101082500_19000101164500.parquet",
        "container/partitioned/year=1900/month=1/day=1/19000101164500_19000102000000.parquet",
        "container/partitioned/year=1900/month=1/day=2/19000102000000_19000102000500.parquet",
    ]

    df = _sorted_export_df_fs(path, filesystem)
    assert df.columns.tolist() == [
        "time",
        "element_type",
        "element_name",
        "attribute",
        "value",
        "year",
        "month",
        "day",
    ]


def test_to_parquet_partitioned_filenames_and_overwrite(outfile, tmp_path):
    path = tmp_path / "overwrite-daily"

    outfile.to_parquet(
        path,
        node_attributes=["invert_depth"],
        nodes=["JUNC3"],
        link_attributes=[],
        subcatchment_attributes=[],
        system_attributes=[],
        partition_freq="D",
        row_batch_size=100,
    )
    first_files = _dataset_files(path)

    outfile.to_parquet(
        path,
        node_attributes=["invert_depth"],
        nodes=["JUNC3"],
        link_attributes=[],
        subcatchment_attributes=[],
        system_attributes=[],
        partition_freq="D",
        row_batch_size=100,
    )
    second_files = _dataset_files(path)

    assert second_files == first_files
    assert first_files == [
        "year=1900/month=1/day=1/19000101000500_19000101082500.parquet",
        "year=1900/month=1/day=1/19000101082500_19000101164500.parquet",
        "year=1900/month=1/day=1/19000101164500_19000102000000.parquet",
        "year=1900/month=1/day=2/19000102000000_19000102000500.parquet",
    ]


def test_to_parquet_partitioned_preloaded_matches_streamed(
    outfile,
    preloaded_outfile,
    tmp_path,
):
    streamed_path = tmp_path / "streamed-daily"
    preloaded_path = tmp_path / "preloaded-daily"

    outfile.to_parquet(
        streamed_path,
        link_attributes=["flow_rate", "flow_depth"],
        links=["COND4"],
        node_attributes=[],
        subcatchment_attributes=[],
        system_attributes=[],
        partition_freq="D",
        row_batch_size=13,
    )
    preloaded_outfile.to_parquet(
        preloaded_path,
        link_attributes=["flow_rate", "flow_depth"],
        links=["COND4"],
        node_attributes=[],
        subcatchment_attributes=[],
        system_attributes=[],
        partition_freq="D",
        row_batch_size=13,
    )

    assert_frame_equal(_sorted_export_df(streamed_path), _sorted_export_df(preloaded_path))


def test_to_parquet_invalid_partition_freq(outfile, tmp_path):
    with pytest.raises(ValueError, match="partition_freq"):
        outfile.to_parquet(tmp_path / "invalid", partition_freq="6H")


def test_to_parquet_missing_pyarrow(outfile, tmp_path, monkeypatch):
    original_import_module = importlib.import_module

    def _raise_for_pyarrow(name, package=None):
        if name.startswith("pyarrow"):
            raise ImportError("missing pyarrow")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _raise_for_pyarrow)

    with pytest.raises(ImportError, match="pyarrow is required"):
        outfile.to_parquet(tmp_path / "missing.parquet")


# def test_elementIndex(
#     elementID, indexSquence, elementType, expectedIndex, out, request
# ):
#     outfile = request.getfixturevalue(out)
#     assert outfile._elementIndex(elementID,indexSquence) == expectedIndex
