"""Tests for `swmm-pandas` package."""

import os
from datetime import datetime
import pathlib

import numpy as np
import pandas as pd
import pytest

from swmm.toolkit.shared_enum import SubcatchAttribute
from swmm.pandas import Output

_HERE = pathlib.Path(__file__).parent
test_out_path = str(_HERE / "data" / "Model.out")


@pytest.fixture(scope="module")
def outfile():
    out = Output(test_out_path)
    yield out
    out._close()


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


# def test_elementIndex(
#     elementID, indexSquence, elementType, expectedIndex, out, request
# ):
#     outfile = request.getfixturevalue(out)
#     assert outfile._elementIndex(elementID,indexSquence) == expectedIndex
