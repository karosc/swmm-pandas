"""Tests for `swmm-pandas` input class."""

import pathlib
from swmm.pandas import Input
import swmm.pandas.input._section_classes as sc
import unittest
import datetime
import pandas as pd
import numpy.testing as nptest
from textwrap import dedent

_HERE = pathlib.Path(__file__).parent


class InuptTest(unittest.TestCase):
    def setUp(self):
        self.test_base_model_path = str(_HERE / "data" / "Model.inp")
        self.test_groundwater_model_path = str(_HERE / "data" / "Groundwater_Model.inp")
        self.test_street_model_path = str(_HERE / "data" / "Inlet_Drains_Model.inp")
        self.test_pump_model_path = str(_HERE / "data" / "Pump_Control_Model.inp")
        self.test_drainage_model_path = str(_HERE / "data" / "Site_Drainage_Model.inp")

        self.test_base_model = Input(self.test_base_model_path)
        # self.test_groundwater_model = Input(self.test_groundwater_model_path)
        # self.test_street_model = Input(self.test_street_model_path)
        # self.test_pump_model = Input(self.test_pump_model_path)
        # self.test_drainage_model = Input(self.test_drainage_model_path)

    def test_title(self):
        inp = self.test_base_model
        self.assertEqual(inp.title, "SWMM is the best!")
        self.assertEqual(
            inp.title.to_swmm_string(), ";;Project Title/Notes\nSWMM is the best!"
        )

    def test_options(self):
        inp = self.test_base_model

        inp.option.loc["ROUTING_STEP", "Value"] = 50
        inp.option.loc["ROUTING_STEP", "desc"] = "Updated routing step"

        self.assertEqual(len(inp.option), 33)
        self.assertEqual(inp.option.index.name, "Option")
        self.assertEqual(
            inp.option.to_swmm_string().split("\n")[-2].strip(),
            "THREADS              4",
        )
        self.assertEqual(
            inp.option.to_swmm_string().split("\n")[21].strip(), ";Updated routing step"
        )
        self.assertEqual(
            inp.option.to_swmm_string().split("\n")[22].strip(),
            "ROUTING_STEP         50",
        )

    def test_report(self):
        inp = self.test_base_model

        self.assertEqual(
            inp.report.NODES,
            [
                "JUNC1",
                "JUNC2",
                "JUNC3",
                "JUNC4",
                "JUNC5",
                "JUNC6",
                "OUT1",
                "OUT2",
                "OUT3",
            ],
        )

        self.assertEqual(inp.report.INPUT, "YES")

        inp.report.LINKS = ["NONE"]
        # check that edit worked
        self.assertEqual(inp.report.to_swmm_string().split("\n")[7], "LINKS NONE")
        # check that input file string limits 5 swmm objects per line
        self.assertEqual(len(inp.report.to_swmm_string().split("\n")[5].split()), 6)

    def test_event(self):
        inp = self.test_base_model

        # check that events are parsed as datetimes
        self.assertIsInstance(inp.event.Start[0], datetime.datetime)
        self.assertIsInstance(inp.event.End[0], datetime.datetime)

        inp.event.loc[0, "Start"] = datetime.datetime(1900, 1, 2)
        inp.event.loc[0, "desc"] = "my first event\nthis is my event"

        # check edit worked
        self.assertEqual(
            inp.event.to_swmm_string().split("\n")[4].split()[0],
            "01/02/1900",
        )

        # check for double line comment description
        self.assertEqual(inp.event.to_swmm_string().split("\n")[2], ";my first event")

        self.assertEqual(inp.event.to_swmm_string().split("\n")[3], ";this is my event")

    def test_files(self):
        # implement better container for files
        ...

    def test_raingages(self):
        inp = self.test_base_model

        self.assertIsInstance(inp.raingage, pd.DataFrame)
        # check some data is loaded
        self.assertEqual(inp.raingage.loc["SCS_Type_III_3in", "Interval"], "0:15")

        # check columns excluded from inp file are added
        nptest.assert_equal(
            inp.raingage.columns.values,
            [
                "Format",
                "Interval",
                "SCF",
                "Source_Type",
                "Source",
                "Station",
                "Units",
                "desc",
            ],
        )

        nptest.assert_equal(
            inp.raingage.loc["RG1"].values,
            [
                "VOLUME",
                "0:05",
                "1.0",
                "FILE",
                "rain.dat",
                "RG1",
                "IN",
                "fake raingage for testing swmm.pandas",
            ],
        )

        nptest.assert_equal(
            inp.raingage.loc["SCS_Type_III_3in"].values,
            ["VOLUME", "0:15", "1.0", "TIMESERIES", "SCS_Type_III_3in", "", "", ""],
        )

        inp.raingage.loc["new_rg"] = [
            "VOLUME",
            "0:5",
            1,
            "FILE",
            "MYFILE",
            "RG1",
            "Inches",
            "my_new_gage",
        ]
        self.assertEqual(
            inp.raingage.to_swmm_string().split("\n")[5].strip(), ";my_new_gage"
        )
        self.assertEqual(
            inp.raingage.to_swmm_string().split("\n")[6].strip(),
            "new_rg            VOLUME  0:5       1    FILE         MYFILE            RG1      Inches",
        )

    def test_evap(self):
        test_cases = [
            dedent(
                """
                    CONSTANT         0.0
                    DRY_ONLY         NO
                """
            ),
            dedent(
                """
                    MONTHLY          1  2  3  4  5  6  7  8  7  6  5  4
                    DRY_ONLY         NO
                    RECOVERY         evap_recovery_pattern
                """
            ),

            dedent(
                """
                    TIMESERIES       evap_timeseries
                    DRY_ONLY         YES
                    RECOVERY         evap_recovery_pattern
                """
            ),
            dedent(
                """
                    TEMPERATURE
                    DRY_ONLY         YES
                    RECOVERY         evap_recovery_pattern
                """
            )
        ]

        for case in test_cases:
            evap = sc.Evap.from_section_text(case)
            self.assertIsInstance(evap,pd.DataFrame)
            self.assertEqual(len(evap.columns),13)
        
        assert 'TEMPERATURE' in evap.index
        evap.drop('TEMPERATURE',inplace=True)
        evap.loc['MONTHLY'] = [''] * evap.shape[1]
        evap.loc['MONTHLY','param1':'param12'] = range(12)
        nptest.assert_equal(
            evap.to_swmm_string().split('\n')[4].split(),
            ['MONTHLY', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        )