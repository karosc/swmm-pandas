"""Tests for `swmm-pandas` InputFile class."""

import datetime
import pathlib
import unittest
from textwrap import dedent

import numpy.testing as nptest
import numpy as np
import pandas as pd
import swmm.pandas.input._section_classes as sc
from swmm.pandas import InputFile, Report
from swmm.toolkit import solver

pd.set_option("future.no_silent_downcasting", True)
_HERE = pathlib.Path(__file__).parent


@unittest.skip("Not Ready")
class InputFileTest(unittest.TestCase):
    def setUp(self):
        self.test_base_model_path = str(_HERE / "data" / "bench_inp.inp")
        self.test_groundwater_model_path = str(_HERE / "data" / "Groundwater_Model.inp")
        self.test_street_model_path = str(_HERE / "data" / "Inlet_Drains_Model.inp")
        self.test_pump_model_path = str(_HERE / "data" / "Pump_Control_Model.inp")
        self.test_drainage_model_path = str(_HERE / "data" / "Site_Drainage_Model.inp")
        self.test_lid_model_path = str(_HERE / "data" / "LID_Model.inp")
        self.test_det_pond_model_path = str(_HERE / "data" / "Detention_Pond_Model.inp")
        self.test_inlet_model_path = str(_HERE / "data" / "Inlet_Drains_Model.inp")
        self.test_divider_model_path = str(_HERE / "data" / "Divider_Example.inp")
        self.test_outlet_model_path = str(_HERE / "data" / "Outlet_Example.inp")

        self.test_base_model = InputFile(self.test_base_model_path)
        self.test_lid_model = InputFile(self.test_lid_model_path)
        self.test_det_pond_model = InputFile(self.test_det_pond_model_path)
        self.test_street_model = InputFile(self.test_street_model_path)
        self.test_site_drainage_model = InputFile(self.test_drainage_model_path)
        self.test_divider_model = InputFile(self.test_divider_model_path)
        self.test_outlet_model = InputFile(self.test_outlet_model_path)

        self.maxDiff = 1_000_000

        # self.test_groundwater_model = InputFile(self.test_groundwater_model_path)
        # self.test_street_model = InputFile(self.test_street_model_path)
        # self.test_pump_model = InputFile(self.test_pump_model_path)
        # self.test_drainage_model = InputFile(self.test_drainage_model_path)

    # def testConduit(self):
    #     inp = self.test_base_model

    def testRemovals(self):
        inp = self.test_base_model
        inp.conduit.drop("COND5")
        inp._sync()
        self.assertRaises(KeyError, inp.inp.conduit.loc["COND5"])
