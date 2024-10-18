"""Tests for `swmm-pandas` input class."""

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


class InputTest(unittest.TestCase):
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

    def test_title(self):
        inp = self.test_base_model
        self.assertEqual(inp.title, "SWMM is the best!")
        self.assertEqual(
            inp.title.to_swmm_string(),
            ";;Project Title/Notes\nSWMM is the best!",
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
            inp.option.to_swmm_string().split("\n")[21].strip(),
            ";Updated routing step",
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

        inp = self.test_lid_model
        self.assertEqual(
            len(inp.report.LID),
            3,
        )
        self.assertEqual(
            inp.report.LID[0].Name,
            "Planters",
        )

        self.assertEqual(
            inp.report.LID[1].Subcatch,
            "S1",
        )

        self.assertEqual(
            inp.report.LID[1].Fname,
            "S1_lid_it.rpt",
        )

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
            inp.raingage.loc["RG1"].tolist(),
            [
                "VOLUME",
                "0:05",
                1.0,
                "FILE",
                "rain.dat",
                "RG1",
                "IN",
                "fake raingage for testing swmm.pandas",
            ],
        )

        nptest.assert_equal(
            inp.raingage.loc["SCS_Type_III_3in"].tolist(),
            ["VOLUME", "0:15", 1.0, "TIMESERIES", "SCS_Type_III_3in", "", "", ""],
        )

        inp.raingage.loc["new_rg"] = [
            "VOLUME",
            "0:5",
            1.0,
            "FILE",
            "MYFILE",
            "RG1",
            "Inches",
            "my_new_gage",
        ]
        self.assertEqual(
            inp.raingage.to_swmm_string().split("\n")[8].strip(),
            ";my_new_gage",
        )
        self.assertEqual(
            inp.raingage.to_swmm_string().split("\n")[9].strip(),
            "new_rg            VOLUME  0:5       1.0  FILE         MYFILE            RG1      Inches",
        )

    def test_evap(self):
        test_cases = [
            dedent(
                """
                    CONSTANT         0.0
                    DRY_ONLY         NO
                """,
            ),
            dedent(
                """
                    MONTHLY          1  2  3  4  5  6  7  8  7  6  5  4
                    DRY_ONLY         NO
                    RECOVERY         evap_recovery_pattern
                """,
            ),
            dedent(
                """
                    TIMESERIES       evap_timeseries
                    DRY_ONLY         YES
                    RECOVERY         evap_recovery_pattern
                """,
            ),
            dedent(
                """
                    TEMPERATURE
                    DRY_ONLY         YES
                    RECOVERY         evap_recovery_pattern
                """,
            ),
        ]

        for case in test_cases:
            evap = sc.Evap.from_section_text(case)
            self.assertIsInstance(evap, pd.DataFrame)
            self.assertEqual(len(evap.columns), 13)

        assert "TEMPERATURE" in evap.index
        evap.drop("TEMPERATURE", inplace=True)
        evap.loc["MONTHLY"] = [""] * evap.shape[1]
        evap.loc["MONTHLY", "param1":"param12"] = range(12)
        nptest.assert_equal(
            evap.to_swmm_string().split("\n")[4].split(),
            ["MONTHLY", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
        )

    def test_temperature(self):
        inp = self.test_base_model

        # assert type and shape
        self.assertIsInstance(inp.temperature, pd.DataFrame)
        self.assertEqual(
            len(inp.temperature.columns),
            14,
        )
        nptest.assert_equal(
            inp.temperature.index.values,
            ["TIMESERIES", "WINDSPEED", "SNOWMELT", "ADC", "ADC"],
        )

        # assert modifications to df end up in swmm string
        inp.temperature.drop("WINDSPEED", inplace=True)
        inp.temperature.loc["WINDSPEED", "param1"] = "FILE"
        inp.temperature.loc["FILE", "param1":"param3"] = [
            "./climate.dat",
            "1/1/1900",
            "F",
        ]

        self.assertEqual(
            inp.temperature.to_swmm_string().split("\n")[6].strip(),
            "WINDSPEED   FILE",
        )
        self.assertEqual(
            inp.temperature.to_swmm_string().split("\n")[7].strip(),
            "FILE        ./climate.dat  1/1/1900  F",
        )

    def test_adjustments(self):
        inp = self.test_base_model

        # assert type and shape
        self.assertIsInstance(inp.adjustments, pd.DataFrame)
        nptest.assert_equal(
            inp.adjustments.columns.values,
            [
                "Subcatchment",
                "Pattern",
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
                "desc",
            ],
        )

        # test edits make it into swmm string
        inp.adjustments.loc["EVAPORATION", "May":"Nov"] = np.nan
        inp.adjustments.loc["EVAPORATION", "Jan":"Dec"] = (
            inp.adjustments.loc["EVAPORATION", "Jan":"Dec"].astype(float).interpolate()
        )

        self.assertEqual(
            inp.adjustments.to_swmm_string(),
            dedent(
                """\
                    ;;Parameter   Subcatchment  Pattern  Jan  Feb  Mar   Apr   May     Jun    Jul     Aug   Sep     Oct    Nov    Dec  
                    ;;----------  ------------  -------  ---  ---  ----  ----  ------  -----  ------  ----  ------  -----  -----  ---  
                    TEMPERATURE                          1    2    -3    -4    3       5      0.875   -10   9       1      -0.5   1    
                    EVAPORATION                          1.0  2.0  -3.0  -4.0  -3.375  -2.75  -2.125  -1.5  -0.875  -0.25  0.375  1.0  
                    RAINFALL                             1.0  2    -3    -4    3       5      0.875   -10   9       1.0    -0.5   1.0  
                    CONDUCTIVITY                         1.0  2    -3    -4    3       5      0.875   -10   9       1.0    -0.5   1.0  
                    N-PERV        SUB3          Monthly                                                                                
                    DSTORE        SUB3          Monthly                                                                                
                """,
            ),
        )

    def test_subcatchments(self) -> None:
        inp = self.test_base_model

        self.assertIsInstance(inp.subcatchment, pd.DataFrame)
        self.assertEqual(inp.subcatchment.shape, (3, 9))
        nptest.assert_equal(
            inp.subcatchment.columns.values,
            [
                "RainGage",
                "Outlet",
                "Area",
                "PctImp",
                "Width",
                "Slope",
                "CurbLeng",
                "SnowPack",
                "desc",
            ],
        )
        inp.subcatchment["Width"] = (inp.subcatchment.Area**0.6).astype(float).round(3)
        self.assertEqual(
            inp.subcatchment.to_swmm_string(),
            dedent(
                """\
                ;;Name  RainGage          Outlet  Area  PctImp  Width  Slope  CurbLeng  SnowPack  
                ;;----  ----------------  ------  ----  ------  -----  -----  --------  --------  
                SUB1    SCS_Type_III_3in  JUNC1   5     30.83   2.627  0.5    0         SNOW1     
                SUB2    SCS_Type_III_3in  JUNC2   17    40.74   5.474  0.5    0         SNOW1     
                SUB3    SCS_Type_III_3in  JUNC4   38    62.21   8.869  0.5    0         SNOW1     
                """,
            ),
        )

    def test_subareas(self) -> None:
        inp = self.test_base_model

        self.assertIsInstance(inp.subarea, pd.DataFrame)
        self.assertEqual(inp.subarea.shape, (3, 8))
        nptest.assert_equal(
            inp.subarea.columns.values,
            [
                "Nimp",
                "Nperv",
                "Simp",
                "Sperv",
                "PctZero",
                "RouteTo",
                "PctRouted",
                "desc",
            ],
        )
        self.maxDiff = 9999
        inp.subarea.loc["SUB3", "PctRouted"] = 20
        self.assertEqual(
            inp.subarea.to_swmm_string(),
            dedent(
                """\
                    ;;Subcatchment  Nimp  Nperv  Simp  Sperv  PctZero  RouteTo   PctRouted  
                    ;;------------  ----  -----  ----  -----  -------  --------  ---------  
                    SUB1            0.05  0.2    0.05  0.1    25       PERVIOUS  50         
                    SUB2            0.05  0.2    0.05  0.1    25       PERVIOUS  50         
                    SUB3            0.05  0.2    0.05  0.1    25       PERVIOUS  20         
                """,
            ),
        )

    def test_infil(self) -> None:
        inp = self.test_base_model

        self.assertIsInstance(inp.infil, pd.DataFrame)
        self.assertEqual(inp.infil.shape, (3, 7))
        nptest.assert_equal(
            inp.infil.columns.values,
            [
                "param1",
                "param2",
                "param3",
                "param4",
                "param5",
                "Method",
                "desc",
            ],
        )

        # test differing methods are parsed
        nptest.assert_equal(
            inp.infil.values,
            np.array(
                [
                    [4.3, 0.86, 0.23, "", "", "", ""],
                    [4.3, 0.86, 0.23, "", "", "MODIFIED_GREEN_AMPT", ""],
                    [4.3, 0.86, 0.23, 0.04, 2, "HORTON", ""],
                ],
                dtype=object,
            ),
        )

        # test assignment
        inp.infil.loc["FAKE_SUB"] = [0, 0, 0, 0, 0, "", "This is fake"]
        self.assertEqual(
            inp.infil.to_swmm_string(),
            dedent(
                """\
                    ;;Subcatchment  param1  param2  param3  param4  param5  Method               
                    ;;------------  ------  ------  ------  ------  ------  -------------------  
                    SUB1            4.3     0.86    0.23                                         
                    SUB2            4.3     0.86    0.23                    MODIFIED_GREEN_AMPT  
                    SUB3            4.3     0.86    0.23    0.04    2       HORTON               
                    ;This is fake
                    FAKE_SUB        0.0     0.0     0.0     0       0                            
                """,
            ),
        )

    def test_lid_control(self) -> None:
        inp = self.test_lid_model
        self.assertIsInstance(inp.lid_control, pd.DataFrame)

        self.assertEqual(
            inp.lid_control.shape,
            (27, 9),
        )

        nptest.assert_equal(
            inp.lid_control.index.unique().to_numpy(),
            [
                "GreenRoof",
                "PorousPave",
                "Planters",
                "InfilTrench",
                "RainBarrels",
                "Swale",
            ],
        )

    def test_lid_usage(self) -> None:
        inp = self.test_lid_model
        self.assertIsInstance(
            inp.lid_usage,
            pd.DataFrame,
        )

        self.assertEqual(
            inp.lid_usage.reset_index().shape,
            (8, 12),
        )

        inp.lid_usage.loc[(slice(None), "Swale"), "Width"] = 100
        inp.lid_usage.loc[(slice(None), "Swale"), "desc"] = "Update width"

        self.assertMultiLineEqual(
            inp.lid_usage.to_swmm_string(),
            dedent(
                """\
                    ;;Subcatchment  LIDProcess   Number  Area      Width  InitSat  FromImp  ToPerv  RptFile  DrainTo  FromPerv  
                    ;;------------  -----------  ------  --------  -----  -------  -------  ------  -------  -------  --------  
                    S1              InfilTrench  4       532.0     133    0        40       0       *        *        0         
                    S1              RainBarrels  32      5.0       0      0        17       1       *        *        0         
                    S4              Planters     30      500.0     0      0        80       0       *        *        0         
                    S5              PorousPave   1       232872.0  683    0        0        0       *        *        0         
                    S5              GreenRoof    1       18400.0   136    0        0        0       *        *        0         
                    ;Update width
                    Swale3          Swale        1       14374.8   100    0        0        0       *        *        0         
                    ;Update width
                    Swale4          Swale        1       21780.0   100    0        0        0       *        *        0         
                    ;Update width
                    Swale6          Swale        1       17859.6   100    0        0        0       *        *        0         
                """,
            ),
        )

    def test_aquifers(self) -> None:
        inp = self.test_base_model

        self.assertEqual(
            inp.aquifer.shape,
            (3, 14),
        )

        inp.aquifer.loc["SUB3", "FC"] = 10

        self.assertMultiLineEqual(
            inp.aquifer.to_swmm_string(),
            dedent(
                """\
                    ;;Name  Por   WP    FC    Ksat  Kslope  Tslope  ETu  ETs  Seep  Ebot    Egw    Umc   ETupat  
                    ;;----  ----  ----  ----  ----  ------  ------  ---  ---  ----  ------  -----  ----  ------  
                    SUB1    0.46  0.13  0.28  0.8   5       20      0.7  10   0     -39.3   1.5    0.23          
                    SUB2    0.46  0.13  0.28  0.8   5       20      0.7  10   0     -36.75  4.5    0.23          
                    SUB3    0.46  0.13  10.0  0.8   5       20      0.7  10   0     -4.53   36.57  0.23          
                """,
            ),
        )

    def test_groundwater(self) -> None:
        inp = self.test_base_model

        self.assertEqual(
            inp.groundwater.shape,
            (3, 14),
        )

        inp.groundwater.loc[:, "Egwt"] = 100
        inp.groundwater.loc[:, "desc"] = "update Egwt"

        self.assertMultiLineEqual(
            inp.groundwater.to_swmm_string(),
            dedent(
                """\
                    ;;Subcatchment  Aquifer  Node   Esurf  A1     B1   A2  B2  A3  Dsw  Egwt  Ebot    Wgr     Umc    
                    ;;------------  -------  -----  -----  -----  ---  --  --  --  ---  ----  ------  ------  -----  
                    ;update Egwt
                    SUB1            SUB1     JUNC1  10.7   0.001  1.5  0   0   0   0    100   -39.3   2.521   0.276  
                    ;update Egwt
                    SUB2            SUB2     JUNC2  5.16   0.001  1.5  0   0   0   0    100   -44.84  -0.029  0.275  
                    ;update Egwt
                    SUB3            SUB3     JUNC4  8.55   0.001  1.5  0   0   0   0    100   -41.45  -3.616  0.279  
                """,
            ),
        )

    def test_gwf(self) -> None:
        inp = self.test_base_model

        self.assertEqual(
            inp.gwf.shape,
            (2, 2),
        )

        inp.gwf.loc[("SUB3", "LATERAL"), :] = [
            "0.001*Hgw + 0.05*(Hgw-5)*STEP(Hgw-5)",
            "add gwf for SUB3",
        ]

        self.assertMultiLineEqual(
            inp.gwf.to_swmm_string(),
            dedent(
                """\
                    ;;Subcatch  Type     Expr                                  
                    ;;--------  -------  ------------------------------------  
                    SUB1        LATERAL  0.001*Hgw+0.05*(Hgw-5)*STEP(Hgw-5)    
                    SUB2        DEEP     0.002                                 
                    ;add gwf for SUB3
                    SUB3        LATERAL  0.001*Hgw + 0.05*(Hgw-5)*STEP(Hgw-5)  
                """,
            ),
        )

    def test_snowpacks(self):
        inp = self.test_base_model

        self.assertEqual(
            inp.snowpack.reset_index().shape,
            (4, 10),
        )

        inp.snowpack.loc[("SNOW1", "REMOVAL"), "param1"] = 4
        inp.snowpack.loc[("SNOW1", "REMOVAL"), "desc"] = "Update plow depth"

        self.assertMultiLineEqual(
            inp.snowpack.to_swmm_string(),
            dedent(
                """\
                    ;;Name  Surface     param1  param2  param3  param4  param5  param6  param7  
                    ;;----  ----------  ------  ------  ------  ------  ------  ------  ------  
                    SNOW1   PLOWABLE    0.005   0.007   24.0    0.2     0.0     0.0     0.1     
                    SNOW1   IMPERVIOUS  0.005   0.007   24.0    0.2     0.0     0.0     2.0     
                    SNOW1   PERVIOUS    0.004   0.004   25.0    0.2     0.0     0.0     2.0     
                    ;Update plow depth
                    SNOW1   REMOVAL     4.0     0.0     0.0     1.0     0.0     0.0             
                """,
            ),
        )

    def test_junctions(self):
        inp = self.test_base_model
        self.assertEqual(
            inp.junc.reset_index().shape,
            (5, 7),
        )

        inp.junc.loc["JUNC4", "Elevation"] -= 5
        inp.junc.loc["JUNC4", "desc"] = "dropped invert 5ft"

        self.assertMultiLineEqual(
            inp.junc.to_swmm_string(),
            dedent(
                """\
                    ;;Name  Elevation  MaxDepth  InitDepth  SurDepth  Aponded  
                    ;;----  ---------  --------  ---------  --------  -------  
                    JUNC1   1.5        10.25     0          0         5000     
                    JUNC2   -1.04      6.2       0          0         5000     
                    JUNC3   -3.47      11.5      0          0         5000     
                    ;dropped invert 5ft
                    JUNC4   -10.25     13.8      0          0         5000     
                    JUNC6   0.0        9.0       0          200       0        
                """,
            ),
        )

    def test_outfalls(self):
        inp = self.test_base_model
        self.assertEqual(
            inp.outfall.reset_index().shape,
            (3, 7),
        )

        inp.outfall.loc["OUT1", "TYPE"] = "NORMAL"
        inp.outfall.loc["OUT1", "desc"] = "changed to normal outfall"

        self.assertMultiLineEqual(
            inp.outfall.to_swmm_string(),
            dedent(
                """\
                    ;;Name  Elevation  Type        StageData    Gated  RouteTo  
                    ;;----  ---------  ----------  -----------  -----  -------  
                    ;changed to normal outfall
                    OUT1    0.1        FREE                     NO              
                    OUT2    -1.04      FREE                     NO              
                    OUT3    0.0        TIMESERIES  head_series  YES    SUB1     
                """
            ),
        )

    def test_storage(self):
        inp = self.test_base_model
        self.assertEqual(
            inp.storage.reset_index().shape,
            (2, 15),
        )

        inp.storage.loc["STOR1", "A1_L"] = 200
        inp.storage.loc["STOR1", "desc"] = "shrunk store"

        self.assertMultiLineEqual(
            inp.storage.to_swmm_string(),
            dedent(
                """\
                    ;;Name  Elev    MaxDepth  InitDepth  Shape       CurveName  A1_L  A2_W  A0_Z  SurDepth  Fevap  Psi  Ksat  IMD  
                    ;;----  ------  --------  ---------  ----------  ---------  ----  ----  ----  --------  -----  ---  ----  ---  
                    JUNC5   -6.5    13.2      0          TABULAR     Store1                       0         2      2    2     0.5  
                    ;shrunk store
                    STOR1   -15.25  21.75     0          FUNCTIONAL             200   1     2     10        3                      
                """
            ),
        )

    def test_conduit(self):
        inp = self.test_base_model
        self.assertEqual(
            inp.conduit.reset_index().shape,
            (8, 10),
        )

        inp.conduit.loc["COND3", "FromNode"] = "JUNC1"
        inp.conduit.loc["COND3", "desc"] = "update from node"

        self.assertMultiLineEqual(
            inp.conduit.to_swmm_string(),
            dedent(
                """\
                    ;;Name  FromNode  ToNode  Length   Roughness  InOffset  OutOffset  InitFlow  MaxFlow  
                    ;;----  --------  ------  -------  ---------  --------  ---------  --------  -------  
                    COND1   JUNC1     JUNC2   932.363  0.015      0         0.25       0         0        
                    ;cond2 comment
                    COND2   JUNC2     JUNC3   599.52   0.019      0         0.25       0         0        
                    ;update from node
                    COND3   JUNC1     JUNC4   541.1    0.019      0         0.5        0         0        
                    COND4   JUNC4     JUNC5   732.48   0.019      0         0.0        0         0        
                    COND5   JUNC5     STOR1   64.72    0.019      0         8.74       0         0        
                    COND6   JUNC6     OUT1    37.72    0.015      0         0.0        0         0        
                    COND7   JUNC5     STOR1   37.72    0.015      0         0.0        0         0        
                    COND8   JUNC5     STOR1   37.72    0.015      0         0.0        0         0        
                """
            ),
        )

    def test_pump(self):
        inp = self.test_base_model

        self.assertEqual(inp.pump.reset_index().shape, (1, 8))

        inp.pump.loc["PUMP1", "Status"] = "OFF"
        inp.pump.add_element(
            Name="PUMP2",
            FromNode="STOR1",
            ToNode="JUNC6",
            PumpCurve="P1",
            Status="ON",
        )

        self.assertMultiLineEqual(
            inp.pump.to_swmm_string(),
            dedent(
                """\
                    ;;Name  FromNode  ToNode  PumpCurve  Status  Startup  Shutoff  
                    ;;----  --------  ------  ---------  ------  -------  -------  
                    PUMP1   STOR1     JUNC6   P1         OFF     1.3      0.3      
                    PUMP2   STOR1     JUNC6   P1         ON                        
                """
            ),
        )

    def test_orifice(self):
        inp = self.test_det_pond_model

        self.assertEqual(inp.orifice.reset_index().shape, (1, 9))

        inp.orifice.loc["O1", "Gated"] = "YES"
        inp.orifice.add_element(
            Name="O2",
            FromNode="SU1",
            ToNode="J_out",
            Type="SIDE",
            Offset=1.25,
            Gate=False,
            CloseTime=6,
            desc="a new orifice!",
        )

        self.assertMultiLineEqual(
            inp.orifice.to_swmm_string(),
            dedent(
                """\
                    ;;Name  FromNode  ToNode  Type  Offset  Qcoeff  Gated  CloseTime  
                    ;;----  --------  ------  ----  ------  ------  -----  ---------  
                    O1      SU1       J_out   SIDE  0.0     0.65    YES    0          
                    ;a new orifice!
                    O2      SU1       J_out   SIDE  1.25                   6          
                """
            ),
        )

    def test_xsect(self):
        inp = self.test_base_model

        self.assertEqual(inp.xsections.reset_index().shape, (9, 10))
        inp.xsections.loc["COND7", "Barrels"] = 1
        inp.xsections.loc["COND7", "desc"] = "changed to single barrel"

        self.assertMultiLineEqual(
            inp.xsections.to_swmm_string(),
            dedent(
                """\
                    ;;Link  Shape            Geom1  Curve        Geom2  Geom3  Geom4  Barrels  Culvert  
                    ;;----  ---------------  -----  -----------  -----  -----  -----  -------  -------  
                    COND1   CIRCULAR         1.0                 0.0    0      0      1        0        
                    COND2   FILLED_CIRCULAR  1.25                0.5    0      0      1        0        
                    COND3   FILLED_CIRCULAR  1.5                 0.5    0      0      1        0        
                    COND4   FILLED_CIRCULAR  2.0                 0.5    0      0      1        0        
                    COND5   FILLED_CIRCULAR  2.0                 1.0    0      0      1        0        
                    COND6   FORCE_MAIN       1.0                 130.0  0      0      1        0        
                    ;changed to single barrel
                    COND7   CUSTOM           10.0   COND7_curve         0      0      1                 
                    COND8   IRREGULAR               Transect                                            
                    WR1     RECT_OPEN        3.2                 3.0    0      0                        
                """
            ),
        )

    def test_street(self):
        inp = self.test_street_model

        self.assertEqual(inp.street.reset_index().shape, (2, 12))

        inp.street.loc["FullStreet", "nRoad"] = 0.012
        inp.street.loc["FullStreet", "desc"] = "lowered road n-value"

        self.assertMultiLineEqual(
            inp.street.to_swmm_string(),
            dedent(
                """\
                    ;;Name      Tcrown  Hcurb  Sroad  nRoad  Hdep  Wdep  Sides  Wback  Sback  nBack  
                    ;;--------  ------  -----  -----  -----  ----  ----  -----  -----  -----  -----  
                    HalfStreet  20      0.5    4      0.016  0     0     1      20     4      0.016  
                    ;lowered road n-value
                    FullStreet  20      0.5    4      0.012  0     0     2      20     4      0.016  
                """
            ),
        )

    def test_inlet(self):
        inp = self.test_street_model

        self.assertEqual(inp.inlet.reset_index().shape, (2, 8))

        inp.inlet.loc[("ComboInlet", "GRATE"), "param4"] = 0.5
        inp.inlet.loc[("ComboInlet", "GRATE"), "param5"] = 0.3
        inp.inlet.loc[("ComboInlet", "GRATE"), "desc"] = (
            "update pct open and splace velocity of grate"
        )

        self.assertMultiLineEqual(
            inp.inlet.to_swmm_string(),
            dedent(
                """\
                    ;;Name      Type   param1  param2  param3      param4  param5  
                    ;;--------  -----  ------  ------  ----------  ------  ------  
                    ;update pct open and splace velocity of grate
                    ComboInlet  GRATE  2       2.0     P_BAR-50    0.5     0.3     
                    ComboInlet  CURB   2       0.5     HORIZONTAL                  
                """
            ),
        )

    def test_inlet_usage(self):
        inp = self.test_street_model

        self.assertEqual(inp.inlet_usage.reset_index().shape, (4, 10))

        inp.inlet_usage.loc["Street5", "Placement"] = "ON_SAG"
        inp.inlet_usage.loc["Street5", "desc"] = "updated placement"

        self.assertMultiLineEqual(
            inp.inlet_usage.to_swmm_string(),
            dedent(
                """\
                    ;;Conduit  Inlet       Node  Number  %Clogged  MaxFlow  hDStore  wDStore  Placement  
                    ;;-------  ----------  ----  ------  --------  -------  -------  -------  ---------  
                    Street1    ComboInlet  J1    1       0         0        0        0                   
                    Street3    ComboInlet  J2a   1       0         0        0        0                   
                    Street4    ComboInlet  J2    1       0         0        0        0                   
                    ;updated placement
                    Street5    ComboInlet  J11   2       0         0        0        0        ON_SAG     
                """
            ),
        )

    def test_pollutant(self):
        inp = self.test_base_model

        self.assertEqual(inp.pollutants.reset_index().shape, (3, 12))

        inp.pollutants.loc["Sewage", "Cinit"] = 100
        inp.pollutants.loc["Sewage", "desc"] = "updated initial conc"

        self.assertMultiLineEqual(
            inp.pollutants.to_swmm_string(),
            dedent(
                """\
                    ;;Name       Units  Crain  Cgw  Crdii  Kdecay  SnowOnly  CoPollutant  CoFrac  Cdwf  Cinit  
                    ;;---------  -----  -----  ---  -----  ------  --------  -----------  ------  ----  -----  
                    Groundwater  MG/L   0      100  0      0       NO        *            0.0     0     0      
                    Rainfall     MG/L   100    0    0      0       NO        *            0.0     0     0      
                    ;updated initial conc
                    Sewage       MG/L   0      0    0      0       NO        *            0.0     100   100    
                """
            ),
        )

    def test_landuse(self):
        inp = self.test_site_drainage_model

        self.assertEqual(inp.landuse.reset_index().shape, (4, 5))

        inp.landuse.loc["Residential_1", "SweepInterval"] = 7
        inp.landuse.loc["Residential_1", "Availability"] = 0.5
        inp.landuse.loc["Residential_1", "LastSweep"] = 3
        inp.landuse.loc["Residential_1", "desc"] = "set weekly street sweeping"

        self.assertMultiLineEqual(
            inp.landuse.to_swmm_string(),
            dedent(
                """\
                    ;;Name         SweepInterval  Availability  LastSweep  
                    ;;-----------  -------------  ------------  ---------  
                    ;set weekly street sweeping
                    Residential_1  7              0.5           3          
                    Residential_2  0              0.0           0          
                    Commercial     0              0.0           0          
                    Undeveloped    0              0.0           0          
                """
            ),
        )

    def test_coverage(self):
        inp = self.test_site_drainage_model

        self.assertEqual(inp.coverage.reset_index().shape, (10, 4))

        inp.coverage.loc[("S5", "Commercial"), "Percent"] = 50
        inp.coverage.loc[("S5", "Commercial"), "desc"] = "reduced to 50%"
        inp.coverage.loc[("S5", "Residential_2"), "Percent"] = 50
        inp.coverage.loc[("S5", "Residential_2"), "desc"] = "added"

        self.assertMultiLineEqual(
            inp.coverage.to_swmm_string(),
            dedent(
                """\
                    ;;Subcatchment  landuse        Percent  
                    ;;------------  -------------  -------  
                    S1              Residential_1  100      
                    S2              Residential_1  27       
                    S2              Residential_2  73       
                    S3              Residential_1  27       
                    S3              Residential_2  32       
                    S4              Residential_1  9        
                    S4              Residential_2  30       
                    S4              Commercial     26       
                    ;reduced to 50%
                    S5              Commercial     50       
                    S6              Commercial     100      
                    ;added
                    S5              Residential_2  50       
                """
            ),
        )

    def test_loadings(self):
        inp = self.test_base_model

        self.assertEqual(inp.loading.reset_index().shape, (3, 4))

        inp.loading.loc[("SUB1", "Ranfall"), "InitBuildup"] = 10
        inp.loading.loc[("SUB1", "Ranfall"), "desc"] = "bumped initial conc"

        self.assertMultiLineEqual(
            inp.loading.to_swmm_string(),
            dedent(
                """\
                    ;;Subcatchment  Pollutant  InitBuildup  
                    ;;------------  ---------  -----------  
                    SUB1            Rainfall   1.0          
                    SUB2            Rainfall   1.4          
                    SUB3            Rainfall   1.1          
                    ;bumped initial conc
                    SUB1            Ranfall    10.0         
                """
            ),
        )

    def test_buildup(self):
        inp = self.test_site_drainage_model

        self.assertEqual(inp.buildup.reset_index().shape, (4, 8))

        inp.buildup.loc[("Residential_1", "TSS"), "C1"] = 0.22
        inp.buildup.loc[("Residential_1", "TSS"), "desc"] = "increased rate"
        inp.buildup.add_element(
            Landuse="Residential_3",
            Pollutant="TSS",
            FuncType="SAT",
            C1=0.1,
            C2=0,
            C3=1,
            PerUnit="AREA",
        )
        self.assertMultiLineEqual(
            inp.buildup.to_swmm_string(),
            dedent(
                """\
                    ;;Landuse      Pollutant  FuncType  C1    C2   C3   PerUnit  
                    ;;-----------  ---------  --------  ----  ---  ---  -------  
                    ;increased rate
                    Residential_1  TSS        EXP       0.22  0.5  0.0  CURB     
                    Residential_2  TSS        EXP       0.13  0.5  0.0  CURB     
                    Commercial     TSS        EXP       0.15  0.2  0.0  CURB     
                    Undeveloped    TSS        NONE      0.0   0.0  0.0  AREA     
                    Residential_3  TSS        SAT       0.1   0.0  1.0  AREA     
                """
            ),
        )

    def test_washoff(self):
        inp = self.test_site_drainage_model

        self.assertEqual(inp.washoff.reset_index().shape, (4, 8))

        inp.washoff.loc[("Residential_1", "TSS"), "SweepRmvl"] = 0.6
        inp.washoff.loc[("Residential_1", "TSS"), "desc"] = "set sweep removal eff"
        inp.washoff.add_element(
            Landuse="Residential_3", Pollutant="TSS", FuncType="EMC", C1=55
        )
        self.assertMultiLineEqual(
            inp.washoff.to_swmm_string(),
            dedent(
                """\
                    ;;Landuse      Pollutant  FuncType  C1   C2   SweepRmvl  BmpRmvl  
                    ;;-----------  ---------  --------  ---  ---  ---------  -------  
                    ;set sweep removal eff
                    Residential_1  TSS        EXP       2    1.8  0.6        0.0      
                    Residential_2  TSS        EXP       4    2.2  0.0        0.0      
                    Commercial     TSS        EXP       4    2.2  0.0        0.0      
                    Undeveloped    TSS        RC        500  2.0  0.0        0.0      
                    Residential_3  TSS        EMC       55                            
                """
            ),
        )

    def test_treatment(self):
        inp = self.test_base_model

        self.assertEqual(inp.treatment.reset_index().shape, (2, 4))

        inp.treatment.loc[("STOR1", "Rainfall"), "Func"] = (
            "C = Rainfall * exp(-0.1*HRT)"
        )
        inp.treatment.loc[("STOR1", "Rainfall"), "desc"] = "modded equation"
        inp.treatment.add_element(
            Node="STOR1",
            Pollutant="Sewage",
            Func="R = 0.1 * R_Rainfall",
            desc="added treatment for sewage",
        )

        self.assertMultiLineEqual(
            inp.treatment.to_swmm_string(),
            dedent(
                """\
                    ;;Node  Pollutant    Func                          
                    ;;----  -----------  ----------------------------  
                    ;modded equation
                    STOR1   Rainfall     C = Rainfall * exp(-0.1*HRT)  
                    ;groundwater removed with rainfall
                    STOR1   Groundwater  R = 0.2 * R_Rainfall          
                    ;added treatment for sewage
                    STOR1   Sewage       R = 0.1 * R_Rainfall          
                """
            ),
        )

    def test_inflows(self):
        inp = self.test_base_model

        self.assertEqual(inp.inflow.reset_index().shape, (3, 9))
        inp.inflow.loc[("JUNC3", "Sewage"), "Pattern"] = "HOURLY"
        inp.inflow.loc[("JUNC3", "Sewage"), "desc"] = "added hourly poll pattern"
        inp.inflow.add_element(
            Node="JUNC4",
            Constituent="FLOW",
            Type="FLOW",
            Baseline=10,
            desc="new inflow!",
        )

        self.assertMultiLineEqual(
            inp.inflow.to_swmm_string(),
            dedent(
                """\
                    ;;Node  Constituent  TimeSeries       Type    Mfactor  Sfactor  Baseline  Pattern  
                    ;;----  -----------  ---------------  ------  -------  -------  --------  -------  
                    JUNC1   FLOW         "inflow_series"  FLOW    1.0      1.0      0.25      HOURLY   
                    JUNC2   FLOW         ""               FLOW    1.0      1.0      10.0               
                    ;added hourly poll pattern
                    JUNC3   Sewage       ""               CONCEN  1.0      1.0      100.0     HOURLY   
                    ;new inflow!
                    JUNC4   FLOW         ""               FLOW    1.0      1.0      10.0               
                """
            ),
        )

    def test_dwf(self):
        inp = self.test_base_model

        self.assertEqual(inp.inflow.reset_index().shape, (3, 9))
        inp.dwf.loc[("JUNC2", "FLOW"), "Pat4"] = "HOURLY2"
        inp.dwf.loc[("JUNC2", "FLOW"), "desc"] = "added second pattern"
        inp.dwf.add_element(
            Node="JUNC1",
            Constituent="FLOW",
            AvgValue=1,
            Pat3="HOURLY",
            desc="testing a pattern addition",
        )

        self.assertMultiLineEqual(
            inp.dwf.to_swmm_string(),
            dedent(
                """\
                    ;;Node  Constituent  AvgValue  Pat1      Pat2  Pat3      Pat4       
                    ;;----  -----------  --------  --------  ----  --------  ---------  
                    ;added second pattern
                    JUNC2   FLOW         0.2       "HOURLY"  ""    ""        "HOURLY2"  
                    JUNC4   FLOW         0.7       "HOURLY"  ""    ""        ""         
                    ;testing a pattern addition
                    JUNC1   FLOW         1.0       ""        ""    "HOURLY"  ""         
                """
            ),
        )

    def test_rdii(self):
        inp = self.test_base_model
        self.assertEqual(inp.rdii.reset_index().shape, (3, 4))

        inp.rdii.loc["JUNC2", "SewerArea"] = 1244.282478391
        inp.rdii.loc["JUNC2", "desc"] = "bumped acreage"

        self.assertMultiLineEqual(
            inp.rdii.to_swmm_string(),
            dedent(
                """\
                    ;;Node  UHgroup  SewerArea       
                    ;;----  -------  --------------  
                    JUNC1   HydrC    10.0            
                    ;bumped acreage
                    JUNC2   HydrA    1244.282478391  
                    ;this rdii has high precision
                    JUNC3   HydrB    5.213837        
                """
            ),
        )

    def test_hydrographs(self):
        inp = self.test_base_model
        self.assertEqual(inp.hydrographs.reset_index().shape, (75, 10))

        inp.hydrographs.loc["HydrB", "R"] *= 100
        inp.hydrographs.rain_gauges["HydrB"] = "RainGauge_C"

        self.assertMultiLineEqual(
            inp.hydrographs.loc["HydrB":"HydrB"].to_swmm_string(),
            dedent(
                """\
                    ;;Name  Month_RG     Response  R  T           K          IA_max  IA_rec  IA_ini  
                    ;;----  -----------  --------  -  ----------  ---------  ------  ------  ------  
                    HydrB   RainGauge_C                                                              
                    HydrB   Jan          Short        0.575       0.825      0.0     0.0     0.0     
                    HydrB   Jan          Medium       5.5         7.275      0.0     0.0     0.0     
                    HydrB   Jan          Long         7.0         17.0       0.0     0.0     0.0     
                    HydrB   Feb          Short        0.63125     0.9354165  0.0     0.0     0.0     
                    HydrB   Feb          Medium       5.19833325  7.38125    0.0     0.0     0.0     
                    HydrB   Feb          Long         7.08333325  21.166665  0.0     0.0     0.0     
                    HydrB   Mar          Short        0.6875      1.045833   0.0     0.0     0.0     
                    HydrB   Mar          Medium       4.8966665   7.4875     0.0     0.0     0.0     
                    HydrB   Mar          Long         7.1666665   25.33333   0.0     0.0     0.0     
                    HydrB   Apr          Short        0.74375     1.1562495  0.0     0.0     0.0     
                    HydrB   Apr          Medium       4.59499975  7.59375    0.0     0.0     0.0     
                    HydrB   Apr          Long         7.24999975  29.499995  0.0     0.0     0.0     
                    HydrB   May          Short        0.8         1.266666   0.0     0.0     0.0     
                    HydrB   May          Medium       4.293333    7.7        0.0     0.0     0.0     
                    HydrB   May          Long         7.333333    33.66666   0.0     0.0     0.0     
                    HydrB   Jun          Short        0.7         1.033333   0.0     0.0     0.0     
                    HydrB   Jun          Medium       5.0         6.8        0.0     0.0     0.0     
                    HydrB   Jun          Long         8.0         28.33333   0.0     0.0     0.0     
                    HydrB   Jul          Short        0.7         1.033333   0.0     0.0     0.0     
                    HydrB   Jul          Medium       5.0         6.8        0.0     0.0     0.0     
                    HydrB   Jul          Long         8.0         28.33333   0.0     0.0     0.0     
                    HydrB   Aug          Short        0.7         1.033333   0.0     0.0     0.0     
                    HydrB   Aug          Medium       5.0         6.8        0.0     0.0     0.0     
                    HydrB   Aug          Long         8.0         28.33333   0.0     0.0     0.0     
                    HydrB   Sep          Short        0.5         1.2166665  0.0     0.0     0.0     
                    HydrB   Sep          Medium       3.75        12.65      0.0     0.0     0.0     
                    HydrB   Sep          Long         10.75       24.166665  0.0     0.0     0.0     
                    HydrB   Oct          Short        0.3         1.4        0.0     0.0     0.0     
                    HydrB   Oct          Medium       2.5         18.5       0.0     0.0     0.0     
                    HydrB   Oct          Long         13.5        20.0       0.0     0.0     0.0     
                    HydrB   Nov          Short        0.4375      1.1125     0.0     0.0     0.0     
                    HydrB   Nov          Medium       4.0         12.8875    0.0     0.0     0.0     
                    HydrB   Nov          Long         10.25       18.5       0.0     0.0     0.0     
                    HydrB   Dec          Short        0.575       0.825      0.0     0.0     0.0     
                    HydrB   Dec          Medium       5.5         7.275      0.0     0.0     0.0     
                    HydrB   Dec          Long         7.0         17.0       0.0     0.0     0.0     
                """
            ),
        )

    def test_curves(self):
        inp = self.test_base_model
        self.assertEqual(inp.curves.reset_index().shape, (9, 5))

        inp.curves.loc[("P1", 2), :] = [10, 10, "extended the curve"]

        self.assertMultiLineEqual(
            inp.curves.to_swmm_string(),
            dedent(
                """\
                    ;;Name       Type     X_Value  Y_Value  
                    ;;---------  -------  -------  -------  
                    COND7_curve  SHAPE    0.1      2.0      
                    COND7_curve           0.2      5.0      
                    COND7_curve           0.5      10.0     
                    COND7_curve           1.0      10.0     
                    P1           PUMP5    0.0      0.0      
                    P1                    7.0      5.8      
                    ;extended the curve
                    P1                    10.0     10.0     
                    Store1       STORAGE  1.0      20.0     
                    Store1                2.0      30.0     
                    Store1                3.0      40.0     
                """
            ),
        )

    def test_timeseries(self):
        inp = self.test_base_model
        self.assertEqual(len(inp.timeseries), 6)
        self.assertEqual(inp.timeseries["SCS_Type_III_3in"].shape, (96, 2))
        inp.timeseries["file_series"].comment = "this is a updated comment"
        inp.timeseries["SCS_Type_III_3in"]["value"] *= 10
        inp.timeseries["SCS_Type_III_3in"].attrs[
            "desc"
        ] = "SCS_Type_III_3in design storm, total rainfall = 30 in, rain units = in."

        idx = pd.date_range("1/1/1900", "2/1/1900", freq="1D", name="time")
        vals = [100] * len(idx)
        df = pd.DataFrame(
            index=idx,
            data=dict(value=vals),
        )
        df.attrs["desc"] = "a new timeseries"
        inp.timeseries["new_ts"] = df

        with open(_HERE / "data" / "timeseries_benchmark.dat") as bench_file:
            bench_text = bench_file.read()
            self.assertMultiLineEqual(inp.timeseries.to_swmm_string(), bench_text)

    def test_weir(self):
        inp = self.test_base_model
        self.assertEqual(inp.weir.reset_index().shape, (1, 14))

        inp.weir.loc["WR1", "Type"] = "SIDE"
        inp.weir.loc["WR1", "desc"] = "changed weir type"

        self.assertMultiLineEqual(
            inp.weir.to_swmm_string(),
            dedent(
                """\
                    ;;Name  FromNode  ToNode  Type  CrestHt  Qcoeff  Gated  EndCon  EndCoeff  Surcharge  RoadWidth  RoadSurf  CoeffCurve  
                    ;;----  --------  ------  ----  -------  ------  -----  ------  --------  ---------  ---------  --------  ----------  
                    ;changed weir type
                    WR1     JUNC2     OUT2    SIDE  3        3.33    NO     0       0         YES                                         
                """
            ),
        )

    def test_divider(self):
        inp = self.test_divider_model
        self.assertEqual(inp.divider.reset_index().shape, (4, 13))

        inp.divider.loc["KRO1004", "Ymax"] = 100
        inp.divider.loc["KRO1004", "desc"] = "bumped ymax"

        self.assertMultiLineEqual(
            inp.divider.to_swmm_string(),
            dedent(
                """\
                    ;;Name   Elevation  DivLink  DivType   DivCurve  Qmin  Height  Cd    Ymax  Y0  Ysur  Apond  
                    ;;-----  ---------  -------  --------  --------  ----  ------  ----  ----  --  ----  -----  
                    KRO1003  594.73     C3       TABULAR   Outflow                       5     0   0     0      
                    ;bumped ymax
                    KRO1004  584.0      C2       WEIR                0.2   5       3.33  100   0   0     0      
                    KRO1010  584.82     C1       CUTOFF              0.2                 11    0   0     0      
                    KRO4008  583.48     C4       OVERFLOW                                10    0   0     0      
                """
            ),
        )

    def test_outlet(self):
        inp = self.test_outlet_model
        self.assertEqual(inp.outlet.reset_index().shape, (4, 10))

        inp.outlet.loc[8060, "Qcoeff"] = 100
        inp.outlet.loc[8060, "desc"] = "bumped Qcoeff"

        self.assertMultiLineEqual(
            inp.outlet.to_swmm_string(),
            dedent(
                """\
                    ;;Name  FromNode  ToNode  Offset  Type              CurveName    Qcoeff  Qexpon  Gated  
                    ;;----  --------  ------  ------  ----------------  -----------  ------  ------  -----  
                    1030    10309     10208   0       TABULAR/HEAD      Outlet_head                  NO     
                    1600    16109     16009   0       FUNCTIONAL/HEAD                10      0.5     NO     
                    ;bumped Qcoeff
                    8060    80608     82309   0       FUNCTIONAL/DEPTH               100     0.5     NO     
                    8130    81309     15009   0       TABULAR/DEPTH     Outlet_head                  NO     
                """
            ),
        )

    def test_losses(self):
        inp = self.test_base_model

        self.assertEqual(inp.losses.reset_index().shape, (2, 7))

        inp.losses.loc["COND2", "Kavg"] = 10_000
        inp.losses.loc["COND2", "desc"] = "added outrageous losses"

        self.assertMultiLineEqual(
            inp.losses.to_swmm_string(),
            dedent(
                """\
                    ;;Link  Kentry  Kexit  Kavg   FlapGate  Seepage  
                    ;;----  ------  -----  -----  --------  -------  
                    COND1   1       2      3      NO        4        
                    ;added outrageous losses
                    COND2   0       0      10000  YES       0        
                """
            ),
        )

    def test_patterns(self):
        inp = self.test_base_model

        self.assertEqual(inp.patterns.reset_index().shape, (84, 4))

        inp.patterns.loc[("Monthly", slice(None)), "Multiplier"] *= 100
        inp.patterns.loc[("Monthly", 0), "desc"] = "A massive monthly pattern"

        with open(_HERE / "data" / "pattern_benchmark.dat") as bench_file:
            bench_text = bench_file.read()
            self.assertMultiLineEqual(inp.patterns.to_swmm_string(), bench_text)

    def test_polygons(self):
        inp = self.test_base_model

        self.assertEqual(inp.polygons.reset_index().shape, (63, 4))

        inp.polygons.iloc[0, 0] = 10_000
        inp.polygons.iloc[0, 2] = "crazy X value"

        swmm_text = inp.polygons.to_swmm_string()

        self.assertIn("crazy X value", swmm_text)
        self.assertIn("10000", swmm_text)

    def test_coord(self):
        inp = self.test_base_model

        self.assertEqual(inp.coordinates.reset_index().shape, (10, 4))

        inp.coordinates.iloc[0, 0] = 10_000
        inp.coordinates.iloc[0, 2] = "crazy X value"

        swmm_text = inp.coordinates.to_swmm_string()

        self.assertIn("crazy X value", swmm_text)
        self.assertIn("10000", swmm_text)

    def test_vert(self):
        inp = self.test_det_pond_model

        self.assertEqual(inp.vertices.reset_index().shape, (25, 4))

        inp.vertices.iloc[0, 0] = 10_000
        inp.vertices.iloc[0, 2] = "crazy X value"

        swmm_text = inp.vertices.to_swmm_string()

        self.assertIn("crazy X value", swmm_text)
        self.assertIn("10000", swmm_text)

    def test_tags(self):
        inp = self.test_det_pond_model

        self.assertEqual(inp.tags.reset_index().shape, (11, 4))

        inp.tags.loc[("Link", "C5"), "Tag"] = "a_new_kind_of_GI"
        inp.tags.loc[("Link", "C5"), "desc"] = "this is a new test GI"

        self.assertMultiLineEqual(
            inp.tags.to_swmm_string(),
            dedent(
                """\
                    ;;Element  Name  Tag               
                    ;;-------  ----  ----------------  
                    Link       C1    Swale             
                    Link       C2    Gutter            
                    Link       C3    Culvert           
                    Link       C4    Swale             
                    ;this is a new test GI
                    Link       C5    a_new_kind_of_GI  
                    Link       C6    Swale             
                    Link       C7    Culvert           
                    Link       C8    Swale             
                    Link       C9    Swale             
                    Link       C10   Swale             
                    Link       C11   Culvert           
                """
            ),
        )

    def test_rerun(self):
        bench_inp = self.test_base_model_path
        bench_rpt = bench_inp.replace("inp", "rpt")
        bench_out = bench_inp.replace("inp", "out")

        test_inp = str(
            pathlib.Path(self.test_base_model_path).parent / "test_model.inp"
        )
        test_rpt = test_inp.replace("inp", "rpt")
        test_out = test_inp.replace("inp", "out")

        solver.swmm_run(
            bench_inp,
            bench_rpt,
            bench_out,
        )

        inp = InputFile(bench_inp)
        inp.to_file(test_inp)

        solver.swmm_run(
            test_inp,
            test_rpt,
            test_out,
        )

        rptb = Report(bench_rpt)
        rptt = Report(test_rpt)

        for attr in dir(rptb):
            if not attr.startswith(("_", "analysis")):
                bench = getattr(rptb, attr)
                test = getattr(rptt, attr)
                pd.testing.assert_frame_equal(bench, test)
