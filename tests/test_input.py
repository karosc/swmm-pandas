"""Tests for `swmm-pandas` input class."""

import datetime
import pathlib
import unittest
from textwrap import dedent

import numpy.testing as nptest
import numpy as np
import pandas as pd
import swmm.pandas.input._section_classes as sc
from swmm.pandas import Input

_HERE = pathlib.Path(__file__).parent


class InputTest(unittest.TestCase):
    def setUp(self):
        self.test_base_model_path = str(_HERE / "data" / "Model.inp")
        self.test_groundwater_model_path = str(_HERE / "data" / "Groundwater_Model.inp")
        self.test_street_model_path = str(_HERE / "data" / "Inlet_Drains_Model.inp")
        self.test_pump_model_path = str(_HERE / "data" / "Pump_Control_Model.inp")
        self.test_drainage_model_path = str(_HERE / "data" / "Site_Drainage_Model.inp")
        self.test_lid_model_path = str(_HERE / "data" / "LID_Model.inp")
        self.test_det_pond_model_path = str(_HERE / "data" / "Detention_Pond_Model.inp")
        self.test_inlet_model_path = str(_HERE / "data" / "Inlet_Drains_Model.inp")

        self.test_base_model = Input(self.test_base_model_path)
        self.test_lid_model = Input(self.test_lid_model_path)
        self.test_det_pond_model = Input(self.test_det_pond_model_path)
        self.test_street_model = Input(self.test_street_model_path)
        self.test_site_drainage_model = Input(self.test_drainage_model_path)

        self.maxDiff = 1000

        # self.test_groundwater_model = Input(self.test_groundwater_model_path)
        # self.test_street_model = Input(self.test_street_model_path)
        # self.test_pump_model = Input(self.test_pump_model_path)
        # self.test_drainage_model = Input(self.test_drainage_model_path)

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
        inp.adjustments.loc["EVAPORATION", "May":"Nov"] = pd.NA
        inp.adjustments.loc["EVAPORATION", "Jan":"Dec"] = (
            inp.adjustments.loc["EVAPORATION", "Jan":"Dec"].astype(float).interpolate()
        )

        self.assertEqual(
            inp.adjustments.to_swmm_string().split("\n")[3].strip(),
            "EVAPORATION   1.0  2    -3   -4   -3.375  -2.75  -2.125  -1.5   -0.875  -0.25  0.375  1.0",
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
        inp.subcatchment["Width"] = (inp.subcatchment.Area**0.6).round(3)
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
        self.maxDiff
        self.assertMultiLineEqual(
            inp.lid_usage.to_swmm_string(),
            dedent(
                """\
                    ;;Subcatchment  LIDProcess   Number  Area      Width  InitSat  FromImp  ToPerv  RptFile  DrainTo  FromPerv  
                    ;;------------  -----------  ------  --------  -----  -------  -------  ------  -------  -------  --------  
                    S1              InfilTrench  4       532       133    0        40       0       *        *        0         
                    S1              RainBarrels  32      5         0      0        17       1       *        *        0         
                    S4              Planters     30      500       0      0        80       0       *        *        0         
                    S5              PorousPave   1       232872    683    0        0        0       *        *        0         
                    S5              GreenRoof    1       18400     136    0        0        0       *        *        0         
                    ;Update width
                    Swale3          Swale        1       14374.80  100    0        0        0       *        *        0         
                    ;Update width
                    Swale4          Swale        1       21780.00  100    0        0        0       *        *        0         
                    ;Update width
                    Swale6          Swale        1       17859.60  100    0        0        0       *        *        0         
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
                    ;;Name  Surface     param1    param2    param3     param4    param5    param6    param7    
                    ;;----  ----------  --------  --------  ---------  --------  --------  --------  --------  
                    SNOW1   PLOWABLE    0.005000  0.007000  24.000000  0.200000  0.000000  0.000000  0.100000  
                    SNOW1   IMPERVIOUS  0.005000  0.007000  24.000000  0.200000  0.000000  0.000000  2.000000  
                    SNOW1   PERVIOUS    0.004000  0.004000  25.000000  0.200000  0.000000  0.000000  2.000000  
                    ;Update plow depth
                    SNOW1   REMOVAL     4         0         0          1         0.000000  0.000000            
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
            (6, 10),
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

        self.assertEqual(inp.xsections.reset_index().shape, (8, 10))
        inp.xsections.loc["COND7", "Barrels"] = 1
        inp.xsections.loc["COND7", "desc"] = "changed to single barrel"

        self.assertMultiLineEqual(
            inp.xsections.to_swmm_string(),
            dedent(
                """\
                    ;;Link  Shape            Geom1  Curve   Geom2  Geom3  Geom4  Barrels  Culvert  
                    ;;----  ---------------  -----  ------  -----  -----  -----  -------  -------  
                    COND1   CIRCULAR         1.0            0      0      0      1        0        
                    COND2   FILLED_CIRCULAR  1.25           0.5    0      0      1        0        
                    COND3   FILLED_CIRCULAR  1.5            0.5    0      0      1        0        
                    COND4   FILLED_CIRCULAR  2.0            0.5    0      0      1        0        
                    COND5   FILLED_CIRCULAR  2.0            1      0      0      1        0        
                    COND5   FILLED_CIRCULAR  2.0            1      0      0      1        0        
                    ;changed to single barrel
                    COND7   CUSTOM           10.0   Store1                       1                 
                    WR1     RECT_OPEN        3.2            3      0      0                        
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
                    S1              Residential_1  100.0    
                    S2              Residential_1  27.0     
                    S2              Residential_2  73.0     
                    S3              Residential_1  27.0     
                    S3              Residential_2  32.0     
                    S4              Residential_1  9.0      
                    S4              Residential_2  30.0     
                    S4              Commercial     26.0     
                    ;reduced to 50%
                    S5              Commercial     50.0     
                    S6              Commercial     100.0    
                    ;added
                    S5              Residential_2  50.0     
                """
            ),
        )
