import pathlib
import pytest
from swmm.pandas import Report
from pandas import Timedelta, Timestamp, DataFrame
from numpy import allclose, all, array, nan, float64


_HERE = pathlib.Path(__file__).parent
test_rpt_path = str(_HERE / "data" / "Model.rpt")


@pytest.fixture(scope="module")
def rptfile():
    return Report(test_rpt_path)


def test_analysis_options(rptfile):
    reference = DataFrame(
        {
            "Setting": {
                "Flow Units": "CFS",
                "Rainfall/Runoff": "YES",
                "RDII": "NO",
                "Snowmelt": "YES",
                "Groundwater": "YES",
                "Flow Routing": "YES",
                "Ponding Allowed": "YES",
                "Water Quality": "YES",
                "Infiltration Method": "GREEN_AMPT",
                "Flow Routing Method": "DYNWAVE",
                "Surcharge Method": "EXTRAN",
                "Starting Date": "01/01/1900 00:00:00",
                "Ending Date": "01/02/1900 00:00:00",
                "Antecedent Dry Days": "0.0",
                "Report Time Step": "00:05:00",
                "Wet Time Step": "00:05:00",
                "Dry Time Step": "01:00:00",
                "Routing Time Step": "15.00 sec",
                "Variable Time Step": "YES",
                "Maximum Trials": "10",
                "Number of Threads": "1",
                "Head Tolerance": "0.005000 ft",
            }
        },
    )["Setting"]
    print(type(reference))
    reference.index.name = "Option"
    opts = rptfile.analysis_options
    assert all(opts.sort_index() == reference.sort_index())


def test_runoff_quantity_continuity(rptfile):
    reference = array(
        [
            [0.0, 0.0],
            [14.998, 3.0],
            [0.0, 0.0],
            [9.339, 1.868],
            [5.489, 1.098],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.187, 0.037],
            [-0.108, nan],
        ]
    )

    test = rptfile.runoff_quantity_continuity

    assert allclose(test, reference, equal_nan=True)


def test_runoff_quality_continuity(rptfile):
    reference = array(
        [
            [0.000000e00, 0.000000e00, 0.000000e00],
            [0.000000e00, 0.000000e00, 0.000000e00],
            [0.000000e00, 4.075669e03, 0.000000e00],
            [0.000000e00, 0.000000e00, 0.000000e00],
            [0.000000e00, 2.537844e03, 0.000000e00],
            [0.000000e00, 0.000000e00, 0.000000e00],
            [0.000000e00, 1.491499e03, 0.000000e00],
            [0.000000e00, 5.071200e01, 0.000000e00],
            [0.000000e00, -1.080000e-01, 0.000000e00],
        ]
    )

    test = rptfile.runoff_quality_continuity
    assert allclose(test, reference)


def test_groundwater_continuity(rptfile):
    reference = array(
        [
            [1.272478e03, 2.544960e02],
            [9.339000e00, 1.868000e00],
            [0.000000e00, 0.000000e00],
            [0.000000e00, 0.000000e00],
            [0.000000e00, 0.000000e00],
            [2.770000e-01, 5.500000e-02],
            [1.281540e03, 2.563080e02],
            [0.000000e00, nan],
        ]
    )

    test = rptfile.groundwater_continuity
    assert allclose(test, reference, equal_nan=True)


def test_flow_routing_continuity(rptfile):
    reference = array(
        [
            [1.787, 0.582],
            [5.487, 1.788],
            [0.277, 0.09],
            [0.0, 0.0],
            [0.0, 0.0],
            [6.977, 2.274],
            [0.529, 0.172],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.055, 0.018],
            [-0.158, nan],
        ]
    )

    test = rptfile.flow_routing_continuity
    assert allclose(test, reference, equal_nan=True)


def test_quality_routing_continuity(rptfile):
    reference = array(
        [
            [0.000000e00, 0.000000e00, 4.854910e02],
            [0.000000e00, 1.490454e03, 0.000000e00],
            [7.523300e01, 0.000000e00, 0.000000e00],
            [0.000000e00, 0.000000e00, 0.000000e00],
            [0.000000e00, 0.000000e00, 0.000000e00],
            [7.346500e01, 1.347329e03, 4.751520e02],
            [9.100000e-01, 1.361150e02, 6.717000e00],
            [0.000000e00, 0.000000e00, 0.000000e00],
            [0.000000e00, 0.000000e00, 0.000000e00],
            [0.000000e00, 0.000000e00, 0.000000e00],
            [2.271000e00, 5.967000e00, 6.828000e00],
            [-1.878000e00, 7.000000e-02, -6.600000e-01],
        ]
    )

    test = rptfile.quality_routing_continuity
    assert allclose(test, reference)


def test_highest_continuity_errors(rptfile):
    reference = array([["Node", 1.52]], dtype=object)
    test = rptfile.highest_continuity_errors.to_numpy()
    assert (reference == test).all()


def test_time_step_critical_elements(rptfile):
    reference = array([["Link", 60.11], ["Link", 39.32]], dtype=object)
    test = rptfile.time_step_critical_elements.to_numpy()
    assert (reference == test).all()


def test_highest_flow_instability_indexes(rptfile):
    reference = array([["Link", 1]], dtype=object)
    test = rptfile.highest_flow_instability_indexes.to_numpy()
    assert (reference == test).all()


def test_runoff_summary(rptfile):
    reference = array(
        [
            [3.0, 0.0, 0.0, 2.28, 0.91, 0.25, 0.71, 0.1, 2.96, 0.236],
            [3.0, 0.0, 0.0, 2.12, 1.2, 0.26, 0.86, 0.4, 11.17, 0.287],
            [3.0, 0.0, 0.0, 1.7, 1.82, 0.34, 1.25, 1.29, 32.64, 0.418],
        ]
    )

    test = rptfile.runoff_summary
    assert allclose(test, reference)


def test_groundwater_summary(rptfile):
    reference = array(
        [
            [2.28, 0.0, 0.0, 0.04, 0.01, 0.28, 2.88, 0.28, 3.37],
            [2.12, 0.0, 0.0, 0.04, 0.04, 0.28, 0.32, 0.28, 0.79],
            [1.7, 0.0, 0.0, 0.07, 0.13, 0.28, -3.31, 0.28, -2.93],
        ]
    )

    test = rptfile.groundwater_summary
    assert allclose(test, reference)


def test_washoff_summary(rptfile):
    reference = array([[0.0, 80.229, 0.0], [0.0, 331.693, 0.0], [0.0, 1079.577, 0.0]])

    test = rptfile.washoff_summary
    assert allclose(test, reference)


def test_node_depth_summary(rptfile):
    reference = DataFrame(
        {
            "Type": {
                "JUNC1": "JUNCTION",
                "JUNC2": "JUNCTION",
                "JUNC3": "JUNCTION",
                "JUNC4": "JUNCTION",
                "JUNC5": "JUNCTION",
                "JUNC6": "JUNCTION",
                "OUT1": "OUTFALL",
                "OUT2": "OUTFALL",
                "STOR1": "STORAGE",
            },
            "Average_Depth_Feet": {
                "JUNC1": 0.75,
                "JUNC2": 1.61,
                "JUNC3": 3.74,
                "JUNC4": 5.1,
                "JUNC5": 4.66,
                "JUNC6": 1.14,
                "OUT1": 0.7,
                "OUT2": 0.0,
                "STOR1": 7.5,
            },
            "Maximum_Depth_Feet": {
                "JUNC1": 10.25,
                "JUNC2": 4.39,
                "JUNC3": 13.49,
                "JUNC4": 21.71,
                "JUNC5": 15.03,
                "JUNC6": 1.66,
                "OUT1": 0.95,
                "OUT2": 0.0,
                "STOR1": 21.75,
            },
            "Maximum_HGL_Feet": {
                "JUNC1": 11.75,
                "JUNC2": 3.35,
                "JUNC3": 10.02,
                "JUNC4": 16.46,
                "JUNC5": 8.53,
                "JUNC6": 1.66,
                "OUT1": 1.05,
                "OUT2": -1.04,
                "STOR1": 6.5,
            },
            "Time_of_Max": {
                "JUNC1": Timedelta("0 days 12:30:00"),
                "JUNC2": Timedelta("0 days 12:30:00"),
                "JUNC3": Timedelta("0 days 13:44:00"),
                "JUNC4": Timedelta("0 days 12:45:00"),
                "JUNC5": Timedelta("0 days 13:14:00"),
                "JUNC6": Timedelta("0 days 11:40:00"),
                "OUT1": Timedelta("0 days 11:40:00"),
                "OUT2": Timedelta("0 days 00:00:00"),
                "STOR1": Timedelta("0 days 12:24:00"),
            },
            "Reported_Max_Depth_Feet": {
                "JUNC1": 10.25,
                "JUNC2": 4.39,
                "JUNC3": 13.49,
                "JUNC4": 21.71,
                "JUNC5": 15.03,
                "JUNC6": 1.65,
                "OUT1": 0.95,
                "OUT2": 0.0,
                "STOR1": 21.75,
            },
        }
    )

    test = rptfile.node_depth_summary
    for col in test:
        if test[col].dtype == float64:
            assert allclose(test[col], reference[col])
        else:
            assert all(test[col] == reference[col])


def test_node_inflow_summary(rptfile):
    reference = DataFrame(
        {
            "Type": {
                "JUNC1": "JUNCTION",
                "JUNC2": "JUNCTION",
                "JUNC3": "JUNCTION",
                "JUNC4": "JUNCTION",
                "JUNC5": "JUNCTION",
                "JUNC6": "JUNCTION",
                "OUT1": "OUTFALL",
                "OUT2": "OUTFALL",
                "STOR1": "STORAGE",
            },
            "Maximum_Lateral_Inflow_CFS": {
                "JUNC1": 2.96,
                "JUNC2": 11.42,
                "JUNC3": 0.0,
                "JUNC4": 33.53,
                "JUNC5": 0.0,
                "JUNC6": 0.0,
                "OUT1": 0.0,
                "OUT2": 0.0,
                "STOR1": 0.0,
            },
            "Maximum_Total_Inflow_CFS": {
                "JUNC1": 2.96,
                "JUNC2": 16.41,
                "JUNC3": 5.0,
                "JUNC4": 33.53,
                "JUNC5": 12.32,
                "JUNC6": 5.8,
                "OUT1": 5.8,
                "OUT2": 16.39,
                "STOR1": 12.32,
            },
            "Time_of_Max": {
                "JUNC1": Timedelta("0 days 12:30:00"),
                "JUNC2": Timedelta("0 days 12:30:00"),
                "JUNC3": Timedelta("0 days 12:41:00"),
                "JUNC4": Timedelta("0 days 12:30:00"),
                "JUNC5": Timedelta("0 days 11:51:00"),
                "JUNC6": Timedelta("0 days 11:40:00"),
                "OUT1": Timedelta("0 days 11:40:00"),
                "OUT2": Timedelta("0 days 12:30:00"),
                "STOR1": Timedelta("0 days 11:51:00"),
            },
            "Lateral_Inflow_Volume_10^6_gal": {
                "JUNC1": 0.102,
                "JUNC2": 0.545,
                "JUNC3": 0.0,
                "JUNC4": 1.81,
                "JUNC5": 0.0,
                "JUNC6": 0.0,
                "OUT1": 0.0,
                "OUT2": 0.0,
                "STOR1": 0.0,
            },
            "Total_Inflow_Volume_10^6_gal": {
                "JUNC1": 0.102,
                "JUNC2": 0.857,
                "JUNC3": 0.502,
                "JUNC4": 2.1,
                "JUNC5": 1.87,
                "JUNC6": 1.7,
                "OUT1": 1.7,
                "OUT2": 0.5760000000000001,
                "STOR1": 1.86,
            },
            "Flow_Balance_Error_Percent": {
                "JUNC1": 0.461,
                "JUNC2": 0.5529999999999999,
                "JUNC3": 1.54,
                "JUNC4": 0.013000000000000001,
                "JUNC5": 0.405,
                "JUNC6": 0.198,
                "OUT1": 0.0,
                "OUT2": 0.0,
                "STOR1": -0.8909999999999999,
            },
        }
    )
    test = rptfile.node_inflow_summary
    for col in test:
        if test[col].dtype == float64:
            assert allclose(test[col], reference[col])
        else:
            assert all(test[col] == reference[col])


def test_node_surcharge_summary(rptfile):
    reference = DataFrame(
        {
            "Type": {
                "JUNC1": "JUNCTION",
                "JUNC3": "JUNCTION",
                "JUNC4": "JUNCTION",
                "JUNC5": "JUNCTION",
                "JUNC6": "JUNCTION",
            },
            "Hours_Surcharged": {
                "JUNC1": 1.49,
                "JUNC3": 4.15,
                "JUNC4": 4.48,
                "JUNC5": 4.78,
                "JUNC6": 8.01,
            },
            "Max. Height_Above_Crown_Feet": {
                "JUNC1": 9.251,
                "JUNC3": 11.99,
                "JUNC4": 19.712,
                "JUNC5": 13.026,
                "JUNC6": 0.659,
            },
            "Min. Depth_Below_Rim_Feet": {
                "JUNC1": 0.0,
                "JUNC3": 0.0,
                "JUNC4": 0.0,
                "JUNC5": 0.0,
                "JUNC6": 7.341,
            },
        }
    )

    test = rptfile.node_surchage_summary
    for col in test:
        if test[col].dtype == float64:
            assert allclose(test[col], reference[col])
        else:
            assert all(test[col] == reference[col])


def test_node_flooding_summary(rptfile):
    reference = DataFrame(
        {
            "Hours_Flooded": {
                "JUNC1": 0.07,
                "JUNC3": 3.19,
                "JUNC4": 3.1,
                "JUNC5": 3.08,
                "STOR1": 2.47,
            },
            "Maximum_Rate_CFS": {
                "JUNC1": 0.87,
                "JUNC3": 2.8,
                "JUNC4": 18.33,
                "JUNC5": 4.37,
                "STOR1": 4.21,
            },
            "Time_of_Max": {
                "JUNC1": Timedelta("0 days 11:49:00"),
                "JUNC3": Timedelta("0 days 12:38:00"),
                "JUNC4": Timedelta("0 days 12:15:00"),
                "JUNC5": Timedelta("0 days 12:24:00"),
                "STOR1": Timedelta("0 days 13:14:00"),
            },
            "Total_Flood_Volume_10^6_gal": {
                "JUNC1": 0.0,
                "JUNC3": 0.081,
                "JUNC4": 0.318,
                "JUNC5": 0.07400000000000001,
                "STOR1": 0.172,
            },
            "Maximum_Ponded_Depth_Feet": {
                "JUNC1": 0.001,
                "JUNC3": 1.99,
                "JUNC4": 7.912000000000001,
                "JUNC5": 1.8259999999999998,
                "STOR1": 0.0,
            },
        }
    )

    test = rptfile.node_flooding_summary
    for col in test:
        if test[col].dtype == float64:
            assert allclose(test[col], reference[col])
        else:
            assert all(test[col] == reference[col])


def test_storage_volume_summary(rptfile):
    reference = DataFrame(
        {
            "Average_Volume_1000_ft3": {"STOR1": 3.0980000000000003},
            "Avg_Pcnt_Full": {"STOR1": 34},
            "Evap_Pcnt_Loss": {"STOR1": 0},
            "Exfil_Pcnt_Loss": {"STOR1": 0},
            "Maximum_Volume_1000_ft3": {"STOR1": 8.982999999999999},
            "Max_Pcnt_Full": {"STOR1": 100},
            "Time_of_Max": {"STOR1": Timedelta("0 days 12:24:00")},
            "Maximum_Outflow_CFS": {"STOR1": 5.8},
        }
    )

    test = rptfile.storage_volume_summary
    for col in test:
        if test[col].dtype == float64:
            assert allclose(test[col], reference[col])
        else:
            assert all(test[col] == reference[col])


def test_outfall_loading_summary(rptfile):
    reference = array(
        [
            [
                9.88800e01,
                3.26000e00,
                5.80000e00,
                1.69800e00,
                6.92630e01,
                8.99065e02,
                4.48553e02,
            ],
            [
                2.54900e01,
                5.81000e00,
                1.63900e01,
                5.76000e-01,
                4.20700e00,
                4.48265e02,
                2.66240e01,
            ],
        ]
    )

    test = rptfile.outfall_loading_summary

    assert allclose(test, reference)


def test_link_flow_summary(rptfile):
    reference = DataFrame(
        {
            "Type": {
                "COND1": "CONDUIT",
                "COND2": "CONDUIT",
                "COND3": "CONDUIT",
                "COND4": "CONDUIT",
                "COND5": "CONDUIT",
                "COND6": "CONDUIT",
                "PUMP1": "PUMP",
                "WR1": "WEIR",
            },
            "Maximum_Flow_CFS": {
                "COND1": 2.93,
                "COND2": 2.48,
                "COND3": 5.0,
                "COND4": 12.32,
                "COND5": 12.32,
                "COND6": 5.8,
                "PUMP1": 5.8,
                "WR1": 16.39,
            },
            "Time_of_Max": {
                "COND1": Timedelta("0 days 12:30:00"),
                "COND2": Timedelta("0 days 13:45:00"),
                "COND3": Timedelta("0 days 12:41:00"),
                "COND4": Timedelta("0 days 11:51:00"),
                "COND5": Timedelta("0 days 11:51:00"),
                "COND6": Timedelta("0 days 11:40:00"),
                "PUMP1": Timedelta("0 days 11:40:00"),
                "WR1": Timedelta("0 days 12:30:00"),
            },
            "Maximum_Veloc_ft/sec": {
                "COND1": 3.74,
                "COND2": 3.23,
                "COND3": 4.0,
                "COND4": 4.87,
                "COND5": 7.84,
                "COND6": 7.44,
                "PUMP1": 1.0,
                "WR1": 0.43,
            },
            "Max/_Full_Flow": {
                "COND1": 1.92,
                "COND2": 1.86,
                "COND3": 2.42,
                "COND4": 2.73,
                "COND5": 17.51,
                "COND6": 2.41,
                "PUMP1": nan,
                "WR1": nan,
            },
            "Max/_Full_Depth": {
                "COND1": 1.0,
                "COND2": 1.0,
                "COND3": 1.0,
                "COND4": 1.0,
                "COND5": 1.0,
                "COND6": 0.97,
                "PUMP1": nan,
                "WR1": nan,
            },
        }
    )

    test = rptfile.link_flow_summary

    for col in test:
        if test[col].dtype == float64:
            assert allclose(test[col], reference[col], equal_nan=True)
        else:
            assert all(test[col] == reference[col])


def test_flow_classification_summary(rptfile):
    reference = array(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.84, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.31, 0.0, 0.0, 0.68, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.49, 0.06, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.77, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.27, 0.0, 0.0, 0.73, 0.0, 0.0],
            [2.32, 0.01, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    test = rptfile.flow_classification_summary

    assert allclose(reference, test)


def test_conduit_surcharge_summary(rptfile):
    reference = array(
        [
            [1.490e00, 1.490e00, 4.520e00, 7.400e-01, 7.400e-01],
            [4.150e00, 4.520e00, 4.150e00, 3.920e00, 1.600e-01],
            [4.150e00, 4.150e00, 4.480e00, 1.870e00, 1.000e-02],
            [4.480e00, 4.480e00, 4.780e00, 3.750e00, 3.750e00],
            [3.790e00, 4.780e00, 3.790e00, 2.351e01, 3.740e00],
            [1.000e-02, 1.000e-02, 8.000e00, 8.070e00, 1.000e-02],
        ]
    )

    test = rptfile.conduit_surcharge_summary

    assert allclose(reference, test)


def test_pumping_summary(rptfile):
    reference = array([[97.74, 1.0, 0.0, 3.26, 5.8, 1.701, 53.62, 0.0, 18.1]])

    test = rptfile.pumping_summary

    assert allclose(reference, test)


def test_link_pollutant_load_summary(rptfile):
    reference = array(
        [
            [4.389000e00, 7.990000e01, 1.040000e-01],
            [1.744300e01, 2.922210e02, 9.695600e01],
            [1.730500e01, 3.115090e02, 9.600500e01],
            [6.991400e01, 1.037475e03, 4.530560e02],
            [6.948400e01, 1.035609e03, 4.490460e02],
            [6.926300e01, 8.990650e02, 4.485530e02],
            [6.957100e01, 8.992150e02, 4.508830e02],
            [4.207000e00, 4.482650e02, 2.662400e01],
        ]
    )

    test = rptfile.link_pollutant_load_summary

    assert allclose(reference, test)


def test_analysis_begun_and_end(rptfile):
    assert type(rptfile.analysis_begun) == Timestamp
    assert type(rptfile.analysis_end) == Timestamp
