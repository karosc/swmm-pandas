from pytest import raises

from swmm.pandas.units import (
    build_pollutant_unit_map,
    resolve_output_unit,
)


def test_build_pollutant_unit_map():
    assert build_pollutant_unit_map(
        ["groundwater", "sewage", "counts"],
        ["MG", "UG", "COUNT"],
    ) == {
        "groundwater": "mg/l",
        "sewage": "ug/l",
        "counts": "#/l",
    }


def test_resolve_output_unit_for_builtins():
    assert resolve_output_unit("link", "flow_rate", "US", "CFS") == "cfs"
    assert resolve_output_unit("link", "flow_depth", "US", "CFS") == "ft"
    assert resolve_output_unit("sys", "air_temp", "SI", "CMS") == "deg c"


def test_resolve_output_unit_for_pollutants():
    pollutant_units = build_pollutant_unit_map(["groundwater"], ["MG"])
    assert (
        resolve_output_unit(
            "link",
            "groundwater",
            "US",
            "CFS",
            pollutant_units=pollutant_units,
        )
        == "mg/l"
    )


def test_resolve_output_unit_rejects_unknown_attribute():
    with raises(KeyError, match="Unsupported SWMM output attribute mapping"):
        resolve_output_unit("link", "unknown", "US", "CFS")
