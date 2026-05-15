"""Shared output-unit resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

OutputKind = str
UnitLabel = str | None

_UNIT_LABELS: dict[str, dict[str, str]] = {
    "US": {
        "rainfall_intensity": "in/hr",
        "depth": "ft",
        "evaporation": "in/day",
        "elevation": "ft",
        "volume": "ft3",
        "velocity": "ft/s",
        "temperature": "deg f",
        "fraction": "fraction",
    },
    "SI": {
        "rainfall_intensity": "mm/hr",
        "depth": "m",
        "evaporation": "mm/day",
        "elevation": "m",
        "volume": "m3",
        "velocity": "m/s",
        "temperature": "deg c",
        "fraction": "fraction",
    },
}

_FLOW_UNITS: dict[str, str] = {
    "CFS": "cfs",
    "GPM": "gpm",
    "MGD": "mgd",
    "CMS": "cms",
    "LPS": "lps",
    "MLD": "mld",
}

_CONCENTRATION_UNITS: dict[str, UnitLabel] = {
    "MG": "mg/l",
    "UG": "ug/l",
    "COUNT": "#/l",
    "NONE": None,
}

_ATTRIBUTE_UNIT_KEYS: dict[OutputKind, dict[str, str]] = {
    "sub": {
        "rainfall": "rainfall_intensity",
        "snow_depth": "depth",
        "evap_loss": "evaporation",
        "infil_loss": "rainfall_intensity",
        "runoff_rate": "flow",
        "gw_outflow_rate": "flow",
        "gw_table_elev": "elevation",
        "soil_moisture": "fraction",
    },
    "node": {
        "invert_depth": "depth",
        "hydraulic_head": "depth",
        "ponded_volume": "volume",
        "lateral_inflow": "flow",
        "total_inflow": "flow",
        "flooding_losses": "flow",
    },
    "link": {
        "flow_rate": "flow",
        "flow_depth": "depth",
        "flow_velocity": "velocity",
        "flow_volume": "volume",
        "capacity": "fraction",
    },
    "sys": {
        "air_temp": "temperature",
        "rainfall": "rainfall_intensity",
        "snow_depth": "depth",
        "evap_infil_loss": "rainfall_intensity",
        "runoff_flow": "flow",
        "dry_weather_inflow": "flow",
        "gw_inflow": "flow",
        "rdii_inflow": "flow",
        "direct_inflow": "flow",
        "total_lateral_inflow": "flow",
        "flood_losses": "flow",
        "outfall_flows": "flow",
        "volume_stored": "volume",
        "evap_rate": "evaporation",
        "ptnl_evap_rate": "evaporation",
    },
}


def _normalize_unit_label(unit: UnitLabel) -> UnitLabel:
    return unit.lower() if isinstance(unit, str) else None


def resolve_flow_unit(flow_unit: str) -> str:
    """Resolve a SWMM flow unit label.

    Parameters
    ----------
    flow_unit : str
        SWMM flow unit code.

    Returns
    -------
    str
        Normalized lowercase flow unit label.

    Raises
    ------
    KeyError
        If ``flow_unit`` is not a supported SWMM flow unit code.
    """
    try:
        return _FLOW_UNITS[flow_unit.upper()]
    except KeyError as exc:
        raise KeyError(f"Unsupported SWMM flow unit: {flow_unit!r}") from exc


def resolve_concentration_unit(concentration_unit: str) -> UnitLabel:
    """Resolve a SWMM pollutant concentration unit label.

    Parameters
    ----------
    concentration_unit : str
        SWMM concentration unit code.

    Returns
    -------
    UnitLabel
        Normalized lowercase concentration unit label, or ``None`` when the
        pollutant has no unit.

    Raises
    ------
    KeyError
        If ``concentration_unit`` is not a supported SWMM concentration unit
        code.
    """
    try:
        return _CONCENTRATION_UNITS[concentration_unit.upper()]
    except KeyError as exc:
        raise KeyError(
            f"Unsupported SWMM concentration unit: {concentration_unit!r}",
        ) from exc


def build_pollutant_unit_map(
    pollutants: Sequence[str],
    concentration_units: Sequence[str],
) -> dict[str, UnitLabel]:
    """Build a pollutant-to-unit mapping for output unit resolution.

    Parameters
    ----------
    pollutants : Sequence[str]
        Pollutant names from the SWMM model.
    concentration_units : Sequence[str]
        SWMM concentration unit codes aligned with ``pollutants``.

    Returns
    -------
    dict[str, UnitLabel]
        Mapping of lowercase pollutant names to normalized concentration unit
        labels.

    Raises
    ------
    ValueError
        If the pollutant names and concentration unit codes do not align after
        dropping ``"NONE"`` entries from ``concentration_units``.
    """
    concentration_units = [v for v in concentration_units if v != "NONE"]
    if len(pollutants) != len(concentration_units):
        raise ValueError(
            "Pollutant names and concentration units must have the same length",
        )

    return {
        pollutant.lower(): _normalize_unit_label(
            resolve_concentration_unit(concentration_unit),
        )
        for pollutant, concentration_unit in zip(
            pollutants,
            concentration_units,
            strict=True,
        )
    }


def resolve_output_unit(
    kind: OutputKind,
    attribute: str,
    unit_system: str,
    flow_unit: str,
    pollutant_units: Mapping[str, UnitLabel] | None = None,
) -> UnitLabel:
    """Resolve the unit label for a SWMM output attribute.

    Parameters
    ----------
    kind : OutputKind
        SWMM output object kind, such as ``"sub"``, ``"node"``, ``"link"``,
        or ``"sys"``.
    attribute : str
        Output attribute name for the requested object kind.
    unit_system : str
        SWMM unit system code, typically ``"US"`` or ``"SI"``.
    flow_unit : str
        SWMM flow unit code used when the attribute maps to flow.
    pollutant_units : Mapping[str, UnitLabel] | None, optional
        Mapping of lowercase pollutant names to resolved concentration unit
        labels.

    Returns
    -------
    UnitLabel
        Normalized lowercase unit label for the requested output attribute, or
        ``None`` when the attribute has no unit.

    Raises
    ------
    KeyError
        If the output kind, attribute, flow unit, or unit system is not
        supported.
    """
    normalized_kind = kind.lower()
    normalized_attribute = attribute.lower()
    normalized_unit_system = unit_system.upper()

    if pollutant_units is not None and normalized_attribute in pollutant_units:
        return _normalize_unit_label(pollutant_units[normalized_attribute])

    try:
        unit_key = _ATTRIBUTE_UNIT_KEYS[normalized_kind][normalized_attribute]
    except KeyError as exc:
        raise KeyError(
            f"Unsupported SWMM output attribute mapping for {kind!r}/{attribute!r}",
        ) from exc

    if unit_key == "flow":
        return _normalize_unit_label(resolve_flow_unit(flow_unit))

    try:
        return _normalize_unit_label(_UNIT_LABELS[normalized_unit_system][unit_key])
    except KeyError as exc:
        raise KeyError(f"Unsupported SWMM unit system: {unit_system!r}") from exc
