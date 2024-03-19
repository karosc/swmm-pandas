from __future__ import annotations
from swmm.pandas.input._section_classes import *

_sections = {
    # TODO build parser for this table
    "TITLE": Section,
    "OPTION": Option,
    # TODO build parser for this table
    # _section_props(ncols=3, col_names=["Action", "File Type", "File Path"]),
    "FILE": Section,
    "RAINGAGE": Raingage,
    "TEMPERATURE": Temperature,
    "EVAP": Evap,
    "SUBCATCHMENT": Subcatchment,
    "SUBAREA": Subarea,
    "INFIL": Infil,
    "AQUIFER": Aquifer,
    "GROUNDWATER": Groundwater,
    "SNOWPACK": Snowpack,
    "JUNC": Junc,
    "OUTFALL": Outfall,
    "STORAGE": Storage,
    "DIVIDER": Divider,
    "CONDUIT": Conduit,
    "PUMP": Pump,
    "ORIFICE": Orifice,
    "WEIR": Weir,
    "OUTLET": Outlet,
    "XSECT": Xsections,
    # TODO build parser for this table
    "TRANSECT": Section,
    "LOSS": Losses,
    # TODO build parser for this table
    "CONTROL": Section,
    "POLLUT": Pollutants,
    "LANDUSE": LandUse,
    "BUILDUP": Buildup,
    "WASHOFF": Washoff,
    "COVERAGE": Section,
    "INFLOW": Inflow,
    "DWF": DWF,
    # TODO build parser for this table
    "PATTERN": Section,
    "RDII": RDII,
    # TODO build parser for this table
    "HYDROGRAPH": Section,
    # TODO build parser for this table
    "LOADING": Section,
    # TODO build parser for this table
    "TREATMENT": Section,
    # TODO build parser for this table
    "CURVE": Section,
    # TODO build parser for this table
    "TIMESERIES": Section,
    # TODO build parser for this table
    "REPORT": Section,  # _section_props(ncols=2, col_names=["Option", "Value"]),
    "MAP": Map,
    "COORDINATE": Coordinates,
    "VERTICES": Verticies,
    "POLYGON": Polygons,
    "SYMBOL": Symbols,
    "LABEL": Labels,
    "BACKDROP": Backdrop,
    "TAG": Tags,
    "PROFILE": Section,
    "LID_CONTROL": LID_Control,
    "LID_USAGE": LID_Usage,
    "GWF": Section,  # _section_props(ncols=3, col_names=["Subcatchment", "Type", "Expr"]),
    "ADJUSTMENT": Adjustments,
    "EVENT": Section,
}
