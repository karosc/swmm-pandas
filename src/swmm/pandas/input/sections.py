from __future__ import annotations
from swmm.pandas.input._section_classes import *

_sections = {
    # TODO build parser for this table
    "TITLE": Section,
    "OPTION": Option,
    # TODO build parser for this table
    "REPORT": Section,  # _section_props(ncols=2, col_names=["Option", "Value"]),
    "EVENT": Section,
    # TODO build parser for this table
    # _section_props(ncols=3, col_names=["Action", "File Type", "File Path"]),
    "FILE": Section,
    "RAINGAGE": Raingage,
    "EVAP": Evap,
    "TEMPERATURE": Temperature,
    "ADJUSTMENT": Adjustments,
    "SUBCATCHMENT": Subcatchment,
    "SUBAREA": Subarea,
    "INFIL": Infil,
    "LID_CONTROL": LID_Control,
    "LID_USAGE": LID_Usage,
    "AQUIFER": Aquifer,
    "GROUNDWATER": Groundwater,
    "GWF": Section,  # _section_props(ncols=3, col_names=["Subcatchment", "Type", "Expr"]),
    "SNOWPACK": Snowpack,
    "JUNC": Junc,
    "OUTFALL": Outfall,
    "DIVIDER": Divider,
    "STORAGE": Storage,
    "CONDUIT": Conduit,
    "PUMP": Pump,
    "ORIFICE": Orifice,
    "WEIR": Weir,
    "OUTLET": Outlet,
    "XSECT": Xsections,
    # TODO build parser for this table
    "TRANSECT": Section,
    "STREETS": Street,
    "INLETS": Inlet,
    "INLET_USAGE": Inlet_Usage,
    "LOSS": Losses,
    # TODO build parser for this table
    "CONTROL": Section,
    "POLLUT": Pollutants,
    "LANDUSE": LandUse,
    "COVERAGE": Coverage,
    # TODO build parser for this table
    "LOADING": Section,
    "BUILDUP": Buildup,
    "WASHOFF": Washoff,
    "TREATMENT": Section,
    # TODO build parser for this table
    "INFLOW": Inflow,
    "DWF": DWF,
    "RDII": RDII,
    # TODO build parser for this table
    "HYDROGRAPH": Section,
    # TODO build parser for this table
    "CURVE": Section,
    # TODO build parser for this table
    "TIMESERIES": Section,
    # TODO build parser for this table    
    "PATTERN": Section,
        
    "MAP": Map,    
    "POLYGON": Polygons,
    "COORDINATE": Coordinates,
    "VERTICES": Verticies,
    "LABEL": Labels,
    "SYMBOL": Symbols,
    "BACKDROP": Backdrop,
    
    "PROFILE": Section,   
    "TAG": Tags,
}
