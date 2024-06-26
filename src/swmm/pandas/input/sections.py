from __future__ import annotations
from swmm.pandas.input._section_classes import *

_sections = {
    "TITLE": Title,
    "OPTION": Option,
    "REPORT": Report,
    "EVENT": Event,
    "FILE": Files,
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
    "GWF": GWF,  # _section_props(ncols=3, col_names=["Subcatchment", "Type", "Expr"]),
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
    "TRANSECT": SectionDf,
    "STREETS": Street,
    "INLETS": Inlet,
    "INLET_USAGE": Inlet_Usage,
    "LOSS": Losses,
    # TODO build parser for this table
    "CONTROL": SectionDf,
    "POLLUT": Pollutants,
    "LANDUSE": LandUse,
    "COVERAGE": Coverage,
    # TODO build parser for this table
    "LOADING": SectionDf,
    "BUILDUP": Buildup,
    "WASHOFF": Washoff,
    "TREATMENT": SectionDf,
    # TODO build parser for this table
    "INFLOW": Inflow,
    "DWF": DWF,
    "RDII": RDII,
    # TODO build parser for this table
    "HYDROGRAPH": SectionDf,
    # TODO build parser for this table
    "CURVE": SectionDf,
    # TODO build parser for this table
    "TIMESERIES": SectionDf,
    # TODO build parser for this table    
    "PATTERN": SectionDf,
        
    "MAP": Map,    
    "POLYGON": Polygons,
    "COORDINATE": Coordinates,
    "VERTICES": Verticies,
    "LABEL": Labels,
    "SYMBOL": Symbols,
    "BACKDROP": Backdrop,
    
    "PROFILE": SectionDf,   
    "TAG": Tags,
}
