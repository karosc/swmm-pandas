# swmm-pandas input
# scope:
#   - high level api for loading, inspecting, changing, and
#     altering a SWMM input file using pandas dataframes
from __future__ import annotations

from swmm.pandas.input._section_classes import SectionBase, _sections
import swmm.pandas.input._section_classes as sc
import pathlib
import re
import warnings


class Input:
    _section_re = re.compile(R"^\[[\s\S]*?(?=^\[)", re.MULTILINE)
    _section_keys = tuple(_sections.keys())

    title: sc.Title
    "string type"
    option: sc.Option
    "['Option', 'Value', 'desc']"
    report: sc.Report
    "Custom class"
    event: sc.Event
    "['Start', 'End', 'desc']"
    files: sc.Files
    "string type"
    raingage: sc.Raingage
    "['Name', 'Format', 'Interval', 'SCF', 'Source_Type', 'Source', 'Station', 'Units', 'desc']"
    evap: sc.Evap
    "['Type', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9', 'param10', 'param11', 'param12', 'desc']"
    temperature: sc.Temperature
    "['Option', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9', 'param10', 'param11', 'param12', 'param13', 'desc']"
    adjustments: sc.Adjustments
    "['Parameter', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'desc']"
    subcatchment: sc.Subcatchment
    "['Name', 'RainGage', 'Outlet', 'Area', 'PctImp', 'Width', 'Slope', 'CurbLeng', 'SnowPack', 'desc']"
    subarea: sc.Subarea
    "['Subcatchment', 'Nimp', 'Nperv', 'Simp', 'Sperv', 'PctZero', 'RouteTo', 'PctRouted', 'desc']"
    infil: sc.Infil
    "['Subcatchment', 'param1', 'param2', 'param3', 'param4', 'param5', 'Method', 'desc']"
    lid_control: sc.LID_Control
    "['Name', 'Type', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'desc']"
    lid_usage: sc.LID_Usage
    "['Subcatchment', 'LIDProcess', 'Number', 'Area', 'Width', 'InitSat', 'FromImp', 'ToPerv', 'RptFile', 'DrainTo', 'FromPerv', 'desc']"
    aquifer: sc.Aquifer
    "['Name', 'Por', 'WP', 'FC', 'Ksat', 'Kslope', 'Tslope', 'ETu', 'ETs', 'Seep', 'Ebot', 'Egw', 'Umc', 'ETupat', 'desc']"
    groundwater: sc.Groundwater
    "['Subcatchment', 'Aquifer', 'Node', 'Esurf', 'A1', 'B1', 'A2', 'B2', 'A3', 'Dsw', 'Egwt', 'Ebot', 'Wgr', 'Umc', 'desc']"
    gwf: sc.GWF
    "['Subcatch', 'Type', 'Expr', 'desc']"
    snowpack: sc.Snowpack
    "['Name', 'Surface', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'desc']"
    junc: sc.Junc
    "['Name', 'Elevation', 'MaxDepth', 'InitDepth', 'SurDepth', 'Aponded', 'desc']"
    outfall: sc.Outfall
    "['Name', 'Elevation', 'Type', 'StageData', 'Gated', 'RouteTo', 'desc']"
    divider: sc.Divider
    "['Name', 'Elevation', 'DivLink', 'DivType', 'DivCurve', 'Qmin', 'Height', 'Cd', 'Ymax', 'Y0', 'Ysur', 'Apond', 'desc']"
    storage: sc.Storage
    "['Name', 'Elev', 'MaxDepth', 'InitDepth', 'Shape', 'CurveName', 'A1_L', 'A2_W', 'A0_Z', 'SurDepth', 'Fevap', 'Psi', 'Ksat', 'IMD', 'desc']"
    conduit: sc.Conduit
    "['Name', 'FromNode', 'ToNode', 'Length', 'Roughness', 'InOffset', 'OutOffset', 'InitFlow', 'MaxFlow', 'desc']"
    pump: sc.Pump
    "['Name', 'FromNode', 'ToNode', 'PumpCurve', 'Status', 'Startup', 'Shutoff', 'desc']"
    orifice: sc.Orifice
    "['Name', 'FromNode', 'ToNode', 'Type', 'Offset', 'Qcoeff', 'Gated', 'CloseTime', 'desc']"
    weir: sc.Weir
    "['Name', 'FromNode', 'ToNode', 'Type', 'CrestHt', 'Qcoeff', 'Gated', 'EndCon', 'EndCoeff', 'Surcharge', 'RoadWidth', 'RoadSurf', 'CoeffCurve', 'desc']"
    outlet: sc.Outlet
    "['Name', 'FromNode', 'ToNode', 'Offset', 'Type', 'CurveName', 'Qcoeff', 'Qexpon', 'Gated', 'desc']"
    xsections: sc.Xsections
    "['Link', 'Shape', 'Geom1', 'Curve', 'Geom2', 'Geom3', 'Geom4', 'Barrels', 'Culvert', 'desc']"
    transects: sc.Transects
    "string type"
    street: sc.Street
    "['Name', 'Tcrown', 'Hcurb', 'Sroad', 'nRoad', 'Hdep', 'Wdep', 'Sides', 'Wback', 'Sback', 'nBack', 'desc']"
    inlet_usage: sc.Inlet_Usage
    "['Conduit', 'Inlet', 'Node', 'Number', '%Clogged', 'MaxFlow', 'hDStore', 'wDStore', 'Placement', 'desc']"
    inlet: sc.Inlet
    "['Name', 'Type', 'param1', 'param2', 'param3', 'param4', 'param5', 'desc']"
    losses: sc.Losses
    "['Link', 'Kentry', 'Kexit', 'Kavg', 'FlapGate', 'Seepage', 'desc']"
    controls: sc.Controls
    pollutants: sc.Pollutants
    "['Name', 'Units', 'Crain', 'Cgw', 'Crdii', 'Kdecay', 'SnowOnly', 'CoPollutant', 'CoFrac', 'Cdwf', 'Cinit', 'desc']"
    landuse: sc.LandUse
    "['Name', 'SweepInterval', 'Availability', 'LastSweep', 'desc']"
    coverage: sc.Coverage
    "['Subcatchment', 'landuse', 'Percent', 'desc']"
    loading: sc.Loading
    "['Subcatchment', 'Pollutant', 'InitBuildup', 'desc']"
    buildup: sc.Buildup
    "['Landuse', 'Pollutant', 'FuncType', 'C1', 'C2', 'C3', 'PerUnit', 'desc']"
    washoff: sc.Washoff
    "['Landuse', 'Pollutant', 'FuncType', 'C1', 'C2', 'SweepRmvl', 'BmpRmvl', 'desc']"
    treatment: sc.Treatment
    "['Node', 'Pollutant', 'Func', 'desc']"
    inflow: sc.Inflow
    "['Node', 'Constituent', 'TimeSeries', 'Type', 'Mfactor', 'Sfactor', 'Baseline', 'Pattern', 'desc']"
    dwf: sc.DWF
    "['Node', 'Constituent', 'Baseline', 'Pat1', 'Pat2', 'Pat3', 'Pat4', 'desc']"
    rdii: sc.RDII
    "['Node', 'UHgroup', 'SewerArea', 'desc']"
    hydrographs: sc.Hydrographs
    "['Name', 'Month_RG', 'Response', 'R', 'T', 'K', 'IA_max', 'IA_rec', 'IA_ini', 'desc']"
    curves: sc.Curves
    "['Name', 'Type', 'X_Value', 'Y_Value', 'desc']"
    timeseries: sc.Timeseries
    "dict of timeseries dataframes or TimeseriesFile objects."
    patterns: sc.Patterns
    "['Name', 'Type', 'Multiplier', 'desc']"
    map: sc.Map
    "string type"
    polygons: sc.Polygons
    "['Subcatch', 'X', 'Y', 'desc']"
    coordinates: sc.Coordinates
    "['Node', 'X', 'Y', 'desc']"
    vertices: sc.Vertices
    "['Link', 'X', 'Y', 'desc']"
    labels: sc.Labels
    "['Xcoord', 'Ycoord', 'Label', 'Anchor', 'Font', 'Size', 'Bold', 'Italic', 'desc']"
    symbols: sc.Symbols
    "['Gage', 'X', 'Y', 'desc']"
    backdrop: sc.Backdrop
    "string type"
    profile: sc.Profile
    "string type"
    tags: sc.Tags
    "['Element', 'Name', 'Tag', 'desc']"

    def __init__(self, inpfile: str):
        """Base class for a SWMM input file.

        The input object provides an attribute for each section supported the SWMM inp file. The
        section properties are created dynamically at runtime to keep source code dry and concise, but
        typehints provide minimal docstrings for dataframe column names. Most sections are represented
        by a pandas dataframe, with the exception of a few.

        This class was written based on the `SWMM Users Manual`_, any parsing bugs might require bug reports to the
        USEPA repo for updates to the users manual.

        .. DANGER::
            This class provides **minimal to no error checking** on your input file when loading it or writing it.

            When creating new model elements or updating the properties of existing ones, swmm.pandas expects
            that the user knows what they are doing.

            Just because swmm.pandas allows you to do something, does not mean SWMM will accept it.

        .. _SWMM Users Manual: https://www.epa.gov/system/files/documents/2022-04/swmm-users-manual-version-5.2.pdf

        .. code-block:: python

            # Using a the _close method
            >>> from swmm.pandas import Input
            >>> inp = Input('tests/data/bench_inp.inp')
            >>> print(inp.option.head())
                               Value desc
            Option
            FLOW_UNITS           CFS
            INFILTRATION  GREEN_AMPT
            FLOW_ROUTING     DYNWAVE
            LINK_OFFSETS       DEPTH
            MIN_SLOPE              0
            >>> print(inp.junc.head())
                  Elevation MaxDepth InitDepth SurDepth Aponded desc
            Name
            JUNC1       1.5    10.25         0        0    5000
            JUNC2     -1.04      6.2         0        0    5000
            JUNC3     -3.47     11.5         0        0    5000
            JUNC4     -5.25     13.8         0        0    5000
            JUNC6         0        9         0      200       0
            >>> inp.junc['Elevation']+=100
            >>> inp.junc['Elevation']
            Name
            JUNC1    101.5
            JUNC2    98.96
            JUNC3    96.53
            JUNC4    94.75
            JUNC6      100
            Name: Elevation, dtype: object
            >>> inp.to_file('new_inp_file.inp')

        Parameters
        ----------
        inpfile: str
            model inp file path
        """

        self.path: str = inpfile

        self._load_inp_file()
        for sect in _sections.keys():
            # print(sect)
            self._set_section_prop(sect)

    def _load_inp_file(self) -> None:
        with open(self.path, "r") as inp:
            self.text: str = inp.read()

        self._sections: dict[str, SectionBase] = {}
        self._section_texts: dict[str, str] = {}

        for section in self._section_re.findall(self.text):
            name = re.findall(R"^\[(.*)\]", section)[0]

            data = "\n".join(re.findall(R"^(?!;{2,}|\[).+$", section, re.MULTILINE))

            try:
                section_idx = list(
                    (name.lower().startswith(x.lower()) for x in _sections)
                ).index(True)
                section_key = self._section_keys[section_idx]
                self._section_texts[section_key] = data
            except Exception as e:
                print(e)
                self._sections[name] = data
                # self.__setattr__(name.lower(), "Not Implemented")

                print(f"Section {name} not yet supported")

    def _get_section(self, key):
        if key in self._section_texts:
            data = self._section_texts[key]
            return _sections[key].from_section_text(data)

        else:
            return _sections[key]._new_empty()

    @classmethod
    def _set_section_prop(cls, section: str) -> None:
        section_class = _sections[section]
        public_property_name = section_class.__name__.lower()
        private_property_name = f"_{public_property_name}"

        def getter(self):
            if not hasattr(self, private_property_name):
                setattr(self, private_property_name, self._get_section(section))
            return getattr(self, private_property_name)

        def setter(self, obj):
            setattr(self, private_property_name, section_class._newobj(obj))

        setattr(cls, public_property_name, property(getter, setter))

    def to_string(self):
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            out_str = ""
            for sect in _sections.keys():
                section_class = _sections[sect]
                public_property_name = section_class.__name__.lower()
                # private_property_name = f"_{public_property_name}"
                if len(sect_obj := getattr(self, public_property_name)) > 0:
                    sect_string = sect_obj.to_swmm_string()
                    out_str += f"[{sect.upper()}]\n{sect_string}\n\n"
            return out_str

    def to_file(self, path: str | pathlib.Path):
        with open(path, "w") as f:
            f.write(self.to_string())
