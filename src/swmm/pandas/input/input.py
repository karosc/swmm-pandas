# swmm-pandas input
# scope:
#   - high level api for loading, inspecting, changing, and
#     altering a SWMM input file using pandas dataframes
from __future__ import annotations
from tokenize import String
from venv import logger

from swmm.pandas.input._section_classes import SectionBase, _sections
import swmm.pandas.input._section_classes as sc
import pathlib
import re
import warnings
from io import StringIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swmm.pandas.input.model import Input
    from typing import Optional, Self


class InputFile:
    _section_re = re.compile(R"^\[[\s\S]*?(?=^\[|\Z)", re.MULTILINE)
    _section_keys = tuple(_sections.keys())

    def __init__(self, inpfile: Optional[str | pathlib.Path | StringIO] = None):
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
        if inpfile is not None:
            self._inpfile = inpfile
            self._load_inp_file()
        # for sect in _sections.keys():
        #     # print(sect)
        #     self._set_section_prop(sect)

    def _load_inp_file(self) -> None:

        if isinstance(self._inpfile, (str, pathlib.Path)):
            with open(self._inpfile) as inp:
                self.text: str = inp.read()
        elif isinstance(self._inpfile, StringIO):
            self.text: str = self._inpfile.read()
        else:
            raise TypeError(
                f"InputFile class expected string, path, or StringIO, got {type(self._inpfile)}"
            )
        self._sections: dict[str, SectionBase] = {}
        self._section_texts: dict[str, str] = {}

        for section in self._section_re.findall(self.text):
            name = re.findall(R"^\[(.*)\]", section)[0]

            data = "\n".join(re.findall(R"^(?!;{2,}|\[).+$", section, re.MULTILINE))

            try:
                section_idx = list(
                    name.lower().startswith(x.lower()) for x in _sections
                ).index(True)
                section_key = self._section_keys[section_idx]
                self._section_texts[section_key] = data
            except Exception as e:
                logger.error(f"Error parsing section: {name}")
                raise e
                # print(e)
                # self._sections[name] = data
                # self.__setattr__(name.lower(), "Not Implemented")

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
                    try:
                        sect_string = sect_obj.to_swmm_string()
                        out_str += f"[{sect_obj._section_name}]\n{sect_string}\n\n"
                    except Exception as e:
                        print(f"error parsing {sect_obj.__class__.__name__}")
                        raise e

            return out_str

    def to_file(self, path: str | pathlib.Path):
        with open(path, "w") as f:
            f.write(self.to_string())

    # region
    # This section is autgenerated by scripts/generate_input_sections.py

    @property
    def title(self) -> sc.Title:

        if not hasattr(self, "_title"):
            self._title = self._get_section("TITLE")

        return self._title

    @title.setter
    def title(self, obj) -> None:
        self._title = sc.Title._newobj(obj)

    @property
    def option(self) -> sc.Option:
        "('Option')['Value', 'desc']"

        if not hasattr(self, "_option"):
            self._option = self._get_section("OPTION")

        return self._option

    @option.setter
    def option(self, obj) -> None:
        self._option = sc.Option._newobj(obj)

    @property
    def files(self) -> sc.Files:
        "String to hold files section"

        if not hasattr(self, "_files"):
            self._files = self._get_section("FILE")

        return self._files

    @files.setter
    def files(self, obj) -> None:
        self._files = sc.Files._newobj(obj)

    @property
    def raingage(self) -> sc.Raingage:
        "('Name')['Format', 'Interval', 'SCF', 'Source_Type', 'Source', 'Station', 'Units', 'desc']"

        if not hasattr(self, "_raingage"):
            self._raingage = self._get_section("RAINGAGE")

        return self._raingage

    @raingage.setter
    def raingage(self, obj) -> None:
        self._raingage = sc.Raingage._newobj(obj)

    @property
    def evap(self) -> sc.Evap:
        "('Type')['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9', 'param10', 'param11', 'param12', 'desc']"

        if not hasattr(self, "_evap"):
            self._evap = self._get_section("EVAP")

        return self._evap

    @evap.setter
    def evap(self, obj) -> None:
        self._evap = sc.Evap._newobj(obj)

    @property
    def temperature(self) -> sc.Temperature:
        "('Option')['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9', 'param10', 'param11', 'param12', 'param13', 'desc']"

        if not hasattr(self, "_temperature"):
            self._temperature = self._get_section("TEMPERATURE")

        return self._temperature

    @temperature.setter
    def temperature(self, obj) -> None:
        self._temperature = sc.Temperature._newobj(obj)

    @property
    def subcatchment(self) -> sc.Subcatchment:
        "('Name')['RainGage', 'Outlet', 'Area', 'PctImp', 'Width', 'Slope', 'CurbLeng', 'SnowPack', 'desc']"

        if not hasattr(self, "_subcatchment"):
            self._subcatchment = self._get_section("SUBCATCHMENT")

        return self._subcatchment

    @subcatchment.setter
    def subcatchment(self, obj) -> None:
        self._subcatchment = sc.Subcatchment._newobj(obj)

    @property
    def subarea(self) -> sc.Subarea:
        "('Subcatchment')['Nimp', 'Nperv', 'Simp', 'Sperv', 'PctZero', 'RouteTo', 'PctRouted', 'desc']"

        if not hasattr(self, "_subarea"):
            self._subarea = self._get_section("SUBAREA")

        return self._subarea

    @subarea.setter
    def subarea(self, obj) -> None:
        self._subarea = sc.Subarea._newobj(obj)

    @property
    def infil(self) -> sc.Infil:
        "('Subcatchment')['param1', 'param2', 'param3', 'param4', 'param5', 'Method', 'desc']"

        if not hasattr(self, "_infil"):
            self._infil = self._get_section("INFIL")

        return self._infil

    @infil.setter
    def infil(self, obj) -> None:
        self._infil = sc.Infil._newobj(obj)

    @property
    def lid_control(self) -> sc.LID_Control:
        "('Name')['Type', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'desc']"

        if not hasattr(self, "_lid_control"):
            self._lid_control = self._get_section("LID_CONTROL")

        return self._lid_control

    @lid_control.setter
    def lid_control(self, obj) -> None:
        self._lid_control = sc.LID_Control._newobj(obj)

    @property
    def lid_usage(self) -> sc.LID_Usage:
        "('Subcatchment', 'LIDProcess')['Number', 'Area', 'Width', 'InitSat', 'FromImp', 'ToPerv', 'RptFile', 'DrainTo', 'FromPerv', 'desc']"

        if not hasattr(self, "_lid_usage"):
            self._lid_usage = self._get_section("LID_USAGE")

        return self._lid_usage

    @lid_usage.setter
    def lid_usage(self, obj) -> None:
        self._lid_usage = sc.LID_Usage._newobj(obj)

    @property
    def aquifer(self) -> sc.Aquifer:
        "('Name')['Por', 'WP', 'FC', 'Ksat', 'Kslope', 'Tslope', 'ETu', 'ETs', 'Seep', 'Ebot', 'Egw', 'Umc', 'ETupat', 'desc']"

        if not hasattr(self, "_aquifer"):
            self._aquifer = self._get_section("AQUIFER")

        return self._aquifer

    @aquifer.setter
    def aquifer(self, obj) -> None:
        self._aquifer = sc.Aquifer._newobj(obj)

    @property
    def groundwater(self) -> sc.Groundwater:
        "('Subcatchment')['Aquifer', 'Node', 'Esurf', 'A1', 'B1', 'A2', 'B2', 'A3', 'Dsw', 'Egwt', 'Ebot', 'Wgr', 'Umc', 'desc']"

        if not hasattr(self, "_groundwater"):
            self._groundwater = self._get_section("GROUNDWATER")

        return self._groundwater

    @groundwater.setter
    def groundwater(self, obj) -> None:
        self._groundwater = sc.Groundwater._newobj(obj)

    @property
    def gwf(self) -> sc.GWF:
        "('Subcatch', 'Type')['Expr', 'desc']"

        if not hasattr(self, "_gwf"):
            self._gwf = self._get_section("GWF")

        return self._gwf

    @gwf.setter
    def gwf(self, obj) -> None:
        self._gwf = sc.GWF._newobj(obj)

    @property
    def snowpack(self) -> sc.Snowpack:
        "('Name', 'Surface')['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'desc']"

        if not hasattr(self, "_snowpack"):
            self._snowpack = self._get_section("SNOWPACK")

        return self._snowpack

    @snowpack.setter
    def snowpack(self, obj) -> None:
        self._snowpack = sc.Snowpack._newobj(obj)

    @property
    def junc(self) -> sc.Junc:
        "('Name')['Elevation', 'MaxDepth', 'InitDepth', 'SurDepth', 'Aponded', 'desc']"

        if not hasattr(self, "_junc"):
            self._junc = self._get_section("JUNC")

        return self._junc

    @junc.setter
    def junc(self, obj) -> None:
        self._junc = sc.Junc._newobj(obj)

    @property
    def outfall(self) -> sc.Outfall:
        "('Name')['Elevation', 'Type', 'StageData', 'Gated', 'RouteTo', 'desc']"

        if not hasattr(self, "_outfall"):
            self._outfall = self._get_section("OUTFALL")

        return self._outfall

    @outfall.setter
    def outfall(self, obj) -> None:
        self._outfall = sc.Outfall._newobj(obj)

    @property
    def divider(self) -> sc.Divider:
        "('Name')['Elevation', 'DivLink', 'DivType', 'DivCurve', 'Qmin', 'Height', 'Cd', 'Ymax', 'Y0', 'Ysur', 'Apond', 'desc']"

        if not hasattr(self, "_divider"):
            self._divider = self._get_section("DIVIDER")

        return self._divider

    @divider.setter
    def divider(self, obj) -> None:
        self._divider = sc.Divider._newobj(obj)

    @property
    def storage(self) -> sc.Storage:
        "('Name')['Elev', 'MaxDepth', 'InitDepth', 'Shape', 'CurveName', 'A1_L', 'A2_W', 'A0_Z', 'SurDepth', 'Fevap', 'Psi', 'Ksat', 'IMD', 'desc']"

        if not hasattr(self, "_storage"):
            self._storage = self._get_section("STORAGE")

        return self._storage

    @storage.setter
    def storage(self, obj) -> None:
        self._storage = sc.Storage._newobj(obj)

    @property
    def conduit(self) -> sc.Conduit:
        "('Name')['FromNode', 'ToNode', 'Length', 'Roughness', 'InOffset', 'OutOffset', 'InitFlow', 'MaxFlow', 'desc']"

        if not hasattr(self, "_conduit"):
            self._conduit = self._get_section("CONDUIT")

        return self._conduit

    @conduit.setter
    def conduit(self, obj) -> None:
        self._conduit = sc.Conduit._newobj(obj)

    @property
    def pump(self) -> sc.Pump:
        "('Name')['FromNode', 'ToNode', 'PumpCurve', 'Status', 'Startup', 'Shutoff', 'desc']"

        if not hasattr(self, "_pump"):
            self._pump = self._get_section("PUMP")

        return self._pump

    @pump.setter
    def pump(self, obj) -> None:
        self._pump = sc.Pump._newobj(obj)

    @property
    def orifice(self) -> sc.Orifice:
        "('Name')['FromNode', 'ToNode', 'Type', 'Offset', 'Qcoeff', 'Gated', 'CloseTime', 'desc']"

        if not hasattr(self, "_orifice"):
            self._orifice = self._get_section("ORIFICE")

        return self._orifice

    @orifice.setter
    def orifice(self, obj) -> None:
        self._orifice = sc.Orifice._newobj(obj)

    @property
    def weir(self) -> sc.Weir:
        "('Name')['FromNode', 'ToNode', 'Type', 'CrestHt', 'Qcoeff', 'Gated', 'EndCon', 'EndCoeff', 'Surcharge', 'RoadWidth', 'RoadSurf', 'CoeffCurve', 'desc']"

        if not hasattr(self, "_weir"):
            self._weir = self._get_section("WEIR")

        return self._weir

    @weir.setter
    def weir(self, obj) -> None:
        self._weir = sc.Weir._newobj(obj)

    @property
    def outlet(self) -> sc.Outlet:
        "('Name')['FromNode', 'ToNode', 'Offset', 'Type', 'CurveName', 'Qcoeff', 'Qexpon', 'Gated', 'desc']"

        if not hasattr(self, "_outlet"):
            self._outlet = self._get_section("OUTLET")

        return self._outlet

    @outlet.setter
    def outlet(self, obj) -> None:
        self._outlet = sc.Outlet._newobj(obj)

    @property
    def xsections(self) -> sc.Xsections:
        "('Link')['Shape', 'Geom1', 'Curve', 'Geom2', 'Geom3', 'Geom4', 'Barrels', 'Culvert', 'desc']"

        if not hasattr(self, "_xsections"):
            self._xsections = self._get_section("XSECT")

        return self._xsections

    @xsections.setter
    def xsections(self, obj) -> None:
        self._xsections = sc.Xsections._newobj(obj)

    @property
    def transects(self) -> sc.Transects:
        "String to hold transects section."

        if not hasattr(self, "_transects"):
            self._transects = self._get_section("TRANSECT")

        return self._transects

    @transects.setter
    def transects(self, obj) -> None:
        self._transects = sc.Transects._newobj(obj)

    @property
    def street(self) -> sc.Street:
        "('Name')['Tcrown', 'Hcurb', 'Sroad', 'nRoad', 'Hdep', 'Wdep', 'Sides', 'Wback', 'Sback', 'nBack', 'desc']"

        if not hasattr(self, "_street"):
            self._street = self._get_section("STREETS")

        return self._street

    @street.setter
    def street(self, obj) -> None:
        self._street = sc.Street._newobj(obj)

    @property
    def inlet_usage(self) -> sc.Inlet_Usage:
        "('Conduit')['Inlet', 'Node', 'Number', '%Clogged', 'MaxFlow', 'hDStore', 'wDStore', 'Placement', 'desc']"

        if not hasattr(self, "_inlet_usage"):
            self._inlet_usage = self._get_section("INLET_USAGE")

        return self._inlet_usage

    @inlet_usage.setter
    def inlet_usage(self, obj) -> None:
        self._inlet_usage = sc.Inlet_Usage._newobj(obj)

    @property
    def inlet(self) -> sc.Inlet:
        "('Name', 'Type')['param1', 'param2', 'param3', 'param4', 'param5', 'desc']"

        if not hasattr(self, "_inlet"):
            self._inlet = self._get_section("INLET")

        return self._inlet

    @inlet.setter
    def inlet(self, obj) -> None:
        self._inlet = sc.Inlet._newobj(obj)

    @property
    def losses(self) -> sc.Losses:
        "('Link')['Kentry', 'Kexit', 'Kavg', 'FlapGate', 'Seepage', 'desc']"

        if not hasattr(self, "_losses"):
            self._losses = self._get_section("LOSS")

        return self._losses

    @losses.setter
    def losses(self, obj) -> None:
        self._losses = sc.Losses._newobj(obj)

    @property
    def controls(self) -> sc.Controls:
        "Dict of control rules stored as text."

        if not hasattr(self, "_controls"):
            self._controls = self._get_section("CONTROL")

        return self._controls

    @controls.setter
    def controls(self, obj) -> None:
        self._controls = sc.Controls._newobj(obj)

    @property
    def pollutants(self) -> sc.Pollutants:
        "('Name')['Units', 'Crain', 'Cgw', 'Crdii', 'Kdecay', 'SnowOnly', 'CoPollutant', 'CoFrac', 'Cdwf', 'Cinit', 'desc']"

        if not hasattr(self, "_pollutants"):
            self._pollutants = self._get_section("POLLUT")

        return self._pollutants

    @pollutants.setter
    def pollutants(self, obj) -> None:
        self._pollutants = sc.Pollutants._newobj(obj)

    @property
    def landuse(self) -> sc.LandUse:
        "('Name')['SweepInterval', 'Availability', 'LastSweep', 'desc']"

        if not hasattr(self, "_landuse"):
            self._landuse = self._get_section("LANDUSE")

        return self._landuse

    @landuse.setter
    def landuse(self, obj) -> None:
        self._landuse = sc.LandUse._newobj(obj)

    @property
    def coverage(self) -> sc.Coverage:
        "('Subcatchment', 'LandUse')['Percent', 'desc']"

        if not hasattr(self, "_coverage"):
            self._coverage = self._get_section("COVERAGE")

        return self._coverage

    @coverage.setter
    def coverage(self, obj) -> None:
        self._coverage = sc.Coverage._newobj(obj)

    @property
    def loading(self) -> sc.Loading:
        "('Subcatchment', 'Pollutant')['InitBuildup', 'desc']"

        if not hasattr(self, "_loading"):
            self._loading = self._get_section("LOADING")

        return self._loading

    @loading.setter
    def loading(self, obj) -> None:
        self._loading = sc.Loading._newobj(obj)

    @property
    def buildup(self) -> sc.Buildup:
        "('Landuse', 'Pollutant')['FuncType', 'C1', 'C2', 'C3', 'PerUnit', 'desc']"

        if not hasattr(self, "_buildup"):
            self._buildup = self._get_section("BUILDUP")

        return self._buildup

    @buildup.setter
    def buildup(self, obj) -> None:
        self._buildup = sc.Buildup._newobj(obj)

    @property
    def washoff(self) -> sc.Washoff:
        "('Landuse', 'Pollutant')['FuncType', 'C1', 'C2', 'SweepRmvl', 'BmpRmvl', 'desc']"

        if not hasattr(self, "_washoff"):
            self._washoff = self._get_section("WASHOFF")

        return self._washoff

    @washoff.setter
    def washoff(self, obj) -> None:
        self._washoff = sc.Washoff._newobj(obj)

    @property
    def treatment(self) -> sc.Treatment:
        "('Node', 'Pollutant')['Func', 'desc']"

        if not hasattr(self, "_treatment"):
            self._treatment = self._get_section("TREATMENT")

        return self._treatment

    @treatment.setter
    def treatment(self, obj) -> None:
        self._treatment = sc.Treatment._newobj(obj)

    @property
    def inflow(self) -> sc.Inflow:
        "('Node', 'Constituent')['TimeSeries', 'InflowType', 'Mfactor', 'Sfactor', 'Baseline', 'Pattern', 'desc']"

        if not hasattr(self, "_inflow"):
            self._inflow = self._get_section("INFLOW")

        return self._inflow

    @inflow.setter
    def inflow(self, obj) -> None:
        self._inflow = sc.Inflow._newobj(obj)

    @property
    def dwf(self) -> sc.DWF:
        "('Node', 'Constituent')['AvgValue', 'Pat1', 'Pat2', 'Pat3', 'Pat4', 'desc']"

        if not hasattr(self, "_dwf"):
            self._dwf = self._get_section("DWF")

        return self._dwf

    @dwf.setter
    def dwf(self, obj) -> None:
        self._dwf = sc.DWF._newobj(obj)

    @property
    def rdii(self) -> sc.RDII:
        "('Node')['UHgroup', 'SewerArea', 'desc']"

        if not hasattr(self, "_rdii"):
            self._rdii = self._get_section("RDII")

        return self._rdii

    @rdii.setter
    def rdii(self, obj) -> None:
        self._rdii = sc.RDII._newobj(obj)

    @property
    def hydrographs(self) -> sc.Hydrographs:
        "('Name', 'Month_RG', 'Response')['R', 'T', 'K', 'IA_max', 'IA_rec', 'IA_ini', 'desc']"

        if not hasattr(self, "_hydrographs"):
            self._hydrographs = self._get_section("HYDROGRAPH")

        return self._hydrographs

    @hydrographs.setter
    def hydrographs(self, obj) -> None:
        self._hydrographs = sc.Hydrographs._newobj(obj)

    @property
    def curves(self) -> sc.Curves:
        "('Name')['Type', 'X_Value', 'Y_Value', 'desc']"

        if not hasattr(self, "_curves"):
            self._curves = self._get_section("CURVE")

        return self._curves

    @curves.setter
    def curves(self, obj) -> None:
        self._curves = sc.Curves._newobj(obj)

    @property
    def timeseries(self) -> sc.Timeseries:
        "Dict of dataframes or TimeseriesFile dataclass."

        if not hasattr(self, "_timeseries"):
            self._timeseries = self._get_section("TIMESERIES")

        return self._timeseries

    @timeseries.setter
    def timeseries(self, obj) -> None:
        self._timeseries = sc.Timeseries._newobj(obj)

    @property
    def patterns(self) -> sc.Patterns:
        "('Name')['Type', 'Multiplier', 'desc']"

        if not hasattr(self, "_patterns"):
            self._patterns = self._get_section("PATTERN")

        return self._patterns

    @patterns.setter
    def patterns(self, obj) -> None:
        self._patterns = sc.Patterns._newobj(obj)

    @property
    def report(self) -> sc.Report:
        "Data class with attribute for each report option."

        if not hasattr(self, "_report"):
            self._report = self._get_section("REPORT")

        return self._report

    @report.setter
    def report(self, obj) -> None:
        self._report = sc.Report._newobj(obj)

    @property
    def adjustments(self) -> sc.Adjustments:
        "('Parameter')['Subcatchment', 'Pattern', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'desc']"

        if not hasattr(self, "_adjustments"):
            self._adjustments = self._get_section("ADJUSTMENT")

        return self._adjustments

    @adjustments.setter
    def adjustments(self, obj) -> None:
        self._adjustments = sc.Adjustments._newobj(obj)

    @property
    def event(self) -> sc.Event:
        "()['Start', 'End', 'desc']"

        if not hasattr(self, "_event"):
            self._event = self._get_section("EVENT")

        return self._event

    @event.setter
    def event(self, obj) -> None:
        self._event = sc.Event._newobj(obj)

    @property
    def tags(self) -> sc.Tags:
        "('Element', 'Name')['Tag', 'desc']"

        if not hasattr(self, "_tags"):
            self._tags = self._get_section("TAG")

        return self._tags

    @tags.setter
    def tags(self, obj) -> None:
        self._tags = sc.Tags._newobj(obj)

    @property
    def map(self) -> sc.Map:
        "String class to hold map section text."

        if not hasattr(self, "_map"):
            self._map = self._get_section("MAP")

        return self._map

    @map.setter
    def map(self, obj) -> None:
        self._map = sc.Map._newobj(obj)

    @property
    def coordinates(self) -> sc.Coordinates:
        "('Node')['X', 'Y', 'desc']"

        if not hasattr(self, "_coordinates"):
            self._coordinates = self._get_section("COORDINATE")

        return self._coordinates

    @coordinates.setter
    def coordinates(self, obj) -> None:
        self._coordinates = sc.Coordinates._newobj(obj)

    @property
    def vertices(self) -> sc.Vertices:
        "('Link')['X', 'Y', 'desc']"

        if not hasattr(self, "_vertices"):
            self._vertices = self._get_section("VERTICES")

        return self._vertices

    @vertices.setter
    def vertices(self, obj) -> None:
        self._vertices = sc.Vertices._newobj(obj)

    @property
    def polygons(self) -> sc.Polygons:
        "('Elem')['X', 'Y', 'desc']"

        if not hasattr(self, "_polygons"):
            self._polygons = self._get_section("POLYGON")

        return self._polygons

    @polygons.setter
    def polygons(self, obj) -> None:
        self._polygons = sc.Polygons._newobj(obj)

    @property
    def symbols(self) -> sc.Symbols:
        "('Gage')['X', 'Y', 'desc']"

        if not hasattr(self, "_symbols"):
            self._symbols = self._get_section("SYMBOL")

        return self._symbols

    @symbols.setter
    def symbols(self, obj) -> None:
        self._symbols = sc.Symbols._newobj(obj)

    @property
    def labels(self) -> sc.Labels:
        "()['Xcoord', 'Ycoord', 'Label', 'Anchor', 'Font', 'Size', 'Bold', 'Italic', 'desc']"

        if not hasattr(self, "_labels"):
            self._labels = self._get_section("LABEL")

        return self._labels

    @labels.setter
    def labels(self, obj) -> None:
        self._labels = sc.Labels._newobj(obj)

    @property
    def backdrop(self) -> sc.Backdrop:
        "String class to hold backdrop section text."

        if not hasattr(self, "_backdrop"):
            self._backdrop = self._get_section("BACKDROP")

        return self._backdrop

    @backdrop.setter
    def backdrop(self, obj) -> None:
        self._backdrop = sc.Backdrop._newobj(obj)

    @property
    def profile(self) -> sc.Profile:
        "String class to hold profile section text"

        if not hasattr(self, "_profile"):
            self._profile = self._get_section("PROFILE")

        return self._profile

    @profile.setter
    def profile(self, obj) -> None:
        self._profile = sc.Profile._newobj(obj)
