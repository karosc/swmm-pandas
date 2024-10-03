# swmm-pandas input
# scope:
#   - high level api for loading, inspecting, changing, and
#     altering a SWMM input file using pandas dataframes
from __future__ import annotations
import profile

from swmm.pandas.input.sections import _sections
import swmm.pandas.input._section_classes as sc
from typing import Dict
import re


class Input:
    _section_re = re.compile(R"^\[[\s\S]*?(?=^\[)", re.MULTILINE)
    _section_keys = tuple(_sections.keys())

    title: sc.Title = None
    option: sc.Option = None
    report: sc.Report = None
    event: sc.Event = None
    files = sc.Files = None
    raingage: sc.Raingage = None
    evap: sc.Evap = None
    temperature: sc.Temperature = None
    adjustments: sc.Adjustments = None
    subcatchment: sc.Subcatchment = None
    subarea: sc.Subarea = None
    infil: sc.Infil = None
    lid_control = sc.LID_Control = None
    lid_usage = sc.LID_Usage
    aquifer: sc.Aquifer = None
    groundwater: sc.Groundwater = None
    gwf: sc.GWF = None
    snowpack: sc.Snowpack = None
    junc: sc.Junc = None
    outfall: sc.Outfall = None
    divider: sc.Divider = None
    storage: sc.Storage = None
    conduit: sc.Conduit = None
    pump: sc.Pump = None
    orifice: sc.Orifice = None
    weir: sc.Weir = None
    outlet: sc.Outlet = None
    xsections: sc.Xsections = None
    # transects: sc.Transects = None
    street: sc.Street = None
    inlet: sc.Inlet = None
    inlet_usage = sc.Inlet_Usage = None
    losses: sc.Losses = None
    controls: sc.Controls = None
    pollutants: sc.Pollutants = None
    landuse: sc.LandUse = None
    coverage: sc.Coverage = None
    loading: sc.Loading = None
    buildup: sc.Buildup = None
    washoff: sc.Washoff = None
    treatment: sc.Treatment = None
    inflow: sc.Inflow = None
    dwf: sc.DWF = None
    rdii: sc.RDII = None
    hydrographs: sc.Hydrographs = None
    curves: sc.Curves = None
    timeseries: sc.Timeseries = None
    pattern: sc.Pattern = None
    map: sc.Map = (None,)
    polygon: sc.Polygons = None
    coordinate: sc.Coordinates = None
    vertices: sc.Vertices
    labels: sc.Labels = None
    symbol: sc.Symbols = None
    backdrop: sc.Backdrop = None
    profile: sc.Profile = None

    def __init__(self, inpfile: str):
        self.path: str = inpfile

        self._load_inp_file()
        for sect in _sections.keys():
            # print(sect)
            self._set_section_prop(sect)

    def _load_inp_file(self):
        with open(self.path, "r") as inp:
            self.text: str = inp.read()

        self._sections: Dict[str, str] = {}

        # section_data = []
        # section_titles = []

        # # split section data from inp file string
        # for section in section_re.findall(self.text):
        #     section_titles.append(re.findall(R"\[(.*)\]", section)[0])
        #     section_data.append(
        #         "\n".join(re.findall(R"^(?!;|\[).+$", section, re.MULTILINE))
        #     )

        # find all section blocks using regex
        self._section_texts = {}

        for section in self._section_re.findall(self.text):
            name = re.findall(R"^\[(.*)\]", section)[0]

            data = "\n".join(re.findall(R"^(?!;{2,}|\[).+$", section, re.MULTILINE))
            self._section_texts[name] = data

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

    # %% ###########################
    # region SECTION PROPS #########
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

    # def _get_section_text(self,section:str) -> str:
    #     section_class = _sections[section]
    #     public_property_name = section_class.__name__.lower()
    #     private_property_name = f"_{public_property_name}"

    #     if not hasattr(self,private_property_name):

    def to_string(self):
        out_str = ""
        for sect in _sections.keys():
            section_class = _sections[sect]
            public_property_name = section_class.__name__.lower()
            # private_property_name = f"_{public_property_name}"
            if len(sect_obj := getattr(self, public_property_name)) > 0:
                sect_string = sect_obj.to_swmm_string()
                out_str += f"[{sect.upper()}]\n{sect_string}\n\n"
        return out_str

    ############ OPTIONS ###########

    # @property
    # def options(self) -> sc.Option:
    #     if not hasattr(self, "_options_df"):
    #         self._options_df = self._get_section("OPTION")

    #     return self._options_df

    # @options.setter
    # def options(self, obj) -> None:
    #     self._options_df = sc.Option._newobj(obj)

    # ############ FILES ###########

    # @property
    # def files(self) -> sc.Section:
    #     raise NotImplementedError
    #     # if not hasattr(self,'_files_df'):
    #     #     self._files_df = self._get_section('FILES')

    #     # return self._files_df

    # @files.setter
    # def files(self, obj) -> None:
    #     raise NotImplementedError
    #     # self._files_df = sc.Files._newobj(obj)

    # ############ RAINGAGE ##########

    # @property
    # def raingages(self) -> sc.Raingage:
    #     if not hasattr(self, "_raingages_df"):
    #         self._raingages_df = self._get_section("RAINGAGE")

    #     return self._raingages_df

    # @raingages.setter
    # def raingages(self, obj) -> None:
    #     self._raingages_df = sc.Raingage._newobj(obj)

    # ############# EVAP #############

    # @property
    # def evaporation(self) -> sc.Evap:
    #     if not hasattr(self, "_evaporation_df"):
    #         self._evaporation_df = self._get_section("EVAP")

    #     return self._evaporation_df

    # @evaporation.setter
    # def evaporation(self, obj) -> None:
    #     self._evaporation_df = sc.Evap._newobj(obj)

    # ########## TEMPERATURE ##########

    # @property
    # def temperature(self) -> sc.Temperature:
    #     if not hasattr(self, "_temperature_df"):
    #         self._temperature_df = self._get_section("TEMPERATURE")

    #     return self._temperature_df

    # @temperature.setter
    # def temperature(self, obj) -> None:
    #     self._temperature_df = sc.Temperature._newobj(obj)

    # ############ LOSSES ############

    # @property
    # def losses(self) -> sc.Losses:
    #     if not hasattr(self, "_losses_df"):
    #         self._losses_df = self._get_section("LOSS")

    #     return self._losses_df

    # @losses.setter
    # def losses(self, obj) -> None:
    #     self._losses_df = sc.Losses._newobj(obj)

    # ############ REPORT ############

    # @property
    # def report(self) -> sc.Section:
    #     raise NotImplementedError
    #     # if not hasattr(self,'_report_df'):
    #     #     self._report_df = self._get_section('REPORT')

    #     # return self._report_df

    # @report.setter
    # def report(self, obj) -> None:
    #     raise NotImplementedError
    #     # self._report_df = Report._newobj(obj)

    # ########### CONDUITS ###########

    # @property
    # def conduits(self) -> sc.Conduit:
    #     if not hasattr(self, "_conduits_df"):
    #         self._conduits_df = self._get_section("CONDUIT")

    #     return self._conduits_df

    # @conduits.setter
    # def conduits(self, obj) -> None:
    #     self._conduits_df = sc.Conduit._newobj(obj)

    # ############ XSECT #############

    # @property
    # def xsections(self) -> sc.Xsections:
    #     if not hasattr(self, "_xsections_df"):
    #         self._xsections_df = self._get_section("XSECT")

    #     return self._xsections_df

    # @xsections.setter
    # def xsections(self, obj) -> None:
    #     self._xsections_df = sc.Xsections._newobj(obj)

    # ########## LID_USAGE ###########

    # @property
    # def lid_usage(self) -> sc.LID_Usage:
    #     if not hasattr(self, "_lid_usage_df"):
    #         self._lid_usage_df = self._get_section("LID_USAGE")

    #     return self._lid_usage_df

    # @lid_usage.setter
    # def lid_usage(self, obj) -> None:
    #     self._lid_usage_df = sc.LID_Usage._newobj(obj)

    # ########### POLLUT #############

    # @property
    # def pollutants(self) -> sc.Pollutants:
    #     if not hasattr(self, "_pollutants_df"):
    #         self._pollutants_df = self._get_section("POLLUT")

    #     return self._pollutants_df

    # @pollutants.setter
    # def pollutants(self, obj) -> None:
    #     self._pollutants_df = sc.Pollutants._newobj(obj)

    # ############ LANDUSE ############

    # @property
    # def landuses(self) -> sc.LandUse:
    #     if not hasattr(self, "_landuses_df"):
    #         self._landuses_df = self._get_section("LANDUSE")

    #     return self._landuses_df

    # @landuses.setter
    # def landuses(self, obj) -> None:
    #     self._landuses_df = sc.LandUse._newobj(obj)

    # ############ BUILDUP ###########

    # @property
    # def buildup(self) -> sc.Buildup:
    #     if not hasattr(self, "_buildup_df"):
    #         self._buildup_df = self._get_section("BUILDUP")

    #     return self._buildup_df

    # @buildup.setter
    # def buildup(self, obj) -> None:
    #     self._buildup_df = sc.Buildup._newobj(obj)

    # ########### WASHOFF ############

    # @property
    # def washoff(self) -> sc.Washoff:
    #     if not hasattr(self, "_washoff_df"):
    #         self._washoff_df = self._get_section("WASHOFF")

    #     return self._washoff_df

    # @washoff.setter
    # def washoff(self, obj) -> None:
    #     self._washoff_df = sc.Washoff._newobj(obj)

    # ############ COVERAGE ###########

    # @property
    # def coverages(self) -> sc.Section:
    #     raise NotImplementedError
    #     # if not hasattr(self,'_coverages_df'):
    #     #     self._coverages_df = self._get_section('COVERAGE')

    #     # return self._coverages_df

    # @coverages.setter
    # def coverages(self, obj) -> None:
    #     raise NotImplementedError
    #     # self._coverages_df = sc.Coverage._newobj(obj)

    # ############ LOADINGS ###########

    # @property
    # def loadings(self) -> sc.Section:
    #     raise NotImplementedError
    #     # if not hasattr(self,'_loadings_df'):
    #     #     self._loadings_df = self._get_section('COVERAGE')

    #     # return self._loadings_df

    # @loadings.setter
    # def loadings(self, obj) -> None:
    #     raise NotImplementedError
    #     # self._loadings_df = sc.Loadings._newobj(obj)

    # ############ PUMPS #############

    # @property
    # def pumps(self) -> sc.Pump:
    #     if not hasattr(self, "_pumps_df"):
    #         self._pumps_df = self._get_section("PUMP")

    #     return self._pumps_df

    # @pumps.setter
    # def pumps(self, obj) -> None:
    #     self._pumps_df = sc.Pump._newobj(obj)

    # ############ ORIFICE #############

    # @property
    # def orifices(self) -> sc.Orifice:
    #     if not hasattr(self, "_orifices_df"):
    #         self._orifices_df = self._get_section("ORIFICE")

    #     return self._orifices_df

    # @orifices.setter
    # def orifices(self, obj) -> None:
    #     self._orifices_df = sc.Orifice._newobj(obj)

    # ############## WEIR ###############

    # @property
    # def weirs(self) -> sc.Weir:
    #     if not hasattr(self, "_weirs_df"):
    #         self._weirs_df = self._get_section("WEIR")

    #     return self._weirs_df

    # @weirs.setter
    # def weirs(self, obj) -> None:
    #     self._weirs_df = sc.Weir._newobj(obj)

    # ############ JUNCTION #############

    # @property
    # def junctions(self) -> sc.Junc:
    #     if not hasattr(self, "_junctions_df"):
    #         self._junctions_df = self._get_section("JUNC")

    #     return self._junctions_df

    # @junctions.setter
    # def junctions(self, obj) -> None:
    #     self._junctions_df = sc.Junc._newobj(obj)

    # ############ OUTFALL #############

    # @property
    # def outfalls(self) -> sc.Outfall:
    #     if not hasattr(self, "_outfalls_df"):
    #         self._outfalls_df = self._get_section("OUTFALL")

    #     return self._outfalls_df

    # @outfalls.setter
    # def outfalls(self, obj) -> None:
    #     self._outfalls_df = sc.Outfall._newobj(obj)

    # ############ STORAGE #############

    # @property
    # def storage(self) -> sc.Storage:
    #     if not hasattr(self, "_storage_df"):
    #         self._storage_df = self._get_section("STORAGE")

    #     return self._storage_df

    # @storage.setter
    # def storage(self, obj) -> None:
    #     self._storage_df = sc.Storage._newobj(obj)

    # ############ DIVIDER #############

    # @property
    # def dividers(self) -> sc.Section:
    #     # raise NotImplementedError
    #     if not hasattr(self, "_dividers_df"):
    #         self._dividers_df = self._get_section("DIVIDER")

    #     return self._dividers_df

    # @dividers.setter
    # def dividers(self, obj) -> None:
    #     raise NotImplementedError
    #     # self._dividers_df = sc.Divider._newobj(obj)

    # ############ SUBCATCH #############

    # @property
    # def subcatchments(self) -> sc.Subcatchment:
    #     if not hasattr(self, "_subcatchments_df"):
    #         self._subcatchments_df = self._get_section("SUBCATCHMENT")

    #     return self._subcatchments_df

    # @subcatchments.setter
    # def subcatchments(self, obj) -> None:
    #     self._subcatchments_df = sc.Subcatchment._newobj(obj)

    # ############ SUBAREA #############

    # @property
    # def subareas(self) -> sc.Subarea:
    #     if not hasattr(self, "_subareas_df"):
    #         self._subareas_df = self._get_section("SUBAREA")

    #     return self._subareas_df

    # @subareas.setter
    # def subareas(self, obj) -> None:
    #     self._subareas_df = sc.Subarea._newobj(obj)

    # ############# INFIL ##############

    # @property
    # def infiltration(self) -> sc.Infil:
    #     if not hasattr(self, "_infiltration_df"):
    #         self._infiltration_df = self._get_section("INFIL")

    #     return self._infiltration_df

    # @infiltration.setter
    # def infiltration(self, obj) -> None:
    #     self._infiltration_df = sc.Infil._newobj(obj)

    # ############ AQUIFER #############

    # @property
    # def aquifers(self) -> sc.Aquifer:
    #     if not hasattr(self, "_aquifers_df"):
    #         self._aquifers_df = self._get_section("AQUIFER")

    #     return self._aquifers_df

    # @aquifers.setter
    # def aquifers(self, obj) -> None:
    #     self._aquifers_df = sc.Aquifer._newobj(obj)

    # ########### GROUNDWATER ###########

    # @property
    # def groundwater(self) -> sc.Groundwater:
    #     if not hasattr(self, "_groundwater_df"):
    #         self._groundwater_df = self._get_section("GROUNDWATER")

    #     return self._groundwater_df

    # @groundwater.setter
    # def groundwater(self, obj) -> None:
    #     self._groundwater_df = sc.Groundwater._newobj(obj)

    # ########### SNOWPACK ###########

    # @property
    # def snowpack(self) -> sc.Groundwater:
    #     if not hasattr(self, "_groundwater_df"):
    #         self._snowpack_df = self._get_section("GROUNDWATER")

    #     return self._snowpack_df

    # @snowpack.setter
    # def snowpack(self, obj) -> None:
    #     self._snowpack_df = sc.Groundwater._newobj(obj)

    # ############## COORDS #############

    # @property
    # def coordinates(self) -> sc.Coordinates:
    #     if not hasattr(self, "_coordinates_df"):
    #         self._coordinates_df = self._get_section("COORDINATE")

    #     return self._coordinates_df

    # @coordinates.setter
    # def coordinates(self, obj) -> None:
    #     self._coordinates_df = sc.Coordinates._newobj(obj)

    # ############### DWF ###############

    # @property
    # def dwf(self) -> sc.DWF:
    #     if not hasattr(self, "_dwf_df"):
    #         self._dwf_df = self._get_section("DWF")

    #     return self._dwf_df

    # @dwf.setter
    # def dwf(self, obj) -> None:
    #     self._dwf_df = sc.DWF._newobj(obj)

    # ############## RDII ###############

    # @property
    # def rdii(self) -> sc.RDII:
    #     if not hasattr(self, "_rdii_df"):
    #         self._rdii_df = self._get_section("RDII")

    #     return self._rdii_df

    # @rdii.setter
    # def rdii(self, obj) -> None:
    #     self._rdii_df = sc.RDII._newobj(obj)

    # ########## HYDROGRAPHS ############

    # @property
    # def hydrographs(self) -> sc.Section:
    #     raise NotImplementedError
    #     # if not hasattr(self,'_hydrographs_df'):
    #     #     self._hydrographs_df = self._get_section('HYDROGRAPH')

    #     # return self._hydrographs_df

    # @hydrographs.setter
    # def hydrographs(self, obj) -> None:
    #     raise NotImplementedError
    #     # self._hydrographs_df = sc.Hydrographs._newobj(obj)

    # ########### VERTICES #############

    # @property
    # def vertices(self) -> sc.Verticies:
    #     if not hasattr(self, "_vertices_df"):
    #         self._vertices_df = self._get_section("VERTICES")

    #     return self._vertices_df

    # @vertices.setter
    # def vertices(self, obj) -> None:
    #     self._vertices_df = sc.Verticies._newobj(obj)

    # ############ INFLOWS #############

    # @property
    # def inflows(self) -> sc.Inflow:
    #     if not hasattr(self, "_inflows_df"):
    #         self._inflows_df = self._get_section("VERTICES")

    #     return self._inflows_df

    # @inflows.setter
    # def inflows(self, obj) -> None:
    #     self._inflows_df = sc.Inflow._newobj(obj)

    # ############ POLYGON #############

    # @property
    # def polygons(self) -> sc.Polygons:
    #     if not hasattr(self, "_polygons_df"):
    #         self._polygons_df = self._get_section("POLYGON")

    #     return self._polygons_df

    # @polygons.setter
    # def polygons(self, obj) -> None:
    #     self._polygons_df = sc.Polygons._newobj(obj)

    # ############ CURVE #############

    # @property
    # def curves(self) -> sc.Section:
    #     raise NotImplementedError

    #     # if not hasattr(self,'_curves_df'):
    #     #     self._curves_df = self._get_section('CURVE')

    #     # return self._curves_df

    # @curves.setter
    # def curves(self, obj) -> None:
    #     raise NotImplementedError
    #     # self._curves_df = sc.Curve._newobj(obj)

    # ############ TIMESERIES #########

    # @property
    # def timeseries(self) -> sc.Section:
    #     raise NotImplementedError

    #     # if not hasattr(self,'_timeseries_df'):
    #     #     self._timeseries_df = self._get_section('TIMESERIES')

    #     # return self._timeseries_df

    # @timeseries.setter
    # def timeseries(self, obj) -> None:
    #     raise NotImplementedError
    #     # self._timeseries_df = sc.Timeseries._newobj(obj)

    # ############# TAGS ###############

    # @property
    # def tags(self) -> sc.Tags:
    #     if not hasattr(self, "_tags_df"):
    #         self._tags_df = self._get_section("TAG")

    #     return self._tags_df

    # @tags.setter
    # def tags(self, obj) -> None:
    #     self._tags_df = sc.Tags._newobj(obj)

    # endregion SECTION PROPS ######
