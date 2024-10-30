# %%
# swmm-pandas input
# scope:
#   - high level api for loading, inspecting, changing, and
#     altering a SWMM input file using pandas dataframes
from __future__ import annotations

from swmm.pandas.input._section_classes import SectionBase, SectionDf, _sections
from swmm.pandas.input.input import InputFile
import pandas as pd

import swmm.pandas.input._section_classes as sc
import pathlib
import re
from typing import Optional, Callable, Any, TypeVar
import warnings
import copy


T = TypeVar("T")


def object_hasattr(obj: Any, name: str):
    try:
        object.__getattribute__(obj, name)
        return True
    except AttributeError:
        return False


def object_getattr(obj: Any, name: str):
    return object.__getattribute__(obj, name)


class NoAssignmentError(Exception):
    def __init__(self, prop_name):
        self.prop_name = prop_name

    def __str__(self) -> str:
        return f"Cannot assign '{self.prop_name}' property, only mutation is allowed."


class NoAccessError(Exception):
    def __init__(self, prop_name):
        self.prop_name = prop_name

    def __str__(self) -> str:
        return (
            f"Cannot directly edit '{self.prop_name}' property in the Input object.\n"
            f"Use the associated node/link table or use the InputFile object for lower level control. "
        )


def no_setter_property(func: Callable[[Any], T]) -> property:

    def readonly_setter(self: Any, obj: Any) -> None:
        raise NoAssignmentError(func.__name__)

    return property(fget=func, fset=readonly_setter, doc=func.__doc__)


class Input:

    def __init__(self, inpfile: Optional[str | InputFile] = None):
        if isinstance(inpfile, InputFile):
            self._inp = inpfile
        elif isinstance(inpfile, str | pathlib.Path):
            self._inp = InputFile(inpfile)

    ##########################################################
    # region General df constructors and destructors #########
    ##########################################################

    # destructors
    def _general_destructor(
        self, inp_frames: list[pd.DataFrame], output_frame: SectionDf
    ) -> None:

        inp_dfs = []
        output_frame_name = output_frame.__class__.__name__.lower()
        cols = output_frame._data_cols(desc=False)
        for inp_frame in inp_frames:
            inp_df = inp_frame.loc[:, cols]
            inp_dfs.append(inp_df)

        out_df = copy.deepcopy(output_frame)
        inp_df = pd.concat(inp_dfs, axis=0)

        out_df = out_df.reindex(inp_df.index.rename(out_df.index.name))
        out_df.loc[inp_df.index, cols] = inp_df[cols]
        out_df = out_df.dropna(how="all")
        setattr(self._inp, output_frame_name, out_df)

    def _extract_table_and_restore_multi_index(
        self,
        input_frame: pd.DataFrame,
        input_index_name: str,
        output_frame: pd.DataFrame,
        prepend: list[tuple[str, str]] = [],
        append: list[tuple[str, str]] = [],
    ) -> pd.DataFrame:
        cols = output_frame._data_cols(desc=False)
        inp_df = input_frame.loc[:, cols]
        # out_df = copy.deepcopy(output_frame)
        levels = [pd.Index([val], name=nom) for nom, val in prepend]
        levels += [inp_df.index.rename(input_index_name)]
        levels += [pd.Index([val], name=nom) for nom, val in append]

        new_idx = pd.MultiIndex.from_product(levels)
        inp_df.index = new_idx

        # out_df = out_df.reindex(out_df.index.union(inp_df.index))
        # out_df.loc[inp_df.index, cols] = inp_df[cols]
        # out_df = out_df.dropna(how="all")
        return inp_df.dropna(how="all")

    # constructors
    def _general_constructor(self, inp_frames: list[SectionDf]) -> pd.DataFrame:
        left = inp_frames.pop(0).drop("desc", axis=1, errors="ignore")
        for right in inp_frames:
            left = pd.merge(
                left,
                right.drop("desc", axis=1, errors="ignore"),
                left_index=True,
                right_index=True,
                how="left",
            )
        return left

    # endregion General df constructors and destructors ######

    # %% ##########################################################
    # region DESTRUCTORS ##########################################
    # Methods to keep the input file class in sync with this class
    ###############################################################
    def _destruct_tags(self) -> None:
        tagged_dfs = [
            (self.junc, "Node"),
            (self.outfall, "Node"),
            (self.storage, "Node"),
            (self.divider, "Node"),
            (self.conduit, "Link"),
            (self.pump, "Link"),
            (self.weir, "Link"),
            (self.orifice, "Link"),
            (self.outlet, "Link"),
            (self.subcatchment, "Subcatch"),
        ]
        tag_dfs = [
            self._extract_table_and_restore_multi_index(
                input_frame=inp_df,
                input_index_name="Name",
                output_frame=self._inp.tags,
                prepend=[("Element", elem_type)],
            )
            for inp_df, elem_type in tagged_dfs
        ]

        tag_df = pd.concat(tag_dfs, axis=0).sort_index()
        self._inp.tags = self._inp.tags.reindex(tag_df.index)
        self._inp.tags.loc[tag_df.index, "Tag"] = tag_df["Tag"]

    def _destruct_nodes(self) -> None:
        node_dfs = [self.junc, self.outfall, self.storage, self.divider]

        out_dfs = [self._inp.rdii, self._inp.coordinates]
        inflo_dfs = [self._inp.dwf, self._inp.inflow]

        for out_df in out_dfs:
            self._general_destructor(inp_frames=node_dfs, output_frame=out_df)

        for out_df in inflo_dfs:
            output_frame_name = out_df.__class__.__name__.lower()
            out_df = out_df.drop("FLOW", level="Constituent", errors="ignore")
            inp_dfs = [
                self._extract_table_and_restore_multi_index(
                    input_frame=inp_df,
                    input_index_name="Node",
                    output_frame=out_df,
                    append=[("Constituent", "FLOW")],
                )
                for inp_df in node_dfs
            ]
            inp_dfs.append(out_df)

            inp_df = pd.concat(inp_dfs).dropna(how="all").sort_index()
            setattr(self._inp, output_frame_name, inp_df)

    def _destruct_xsect(self) -> None:
        if (
            hasattr(self, "_conduit_full")
            or hasattr(self, "_weir_full")
            or hasattr(self, "_orifice_full")
        ):
            self._general_destructor(
                inp_frames=[self.conduit, self.weir, self.orifice],
                output_frame=self._inp.xsections,
            )

    # endregion DESTRUCTORS ######

    # %% ###########################
    # region Generalized NODES #####
    ################################

    def _node_constructor(self, inp_df: SectionDf) -> pd.DataFrame:
        return self._general_constructor(
            [
                inp_df,
                self._inp.dwf.loc[(slice(None), slice("FLOW", "FLOW")), :].droplevel(
                    "Constituent"
                ),
                self._inp.inflow.loc[(slice(None), slice("FLOW", "FLOW")), :].droplevel(
                    "Constituent"
                ),
                self._inp.rdii,
                self._inp.tags.sort_index()
                .loc[slice("Node", "Node"), slice(None)]
                .droplevel("Element"),
                self._inp.coordinates,
            ]
        )

    # endregion NODES and LINKS ######

    # %% ###########################
    # region MAIN TABLES ###########
    ################################

    ######### JUNCTIONS #########
    @no_setter_property
    def junc(self) -> pd.DataFrame:
        """('Name')['Elevation', 'MaxDepth', 'InitDepth', 'SurDepth', 'Aponded', 'AvgValue', 'Pat1', 'Pat2', 'Pat3', 'Pat4', 'TimeSeries', 'InflowType', 'Mfactor', 'Sfactor', 'Baseline', 'Pattern', 'UHgroup', 'SewerArea', 'Tag', 'X', 'Y']"""
        if not hasattr(self, "_junc_full"):
            self._junc_full = self._node_constructor(self._inp.junc)

        return self._junc_full

    def _junction_destructor(self) -> None:
        if hasattr(self, "_junc_full"):
            self._general_destructor([self.junc], self._inp.junc)

    ######## OUTFALLS #########
    @no_setter_property
    def outfall(self) -> pd.DataFrame:
        """('Name')['Elevation', 'Type', 'StageData', 'Gated', 'RouteTo', 'AvgValue', 'Pat1', 'Pat2', 'Pat3', 'Pat4', 'TimeSeries', 'InflowType', 'Mfactor', 'Sfactor', 'Baseline', 'Pattern', 'UHgroup', 'SewerArea', 'Tag', 'X', 'Y']"""
        if not hasattr(self, "_outfall_full"):
            self._outfall_full = self._node_constructor(self._inp.outfall)

        return self._outfall_full

    def _outfall_destructor(self) -> None:
        if hasattr(self, "_outfall_full"):
            self._general_destructor([self.outfall], self._inp.outfall)

    ######## STORAGE #########
    @no_setter_property
    def storage(self):
        """('Name')['Elev', 'MaxDepth', 'InitDepth', 'Shape', 'CurveName', 'A1_L', 'A2_W', 'A0_Z', 'SurDepth', 'Fevap', 'Psi', 'Ksat', 'IMD', 'AvgValue', 'Pat1', 'Pat2', 'Pat3', 'Pat4', 'TimeSeries', 'InflowType', 'Mfactor', 'Sfactor', 'Baseline', 'Pattern', 'UHgroup', 'SewerArea', 'Tag', 'X', 'Y']"""
        if not hasattr(self, "_storage_full"):
            self._storage_full = self._node_constructor(self._inp.storage)

        return self._storage_full

    def _storage_destructor(self) -> None:
        if hasattr(self, "_storage_full"):
            self._general_destructor([self.storage], self._inp.storage)

    ######## DIVIDER #########
    @no_setter_property
    def divider(self):
        """('Name')['Elevation', 'DivLink', 'DivType', 'DivCurve', 'Qmin', 'Height', 'Cd', 'Ymax', 'Y0', 'Ysur', 'Apond', 'AvgValue', 'Pat1', 'Pat2', 'Pat3', 'Pat4', 'TimeSeries', 'InflowType', 'Mfactor', 'Sfactor', 'Baseline', 'Pattern', 'UHgroup', 'SewerArea', 'Tag', 'X', 'Y']"""
        if not hasattr(self, "_divider_full"):
            self._divider_full = self._node_constructor(self._inp.divider)

        return self._divider_full

    def _divider_destructor(self) -> None:
        if hasattr(self, "_divider_full"):
            self._general_destructor([self.divider], self._inp.divider)

    ######### CONDUITS #########
    @no_setter_property
    def conduit(self) -> pd.DataFrame:
        """('Name')['FromNode', 'ToNode', 'Length', 'Roughness', 'InOffset', 'OutOffset', 'InitFlow', 'MaxFlow', 'Kentry', 'Kexit', 'Kavg', 'FlapGate', 'Seepage', 'Shape', 'Geom1', 'Curve', 'Geom2', 'Geom3', 'Geom4', 'Barrels', 'Culvert', 'Tag']"""
        if not hasattr(self, "_conduit_full"):
            self._conduit_full = self._general_constructor(
                [
                    self._inp.conduit,
                    self._inp.losses,
                    self._inp.xsections,
                    self._inp.tags.sort_index()
                    .loc[slice("Link", "Link"), slice(None)]
                    .droplevel(0),
                ]
            )

        return self._conduit_full

    def _conduit_destructor(self) -> None:
        if hasattr(self, "_conduit_full"):
            for frame in [self._inp.conduit, self._inp.losses]:
                self._general_destructor([self.conduit], frame)

    ######## PUMPS #########
    @no_setter_property
    def pump(self) -> pd.DataFrame:
        """('Name')['FromNode', 'ToNode', 'PumpCurve', 'Status', 'Startup', 'Shutoff', 'Tag']"""
        if not hasattr(self, "_pump_full"):
            self._pump_full = self._general_constructor(
                [
                    self._inp.pump,
                    self._inp.tags.sort_index()
                    .loc[slice("Link", "Link"), slice(None)]
                    .droplevel(0),
                ]
            )

        return self._pump_full

    def _pump_destructor(self) -> None:
        if hasattr(self, "_pump_full"):
            self._general_destructor([self.pump], self._inp.pump)

    ######## WEIRS #########
    @no_setter_property
    def weir(self) -> pd.DataFrame:
        """('Name')['FromNode', 'ToNode', 'Type', 'CrestHt', 'Qcoeff', 'Gated', 'EndCon', 'EndCoeff', 'Surcharge', 'RoadWidth', 'RoadSurf', 'CoeffCurve', 'Shape', 'Geom1', 'Curve', 'Geom2', 'Geom3', 'Geom4', 'Barrels', 'Culvert', 'Tag']"""
        if not hasattr(self, "_weir_full"):
            self._weir_full = self._general_constructor(
                [
                    self._inp.weir,
                    self._inp.xsections,
                    self._inp.tags.sort_index()
                    .loc[slice("Link", "Link"), slice(None)]
                    .droplevel(0),
                ]
            )

        return self._weir_full

    def _weir_destructor(self) -> None:
        if hasattr(self, "_weir_full"):
            self._general_destructor(
                [self.weir],
                self._inp.weir,
            )

    ######## ORIFICES #########
    @no_setter_property
    def orifice(self) -> pd.DataFrame:
        """('Name')['FromNode', 'ToNode', 'Type', 'Offset', 'Qcoeff', 'Gated', 'CloseTime', 'Shape', 'Geom1', 'Curve', 'Geom2', 'Geom3', 'Geom4', 'Barrels', 'Culvert', 'Tag']"""

        if not hasattr(self, "_orifice_full"):
            self._orifice_full = self._general_constructor(
                [
                    self._inp.orifice,
                    self._inp.xsections,
                    self._inp.tags.sort_index()
                    .loc[slice("Link", "Link"), slice(None)]
                    .droplevel(0),
                ]
            )

        return self._orifice_full

    def _orifice_destructor(self) -> None:
        if hasattr(self, "_orifice_full"):
            self._general_destructor(
                [self.orifice],
                self._inp.orifice,
            )

    ######## OULETS #########
    @no_setter_property
    def outlet(self) -> pd.DataFrame:
        """('Name')['FromNode', 'ToNode', 'Offset', 'Type', 'CurveName', 'Qcoeff', 'Qexpon', 'Gated', 'Tag']"""

        if not hasattr(self, "_outlet_full"):
            self._outlet_full = self._general_constructor(
                [
                    self._inp.outlet,
                    self._inp.tags.sort_index()
                    .loc[slice("Link", "Link"), slice(None)]
                    .droplevel(0),
                ]
            )

        return self._outlet_full

    def _outlet_destructor(self) -> None:
        if hasattr(self, "_outlet_full"):
            self._general_destructor(
                [self.outlet],
                self._inp.outlet,
            )

    ####### SUBCATCHMENTS
    @no_setter_property
    def subcatchment(self) -> pd.DataFrame:
        """
        ('Name')['RainGage', 'Outlet', 'Area', 'PctImp', 'Width', 'Slope', 'CurbLeng', 'SnowPack', 'Nimp', 'Nperv', 'Simp', 'Sperv', 'PctZero', 'RouteTo', 'PctRouted', 'Tag']
        """
        if not hasattr(self, "_subcatch_full"):
            self._subcatch_full = self._general_constructor(
                [
                    self._inp.subcatchment,
                    self._inp.subarea,
                    self._inp.tags.sort_index()
                    .loc[slice("Subcatch", "Subcatch"), slice(None)]
                    .droplevel("Element"),
                ]
            )

        return self._subcatch_full

    def _subcatchment_destructor(self) -> None:
        if hasattr(self, "_subcatch_full"):

            self._general_destructor(
                [self.subcatchment],
                self._inp.subcatchment,
            )

            self._general_destructor(
                [self.subcatchment],
                self._inp.subarea,
            )

    # endregion MAIN TABLES ######
    def _clear(self):
        full_tables = filter(lambda x: "_full" in x, dir(self))
        for table in full_tables:
            delattr(self, table)

    def _sync(self):
        # nodes
        self._junction_destructor()
        self._outfall_destructor()
        self._storage_destructor()

        # links
        self._conduit_destructor()
        self._pump_destructor()
        self._orifice_destructor()
        self._weir_destructor()
        self._outlet_destructor()

        # subcatch
        self._subcatchment_destructor()

        # other
        self._destruct_nodes()
        self._destruct_xsect()
        self._destruct_tags()

    def to_file(self, path: str | pathlib.Path):
        self._sync()
        with open(path, "w") as f:
            f.write(self._inp.to_string())

    # region non-component sections
    # This section is autgenerated by scripts/generate_input_sections.py
    @property
    def title(self) -> sc.Title:

        return self._inp.title

    @title.setter
    def title(self, obj) -> None:
        self._inp.title = obj

    @property
    def option(self) -> sc.Option:
        "('Option')['Value', 'desc']"

        return self._inp.option

    @option.setter
    def option(self, obj) -> None:
        self._inp.option = obj

    @property
    def files(self) -> sc.Files:
        "String to hold files section"

        return self._inp.files

    @files.setter
    def files(self, obj) -> None:
        self._inp.files = obj

    @property
    def raingage(self) -> sc.Raingage:
        "('Name')['Format', 'Interval', 'SCF', 'Source_Type', 'Source', 'Station', 'Units', 'desc']"

        return self._inp.raingage

    @raingage.setter
    def raingage(self, obj) -> None:
        self._inp.raingage = obj

    @property
    def evap(self) -> sc.Evap:
        "('Type')['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9', 'param10', 'param11', 'param12', 'desc']"

        return self._inp.evap

    @evap.setter
    def evap(self, obj) -> None:
        self._inp.evap = obj

    @property
    def temperature(self) -> sc.Temperature:
        "('Option')['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9', 'param10', 'param11', 'param12', 'param13', 'desc']"

        return self._inp.temperature

    @temperature.setter
    def temperature(self, obj) -> None:
        self._inp.temperature = obj

    @property
    def subarea(self) -> sc.Subarea:
        "('Subcatchment')['Nimp', 'Nperv', 'Simp', 'Sperv', 'PctZero', 'RouteTo', 'PctRouted', 'desc']"

        return self._inp.subarea

    @subarea.setter
    def subarea(self, obj) -> None:
        self._inp.subarea = obj

    @property
    def infil(self) -> sc.Infil:
        "('Subcatchment')['param1', 'param2', 'param3', 'param4', 'param5', 'Method', 'desc']"

        return self._inp.infil

    @infil.setter
    def infil(self, obj) -> None:
        self._inp.infil = obj

    @property
    def lid_control(self) -> sc.LID_Control:
        "('Name')['Type', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'desc']"

        return self._inp.lid_control

    @lid_control.setter
    def lid_control(self, obj) -> None:
        self._inp.lid_control = obj

    @property
    def lid_usage(self) -> sc.LID_Usage:
        "('Subcatchment', 'LIDProcess')['Number', 'Area', 'Width', 'InitSat', 'FromImp', 'ToPerv', 'RptFile', 'DrainTo', 'FromPerv', 'desc']"

        return self._inp.lid_usage

    @lid_usage.setter
    def lid_usage(self, obj) -> None:
        self._inp.lid_usage = obj

    @property
    def aquifer(self) -> sc.Aquifer:
        "('Name')['Por', 'WP', 'FC', 'Ksat', 'Kslope', 'Tslope', 'ETu', 'ETs', 'Seep', 'Ebot', 'Egw', 'Umc', 'ETupat', 'desc']"

        return self._inp.aquifer

    @aquifer.setter
    def aquifer(self, obj) -> None:
        self._inp.aquifer = obj

    @property
    def groundwater(self) -> sc.Groundwater:
        "('Subcatchment')['Aquifer', 'Node', 'Esurf', 'A1', 'B1', 'A2', 'B2', 'A3', 'Dsw', 'Egwt', 'Ebot', 'Wgr', 'Umc', 'desc']"

        return self._inp.groundwater

    @groundwater.setter
    def groundwater(self, obj) -> None:
        self._inp.groundwater = obj

    @property
    def gwf(self) -> sc.GWF:
        "('Subcatch', 'Type')['Expr', 'desc']"

        return self._inp.gwf

    @gwf.setter
    def gwf(self, obj) -> None:
        self._inp.gwf = obj

    @property
    def snowpack(self) -> sc.Snowpack:
        "('Name', 'Surface')['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'desc']"

        return self._inp.snowpack

    @snowpack.setter
    def snowpack(self, obj) -> None:
        self._inp.snowpack = obj

    @property
    def xsections(self) -> sc.Xsections:
        "('Link')['Shape', 'Geom1', 'Curve', 'Geom2', 'Geom3', 'Geom4', 'Barrels', 'Culvert', 'desc']"

        return self._inp.xsections

    @xsections.setter
    def xsections(self, obj) -> None:
        self._inp.xsections = obj

    @property
    def transects(self) -> sc.Transects:
        "String to hold transects section."

        return self._inp.transects

    @transects.setter
    def transects(self, obj) -> None:
        self._inp.transects = obj

    @property
    def street(self) -> sc.Street:
        "('Name')['Tcrown', 'Hcurb', 'Sroad', 'nRoad', 'Hdep', 'Wdep', 'Sides', 'Wback', 'Sback', 'nBack', 'desc']"

        return self._inp.street

    @street.setter
    def street(self, obj) -> None:
        self._inp.street = obj

    @property
    def inlet_usage(self) -> sc.Inlet_Usage:
        "('Conduit')['Inlet', 'Node', 'Number', '%Clogged', 'MaxFlow', 'hDStore', 'wDStore', 'Placement', 'desc']"

        return self._inp.inlet_usage

    @inlet_usage.setter
    def inlet_usage(self, obj) -> None:
        self._inp.inlet_usage = obj

    @property
    def inlet(self) -> sc.Inlet:
        "('Name', 'Type')['param1', 'param2', 'param3', 'param4', 'param5', 'desc']"

        return self._inp.inlet

    @inlet.setter
    def inlet(self, obj) -> None:
        self._inp.inlet = obj

    @property
    def losses(self) -> sc.Losses:
        "('Link')['Kentry', 'Kexit', 'Kavg', 'FlapGate', 'Seepage', 'desc']"

        return self._inp.losses

    @losses.setter
    def losses(self, obj) -> None:
        self._inp.losses = obj

    @property
    def controls(self) -> sc.Controls:
        "Dict of control rules stored as text."

        return self._inp.controls

    @controls.setter
    def controls(self, obj) -> None:
        self._inp.controls = obj

    @property
    def pollutants(self) -> sc.Pollutants:
        "('Name')['Units', 'Crain', 'Cgw', 'Crdii', 'Kdecay', 'SnowOnly', 'CoPollutant', 'CoFrac', 'Cdwf', 'Cinit', 'desc']"

        return self._inp.pollutants

    @pollutants.setter
    def pollutants(self, obj) -> None:
        self._inp.pollutants = obj

    @property
    def landuse(self) -> sc.LandUse:
        "('Name')['SweepInterval', 'Availability', 'LastSweep', 'desc']"

        return self._inp.landuse

    @landuse.setter
    def landuse(self, obj) -> None:
        self._inp.landuse = obj

    @property
    def coverage(self) -> sc.Coverage:
        "('Subcatchment', 'LandUse')['Percent', 'desc']"

        return self._inp.coverage

    @coverage.setter
    def coverage(self, obj) -> None:
        self._inp.coverage = obj

    @property
    def loading(self) -> sc.Loading:
        "('Subcatchment', 'Pollutant')['InitBuildup', 'desc']"

        return self._inp.loading

    @loading.setter
    def loading(self, obj) -> None:
        self._inp.loading = obj

    @property
    def buildup(self) -> sc.Buildup:
        "('Landuse', 'Pollutant')['FuncType', 'C1', 'C2', 'C3', 'PerUnit', 'desc']"

        return self._inp.buildup

    @buildup.setter
    def buildup(self, obj) -> None:
        self._inp.buildup = obj

    @property
    def washoff(self) -> sc.Washoff:
        "('Landuse', 'Pollutant')['FuncType', 'C1', 'C2', 'SweepRmvl', 'BmpRmvl', 'desc']"

        return self._inp.washoff

    @washoff.setter
    def washoff(self, obj) -> None:
        self._inp.washoff = obj

    @property
    def treatment(self) -> sc.Treatment:
        "('Node', 'Pollutant')['Func', 'desc']"

        return self._inp.treatment

    @treatment.setter
    def treatment(self, obj) -> None:
        self._inp.treatment = obj

    @property
    def inflow(self) -> sc.Inflow:
        "('Node', 'Constituent')['TimeSeries', 'InflowType', 'Mfactor', 'Sfactor', 'Baseline', 'Pattern', 'desc']"

        return self._inp.inflow

    @inflow.setter
    def inflow(self, obj) -> None:
        self._inp.inflow = obj

    @property
    def dwf(self) -> sc.DWF:
        "('Node', 'Constituent')['AvgValue', 'Pat1', 'Pat2', 'Pat3', 'Pat4', 'desc']"

        return self._inp.dwf

    @dwf.setter
    def dwf(self, obj) -> None:
        self._inp.dwf = obj

    @property
    def rdii(self) -> sc.RDII:
        "('Node')['UHgroup', 'SewerArea', 'desc']"

        return self._inp.rdii

    @rdii.setter
    def rdii(self, obj) -> None:
        self._inp.rdii = obj

    @property
    def hydrographs(self) -> sc.Hydrographs:
        "('Name', 'Month_RG', 'Response')['R', 'T', 'K', 'IA_max', 'IA_rec', 'IA_ini', 'desc']"

        return self._inp.hydrographs

    @hydrographs.setter
    def hydrographs(self, obj) -> None:
        self._inp.hydrographs = obj

    @property
    def curves(self) -> sc.Curves:
        "('Name')['Type', 'X_Value', 'Y_Value', 'desc']"

        return self._inp.curves

    @curves.setter
    def curves(self, obj) -> None:
        self._inp.curves = obj

    @property
    def timeseries(self) -> sc.Timeseries:
        "Dict of dataframes or TimeseriesFile dataclass."

        return self._inp.timeseries

    @timeseries.setter
    def timeseries(self, obj) -> None:
        self._inp.timeseries = obj

    @property
    def patterns(self) -> sc.Patterns:
        "('Name')['Type', 'Multiplier', 'desc']"

        return self._inp.patterns

    @patterns.setter
    def patterns(self, obj) -> None:
        self._inp.patterns = obj

    @property
    def report(self) -> sc.Report:
        "Data class with attribute for each report option."

        return self._inp.report

    @report.setter
    def report(self, obj) -> None:
        self._inp.report = obj

    @property
    def adjustments(self) -> sc.Adjustments:
        "('Parameter')['Subcatchment', 'Pattern', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'desc']"

        return self._inp.adjustments

    @adjustments.setter
    def adjustments(self, obj) -> None:
        self._inp.adjustments = obj

    @property
    def event(self) -> sc.Event:
        "()['Start', 'End', 'desc']"

        return self._inp.event

    @event.setter
    def event(self, obj) -> None:
        self._inp.event = obj

    @property
    def tags(self) -> sc.Tags:
        "('Element', 'Name')['Tag', 'desc']"

        return self._inp.tags

    @tags.setter
    def tags(self, obj) -> None:
        self._inp.tags = obj

    @property
    def map(self) -> sc.Map:
        "String class to hold map section text."

        return self._inp.map

    @map.setter
    def map(self, obj) -> None:
        self._inp.map = obj

    @property
    def coordinates(self) -> sc.Coordinates:
        "('Node')['X', 'Y', 'desc']"

        return self._inp.coordinates

    @coordinates.setter
    def coordinates(self, obj) -> None:
        self._inp.coordinates = obj

    @property
    def vertices(self) -> sc.Vertices:
        "('Link')['X', 'Y', 'desc']"

        return self._inp.vertices

    @vertices.setter
    def vertices(self, obj) -> None:
        self._inp.vertices = obj

    @property
    def polygons(self) -> sc.Polygons:
        "('Elem')['X', 'Y', 'desc']"

        return self._inp.polygons

    @polygons.setter
    def polygons(self, obj) -> None:
        self._inp.polygons = obj

    @property
    def symbols(self) -> sc.Symbols:
        "('Gage')['X', 'Y', 'desc']"

        return self._inp.symbols

    @symbols.setter
    def symbols(self, obj) -> None:
        self._inp.symbols = obj

    @property
    def labels(self) -> sc.Labels:
        "()['Xcoord', 'Ycoord', 'Label', 'Anchor', 'Font', 'Size', 'Bold', 'Italic', 'desc']"

        return self._inp.labels

    @labels.setter
    def labels(self, obj) -> None:
        self._inp.labels = obj

    @property
    def backdrop(self) -> sc.Backdrop:
        "String class to hold backdrop section text."

        return self._inp.backdrop

    @backdrop.setter
    def backdrop(self, obj) -> None:
        self._inp.backdrop = obj

    @property
    def profile(self) -> sc.Profile:
        "String class to hold profile section text"

        return self._inp.profile

    @profile.setter
    def profile(self, obj) -> None:
        self._inp.profile = obj

    # endregion non-component sections
