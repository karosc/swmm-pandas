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
            self.inp = inpfile
        elif isinstance(inpfile, str | pathlib.Path):
            self.inp = InputFile(inpfile)

    # def __getattribute__(self, name: str) -> Any:
    #     _self_no_access = [
    #         "tags",
    #         "dwf",
    #         "inflow",
    #         "rdii",
    #         "losses",
    #         "xsections",
    #     ]

    #     if name in _self_no_access:
    #         raise NoAccessError(name)

    #     elif object_hasattr(self, name):
    #         return object_getattr(self, name)

    #     elif object_hasattr(InputFile, name):
    #         return object_getattr(object_getattr(self, "inp"), name)
    #     else:
    #         raise AttributeError(f"'Input' object has no attribute '{name}'")

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
        setattr(self.inp, output_frame_name, out_df)

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
                output_frame=self.inp.tags,
                prepend=[("Element", elem_type)],
            )
            for inp_df, elem_type in tagged_dfs
        ]

        tag_df = pd.concat(tag_dfs, axis=0)
        self.inp.tags = self.inp.tags.reindex(tag_df.index)
        self.inp.tags.loc[self.inp.tags.index, :] = self.inp.tags

    def _destruct_nodes(self) -> None:
        node_dfs = [self.junc, self.outfall, self.storage, self.divider]

        out_dfs = [self.inp.rdii, self.inp.coordinates]
        inflo_dfs = [self.inp.dwf, self.inp.inflow]

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

            inp_df = pd.concat(inp_dfs).dropna(how="all")
            setattr(self.inp, output_frame_name, inp_df)

    def _destruct_xsect(self) -> None:
        if (
            hasattr(self, "_conduit_full")
            or hasattr(self, "_weir_full")
            or hasattr(self, "_orifice_full")
        ):
            self._general_destructor(
                inp_frames=[self.conduit, self.weir, self.orifice],
                output_frame=self.inp.xsections,
            )

    # endregion DESTRUCTORS ######

    # %% ###########################
    # region Generalized NODES #####
    ################################

    def _node_constructor(self, inp_df: SectionDf) -> pd.DataFrame:
        return self._general_constructor(
            [
                inp_df,
                self.inp.dwf.loc[(slice(None), slice("FLOW", "FLOW")), :].droplevel(
                    "Constituent"
                ),
                self.inp.inflow.loc[(slice(None), slice("FLOW", "FLOW")), :].droplevel(
                    "Constituent"
                ),
                self.inp.rdii,
                self.inp.tags.loc[slice("Node", "Node"), slice(None)].droplevel(
                    "Element"
                ),
                self.inp.coordinates,
            ]
        )

    # endregion NODES and LINKS ######

    # %% ###########################
    # region MAIN TABLES ###########
    ################################

    ######### JUNCTIONS #########
    @no_setter_property
    def junc(self) -> pd.DataFrame:
        if not hasattr(self, "_junc_full"):
            self._junc_full = self._node_constructor(self.inp.junc)

        return self._junc_full

    def _junction_destructor(self) -> None:
        if hasattr(self, "_junc_full"):
            self._general_destructor([self.junc], self.inp.junc)

    ######## OUTFALLS #########
    @no_setter_property
    def outfall(self) -> pd.DataFrame:
        if not hasattr(self, "_outfall_full"):
            self._outfall_full = self._node_constructor(self.inp.outfall)

        return self._outfall_full

    def _outfall_destructor(self) -> None:
        if hasattr(self, "_outfall_full"):
            self._general_destructor([self.outfall], self.inp.outfall)

    ######## STORAGE #########
    @no_setter_property
    def storage(self):
        if not hasattr(self, "_storage_full"):
            self._storage_full = self._node_constructor(self.inp.storage)

        return self._storage_full

    def _storage_destructor(self) -> None:
        if hasattr(self, "_storage_full"):
            self._general_destructor([self.storage], self.inp.storage)

    ######## DIVIDER #########
    @no_setter_property
    def divider(self):
        if not hasattr(self, "_divider_full"):
            self._divider_full = self._node_constructor(self.inp.divider)

        return self._divider_full

    def _divider_destructor(self) -> None:
        if hasattr(self, "_divider_full"):
            self._general_destructor([self.divider], self.inp.divider)

    ######### CONDUITS #########
    @no_setter_property
    def conduit(self) -> pd.DataFrame:
        if not hasattr(self, "_conduit_full"):
            self._conduit_full = self._general_constructor(
                [
                    self.inp.conduit,
                    self.inp.losses,
                    self.inp.xsections,
                    self.inp.tags.loc[slice("Link", "Link"), slice(None)].droplevel(0),
                ]
            )

        return self._conduit_full

    def _conduit_destructor(self) -> None:
        if hasattr(self, "_conduit_full"):
            for frame in [self.inp.conduit, self.inp.losses]:
                self._general_destructor([self.conduit], frame)

    ######## PUMPS #########
    @no_setter_property
    def pump(self) -> pd.DataFrame:
        if not hasattr(self, "_pump_full"):
            self._pump_full = self._general_constructor(
                [
                    self.inp.pump,
                    self.inp.tags.loc[slice("Link", "Link"), slice(None)].droplevel(0),
                ]
            )

        return self._pump_full

    def _pump_destructor(self) -> None:
        if hasattr(self, "_pump_full"):
            self._general_destructor([self.pump], self.inp.pump)

    ######## WEIRS #########
    @no_setter_property
    def weir(self) -> pd.DataFrame:
        if not hasattr(self, "_weir_full"):
            self._weir_full = self._general_constructor(
                [
                    self.inp.weir,
                    self.inp.xsections,
                    self.inp.tags.loc[slice("Link", "Link"), slice(None)].droplevel(0),
                ]
            )

        return self._weir_full

    def _weir_destructor(self) -> None:
        if hasattr(self, "_weir_full"):
            self._general_destructor(
                [self.weir],
                self.inp.weir,
            )

    ######## ORIFICES #########
    @no_setter_property
    def orifice(self) -> pd.DataFrame:
        if not hasattr(self, "_orifice_full"):
            self._orifice_full = self._general_constructor(
                [
                    self.inp.orifice,
                    self.inp.xsections,
                    self.inp.tags.loc[slice("Link", "Link"), slice(None)].droplevel(0),
                ]
            )

        return self._orifice_full

    def _orifice_destructor(self) -> None:
        if hasattr(self, "_orifice_full"):
            self._general_destructor(
                [self.orifice],
                self.inp.orifice,
            )

    ######## OULETS #########
    @no_setter_property
    def outlet(self) -> pd.DataFrame:
        if not hasattr(self, "_outlet_full"):
            self._outlet_full = self._general_constructor(
                [
                    self.inp.outlet,
                    self.inp.tags.loc[slice("Link", "Link"), slice(None)].droplevel(0),
                ]
            )

        return self._outlet_full

    def _outlet_destructor(self) -> None:
        if hasattr(self, "_outlet_full"):
            self._general_destructor(
                [self.outlet],
                self.inp.outlet,
            )

    ####### SUBCATCHMENTS
    @no_setter_property
    def subcatchment(self) -> pd.DataFrame:
        if not hasattr(self, "_subcatch_full"):
            self._subcatch_full = self._general_constructor(
                [
                    self.inp.subcatchment,
                    self.inp.subarea,
                    self.inp.tags.loc[
                        slice("Subcatch", "Subcatch"), slice(None)
                    ].droplevel("Element"),
                ]
            )

        return self._subcatch_full

    def _subcatchment_destructor(self) -> None:
        if hasattr(self, "_subcatch_full"):

            self._general_destructor(
                [self.subcatchment],
                self.inp.subcatchment,
            )

            self._general_destructor(
                [self.subcatchment],
                self.inp.subarea,
            )

    # endregion MAIN TABLES ######

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
            f.write(self.inp.to_string())
