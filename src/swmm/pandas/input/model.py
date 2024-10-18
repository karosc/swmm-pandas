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

    def __init__(self, inpfile: Optional[str | Input] = None):
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
        self, inp_frame: pd.DataFrame, output_frames: list[SectionDf]
    ) -> None:
        for output_frame in output_frames:
            output_frame_name = output_frame.__class__.__name__.lower()
            cols = output_frame._data_cols(desc=False)
            inp_df = inp_frame.loc[:, cols]
            out_df = copy.deepcopy(output_frame)

            out_df = out_df.reindex(
                out_df.index.union(inp_df.index).rename(out_df.index.name)
            )
            out_df.loc[inp_df.index, cols] = inp_df[list(cols)]
            out_df = out_df.dropna(how="all")
            setattr(self.inp, output_frame_name, out_df)

    def _destruct_tags(
        self,
        input_frame: pd.DataFrame,
        element_type: str,
    ) -> None:
        tag_df = self._extract_table_and_restore_multi_index(
            input_frame=input_frame,
            input_index_name="Name",
            output_frame=self.inp.tags,
            prepend=[("Element", element_type)],
        )
        self.inp.tags = tag_df

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
        out_df = copy.deepcopy(output_frame)
        levels = [pd.Index([val], name=nom) for nom, val in prepend]
        levels += [inp_df.index.rename(input_index_name)]
        levels += [pd.Index([val], name=nom) for nom, val in append]

        new_idx = pd.MultiIndex.from_product(levels)
        inp_df.index = new_idx

        out_df = out_df.reindex(out_df.index.union(inp_df.index))
        out_df.loc[inp_df.index, cols] = inp_df[cols]
        out_df = out_df.dropna(how="all")
        return out_df

    # constructors
    def _general_constructor(self, inp_frames: list[SectionDf]) -> pd.DataFrame:
        left = inp_frames.pop(0).drop("desc", axis=1)
        for right in inp_frames:
            left = pd.merge(
                left,
                right.drop("desc", axis=1),
                left_index=True,
                right_index=True,
                how="left",
            )
        return left

    # endregion General df constructors and destructors ######

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

    def _node_destructor(self, inp_df: pd.DataFrame, out_df: SectionDf) -> None:
        self._general_destructor(
            inp_df,
            [
                out_df,
                self.inp.rdii,
                self.inp.coordinates,
            ],
        )

        self._destruct_tags(inp_df, "Node")

        self.inp.dwf = self._extract_table_and_restore_multi_index(
            input_frame=inp_df,
            input_index_name="Node",
            output_frame=self.inp.dwf,
            append=[("Constituent", "FLOW")],
        )

        self.inp.inflow = self._extract_table_and_restore_multi_index(
            input_frame=inp_df,
            input_index_name="Node",
            output_frame=self.inp.inflow,
            append=[("Constituent", "FLOW")],
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
            self._node_destructor(self.junc, self.inp.junc)

    ######## OUTFALLS #########
    @no_setter_property
    def outfall(self) -> pd.DataFrame:
        if not hasattr(self, "_outfall_full"):
            self._outfall_full = self._node_constructor(self.inp.outfall)

        return self._outfall_full

    def _outfall_destructor(self) -> None:
        if hasattr(self, "_outfall_full"):
            self._node_destructor(self.outfall, self.inp.outfall)

    ######## STORAGE #########
    @no_setter_property
    def storage(self):
        if not hasattr(self, "_storage_full"):
            self._storage_full = self._node_constructor(self.inp.storage)

        return self._storage_full

    def _storage_destructor(self) -> None:
        if hasattr(self, "_storage_full"):
            self._node_destructor(self.storage, self.inp.storage)

    ######## DIVIDER #########
    @no_setter_property
    def divider(self):
        if not hasattr(self, "_divider_full"):
            self._divider_full = self._node_constructor(self.inp.divider)

        return self._storage_full

    def _storage_destructor(self) -> None:
        if hasattr(self, "_divider_full"):
            self._node_destructor(self.divider, self.inp.divider)

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
            self._general_destructor(
                self.conduit,
                [
                    self.inp.conduit,
                    self.inp.losses,
                    self.inp.xsections,
                ],
            )
            self._destruct_tags(self.conduit, "Link")

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
            self._general_destructor(
                self.pump,
                [
                    self.inp.pump,
                ],
            )
            self._destruct_tags(self.pump, "Link")

    ######## WEIRS #########
    @no_setter_property
    def weir(self) -> pd.DataFrame:
        if not hasattr(self, "_weir_full"):
            self._weir_full = self._general_constructor(
                [
                    self.inp.weir,
                    self.inp.tags.loc[slice("Link", "Link"), slice(None)].droplevel(0),
                ]
            )

        return self._weir_full

    def _weir_destructor(self) -> None:
        if hasattr(self, "_weir_full"):
            self._general_destructor(
                self.weir,
                [
                    self.inp.weir,
                ],
            )
            self._destruct_tags(self.weir, "Link")

    ######## ORIFICES #########
    @no_setter_property
    def orifice(self) -> pd.DataFrame:
        if not hasattr(self, "_orifice_full"):
            self._orifice_full = self._general_constructor(
                [
                    self.inp.orifice,
                    self.inp.tags.loc[slice("Link", "Link"), slice(None)].droplevel(0),
                ]
            )

        return self._orifice_full

    def _orifice_destructor(self) -> None:
        if hasattr(self, "_orifice_full"):
            self._general_destructor(
                self.orifice,
                [
                    self.inp.orifice,
                ],
            )
            self._destruct_tags(self.orifice, "Link")

    ######## OULETS #########
    @no_setter_property
    def outlet(self) -> pd.DataFrame:
        if not hasattr(self, "_outlet_full"):
            self._outlet_full = self._general_constructor(
                [
                    self.outlet,
                    self.inp.tags.loc[slice("Link", "Link"), slice(None)].droplevel(0),
                ]
            )

        return self._outlet_full

    def _outlet_destructor(self) -> None:
        if hasattr(self, "_outlet_full"):
            self._general_destructor(
                self.outlet,
                [
                    self.inp.outlet,
                ],
            )
            self._destruct_tags(self.outlet, "Link")

    ####### SUBCATCHMENTS
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

    def to_file(self, path: str | pathlib.Path):
        self._sync()
        with open(path, "w") as f:
            f.write(self.inp.to_string())
