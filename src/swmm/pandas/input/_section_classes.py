from __future__ import annotations

import logging
from numbers import Number
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from calendar import month_abbr
import re
import textwrap
import copy
import pandas as pd
from pandas._libs.missing import NAType
import numpy as np

if TYPE_CHECKING:
    from typing import Self, TypeGuard
    from collections.abc import Iterable, Iterator

TRow = list[str | float | int | pd.Timestamp | pd.Timedelta | NAType]

_logger = logging.getLogger(__name__)


class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


# class ClassPropertyDescriptor(object):

#     def __init__(self, fget, fset=None):
#         self.fget = fget
#         self.fset = fset

#     def __get__(self, obj, klass=None):
#         if klass is None:
#             klass = type(obj)
#         return self.fget.__get__(obj, klass)()

#     def __set__(self, obj, value):
#         if not self.fset:
#             raise AttributeError("can't set attribute")
#         type_ = type(obj)
#         return self.fset.__get__(obj, type_)(value)

#     def setter(self, func):
#         if not isinstance(func, (classmethod, staticmethod)):
#             func = classmethod(func)
#         self.fset = func
#         return self

# def classproperty(func):
#     if not isinstance(func, (classmethod, staticmethod)):
#         func = classmethod(func)

#     return ClassPropertyDescriptor(func)


def _coerce_numeric(data: str) -> str | float | int:
    try:
        number = float(data)
        number = int(number) if number.is_integer() and "." not in data else number
        return number
        # if str(number) == data:
        #     return number
    except ValueError:
        pass

    return data


def _strip_comment(line: str) -> tuple[str, str]:
    """
    Splits a line into its data and comments


    Examples
    --------
    >>> _strip_comment(" JUNC1  1.5  10.25  0  0  5000 ; This is my fav junction ")
    ["JUNC1  1.5  10.25  0  0  5000 ", "This is my fav junction"]


    """
    try:
        return line[: line.index(";")].strip(), line[line.index(";") + 1 :].strip()

    except ValueError:
        return line, ""


def _is_line_comment(line: str) -> bool:
    """Determines if a line in the inp file is a comment line"""
    try:
        return line.strip()[0] == ";"
    except IndexError:
        return False


def _is_data(line: str):
    """
    Determines if an inp file line has data by checking if the line
    is a table header (starting with `;;`) or a section header (starting with a `[`)
    """
    if len(line) == 0 or line.strip()[0:2] == ";;" or line.strip()[0] == "[":
        return False
    return True


def comment_formatter(line: str):
    if len(line) > 0:
        line = ";" + line.strip().strip("\n").strip()
        line = line.replace("\n", "\n;") + "\n"
    return line


class SectionSeries(pd.Series):
    @property
    def _constructor(self):
        return SectionSeries

    @property
    def _constructor_expanddim(self):
        return SectionDf

    # def _constructor_from_mgr(self, mgr, axes) -> Self:
    #     # required override for pandas
    #     return self.__class__._from_mgr(mgr, axes)


class SectionBase(ABC):

    _section_name: str

    @classmethod
    @abstractmethod
    def from_section_text(cls, text: str, *args, **kwargs) -> Self: ...

    @classmethod
    @abstractmethod
    def _from_section_text(cls, text: str, *args, **kwargs) -> Self: ...
    @classmethod
    @abstractmethod
    def _new_empty(cls) -> Self: ...

    @classmethod
    @abstractmethod
    def _newobj(cls, *args, **kwargs) -> Self: ...

    @abstractmethod
    def to_swmm_string(self) -> str: ...


class SectionText(SectionBase, str):
    @classmethod
    def from_section_text(cls, text: str) -> Self:
        """Construct an instance of the class from the section inp text"""
        return cls._from_section_text(text)

    @classmethod
    def _from_section_text(cls, text: str) -> Self:
        return cls(text)

    @classmethod
    def _new_empty(cls) -> Self:
        return cls("")

    @classmethod
    def _newobj(cls, *args, **kwargs) -> Self:
        return cls(*args, **kwargs)

    def to_swmm_string(self) -> str:
        return ";;Project Title/Notes\n" + self


class SectionDf(SectionBase, pd.DataFrame):
    _metadata = ["_ncol", "_headings", "headings"]
    _ncol: int = 0
    _headings: list[str] = []
    _index_col: list[str] | str | None = None

    @classmethod
    def _data_cols(cls, desc: bool = True) -> list[str]:
        if isinstance(cls._index_col, str):
            idx = [copy.deepcopy(cls._index_col)]
        else:
            idx = copy.deepcopy(cls._index_col)

        if not desc:
            idx.append("desc")

        return [col for col in cls.headings if col not in idx]

    @classproperty
    def headings(cls) -> list[str]:
        return (
            cls._headings
            + [f"param{i+1}" for i in range(cls._ncol - len(cls._headings))]
            + ["desc"]
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._validate_headings()

    @classmethod
    def from_section_text(cls, text: str) -> Self:
        """Construct an instance of the class from the section inp text"""
        raise NotImplementedError

    @classmethod
    def _from_section_text(cls, text: str, ncols: int) -> Self:
        """

        Parse the SWMM section t ext into a dataframe

        This is a generic parser that assumes the SWMM section is tabular with the each row
        having the same number of tokens (i.e. columns). Comments preceeding a row in the inp file
        are added to the dataframe in a comments column.

        """
        rows = text.split("\n")
        data: list[TRow] = []
        line_comment = ""
        for row in rows:
            # check if row contains data
            if not _is_data(row):
                continue

            elif _is_line_comment(row):
                line_comment += _strip_comment(row)[1] + "\n"
                continue

            line, comment = _strip_comment(row)
            if len(comment) > 0:
                line_comment += comment + "\n"

            # split row into tokens coercing numerics into floats
            split_data = [_coerce_numeric(val) for val in line.split()]

            # parse tokenzied data into uniform tabular shape so each
            # row has the same number of columns
            table_data = cls._tabulate(split_data)

            data += cls._get_rows(
                table_data=table_data,
                ncols=ncols,
                line_comment=line_comment,
            )
            line_comment = ""

        # instantiate DataFrame
        df = cls(data=data, columns=cls.headings, dtype=object)
        return cls(df.set_index(cls._index_col)) if cls._index_col else df

        # if cls._index_col is not None:
        #     df.set_index(cls._index_col)
        # return df

    @staticmethod
    def _is_nested_list(l: TRow | list[TRow]) -> TypeGuard[list[TRow]]:
        return isinstance(l[0], list)

    @staticmethod
    def _is_not_nested_list(l: TRow | list[TRow]) -> TypeGuard[TRow]:
        return not isinstance(l[0], list)

    @classmethod
    def _get_rows(
        cls,
        table_data: TRow | list[TRow],
        ncols: int,
        line_comment: str,
    ) -> list[TRow]:

        _table_data: list[TRow]
        if cls._is_nested_list(table_data):
            _table_data = table_data
        elif cls._is_not_nested_list(table_data):
            _table_data = [table_data]
        else:
            raise Exception(f"Error parsing row {table_data}")

        rows: list[TRow] = []
        for row in _table_data:
            # create and empty row
            row_data: TRow = [""] * (ncols + 1)
            # assign data to row
            row_data[:ncols] = row
            # add comments to last column
            row_data[-1] = line_comment.strip("\n")
            rows.append(row_data)
        return rows

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        """
        Function to convert tokenized data into a table row with an expected number of columns

        This function allows the parser to accomodate lines in a SWWM section that might have
        different numbers of tokens.

        This is the generic version of the method that assumes all tokens in the line
        are assign the front of the table row and any left over spaces in the row are left
        blank. Various sections require custom implementations of thie method.

        """
        out: TRow = [""] * cls._ncol
        out[: len(line)] = line
        return out

    @classmethod
    def _new_empty(cls) -> Self:
        """Construct and empty instance"""
        df = cls(data=[], columns=cls.headings)
        return df.set_index(cls._index_col) if cls._index_col else df

    @classmethod
    def _newobj(cls, *args, **kwargs) -> Self:
        df = cls(*args, **kwargs)
        return df

    def _validate_headings(self) -> None:
        missing = []
        for heading in self.headings:
            if heading not in self.reset_index().columns:
                missing.append(heading)
        if len(missing) > 0:
            # print('cols: ',self.columns)
            raise ValueError(
                f"{self.__class__.__name__} section is missing columns {missing}"
            )
            # self.reindex(self.headings,inplace=True)

    def add_element(self, **kwargs) -> Self:
        # Create a new row with NaN values for all columns
        headings = self.headings.copy()
        idx_name: str | tuple[str, ...]
        try:
            if isinstance(self._index_col, str):
                idx_name = self._index_col
                idx = kwargs[idx_name]
                headings.remove(idx_name)
                kwargs.pop(idx_name)

            elif isinstance(self._index_col, (list, tuple)):
                idx_name = tuple(self._index_col)
                idx = []
                for col in idx_name:
                    idx.append(kwargs[col])
                    headings.remove(col)
                    kwargs.pop(col)
                idx = tuple(idx)

        except KeyError:
            raise KeyError(
                f"Missing index column {self._index_col!r} in provided values. Please provide a value for {self._index_col!r}"
            )
        new_row = pd.Series(index=headings, name=idx, dtype=object)

        # Update the new row with provided values
        for col, value in kwargs.items():
            if col in headings:
                new_row.loc[col] = value
            else:
                print(
                    f"Warning: Column '{col}' not found in the DataFrame. Skipping this value."
                )
        # Append the new row to the DataFrame
        self.loc[idx, :] = new_row
        return self

    @property
    def _constructor(self):
        # required override for pandas
        # https://pandas.pydata.org/docs/development/extending.html#override-constructor-properties
        return self.__class__

    @property
    def _constructor_sliced(self):
        # required override for pandas
        # https://pandas.pydata.org/docs/development/extending.html#override-constructor-properties
        return SectionSeries

    def _constructor_from_mgr(self, mgr, axes) -> Self:
        # required override for pandas
        return self.__class__._from_mgr(mgr, axes)

    def _constructor_sliced_from_mgr(self, mgr, axes) -> SectionSeries:
        # required override for pandas
        return SectionSeries._from_mgr(mgr, axes)

    def to_swmm_string(self) -> str:
        """Create a string representation of section"""
        self._validate_headings()
        # reset index
        out_df = (
            self.reset_index(self._index_col)
            .reindex(self.headings, axis=1)
            .infer_objects(copy=False)
            .fillna("")
        )

        # determine the longest variable in each column of the table
        # used to figure out how wide to make the columns
        max_data = (
            out_df.astype(str)
            .map(
                len,
            )
            .max()
        )
        # determine the length of the header names
        max_header = out_df.columns.to_series().apply(len)

        max_header.iloc[
            0
        ] += 2  # add 2 to first header to account for comment formatting

        # determine the column widths by finding the max legnth out of data
        # and headers
        col_widths = pd.concat([max_header, max_data], axis=1).max(axis=1) + 2

        # create format strings for header, divider, and data
        header_format = ""
        header_divider = ""
        data_format = ""
        for i, col in enumerate(col_widths.drop("desc")):
            data_format += f"{{:<{col}}}"
            header_format += f";;{{:<{col-2}}}" if i == 0 else f"{{:<{col}}}"
            header_divider += f";;{'-'*(col-4)}  " if i == 0 else f"{'-'*(col-2)}  "
        data_format += "\n"
        header_format += "\n"
        header_divider += "\n"

        # loop over data and format each each row of data as a string
        outstr = ""
        for i, row in enumerate(out_df.drop("desc", axis=1).values):
            desc = out_df.loc[i, "desc"]
            if (not pd.isna(desc)) and (len(strdesc := str(desc)) > 0):
                outstr += comment_formatter(strdesc)
            outstr += data_format.format(*row)

        header = header_format.format(*out_df.drop("desc", axis=1).columns)
        # concatenate the header, divider, and data
        return header + header_divider + outstr


class Title(SectionText):
    _section_name = "TITLE"


class Option(SectionDf):
    """
    Index: Option
    Columns: Value
    """

    _section_name = "OPTIONS"
    _ncol = 2
    _headings = ["Option", "Value"]
    _index_col = "Option"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)

    def _ipython_key_completions_(self):
        return list(["Value"])


class Report(SectionBase):
    _section_name = "REPORT"

    @dataclass
    class LIDReportEntry:
        Name: str
        Subcatch: str
        Fname: str

    class LIDReport(list[LIDReportEntry]):
        def __init__(self, entries: Iterable[Report.LIDReportEntry]):
            for i in entries:
                if not isinstance(i, Report.LIDReportEntry):
                    raise ValueError(
                        f"LIDReport is instantiated with a sequence of LIDReportEntries, got {type(i)}",
                    )
            super().__init__(entries)

        def add(self, lid_name: str, subcatch: str, Fname: str) -> None:
            self.append(
                Report.LIDReportEntry(
                    Name=lid_name,
                    Subcatch=subcatch,
                    Fname=Fname,
                ),
            )

        def delete(self, lid_name: str) -> None:
            for i, v in enumerate(self):
                if v.Name == lid_name:
                    break
            self.pop(i)

        def __repr__(self):
            rep = "LIDReportList(\n"
            for lid in self:
                rep += f"    {lid}.__repr__()\n"
            return f"{rep})"

    def __init__(
        self,
        disabled: str | None = None,
        input: str | None = None,
        continuity: str | None = None,
        flowstats: str | None = None,
        controls: str | None = None,
        averages: str | None = None,
        subcatchments: list[str] = [],
        nodes: list[str] = [],
        links: list[str] = [],
        lids: list[dict] = [],
    ):
        self.DISABLED = disabled
        self.INPUT = input
        self.CONTINUITY = continuity
        self.FLOWSTATS = flowstats
        self.CONTROLS = controls
        self.AVERAGES = averages
        self.SUBCATCHMENTS = subcatchments
        self.NODES = nodes
        self.LINKS = links
        self.LID = self.LIDReport([])

        for lid in lids:
            self.LID.add(lid["name"], lid["subcatch"], lid["fname"])

    @classmethod
    def from_section_text(cls, text: str, *args, **kwargs) -> Self:
        rows = text.split("\n")

        obj = cls()

        for row in rows:
            # check if row contains data
            if not _is_data(row):
                continue

            if ";" in row:
                warnings.warn(
                    "swmm.pandas does not currently support comments in the [REPORT] section. Truncating...",
                )
                if _is_line_comment(row):
                    continue

            tokens = row.split()
            report_type = tokens[0].upper()
            if not hasattr(obj, report_type):
                warnings.warn(
                    f"{report_type} is not a supported report type, skipping..."
                )
                continue
            elif report_type in ("SUBCATCHMENTS", "NODES", "LINKS"):
                setattr(
                    obj,
                    report_type,
                    getattr(obj, report_type) + tokens[1:],
                )
            elif report_type == "LID":
                obj.LID.add(
                    lid_name=tokens[1],
                    subcatch=tokens[2],
                    Fname=tokens[3],
                )
            else:
                setattr(obj, report_type, tokens[1])

        return obj

    @classmethod
    def _from_section_text(cls, text: str, *args, **kwargs) -> Self:
        raise NotImplementedError

    @classmethod
    def _new_empty(cls) -> Self:
        return cls()

    @classmethod
    def _newobj(cls, *args, **kwargs) -> Self:
        return cls(*args, **kwargs)

    def to_swmm_string(self) -> str:
        return ";;Reporting Options\n" + self.__repr__()

    def __repr__(self) -> str:
        out_str = ""
        for switch in (
            "DISABLED",
            "INPUT",
            "CONTINUITY",
            "FLOWSTATS",
            "CONTROLS",
            "AVERAGES",
        ):
            if (value := getattr(self, switch)) is not None:
                out_str += f"{switch} {value}\n"

        for seq in ("SUBCATCHMENTS", "NODES", "LINKS"):
            if len(items := getattr(self, seq)) > 0:
                i = 0
                while i < len(items):
                    out_str += f"{seq} {' '.join(items[i:i+5])}\n"
                    i += 5
        if len(self.LID) > 0:
            for lid in self.LID:
                out_str += f"LID {lid.Name} {lid.Subcatch} {lid.Fname}\n"

        return out_str

    def __len__(self):
        length = 0
        for switch in ("DISABLED", "INPUT", "CONTINUITY", "FLOWSTATS", "CONTROLS"):
            if getattr(self, switch) is not None:
                length += 1

        for seq in ("SUBCATCHMENTS", "NODES", "LINKS"):
            length += len(getattr(self, seq))

        length += len(self.LID)
        return length


class Files(SectionText):
    _section_name = "FILES"


class Event(SectionDf):
    _section_name = "EVENT"
    _ncol = 2
    _headings = ["Start", "End"]

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        out: TRow = [""] * cls._ncol
        if len(line) != 4:
            raise ValueError(f"Event lines must have 4 values but found {len(line)}")

        start_time = " ".join(line[:2])  # type: ignore
        end_time = " ".join(line[2:])  # type: ignore

        try:
            out[0] = pd.to_datetime(start_time)
            out[1] = pd.to_datetime(end_time)
            return out
        except Exception as e:
            print(f"Error parsing event dates: {start_time}  or   {end_time}")
            raise e

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)

    def to_swmm_string(self) -> str:
        df = self.copy()

        df["Start"] = pd.to_datetime(df["Start"]).dt.strftime("%m/%d/%Y %H:%M")
        df["End"] = pd.to_datetime(df["End"]).dt.strftime("%m/%d/%Y %H:%M")
        return super(Event, df).to_swmm_string()


class Raingage(SectionDf):
    _section_name = "RAINGAGES"
    _ncol = 8
    _headings = [
        "Name",
        "Format",
        "Interval",
        "SCF",
        "Source_Type",
        "Source",
        "Station",
        "Units",
    ]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Evap(SectionDf):
    _section_name = "EVAPORATION"
    _ncol = 13
    _headings = ["Type"]
    _index_col = "Type"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Temperature(SectionDf):
    _section_name = "TEMPERATURE"
    _ncol = 14
    _headings = ["Option"]
    _index_col = "Option"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Subcatchment(SectionDf):
    _section_name = "SUBCATCHMENTS"
    _ncol = 9
    _headings = [
        "Name",
        "RainGage",
        "Outlet",
        "Area",
        "PctImp",
        "Width",
        "Slope",
        "CurbLeng",
        "SnowPack",
    ]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Subarea(SectionDf):
    _section_name = "SUBAREAS"
    _ncol = 8
    _headings = [
        "Subcatchment",
        "Nimp",
        "Nperv",
        "Simp",
        "Sperv",
        "PctZero",
        "RouteTo",
        "PctRouted",
    ]
    _index_col = "Subcatchment"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Infil(SectionDf):
    _section_name = "INFILTRATION"
    _ncol = 7
    _headings = [
        "Subcatchment",
        "param1",
        "param2",
        "param3",
        "param4",
        "param5",
        "Method",
    ]
    _index_col = "Subcatchment"
    _infiltration_methods = (
        "HORTON",
        "MODIFIED_HORTON",
        "GREEN_AMPT",
        "MODIFIED_GREEN_AMPT",
        "CURVE_NUMBER",
    )

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        out: TRow = [""] * cls._ncol

        # pop first entry in the line (subcatch name)
        out[0] = line.pop(0)

        # add catchment specific method if present
        if line[-1] in cls._infiltration_methods:
            out[cls._headings.index("Method")] = line.pop(-1)

        # add params
        out[1 : 1 + len(line)] = line
        return out


class Aquifer(SectionDf):
    _section_name = "AQUIFERS"
    _ncol = 14
    _headings = [
        "Name",
        "Por",
        "WP",
        "FC",
        "Ksat",
        "Kslope",
        "Tslope",
        "ETu",
        "ETs",
        "Seep",
        "Ebot",
        "Egw",
        "Umc",
        "ETupat",
    ]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Groundwater(SectionDf):
    _section_name = "GROUNDWATER"
    _ncol = 14
    _headings = [
        "Subcatchment",
        "Aquifer",
        "Node",
        "Esurf",
        "A1",
        "B1",
        "A2",
        "B2",
        "A3",
        "Dsw",
        "Egwt",
        "Ebot",
        "Wgr",
        "Umc",
    ]
    _index_col = "Subcatchment"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class GWF(SectionDf):
    _section_name = "GWF"
    _ncol = 3
    _headings = [
        "Subcatch",
        "Type",
        "Expr",
    ]
    _index_col = ["Subcatch", "Type"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)

    @classmethod
    def _tabulate(cls, line: list[str | float]) -> TRow | list[TRow]:
        out: TRow = [""] * cls._ncol
        out[0] = line.pop(0)
        out[1] = line.pop(0)
        out[2] = "".join([str(s).strip() for s in line])
        return out


class Snowpack(SectionDf):
    _section_name = "SNOWPACKS"
    _ncol = 9
    _headings = ["Name", "Surface"]
    _index_col = ["Name", "Surface"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Junc(SectionDf):
    _section_name = "JUNCTIONS"
    _ncol = 6
    _headings = [
        "Name",
        "Elevation",
        "MaxDepth",
        "InitDepth",
        "SurDepth",
        "Aponded",
    ]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Outfall(SectionDf):
    _section_name = "OUTFALLS"
    _ncol = 6
    _headings = ["Name", "Elevation", "Type", "StageData", "Gated", "RouteTo"]
    _index_col = "Name"

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        out: TRow = [""] * cls._ncol

        # pop first three entries in the line
        # (required entries for every outfall type)
        out[:3] = line[:3]
        outfall_type = str(out[2]).lower()
        del line[:3]
        try:
            if outfall_type in ("free", "normal"):
                out[4 : 4 + len(line)] = line
                return out
            else:
                out[3 : 3 + len(line)] = line
                return out
        except Exception as e:
            print("Error parsing Outfall line: {line}")
            raise e

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Storage(SectionDf):
    _section_name = "STORAGE"
    _ncol = 14
    _headings = [
        "Name",
        "Elev",
        "MaxDepth",
        "InitDepth",
        "Shape",
        "CurveName",
        "A1_L",
        "A2_W",
        "A0_Z",
        "SurDepth",
        "Fevap",
        "Psi",
        "Ksat",
        "IMD",
    ]
    _index_col = "Name"

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        out: TRow = [""] * cls._ncol
        out[: cls._headings.index("CurveName")] = line[:5]
        line = line[5:]
        shape = str(out[cls._headings.index("Shape")]).lower()
        if shape in ("functional", "cylindrical", "conical", "paraboloid", "pyramidal"):
            out[6 : 6 + len(line)] = line
            return out
        elif shape == "tabular":
            out[cls._headings.index("CurveName")] = line.pop(0)
            out[
                cls._headings.index("SurDepth") : cls._headings.index("SurDepth")
                + len(line)
            ] = line
            return out
        else:
            raise ValueError(f"Unexpected line in storage section ({line})")

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Divider(SectionDf):
    _section_name = "DIVIDERS"
    _ncol = 12
    _headings = [
        "Name",
        "Elevation",
        "DivLink",
        "DivType",
        "DivCurve",
        "Qmin",
        "Height",
        "Cd",
        "Ymax",
        "Y0",
        "Ysur",
        "Apond",
    ]
    _index_col = "Name"

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        out: TRow = [""] * cls._ncol

        # pop first four entries in the line
        # (required entries for every Divider type)
        out[:4] = line[:4]
        div_type = str(out[3]).lower()
        del line[:4]
        try:
            if div_type == "overflow":
                out[8 : 8 + len(line)] = line

            elif div_type == "cutoff":
                out[5] = line.pop(0)
                out[8 : 8 + len(line)] = line
            elif div_type == "tabular":
                out[4] = line.pop(0)
                out[8 : 8 + len(line)] = line
            elif div_type == "weir":
                out[5 : 5 + len(line)] = line
            else:
                raise ValueError(f"Unexpected divider type: {div_type!r}")
            return out

        except Exception as e:
            print("Error parsing Divider line: {line!r}")
            raise e

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Conduit(SectionDf):
    _section_name = "CONDUITS"
    _ncol = 9
    _headings = [
        "Name",
        "FromNode",
        "ToNode",
        "Length",
        "Roughness",
        "InOffset",
        "OutOffset",
        "InitFlow",
        "MaxFlow",
    ]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Pump(SectionDf):
    _section_name = "PUMPS"
    _ncol = 7
    _headings = [
        "Name",
        "FromNode",
        "ToNode",
        "PumpCurve",
        "Status",
        "Startup",
        "Shutoff",
    ]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Orifice(SectionDf):
    _section_name = "ORIFICES"
    _ncol = 8
    _headings = [
        "Name",
        "FromNode",
        "ToNode",
        "Type",
        "Offset",
        "Qcoeff",
        "Gated",
        "CloseTime",
    ]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Weir(SectionDf):
    _section_name = "WEIRS"
    _ncol = 13
    _headings = [
        "Name",
        "FromNode",
        "ToNode",
        "Type",
        "CrestHt",
        "Qcoeff",
        "Gated",
        "EndCon",
        "EndCoeff",
        "Surcharge",
        "RoadWidth",
        "RoadSurf",
        "CoeffCurve",
    ]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Outlet(SectionDf):
    _section_name = "OUTLETS"
    _ncol = 9
    _headings = [
        "Name",
        "FromNode",
        "ToNode",
        "Offset",
        "Type",
        "CurveName",
        "Qcoeff",
        "Qexpon",
        "Gated",
    ]
    _index_col = "Name"

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        out: TRow = [""] * cls._ncol
        out[: cls._headings.index("CurveName")] = line[:5]
        line = line[5:]

        if "functional" in str(out[cls._headings.index("Type")]).lower():
            out[6 : 6 + len(line)] = line
            return out
        elif "tabular" in str(out[cls._headings.index("Type")]).lower():
            out[cls._headings.index("CurveName")] = line[0]
            if len(line) > 1:
                out[cls._headings.index("Gated")] = line[1]
            return out
        else:
            raise ValueError(f"Unexpected line in outlet section ({line})")

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Xsections(SectionDf):
    _section_name = "XSECTIONS"
    _shapes = (
        "CIRCULAR",
        "FORCE_MAIN",
        "FILLED_CIRCULAR",
        "DUMMY",
        "RECT_CLOSED",
        "RECT_OPEN",
        "TRAPEZOIDAL",
        "TRIANGULAR",
        "HORIZ_ELLIPSE",
        "VERT_ELLIPSE",
        "ARCH",
        "PARABOLIC",
        "POWER",
        "RECT_TRIANGULAR",
        "RECT_ROUND",
        "MODBASKETHANDLE",
        "EGG",
        "HORSESHOE",
        "GOTHIC",
        "CATENARY",
        "SEMIELLIPTICAL",
        "BASKETHANDLE",
        "SEMICIRCULAR",
        "CUSTOM",
    )

    _ncol = 9
    _headings = [
        "Link",
        "Shape",
        "Geom1",
        "Curve",
        "Geom2",
        "Geom3",
        "Geom4",
        "Barrels",
        "Culvert",
    ]
    _index_col = "Link"

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        out: TRow = [""] * cls._ncol
        out[:2] = line[:2]
        line = line[2:]

        if str(out[1]).lower() == "custom" and len(line) >= 2:
            out[cls._headings.index("Curve")], out[cls._headings.index("Geom1")] = (
                line[1],
                line[0],
            )
            ## TODO: Fix this depending on results from https://github.com/USEPA/Stormwater-Management-Model/issues/193
            # out[cls.headings.index("Barrels")] = line[2] if len(line) > 2 else 1
            out[
                cls._headings.index("Geom3") : cls._headings.index("Geom3")
                + len(line)
                - 2
            ] = line[2:]
            return out
        elif str(out[1]).lower() == "irregular":
            out[cls._headings.index("Curve")] = line[0]
            return out
        elif str(out[1]).upper() in cls._shapes:
            out[cls._headings.index("Geom1")] = line.pop(0)
            out[
                cls._headings.index("Geom2") : cls._headings.index("Geom2") + len(line)
            ] = line
            return out
        else:
            raise ValueError(f"Unexpected line in xsection section ({line})")

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)

    def to_swmm_string(self) -> str:
        df = self.copy(deep=True)

        # fill geoms
        mask = df["Shape"].isin(self._shapes)
        geom_cols = [f"Geom{i}" for i in range(1, 5)]
        df.loc[mask, geom_cols] = (
            df.loc[mask, geom_cols]
            .infer_objects(copy=False)
            .fillna(0)
            .infer_objects(copy=False)
        )
        df.loc[mask, geom_cols] = (
            df.loc[mask, geom_cols]
            .infer_objects(copy=False)
            .replace("", 0)
            .infer_objects(copy=False)
        )

        # fix custom shapes, Geom2 needs to be empty since the curve goes there
        mask = df["Shape"].astype(str).str.upper() == "CUSTOM"
        df.loc[mask, "Geom2"] = ""

        return super(Xsections, df).to_swmm_string()


class Street(SectionDf):
    _section_name = "STREETS"
    _ncol = 11
    _headings = [
        "Name",
        "Tcrown",
        "Hcurb",
        "Sroad",
        "nRoad",
        "Hdep",
        "Wdep",
        "Sides",
        "Wback",
        "Sback",
        "nBack",
    ]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Transects(SectionText):
    _section_name = "TRANSECTS"


class Timeseries(SectionBase):
    _section_name = "TIMESERIES"

    def __init__(self, ts: dict):
        self._timeseries = ts

    @dataclass
    class TimeseriesFile:
        name: str
        Fname: str
        desc: str = ""

        def to_swmm(self):
            desc = comment_formatter(self.desc)
            return f"{self.desc}{self.name}  FILE  {self.Fname}\n\n"

    @staticmethod
    def _timeseries_to_swmm_dat(df, name):
        def df_time_formatter(x):
            if isinstance(x, pd.Timedelta):
                total_seconds = x.total_seconds()
                hours = int(total_seconds // 3600)  # Get the total hours
                minutes = int((total_seconds % 3600) // 60)  # Get the remaining minutes
                return f"{hours:02}:{minutes:02}"
            elif isinstance(x, pd.Timestamp):
                return x.strftime("%m/%d/%Y %H:%M")
            elif isinstance(x, (float, int)):
                return x

        def df_comment_formatter(x):
            if len(x) > 0:
                return comment_formatter(x).strip("\n")
            else:
                return ""

        df["name"] = name

        if len(comment := df.attrs.get("desc", "")) > 0:
            comment_line = df_comment_formatter(comment) + "\n"
        else:
            comment_line = ""
        return (
            comment_line
            + df.reset_index(names="time")
            .reindex(["name", "time", "value", "desc"], axis=1)
            .fillna("")
            .to_string(
                formatters=dict(time=df_time_formatter, desc=df_comment_formatter),
                index=False,
                header=False,
            )
            + "\n\n"
        )

    @classmethod
    def from_section_text(cls, text: str):
        def is_valid_time_format(time_string):
            pattern = r"^\d+:\d+$"
            return bool(re.match(pattern, time_string))

        def is_valid_date(date_str):
            # Regex pattern to match mm/dd/yyyy, m/d/yyyy, m/dd/yyyy, or mm/d/yyyy
            pattern = r"^(0?[1-9]|1[0-2])/([0-2]?[0-9]|3[01])/(\d{4})$"

            # Check if the date string matches the pattern
            match = re.match(pattern, date_str)

            return bool(match)

        timeseries: dict[str, pd.DataFrame | Timeseries.TimeseriesFile] = {}

        rows = text.split("\n")
        line_comment = ""
        ts_comment = ""
        current_time_series_name = ""
        current_time_series_data: list[TRow] = []
        for row in rows:
            # check if row contains data
            if not _is_data(row):
                continue

            elif _is_line_comment(row):
                line_comment += _strip_comment(row)[1] + "\n"
                continue

            line, comment = _strip_comment(row)
            if len(comment) > 0:
                line_comment += comment + "\n"

            # split row into tokens coercing numerics into floats
            split_data = [_coerce_numeric(val) for val in line.split()]

            ts_name = str(split_data.pop(0))
            if ts_name != current_time_series_name:

                if len(current_time_series_data) > 0:
                    df = pd.DataFrame(
                        current_time_series_data, columns=["time", "value", "desc"]
                    ).set_index("time")
                    df.attrs["desc"] = ts_comment
                    timeseries[current_time_series_name] = df

                current_time_series_name = ts_name
                current_time_series_data = []
                ts_comment = line_comment
                line_comment = ""

                if str(split_data[0]).upper() == "FILE" and len(split_data) == 2:
                    timeseries[ts_name] = cls.TimeseriesFile(
                        name=ts_name, Fname=str(split_data[1]), desc=line_comment
                    )
                    continue
            time: pd.Timedelta | pd.Timestamp
            while len(split_data) > 0:
                if isinstance(split_data[0], Number):
                    time = pd.Timedelta(hours=float(split_data.pop(0)))
                    value = float(split_data.pop(0))
                elif is_valid_time_format(split_data[0]):
                    hours, minutes = str(split_data.pop(0)).split(":")
                    time = pd.Timedelta(hours=int(hours), minutes=int(minutes))
                    value = float(split_data.pop(0))
                elif is_valid_date(split_data[0]):
                    date = pd.to_datetime(split_data.pop(0))
                    if not is_valid_time_format(split_data[0]):
                        raise ValueError(
                            f"Error parsing timeseries {ts_name!r} time: {split_data[0]}"
                        )
                    hours, minutes = str(split_data.pop(0)).split(":")
                    _time = pd.Timedelta(hours=int(hours), minutes=int(minutes))
                    time = date + _time
                    value = float(split_data.pop(0))
                else:
                    raise ValueError(f"Error parsing Timeseries row {split_data}")

                current_time_series_data.append([time, value, line_comment])

            line_comment = ""

        # instantiate DataFrame
        return cls(ts=timeseries)

    @classmethod
    def _from_section_text(cls, text: str, *args, **kwargs) -> Self:
        raise NotImplementedError

    @classmethod
    def _new_empty(cls) -> Self:
        return cls(ts={})

    @classmethod
    def _newobj(cls, *args, **kwargs) -> Self:
        return cls(*args, **kwargs)

    def to_swmm_string(self) -> str:
        out_str = textwrap.dedent(
            """\
            ;;Name           Date       Time       Value     
            ;;-------------- ---------- ---------- ----------
        """
        )
        for ts_name, ts_data in self._timeseries.items():
            if isinstance(ts_data, pd.DataFrame):
                out_str += self._timeseries_to_swmm_dat(ts_data, ts_name)
            elif isinstance(ts_data, self.TimeseriesFile):
                out_str += ts_data.to_swmm()
        return out_str

    def add_file_timeseries(self, name: str, Fname: str, comment: str = "") -> Self:
        self._timeseries[name] = self.TimeseriesFile(
            name=name, Fname=Fname, desc=comment
        )
        return self

    def __setitem__(self, key, data) -> None:
        if isinstance(data, pd.DataFrame):
            if "value" not in data.columns:
                raise ValueError(
                    f"Expected 'value' columns in dataframe, got {data.columns!r}"
                )

            self._timeseries[key] = data.reindex(["value", "comment"], axis=1)
        else:
            raise TypeError(
                f"__setitem__ currently only supports dataframes, got {type(data)}. "
                "Use the `add_file_timeseries` method to add file-based timeseries"
            )

    def __getitem__(self, name) -> TimeseriesFile | pd.DataFrame:
        return self._timeseries[name]

    def __repr__(self) -> str:
        longest_name = max(map(len, self._timeseries.keys()))
        width = longest_name + 2
        reprstr = ""
        for name, value in self._timeseries.items():
            if isinstance(value, self.TimeseriesFile):
                reprstr += f"{name:{width}}|  TimeseriesFile(Fname={value.Fname!r}, desc={value.desc!r})\n"
            elif isinstance(value, pd.DataFrame):
                reprstr += f"{name:{width}}|  DataFrame(start={value.index[0]!r}, end={value.index[-1]!r},len={len(value)})\n"
        return reprstr

    def __iter__(self) -> Iterator[TimeseriesFile | pd.DataFrame]:
        return iter(self._timeseries.values())

    def __len__(self) -> int:
        return len(self._timeseries)

    def _ipython_key_completions_(self) -> list[str]:
        """Provide method for the key-autocompletions in IPython.
        See http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
        For the details.
        """

        return list(self._timeseries.keys())


class Patterns(SectionDf):
    _section_name = "PATTERNS"
    _ncol = 3
    _headings = ["Name", "Type", "Multiplier"]
    _index_col = ["Name"]
    _valid_types = [
        "MONTHLY",
        "DAILY",
        "HOURLY",
        "WEEKEND",
    ]

    @classmethod
    def _tabulate(cls, line: list[str | float]) -> TRow | list[TRow]:
        out: list[TRow] = []
        name = line.pop(0)

        pattern_type: str | NAType
        if str(line[0]).upper() in cls._valid_types:
            pattern_type = str(line.pop(0)).upper()
        elif isinstance(line[0], Number):
            pattern_type = pd.NA
        else:
            raise ValueError(f"Error parsing pattern line {[name]+line!r}")

        for value in line:
            row: TRow = [""] * cls._ncol
            float_val = float(value)
            row[0:3] = name, pattern_type, float_val
            out.append(row)
        return out

    @classmethod
    def _validate_pattern_types(cls, df: pd.DataFrame) -> dict[str, str]:
        unique_patterns = df.reset_index()[["Name", "Type"]].dropna().drop_duplicates()
        if unique_patterns["Name"].duplicated().any():
            raise ValueError(
                "Pattern with duplicate types found in input file. "
                "Each pattern must only specify a single type to work with swmm.pandas"
            )
        if not all(
            bools := [pattern in cls._valid_types for pattern in unique_patterns.Type]
        ):
            invalid_patterns = unique_patterns["Type"].loc[~np.array(bools)].to_list()
            raise ValueError(f"Unknown curves {invalid_patterns!r}")

        return unique_patterns.set_index("Name")["Type"].to_dict()

    @classmethod
    def from_section_text(cls, text: str) -> Self:
        df = super()._from_section_text(text, cls._ncol)
        pattern_types = cls._validate_pattern_types(df)
        df = df.reset_index().drop("Type", axis=1)
        df["Pattern_Index"] = df.groupby("Name").cumcount()
        df = cls(df.set_index(["Name", "Pattern_Index"]))
        df.attrs = pattern_types  # type: ignore
        return df

    def to_swmm_string(self) -> str:
        df = self.copy(deep=True)

        # add type back into frame in first row of curve
        type_idx = pd.MultiIndex.from_frame(
            df.index.to_frame()
            .drop("Name", axis=1)
            .groupby("Name")["Pattern_Index"]
            .min()
            .reset_index()
        )
        type_values = type_idx.get_level_values(0).map(df.attrs).to_numpy()
        df.loc[:, "Type"] = ""
        df.loc[type_idx, "Type"] = type_values

        # sort by name and index then drop the curve index field since swmm doesn't use it
        df = df.sort_index(ascending=[True, True])
        df.index = df.index.droplevel("Pattern_Index")
        return super(Patterns, df).to_swmm_string()


class Inlet(SectionDf):
    _section_name = "INLETS"
    _ncol = 7
    _headings = [
        "Name",
        "Type",
    ]
    _index_col = ["Name", "Type"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Inlet_Usage(SectionDf):
    _section_name = "INLET_USAGE"
    _ncol = 9
    _headings = [
        "Conduit",
        "Inlet",
        "Node",
        "Number",
        "%Clogged",
        "MaxFlow",
        "hDStore",
        "wDStore",
        "Placement",
    ]
    _index_col = "Conduit"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Losses(SectionDf):
    _section_name = "LOSSES"
    _ncol = 6
    _headings = ["Link", "Kentry", "Kexit", "Kavg", "FlapGate", "Seepage"]
    _index_col = "Link"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)

    def to_swmm_string(self) -> str:
        df = self.copy(deep=True)

        for col in self._data_cols(desc=False):
            if col != "FlapGate":
                df[col] = df[col].infer_objects(copy=False).fillna(0.0)
            else:
                df[col] = df[col].infer_objects(copy=False).fillna("NO")

        return super(Losses, df).to_swmm_string()


class Controls(SectionText):
    _section_name = "CONTROLS"


class Pollutants(SectionDf):
    _section_name = "POLLUTANTS"
    _ncol = 11
    _headings = [
        "Name",
        "Units",
        "Crain",
        "Cgw",
        "Crdii",
        "Kdecay",
        "SnowOnly",
        "CoPollutant",
        "CoFrac",
        "Cdwf",
        "Cinit",
    ]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class LandUse(SectionDf):
    _section_name = "LANDUSES"
    _ncol = 4
    _headings = ["Name", "SweepInterval", "Availability", "LastSweep"]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)

    def to_swmm_string(self) -> str:
        for col in self.columns:
            self[col] = self[col].infer_objects(copy=False).fillna(0.0)
        return super().to_swmm_string()


class Coverage(SectionDf):
    _section_name = "COVERAGES"
    _ncol = 3
    _headings = ["Subcatchment", "landuse", "Percent"]
    _index_col = ["Subcatchment", "landuse"]

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        if len(line) > 3:
            raise Exception(
                "swmm.pandas doesn't yet support having multiple land "
                "uses on a single coverage line. Separate your land use "
                "coverages onto individual lines first",
            )
        return super()._tabulate(line)

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Loading(SectionDf):
    _section_name = "LOADINGS"
    _ncol = 3
    _headings = ["Subcatchment", "Pollutant", "InitBuildup"]
    _index_col = ["Subcatchment", "Pollutant"]

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        if len(line) > 3:
            raise Exception(
                "swmm.pandas doesn't yet support having multiple pollutants "
                "uses on a single loading line. Separate your pollutant "
                "loadings onto individual lines first",
            )
        return super()._tabulate(line)

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Buildup(SectionDf):
    _section_name = "BUILDUP"
    _ncol = 4
    _headings = ["Landuse", "Pollutant", "FuncType", "C1", "C2", "C3", "PerUnit"]
    _index_col = ["Landuse", "Pollutant"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Washoff(SectionDf):
    _section_name = "WASHOFF"
    _ncol = 4
    _headings = ["Landuse", "Pollutant", "FuncType", "C1", "C2", "SweepRmvl", "BmpRmvl"]
    _index_col = ["Landuse", "Pollutant"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Treatment(SectionDf):
    _section_name = "TREATMENT"
    _ncol = 3
    _headings = ["Node", "Pollutant", "Func"]
    _index_col = ["Node", "Pollutant"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)

    @classmethod
    def _tabulate(cls, line: list[str | float]) -> TRow | list[TRow]:
        node = str(line.pop(0))
        poll = str(line.pop(0))
        eqn = " ".join(str(v) for v in line)
        out: TRow = [node, poll, eqn]
        return out


class Inflow(SectionDf):
    _section_name = "INFLOWS"
    _ncol = 8
    _headings = [
        "Node",
        "Constituent",
        "TimeSeries",
        "InflowType",
        "Mfactor",
        "Sfactor",
        "Baseline",
        "Pattern",
    ]
    _index_col = ["Node", "Constituent"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        return [v.replace('"', "") if isinstance(v, str) else v for v in line]

    def to_swmm_string(self) -> str:
        df = self.copy(deep=True)
        df["Mfactor"] = df["Mfactor"].infer_objects(copy=False).fillna(1.0)
        df["Sfactor"] = df["Sfactor"].infer_objects(copy=False).fillna(1.0)

        # strip out any existing double quotes
        df["TimeSeries"] = df["TimeSeries"].fillna("").str.replace('"', "")
        df["TimeSeries"] = '"' + df["TimeSeries"].astype(str) + '"'
        return super(Inflow, df).to_swmm_string()


class DWF(SectionDf):
    _section_name = "DWF"
    _ncol = 7
    _headings = [
        "Node",
        "Constituent",
        "AvgValue",
        "Pat1",
        "Pat2",
        "Pat3",
        "Pat4",
    ]
    _index_col = ["Node", "Constituent"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        return [v.replace('"', "") if isinstance(v, str) else v for v in line]

    def to_swmm_string(self) -> str:
        df = self.copy(deep=True)
        df["AvgValue"] = df["AvgValue"].infer_objects(copy=False).fillna(0.0)

        for ipat in range(1, 5):
            col = f"Pat{ipat}"
            df[col] = df[col].fillna("").str.replace('"', "")
            df[col] = '"' + df[col].astype(str) + '"'

        return super(DWF, df).to_swmm_string()


class RDII(SectionDf):
    _section_name = "RDII"
    _ncol = 3
    _headings = ["Node", "UHgroup", "SewerArea"]
    _index_col = "Node"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Hydrographs(SectionDf):
    _section_name = "HYDROGRAPHS"
    _ncol = 9
    _headings = [
        "Name",
        "Month_RG",
        "Response",
        "R",
        "T",
        "K",
        "IA_max",
        "IA_rec",
        "IA_ini",
    ]
    _index_col = ["Name", "Month_RG", "Response"]

    @classmethod
    def from_section_text(cls, text: str) -> Self:

        df = super()._from_section_text(text, cls._ncol).reset_index()
        rg_rows = cls._find_rain_gauge_rows(df)
        rgs = df.loc[rg_rows].set_index("Name")["Month_RG"].to_dict()
        df.drop(rg_rows, inplace=True)
        df = cls(df.set_index(cls._index_col).sort_index())
        df.attrs = rgs
        return df

    @property
    def rain_gauges(self) -> dict[str, str]:
        return self.attrs  # type: ignore

    @staticmethod
    def _find_rain_gauge_rows(df) -> pd.Index:
        # Function to check if a row matches the raingauge criteria
        def is_raingauge_row(row):
            return (row != "").sum() == 2

        # Apply the function to each row and get the indices where it's True
        raingauge_indices = df.loc[df.apply(is_raingauge_row, axis=1)].index

        return raingauge_indices

    def to_swmm_string(self) -> str:

        def month_to_number(month):
            try:
                return list(month_abbr).index(month.capitalize())
            except ValueError:
                return -1  # This will sort unrecognized months to the top

        def index_mapper(index):
            if index.name == "Month_RG":
                return index.map(month_to_number)
            else:
                return index

        # add rain gauge rows
        _temp = self.__class__._new_empty()
        for name in self.index.get_level_values("Name").unique():
            try:
                _temp.add_element(
                    Name=name, Month_RG=self.rain_gauges[name], Response=""
                )
            except KeyError:
                raise KeyError(
                    f"Raingauge for hydrograph {name!r} not found in hydrographs.rain_gauges property. "
                    f"Only found {self.rain_gauges!r}"
                )

        df = pd.concat([self, _temp])
        # sort by name, month, and response after adding in raingauges
        df = Hydrographs(df.sort_index(ascending=[True, True, False], key=index_mapper))
        return super(Hydrographs, df).to_swmm_string()


class Curves(SectionDf):
    _section_name = "CURVES"
    _ncol = 4
    _headings = ["Name", "Type", "X_Value", "Y_Value"]
    _index_col = ["Name"]
    _valid_types = [
        "STORAGE",
        "SHAPE",
        "DIVERSION",
        "TIDAL",
        "PUMP1",
        "PUMP2",
        "PUMP3",
        "PUMP4",
        "PUMP5",
        "RATING",
        "CONTROL",
        "WEIR",
    ]

    @classmethod
    def _tabulate(cls, line: list[str | float]) -> TRow | list[TRow]:
        out = []
        name = line.pop(0)

        curve_type: str | NAType
        if str(line[0]).upper() in cls._valid_types:
            curve_type = str(line.pop(0)).upper()
        elif isinstance(line[0], Number):
            curve_type = pd.NA
        else:
            raise ValueError(f"Error parsing curve line {[name]+line!r}")

        for chunk in range(0, len(line), 2):
            row: TRow = [""] * cls._ncol
            x_value, y_value = line[chunk : chunk + 2]
            row[0:4] = name, curve_type, x_value, y_value
            out.append(row)
        return out

    @classmethod
    def _validate_curve_types(cls, df: pd.DataFrame) -> dict[str, str]:
        unique_curves = df.reset_index()[["Name", "Type"]].dropna().drop_duplicates()
        if unique_curves["Name"].duplicated().any():
            raise ValueError(
                "Curve with duplicate types found in input file. "
                "Each curve must only specify a single type to work with swmm.pandas"
            )
        if not all(
            bools := [curve in cls._valid_types for curve in unique_curves.Type]
        ):
            invalid_curves = unique_curves["Type"].loc[~np.array(bools)].to_list()
            raise ValueError(f"Unknown curves {invalid_curves!r}")

        return unique_curves.set_index("Name")["Type"].to_dict()

    @classmethod
    def from_section_text(cls, text: str) -> Self:
        df = super()._from_section_text(text, cls._ncol)
        curve_types = cls._validate_curve_types(df)
        df = df.reset_index().drop("Type", axis=1)
        df["Curve_Index"] = df.groupby("Name").cumcount()
        df = cls(df.set_index(["Name", "Curve_Index"]))
        df.attrs = curve_types  # type: ignore
        return df

    def to_swmm_string(self) -> str:
        df = self.copy(deep=True)

        # add type back into frame in first row of curve
        type_idx = pd.MultiIndex.from_frame(
            df.index.to_frame()
            .drop("Name", axis=1)
            .groupby("Name")["Curve_Index"]
            .min()
            .reset_index()
        )
        type_values = type_idx.get_level_values(0).map(df.attrs).to_numpy()
        df.loc[:, "Type"] = ""
        df.loc[type_idx, "Type"] = type_values

        # sort by name and index then drop the curve index field since swmm doesn't use it
        df = Curves(df.sort_index(ascending=[True, True]))
        df.index = df.index.droplevel("Curve_Index")
        return super(Curves, df).to_swmm_string()


class Coordinates(SectionDf):
    _section_name = "COORDINATES"
    _ncol = 3
    _headings = ["Node", "X", "Y"]
    _index_col = "Node"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Vertices(SectionDf):
    _section_name = "VERTICIES"
    _ncol = 3
    _headings = ["Link", "X", "Y"]
    _index_col = "Link"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Polygons(SectionDf):
    _section_name = "POLYGONS"
    _ncol = 3
    _headings = ["Subcatch", "X", "Y"]
    _index_col = "Subcatch"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Symbols(SectionDf):
    _section_name = "SYMBOLS"
    _ncol = 3
    _headings = ["Gage", "X", "Y"]
    _index_col = "Gage"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Labels(SectionDf):
    _section_name = "LABELS"
    _ncol = 8
    _headings = [
        "Xcoord",
        "Ycoord",
        "Label",
        "Anchor",
        "Font",
        "Size",
        "Bold",
        "Italic",
    ]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Tags(SectionDf):
    _section_name = "TAGS"
    _ncol = 3
    _headings = ["Element", "Name", "Tag"]
    _index_col = ["Element", "Name"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Profile(SectionText):
    _section_name = "PROFILE"


class LID_Control(SectionDf):
    _section_name = "LID_CONTROLS"
    _ncol = 9
    _headings = ["Name", "Type"]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)

    @classmethod
    def _tabulate(cls, line: list[str | float]) -> TRow | list[TRow]:
        lid_type = line[1]
        if lid_type == "REMOVALS":
            out: list[TRow] = []
            name = line.pop(0)
            lid_type = line.pop(0)
            for chunk in range(0, len(line), 2):
                row: TRow = [""] * cls._ncol
                pollutant, removal = line[chunk : chunk + 2]
                row[0:4] = name, lid_type, pollutant, removal
                out.append(row)
            return out
        else:
            return super()._tabulate(line)


class LID_Usage(SectionDf):
    _section_name = "LID_USAGE"
    _ncol = 11
    _headings = [
        "Subcatchment",
        "LIDProcess",
        "Number",
        "Area",
        "Width",
        "InitSat",
        "FromImp",
        "ToPerv",
        "RptFile",
        "DrainTo",
        "FromPerv",
    ]

    _index_col = ["Subcatchment", "LIDProcess"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)


class Adjustments(SectionDf):
    _section_name = "ADJUSTMENTS"
    _ncol = 15
    _headings = [
        "Parameter",
        "Subcatchment",
        "Pattern",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    _index_col = "Parameter"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol)

    @classmethod
    def _tabulate(cls, line: list[str | float | int]) -> TRow | list[TRow]:
        out: TRow = [""] * cls._ncol
        out[0] = line.pop(0)
        if str(out[0]).lower() in ["n-perv", "dstore"]:
            out[1 : 1 + len(line)] = line
        else:
            start = cls._headings.index("Jan")
            out[start : start + len(line)] = line
        return out


class Backdrop(SectionText):
    _section_name = "BACKDROP"


# TODO: write custom to_string class
# class Backdrop:
#     @classmethod
#     def __init__(self, text: str):
#         rows = text.split("\n")
#         data = []
#         line_comment = ""
#         for row in rows:
#             if not _is_data(row):
#                 continue

#             elif row.strip()[0] == ";":
#                 print(row)
#                 line_comment += row
#                 continue

#             line, comment = _strip_comment(row)
#             line_comment += comment

#             split_data = [_coerce_numeric(val) for val in row.split()]

#             if split_data[0].upper() == "DIMENSIONS":
#                 self.dimensions = split_data[1:]

#             elif split_data[0].upper() == "FILE":
#                 self.file = split_data[1]

#     def from_section_text(cls, text: str):
#         return cls(text)

#     def __repr__(self) -> str:
#         return f"Backdrop(dimensions = {self.dimensions}, file = {self.file})"


class Map(SectionText):
    _section_name = "MAP"


# TODO: write custom to_string class
# class Map:
#     @classmethod
#     def __init__(self, text: str):
#         rows = text.split("\n")
#         data = []
#         line_comment = ""
#         for row in rows:
#             if not _is_data(row):
#                 continue

#             elif row.strip()[0] == ";":
#                 print(row)
#                 line_comment += row
#                 continue

#             line, comment = _strip_comment(row)
#             line_comment += comment

#             split_data = [_coerce_numeric(val) for val in row.split()]

#             if split_data[0].upper() == "DIMENSIONS":
#                 self.dimensions = split_data[1:]

#             elif split_data[0].upper() == "UNITS":
#                 self.units = split_data[1]

#     @classmethod
#     def from_section_text(cls, text: str):
#         return cls(text)

#     def __repr__(self) -> str:
#         return f"Map(dimensions = {self.dimensions}, units = {self.units})"


_sections: dict[str, type[SectionBase]] = {
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
    "GWF": GWF,
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
    "TRANSECT": Transects,
    "STREETS": Street,
    "INLET_USAGE": Inlet_Usage,
    "INLET": Inlet,
    "LOSS": Losses,
    # TODO build parser for this table
    "CONTROL": Controls,
    "POLLUT": Pollutants,
    "LANDUSE": LandUse,
    "COVERAGE": Coverage,
    "LOADING": Loading,
    "BUILDUP": Buildup,
    "WASHOFF": Washoff,
    "TREATMENT": Treatment,
    "INFLOW": Inflow,
    "DWF": DWF,
    "RDII": RDII,
    "HYDROGRAPH": Hydrographs,
    "CURVE": Curves,
    "TIMESERIES": Timeseries,
    "PATTERN": Patterns,
    "MAP": Map,
    "POLYGON": Polygons,
    "COORDINATE": Coordinates,
    "VERTICES": Vertices,
    "LABEL": Labels,
    "SYMBOL": Symbols,
    "BACKDROP": Backdrop,
    "PROFILE": Profile,
    "TAG": Tags,
}
