from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Value
from typing import Iterable, List, Optional, Self, TypedDict, TypeVar

import pandas as pd

_logger = logging.getLogger(__name__)


class classproperty(object):
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
        number = int(number) if number.is_integer() else number
        if str(number) == data:
            return number
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


class SectionSeries(pd.Series):
    @property
    def _constructor(self):
        return SectionSeries

    @property
    def _constructor_expanddim(self):
        return SectionDf

    def _constructor_from_mgr(self, mgr, axes):
        # required override for pandas
        return self.__class__._from_mgr(mgr, axes)


class SectionBase(ABC):
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
    def _newobj(cls, *args, **kwargs) -> Self: ...

    @abstractmethod
    def to_swmm_string(self) -> str: ...


class SectionText(SectionBase, str):
    @classmethod
    def from_section_text(cls, text: str):
        """Construct an instance of the class from the section inp text"""
        return cls._from_section_text(text)

    @classmethod
    def _from_section_text(cls, text: str):
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
    _ncol = 0
    _headings = []
    _index_col = None

    @classproperty
    def headings(cls):
        return (
            cls._headings
            + [f"param{i+1}" for i in range(cls._ncol - len(cls._headings))]
            + ["desc"]
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._validate_headings()

    @classmethod
    def from_section_text(cls, text: str):
        """Construct an instance of the class from the section inp text"""
        raise NotImplementedError

    @classmethod
    def _from_section_text(cls, text: str, ncols: int, headings: List[str]) -> Self:
        """

        Parse the SWMM section t ext into a dataframe

        This is a generic parser that assumes the SWMM section is tabular with the each row
        having the same number of tokens (i.e. columns). Comments preceeding a row in the inp file
        are added to the dataframe in a comments column.

        """
        rows = text.split("\n")
        data = []
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
        df = cls(data=data, columns=cls.headings)
        return df.set_index(cls._index_col) if cls._index_col else df

        # if cls._index_col is not None:
        #     df.set_index(cls._index_col)
        # return df

    @staticmethod
    def _is_nested_list(l: list) -> bool:
        return isinstance(l[0], list)

    @classmethod
    def _get_rows(
        cls,
        table_data: list[str | float] | list[list[str | float]],
        ncols: int,
        line_comment: str,
    ) -> list[list[str | float]]:
        if not cls._is_nested_list(table_data):
            table_data = [table_data]

        rows = []
        for row in table_data:
            # create and empty row
            row_data = [""] * (ncols + 1)
            # assign data to row
            row_data[:ncols] = row
            # add comments to last column
            row_data[-1] = line_comment.strip("\n")
            rows.append(row_data)
        return rows

    @classmethod
    def _tabulate(cls, line: list[str | float]) -> list[str | float] | list[list[str | float]]:
        """
        Function to convert tokenized data into a table row with an expected number of columns

        This function allows the parser to accomodate lines in a SWWM section that might have
        different numbers of tokens.

        This is the generic version of the method that assumes all tokens in the line
        are assign the front of the table row and any left over spaces in the row are left
        blank. Various sections require custom implementations of thie method.

        """
        out = [""] * cls._ncol
        out[: len(line)] = line
        return out

    @classmethod
    def _new_empty(cls):
        """Construct and empty instance"""
        return cls(data=[], columns=cls.headings)

    @classmethod
    def _newobj(cls, *args, **kwargs):
        df = cls(*args, **kwargs)
        return df

    def _validate_headings(self):
        missing = []
        for heading in self.headings:
            if heading not in self.reset_index().columns:
                missing.append(heading)
        if len(missing) > 0:
            # print('cols: ',self.columns)
            raise ValueError(f"{self.__class__.__name__} section is missing columns {missing}")
            # self.reindex(self.headings,inplace=True)

    def add_element(self, obj):
        other = self.__class__.__newobj__(obj, index=[0])
        return pd.concat([self, other])

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

    def _constructor_from_mgr(self, mgr, axes):
        # required override for pandas
        return self.__class__._from_mgr(mgr, axes)

    def _constructor_sliced_from_mgr(self, mgr, axes):
        # required override for pandas
        return SectionSeries._from_mgr(mgr, axes)

    def to_swmm_string(self):
        """Create a string representation of section"""
        self._validate_headings()
        # reset index
        out_df = (
            self.reset_index(self._index_col)
            .reindex(self.headings, axis=1)
            .infer_objects(copy=False)
            .fillna("")
        )

        def comment_formatter(line: str):
            if len(line) > 0:
                line = ";" + line.strip().strip("\n").strip()
                line = line.replace("\n", "\n;") + "\n"
            return line

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

        max_header.iloc[0] += 2  # add 2 to first header to account for comment formatting

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


class Title(SectionText): ...


class Option(SectionDf):
    """
    Index: Option
    Columns: Value
    """

    _ncol = 2
    _headings = ["Option", "Value"]
    _index_col = "Option"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    def _ipython_key_completions_(self):
        return list(["Value"])


class Report(SectionBase):
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

        def remove(self, lid_name: str) -> None:
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
        disabled: Optional[str] = None,
        input: Optional[str] = None,
        continuity: Optional[str] = None,
        flowstats: Optional[str] = None,
        controls: Optional[str] = None,
        averages: Optional[str] = None,
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
                warnings.warn(f"{report_type} is not a supported report type, skipping...")
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


class Files(SectionText): ...


class Event(SectionDf):
    _ncol = 2
    _headings = ["Start", "End"]

    @classmethod
    def _tabulate(cls, line: list):
        out = [""] * Event._ncol
        if len(line) != 4:
            raise ValueError(f"Event lines must have 4 values but found {len(line)}")

        start_time = " ".join(line[:2])
        end_time = " ".join(line[2:])

        try:
            out[0] = pd.to_datetime(start_time)
            out[1] = pd.to_datetime(end_time)
            return out
        except Exception as e:
            print(f"Error parsing event dates: {start_time}  or   {end_time}")
            raise e

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    def to_swmm_string(self):
        df = self.copy()
        df["Start"] = df["Start"].dt.strftime("%m/%d/%Y %H:%M")
        df["End"] = df["End"].dt.strftime("%m/%d/%Y %H:%M")
        return super(Event, df).to_swmm_string()


class Raingage(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Evap(SectionDf):
    _ncol = 13
    _headings = ["Type"]
    _index_col = "Type"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Temperature(SectionDf):
    _ncol = 14
    _headings = ["Option"]
    _index_col = "Option"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Subcatchment(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Subarea(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Infil(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @classmethod
    def _tabulate(cls, line: list):
        out = [""] * Infil._ncol

        # pop first entry in the line (subcatch name)
        out[0] = line.pop(0)

        # add catchment specific method if present
        if line[-1] in cls._infiltration_methods:
            out[cls._headings.index("Method")] = line.pop(-1)

        # add params
        out[1 : 1 + len(line)] = line
        return out


class Aquifer(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Groundwater(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class GWF(SectionDf):
    _ncol = 3
    _headings = [
        "Subcatch",
        "Type",
        "Expr",
    ]
    _index_col = ["Subcatch", "Type"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @classmethod
    def _tabulate(cls, line: List[str | float]) -> List[str | float] | List[List[str | float]]:
        out = [""] * cls._ncol
        out[0] = line.pop(0)
        out[1] = line.pop(0)
        out[2] = "".join([str(s).strip() for s in line])
        return out


class Snowpack(SectionDf):
    _ncol = 9
    _headings = ["Name", "Surface"]
    _index_col = ["Name", "Surface"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Junc(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Outfall(SectionDf):
    _ncol = 6
    _headings = ["Name", "Elevation", "Type", "StageData", "Gated", "RouteTo"]
    _index_col = "Name"

    @classmethod
    def _tabulate(cls, line: list):
        out = [""] * Outfall._ncol

        # pop first three entries in the line
        # (required entries for every outfall type)
        out[:3] = line[:3]
        outfall_type = out[2].lower()
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Storage(SectionDf):
    _ncol = 14
    _headings = [
        "Name",
        "Elev",
        "MaxDepth",
        "InitDepth",
        "Shape",
        "CurveName",
        "A1/L",
        "A2/W",
        "A0/Z",
        "SurDepth",
        "Fevap",
        "Psi",
        "Ksat",
        "IMD",
    ]
    _index_col = "Name"

    @classmethod
    def _tabulate(cls, line: list):
        out = [""] * Storage._ncol
        out[: cls._headings.index("CurveName")] = line[:5]
        line = line[5:]
        shape = out[cls._headings.index("Shape")].lower()
        if shape in ("functional", "cylindrical", "conical", "paraboloid", "pyramidal"):
            out[6 : 6 + len(line)] = line
            return out
        elif shape == "tabular":
            out[cls._headings.index("CurveName")] = line.pop(0)
            out[cls._headings.index("SurDepth") : cls._headings.index("SurDepth") + len(line)] = (
                line
            )
            return out
        else:
            raise ValueError(f"Unexpected line in storage section ({line})")

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Divider(SectionDf):
    _ncol = 11
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
    def _tabulate(cls, line: list):
        out = [""] * Outfall._ncol

        # pop first four entries in the line
        # (required entries for every Divider type)
        out[:4] = line[:4]
        div_type = out[3].lower()
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Conduit(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Pump(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Orifice(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Weir(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Outlet(SectionDf):
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
    def _tabulate(cls, line: list):
        out = [""] * Outlet._ncol
        out[: cls._headings.index("CurveName")] = line[:5]
        line = line[5:]

        if "functional" in out[cls._headings.index("Type")].lower():
            out[6 : 6 + len(line)] = line
            return out
        elif "tabular" in out[cls._headings.index("Type")].lower():
            out[cls._headings.index("CurveName")] = line[0]
            if len(line) > 1:
                out[cls._headings.index("Gated")] = line[1]
            return out
        else:
            raise ValueError(f"Unexpected line in outlet section ({line})")

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Xsections(SectionDf):
    _shapes = (
        "CIRCULAR",
        "FORCE_MAIN",
        "FILLED_CIRCULAR",
        "Depth",
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
        "Height",
        "RECT_ROUND",
        "Radius",
        "MODBASKETHANDLE",
        "EGG",
        "HORSESHOE",
        "GOTHIC",
        "CATENARY",
        "SEMIELLIPTICAL",
        "BASKETHANDLE",
        "SEMICIRCULAR",
    )

    _ncol = 8
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

    @classmethod
    def _tabulate(cls, line: list):
        out = [""] * Outlet._ncol
        out[:2] = line[:2]
        line = line[2:]

        if out[1].lower() == "custom" and len(line) >= 2:
            out[cls._headings.index("Curve")], out[cls._headings.index("Geom1")] = (
                line[1],
                line[0],
            )
            out[cls.headings.index("Barrels")] = out[2] if len(out) > 2 else 1
            return out
        elif out[1].lower() == "irregular":
            out[cls._headings.index("Curve")] = line[0]
            return out
        elif out[1].upper() in cls._shapes:
            out[cls._headings.index("Geom1")] = line.pop(0)
            out[cls._headings.index("Geom2") : cls._headings.index("Geom2") + len(line)] = line
            return out
        else:
            raise ValueError(f"Unexpected line in xsection section ({line})")

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Street(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Inlet(SectionDf):
    _ncol = 7
    _headings = [
        "Name",
        "Type",
    ]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Inlet_Usage(SectionDf):
    _ncol = 7
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Losses(SectionDf):
    _ncol = 6
    _headings = ["Link", "Kentry", "Kexit", "Kavg", "FlapGate", "Seepage"]
    _index_col = "Link"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Pollutants(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class LandUse(SectionDf):
    _ncol = 4
    _headings = ["Name", "SweepInterval", "Availability", "LastSweep"]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Coverage(SectionDf):
    _ncol = 3
    _headings = ["Subcatchment", "landuse", "Percent"]
    _index_col = ["Subcatchment", "landuse"]

    @classmethod
    def _tabulate(cls, line: list):
        if len(line) > 3:
            raise Exception(
                "swmm.pandas doesn't yet support having multiple land "
                "uses on a single coverage line. Separate your land use "
                "coverages onto individual lines first",
            )
        return super()._tabulate(line)

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Loading(SectionDf):
    _ncol = 3
    _headings = ["Subcatchment", "Pollutant", "InitBuildup"]
    _index_col = ["Subcatchment", "Pollutant"]

    @classmethod
    def _tabulate(cls, line: list):
        if len(line) > 3:
            raise Exception(
                "swmm.pandas doesn't yet support having multiple pollutants "
                "uses on a single loading line. Separate your pollutant "
                "loadings onto individual lines first",
            )
        return super()._tabulate(line)

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Buildup(SectionDf):
    _ncol = 4
    _headings = ["Landuse", "Pollutant", "FuncType", "C1", "C2", "C3", "PerUnit"]
    _index_col = ["Landuse", "Polutant"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Washoff(SectionDf):
    _ncol = 4
    _headings = ["Landuse", "Pollutant", "FuncType", "C1", "C2", "SweepRmvl", "BmpRmvl"]
    _index_col = ["Landuse", "Polutant"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Treatment(SectionDf):
    _ncol = 3
    _headings = ["Node", "Pollutant", "Func"]
    _index_col = ["Node", "Pollutant"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


# TODO needs double quote handler for timeseries heading
class Inflow(SectionDf):
    _ncol = 8
    _headings = [
        "Node",
        "Constituent",
        "TimeSeries",
        "Type",
        "Mfactor",
        "Sfactor",
        "Baseline",
        "Pattern",
    ]
    _index_col = ["Node", "Constituent"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class DWF(SectionDf):
    _ncol = 7
    _headings = [
        "Node",
        "Constituent",
        "Baseline",
        "Pat1",
        "Pat2",
        "Pat3",
        "Pat4",
    ]
    _index_col = ["Node", "Constituent"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class RDII(SectionDf):
    _ncol = 3
    _headings = ["Node", "UHgroup", "SewerArea"]
    _index_col = "Node"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Coordinates(SectionDf):
    _ncol = 3
    _headings = ["Node", "X", "Y"]
    _index_col = "Node"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Verticies(SectionDf):
    _ncol = 3
    _headings = ["Link", "X", "Y"]
    _index_col = "Link"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Polygons(SectionDf):
    _ncol = 3
    _headings = ["Subcatch", "X", "Y"]
    _index_col = "Subcatch"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Symbols(SectionDf):
    _ncol = 3
    _headings = ["Gage", "X", "Y"]
    _index_col = "Gage"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Labels(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Tags(SectionDf):
    _ncol = 3
    _headings = ["Element", "Name", "Tag"]
    _index_col = "Element"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class LID_Control(SectionDf):
    _ncol = 9
    _headings = ["Name", "Type"]
    _index_col = "Name"

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @classmethod
    def _tabulate(cls, line: list[str | float]) -> list[str | float] | list[list[str | float]]:
        lid_type = line[1]
        if lid_type == "REMOVALS":
            out = []
            name = line.pop(0)
            lid_type = line.pop(0)
            for chunk in range(0, len(line), 2):
                row = [""] * cls._ncol
                pollutant, removal = line[chunk : chunk + 2]
                row[0:4] = name, lid_type, pollutant, removal
                out.append(row)
            return out
        else:
            return super()._tabulate(line)


class LID_Usage(SectionDf):
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Adjustments(SectionDf):
    _ncol = 13
    _headings = [
        "Parameter",
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
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Backdrop(SectionText): ...


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


class Map(SectionText): ...


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
