from __future__ import annotations
import pandas as pd
from typing import List
import logging

_logger = logging.getLogger(__name__)


def _coerce_float(data):
    try:
        return float(data)
    except ValueError:
        return data


def _strip_comment(line: str):
    try:
        return line[: line.index(";")], line[line.index(";") :]

    except ValueError:
        return line, ""


def _is_line_comment(line: str):
    try:
        return line.strip()[0] == ";"
    except IndexError:
        return False


def _is_data(line: str):
    if len(line) == 0 or line.strip()[0:2] == ";;" or line.strip()[0] == "[":
        return False
    return True


class SectionSeries(pd.Series):
    @property
    def _constructor(self):
        return SectionSeries

    @property
    def _constructor_expanddim(self):
        return Section


class Section(pd.DataFrame):
    _metadata = ["_ncol", "_headings", "headings"]
    _ncol = 0
    _headings = []

    @classmethod
    @property
    def headings(cls):
        return (
            cls._headings
            + [f"param{i+1}" for i in range(cls._ncol - len(cls._headings))]
            + ["desc"]
        )

    @classmethod
    def from_section_text(cls, text: str):
        return text

    @classmethod
    def _from_section_text(cls, text: str, ncols: int, headings: List[str]):
        rows = text.split("\n")
        data = []
        line_comment = ""
        for row in rows:
            if not _is_data(row):
                continue

            elif row.strip()[0] == ";":
                # print(row)
                line_comment += row.replace(";", "").strip() + "\n"
                continue

            line, comment = _strip_comment(row)
            if len(comment) > 0:
                line_comment += comment.replace(";", "").strip() + "\n"

            row_data = [""] * (ncols + 1)
            # print(row_data)
            split_data = [_coerce_float(val) for val in line.split()]
            row_data[:ncols] = cls._assigner(split_data)
            row_data[-1] = line_comment
            data.append(row_data)
            line_comment = ""

        return cls(data=data, columns=cls.headings)

    @classmethod
    def _assigner(cls, line: list):
        out = [""] * cls._ncol
        out[: len(line)] = line
        return out

    @classmethod
    def _new_empty(cls):
        return cls(data=[], columns=cls.headings)

    @classmethod
    def _newobj(cls, *args, **kwargs):
        df = cls(*args, **kwargs)
        df._validate_headings()
        return df

    def _validate_headings(self):
        missing = []
        for heading in self.headings:
            if heading not in self.columns:
                missing.append(heading)
        if len(missing) > 0:
            # print('cols: ',self.columns)
            raise ValueError(
                f"{self.__class__.__name__} section is missing columns {missing}"
            )
            # self.reindex(self.headings,inplace=True)

    def add_element(self, obj):
        other = self.__class__.__newobj__(obj, index=[0])
        return pd.concat([self, other])

    @property
    def _constructor(self):
        return Section

    @property
    def _constructor_sliced(self):
        return SectionSeries

    def to_swmm_string(self):
        def comment_formatter(line):
            if len(line) > 0:
                line = ";" + line.strip().strip("\n")
                line = line.replace("\n", "\n;") + "\n"
            return line

        max_data = (
            self.astype(str)
            .map(
                len,
            )
            .max()
        )
        max_header = self.columns.to_series().apply(len)
        max_header.iloc[0] += (
            2  # add 2 to first header to account for comment formatting
        )
        col_widths = pd.concat([max_header, max_data], axis=1).max(axis=1) + 2

        data_format = ""
        header_format = ""
        header_sep = ""
        for i, col in enumerate(col_widths.drop("desc")):
            data_format += f"{{:<{col}}}"

            header_format += f";;{{:<{col-2}}}" if i == 0 else f"{{:<{col}}}"
            header_sep += f";;{'-'*(col-4)}  " if i == 0 else f"{'-'*(col-2)}  "
        data_format += "\n"
        header_format += "\n"
        header_sep += "\n"

        outstr = ""
        for i, row in enumerate(self.drop("desc", axis=1).values):
            desc = self.loc[i, "desc"]
            if len(desc) > 0:
                outstr += comment_formatter(desc)
            outstr += data_format.format(*row)

        header = header_format.format(*self.drop("desc", axis=1).columns)

        return header + header_sep + outstr


class Option(Section):
    _ncol = 2
    _headings = ["Option", "Value"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Option

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Evap(Section):
    _ncol = 13
    _headings = ["Type"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Evap

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Temperature(Section):
    _ncol = 14
    _headings = ["Option"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Temperature

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Raingage(Section):
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Raingage

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Subcatchment(Section):
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Subcatchment

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Subarea(Section):
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Subarea

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Infil(Section):
    _ncol = 6
    _headings = ["Subcatchment"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Infil

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Aquifer(Section):
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Aquifer

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Groundwater(Section):
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Groundwater

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Snowpack(Section):
    _ncol = 9
    _headings = ["Name", "Surface"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Snowpack

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Junc(Section):
    _ncol = 6
    _headings = [
        "Name",
        "Elevation",
        "MaxDepth",
        "InitDepth",
        "SurDepth",
        "Aponded",
    ]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Junc

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Outfall(Section):
    _ncol = 6
    _headings = ["Name", "Elevation", "Type", "StageData", "Gated", "RouteTo"]

    @classmethod
    def _assigner(cls, line: list):
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

    @property
    def _constructor(self):
        return Outfall

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Storage(Section):
    _ncol = 14
    _headings = [
        "Name",
        "Elev",
        "MaxDepth",
        "InitDepth",
        "Shape",
        "CurveName",
        "A1",
        "A2",
        "A0",
        "N/A",
        "Fevap",
        "Psi",
        "Ksat",
        "IMD",
    ]

    @classmethod
    def _assigner(cls, line: list):
        out = [""] * Storage._ncol
        out[: cls._headings.index("CurveName")] = line[:5]
        line = line[5:]

        if out[cls._headings.index("Shape")].lower() == "functional":
            out[6 : 6 + len(line)] = line
            return out
        elif out[cls._headings.index("Shape")].lower() == "tabular":
            out[cls._headings.index("CurveName")] = line.pop(0)
            out[cls._headings.index("N/A") : cls._headings.index("N/A") + len(line)] = (
                line
            )
            return out
        else:
            raise ValueError(f"Unexpected line in storage section ({line})")

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Storage

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Divider(Section):
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

    @classmethod
    def _assigner(cls, line: list):
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

    @property
    def _constructor(self):
        return Outfall

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Conduit(Section):
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Conduit

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Pump(Section):
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Pump

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Orifice(Section):
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Orifice

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Weir(Section):
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Weir

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Outlet(Section):
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

    @classmethod
    def _assigner(cls, line: list):
        out = [""] * Outlet._ncol
        out[: cls._headings.index("CurveName")] = line[:5]
        line = line[5:]

        if "functional" in out[cls._headings.index("Type")].lower():
            out[6 : 6 + len(line)] = line
            return out
        elif "tabular" in out[cls._headings.index("Type")].lower():
            out[cls._headings.index("CurveName")] = line[0]
            out[cls._headings.index("Gated")] = line[1]
            return out
        else:
            raise ValueError(f"Unexpected line in outlet section ({line})")

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Outlet

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Losses(Section):
    _ncol = 6
    _headings = ["Link", "Kentry", "Kexit", "Kavg", "FlapGate", "Seepage"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Pollutants(Section):
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Pollutants

    @property
    def _constructor_sliced(self):
        return SectionSeries


class LandUse(Section):
    _ncol = 4
    _headings = ["Name", "SweepInterval", "Availability", "LastSweep"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return LandUse

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Buildup(Section):
    _ncol = 4
    _headings = ["Landuse", "Pollutant", "FuncType", "C1", "C2", "C3", "PerUnit"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Buildup

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Washoff(Section):
    _ncol = 4
    _headings = ["Landuse", "Pollutant", "FuncType", "C1", "C2", "SweepRmvl", "BmpRmvl"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Washoff

    @property
    def _constructor_sliced(self):
        return SectionSeries


# TODO needs double quote handler for timeseries heading
class Inflow(Section):
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Inflow

    @property
    def _constructor_sliced(self):
        return SectionSeries


class DWF(Section):
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return DWF

    @property
    def _constructor_sliced(self):
        return SectionSeries


class RDII(Section):
    _ncol = 3
    _headings = ["Node", "UHgroup", "SewerArea"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return RDII

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Xsections(Section):
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
        "Curve",
        "Geom1",
        "Geom2",
        "Geom3",
        "Geom4",
        "Barrels",
        "Culvert",
    ]

    @classmethod
    def _assigner(cls, line: list):
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
            out[
                cls._headings.index("Geom1") : cls._headings.index("Geom1") + len(line)
            ] = line
            return out
        else:
            raise ValueError(f"Unexpected line in xsection section ({line})")

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Xsections

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Coordinates(Section):
    _ncol = 3
    _headings = ["Node", "X", "Y"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Coordinates

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Verticies(Section):
    _ncol = 3
    _headings = ["Link", "X", "Y"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Verticies

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Polygons(Section):
    _ncol = 3
    _headings = ["Subcatch", "X", "Y"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)


class Symbols(Section):
    _ncol = 3
    _headings = ["Gage", "X", "Y"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Symbols

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Labels(Section):
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

    @property
    def _constructor(self):
        return Labels

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Tags(Section):
    _ncol = 3
    _headings = ["Element", "Name", "Tag"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Tags

    @property
    def _constructor_sliced(self):
        return SectionSeries


class LID_Control(Section):
    _ncol = 9
    _headings = ["Name", "Type"]

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return LID_Control

    @property
    def _constructor_sliced(self):
        return SectionSeries


class LID_Usage(Section):
    _ncol = (11,)
    _headings = (
        [
            "Subcatchment",
            "LIDProcess",
            "Number",
            "Area",
            "Width",
            "InitSat",
            "FromImp",
            "ToPerv",
            "RptFiqle",
            "DrainTo",
            "FromPerv",
        ],
    )

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return LID_Usage

    @property
    def _constructor_sliced(self):
        return SectionSeries


class Adjustments(Section):
    _ncol = ("13",)
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

    @classmethod
    def from_section_text(cls, text: str):
        return super()._from_section_text(text, cls._ncol, cls._headings)

    @property
    def _constructor(self):
        return Adjustments

    @property
    def _constructor_sliced(self):
        return SectionSeries


# TODO: write custom to_string class
class Backdrop:
    @classmethod
    def __init__(self, text: str):
        rows = text.split("\n")
        data = []
        line_comment = ""
        for row in rows:
            if not _is_data(row):
                continue

            elif row.strip()[0] == ";":
                print(row)
                line_comment += row
                continue

            line, comment = _strip_comment(row)
            line_comment += comment

            split_data = [_coerce_float(val) for val in row.split()]

            if split_data[0].upper() == "DIMENSIONS":
                self.dimensions = split_data[1:]

            elif split_data[0].upper() == "FILE":
                self.file = split_data[1]

    def from_section_text(cls, text: str):
        return cls(text)

    def __repr__(self) -> str:
        return f"Backdrop(dimensions = {self.dimensions}, file = {self.file})"


# TODO: write custom to_string class
class Map:
    @classmethod
    def __init__(self, text: str):
        rows = text.split("\n")
        data = []
        line_comment = ""
        for row in rows:
            if not _is_data(row):
                continue

            elif row.strip()[0] == ";":
                print(row)
                line_comment += row
                continue

            line, comment = _strip_comment(row)
            line_comment += comment

            split_data = [_coerce_float(val) for val in row.split()]

            if split_data[0].upper() == "DIMENSIONS":
                self.dimensions = split_data[1:]

            elif split_data[0].upper() == "UNITS":
                self.units = split_data[1]

    @classmethod
    def from_section_text(cls, text: str):
        return cls(text)

    def __repr__(self) -> str:
        return f"Map(dimensions = {self.dimensions}, units = {self.units})"
