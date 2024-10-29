from __future__ import annotations

from aenum import EnumMeta
import os.path
import warnings
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable
from collections.abc import Sequence
from itertools import product
from io import SEEK_END
import struct

from aenum import Enum, EnumMeta, extend_enum
from numpy import asarray, atleast_1d, atleast_2d, concatenate, datetime64
from numpy import integer as npint
from numpy import ndarray, stack, tile, vstack
import numpy.core.records
from pandas.core.api import (
    DataFrame,
    DatetimeIndex,
    Index,
    MultiIndex,
    Timestamp,
    to_datetime,
    IndexSlice,
)
from swmm.toolkit import output, shared_enum

from swmm.pandas.output.structure import Structure
from swmm.pandas.output.tools import arrayish, _enum_get, _enum_keys


def output_open_handler(func):
    """Checks if output file is open before running function.

    Parameters
    ----------
    func: function
        method of Output class
    """

    @wraps(func)
    def inner_function(self, *args, **kwargs):
        if not self._loaded:
            self._open()

        return func(self, *args, **kwargs)

    return inner_function


class Output:
    def __init__(self, binfile: str, preload: bool = False):
        """Base class for a SWMM binary output file.

        The output object provides several options to process timeseries within binary output file.

        Output files should be closed after use prevent memory leaks. Close them explicitly with
        the `_close()` method or deleting the object using `del`, or use it with a context manager.

        .. code-block:: python

            # Using a the _close method
            >>> from swmm.pandas import Output
            >>> out = Output('tests/Model.out')
            >>> print(out.project_size)
            [3, 9, 8, 1, 3]
            >>> out._close() # can also use `del out`
            >>>
            # Using a context manager
            >>> with Output('tests/Model.out') as out:
            ...     print(out.pollutants)
            ('groundwater', 'pol_rainfall', 'sewage')

        Parameters
        ----------
        binfile: str
            model binary file path
        preload: bool
            If true, the whole output file will be preloaded into memory before instantiating the object.
            This is usefull if you plan on extracting lots of data from an output file that can easily
            fit into ram as it significantly can speed up iterative I/O.
        """

        self._period: int
        """number of reporting time steps in the """

        self._report: int
        """out file reporting time step in seconds"""

        self._start: datetime
        """start datetime of the output file records"""

        self._end: datetime
        """end datetime of the output file records"""

        self._timeIndex: DatetimeIndex
        """DatetimeIndex to use for output timeseries"""

        self._project_size: list[int]
        """Array of element count values [nSubcatchments, nNodes, nLinks, nSystems(1), nPollutants]"""

        self._subcatchments: tuple[str, ...]
        """Tuple of subcatchment names in output file"""

        self._links: tuple[str, ...]
        """Tuple of link names in output file"""

        self._pollutants: tuple[str, ...]
        """Tuple of pollutant names in output file"""

        self._handle = None

        self._binfile: str = binfile
        """path to binary output file"""

        self._delete_handle: bool = False
        """Indicates if output file was closed correctly"""

        self._preload: bool = preload

        self._loaded: bool = False
        """Indicates if output file was loaded correctly"""

        self.subcatch_attributes = Enum(
            "subcatch_attributes",
            list(shared_enum.SubcatchAttribute.__members__.keys())[:-1],
            start=0,
        )
        """Subcatchment attribute enumeration: By default has

        'rainfall',
        'snow_depth',
        'evap_loss',
        'infil_loss',
        'runoff_rate',
        'gw_outflow_rate',
        'gw_table_elev',
        'soil_moisture'
        """

        # need copies of enumes to extend them for pollutants
        # basically recreate enums using the keys from shared_enum
        # but drop POLLUT_CONC_0 for each
        #
        # I looked into using swmm.toolkit.output_metadata for this but it
        # extends global enums, which could break having multiple
        # output objects opened in the same python session if they
        # have different pollutant names

        self.node_attributes: shared_enum.NodeAttribute = Enum(
            "node_attributes",
            list(shared_enum.NodeAttribute.__members__.keys())[:-1],
            start=0,
        )
        """Node attribute enumeration: By default has

        'invert_depth',
        'hydraulic_head',
        'ponded_volume',
        'lateral_inflow',
        'total_inflow',
        'flooding_losses'
        """
        self.link_attributes: shared_enum.LinkAttribute = Enum(
            "link_attributes",
            list(shared_enum.LinkAttribute.__members__.keys())[:-1],
            start=0,
        )
        """Link attribute enumeration: By default has

        'flow_rate',
        'flow_depth',
        'flow_velocity',
        'flow_volume',
        'capacity',

        """

        self.system_attributes = shared_enum.SystemAttribute
        """System attribute enumeration: By default has

        'air_temp',
        'rainfall',
        'snow_depth',
        'evap_infil_loss',
        'runoff_flow',
        'dry_weather_inflow',
        'gw_inflow',
        'rdii_inflow',
        'direct_inflow',
        'total_lateral_inflow',
        'flood_losses',
        'outfall_flows',
        'volume_stored',
        'evap_rate',
        'ptnl_evap_rate'

        """

    @staticmethod
    def _elementIndex(
        elementID: str | int | None, indexSquence: Sequence[str], elementType: str
    ) -> int:
        """Validate the index of a model element passed to Output methods. Used to
        convert model element names to their index in the out file.

        Parameters
        ----------
        elementID: str, int
            The name or index of the model element listed in the index_dict dict.
        indexSquence: one of more string
            The ordered sequence against which to validate the index
            (one of self.nodes, self.links, self.subcatchments).
        elementType: str
            The type of model element (e.g. node, link, etc.)
            Only used to print the exception if an attribute cannot be found.

        Returns
        -------
        int
            The integer index of the requested element.

        Raises
        ------
        OutputException
            Exception if element cannot be found in indexSequence.

        """

        if isinstance(elementID, (int, npint)):
            return int(elementID)

        try:
            return indexSquence.index(elementID)

        # since this class can pull multiple attributes and elements in one function
        # call it is probably better to do some pre-validation of input arguments
        # before starting a potentially lengthy data pull
        except ValueError:
            raise ValueError(
                f"{elementType} ID: {elementID} does not exist in model output."
            )

    @staticmethod
    def _validateAttribute(
        attribute: int | str | Sequence[int | str] | None,
        validAttributes: EnumMeta,
    ) -> tuple[list, list]:
        """
        Function to validate attribute arguments of element_series, element_attribute,
        and element_result functions.

        Parameters
        ----------
        attribute: Union[int, str, Sequence[Union[int, str]], None]
            The attribute to validate against validAttributes.
        validAttributes: dict
            THe dict of attributes against which to validate attribute.

        Returns
        -------
        Tuple[list, list]
            Two arrays, one of attribute names and one of attribute indicies.

        """
        # this kind of logic was needed in the series and results functions.
        # not sure if this is the best way, but it felt a bit DRYer to
        # put it into a funciton
        attributeArray: list[EnumMeta | str | int]
        if isinstance(attribute, (type(None), EnumMeta)):
            attributeArray = list(_enum_keys(validAttributes))
        elif isinstance(attribute, arrayish):
            attributeArray = list(attribute)
        elif isinstance(attribute, (int, str)):
            attributeArray = [attribute]
        else:
            raise ValueError(f"Error validating attribute {attribute!r}")

        # allow mixed input of attributes
        # accept string names, integers, or enums values in the same list
        attributeIndexArray = []
        for i, attrib in enumerate(attributeArray):
            if isinstance(attrib, Enum):
                attributeArray[i] = attrib.name.lower()
                attributeIndexArray.append(attrib)

            elif isinstance(attrib, (int, npint)):
                # will raise index error if not in range
                attribName = _enum_keys(validAttributes)[attrib]
                attributeArray[i] = attribName
                attributeIndexArray.append(_enum_get(validAttributes, attribName))

            elif isinstance(attrib, str):
                index = _enum_get(validAttributes, attrib)
                if index is None:
                    raise ValueError(
                        f"Attribute {attrib} not in valid attribute list: {_enum_keys(validAttributes)}"
                    )
                attributeIndexArray.append(index)
            else:
                raise TypeError(
                    f"Input type: {type(attrib)} not valid. Must be one of int, str, or Enum"
                )

        # attributeIndexArray = [validAttributes.get(atr, -1) for atr in attributeArray]

        return attributeArray, attributeIndexArray

    @staticmethod
    def _validateElement(
        element: int | str | Sequence[int | str] | None,
        validElements: Sequence[str],
    ) -> tuple[list[str], list[int]]:
        """
        Function to validate element arguments of element_series, element_attribute,
        and element_result functions.

        Parameters
        ----------
        element: Union[int, str, Sequence[Union[int, str]], None]
            The element name or index or None. If None, return all elements in
            validElements.
        validElements: Sequence[str]
            Tuple of elements against which to validate element.

        Returns
        -------
        Tuple[list, list]
            Two arrays, one of element names and one of element indicies.

        """
        # this kind of logic was needed in the series and results functions
        # not sure if this is the best way, but it felt a bit DRYer to
        # put it into a funciton
        elementArray: list[str | int]
        if element is None:
            elementArray = list(validElements)
        elif isinstance(element, arrayish):
            elementArray = list(element)
        else:
            # ignore typing since types of this output list
            # are reconciled in the next loop. mypy was complaining.
            elementArray = [element]  # type: ignore

        elementIndexArray = []
        elemNameArray = []
        # allow mixed input of elements. string names can be mixed
        # with integer indicies in the same input list
        for i, elem in enumerate(elementArray):
            if isinstance(elem, (int, npint)):
                # will raise index error if not in range
                elemName = validElements[elem]
                elemNameArray.append(elemName)
                elementIndexArray.append(elem)

            elif isinstance(elem, str):
                elementIndexArray.append(Output._elementIndex(elem, validElements, ""))
                elemNameArray.append(elem)
            else:
                raise TypeError(
                    f"Input type {type(elem)} not valid. Must be one of int, str"
                )

        return elemNameArray, elementIndexArray

    @staticmethod
    def _datetime_from_swmm(swmm_datetime):
        remaining_days = swmm_datetime % 1
        days = swmm_datetime - remaining_days
        seconds = remaining_days * 86400
        dt = datetime(year=1899, month=12, day=30) + timedelta(
            days=days, seconds=seconds
        )
        return dt

    def _checkPollutantName(self, name: str) -> str:
        """Check pollutant name against existing attribute dicts.
        Rename and and warn if existing attribute is duplicated.

        Parameters
        ----------
        name: str
            The name of pollutant.

        Returns
        -------
        str
            The validated name of pollutant.
        """

        elems = []
        if name.lower() in _enum_keys(self.subcatch_attributes):
            elems.append("subcatchment")

        if name.lower() in _enum_keys(self.node_attributes):
            elems.append("node")

        if name.lower() in _enum_keys(self.link_attributes):
            elems.append("link")

        if name.lower() in _enum_keys(self.system_attributes):
            elems.append("system")

        if len(elems) > 0:
            warnings.warn(
                f"Pollutent {name} is a duplicate of existing {','.join(elems)} attribute, renaming to pol_{name}"
            )
            return f"pol_{name}"

        return name

    def _open(self) -> bool:
        """Open a binary file.

        Parameters
        ----------

        Returns
        -------
        bool
            True if binary file was opened successfully.

        """
        if not os.path.exists(self._binfile):
            raise ValueError(f"Output file at: '{self._binfile}' does not exist")

        if self._handle is None:
            self._handle = output.init()

        if not self._loaded:
            self._loaded = True
            output.open(self._handle, self._binfile)
            self._start = self._datetime_from_swmm(output.get_start_date(self._handle))
            self._report = output.get_times(self._handle, shared_enum.Time.REPORT_STEP)
            self._period = output.get_times(self._handle, shared_enum.Time.NUM_PERIODS)
            self._end = self._start + timedelta(seconds=self._period * self._report)

            # load pollutants if not already loaded
            if not hasattr(self, "_pollutants"):
                # load pollutant data if it has not before
                total = self.project_size[4]
                self._pollutants = tuple(
                    self._checkPollutantName(
                        self._objectName(shared_enum.ElementType.POLLUT, index).lower()
                    )
                    for index in range(total)
                )

                for i, nom in enumerate(self._pollutants):
                    # extend enums to include pollutants
                    extend_enum(self.subcatch_attributes, nom.upper(), 8 + i)
                    extend_enum(self.node_attributes, nom.upper(), 6 + i)
                    extend_enum(self.link_attributes, nom.upper(), 5 + i)

        if self._preload:
            # respos = output.get_positions(self._handle)[2]
            # self._close()

            subs = list(
                product(
                    ["sub"],
                    range(len(self.subcatchments)),
                    range(len(self.subcatch_attributes)),
                )
            )
            nodes = list(
                product(
                    ["node"],
                    range(len(self.nodes)),
                    range(len(self.node_attributes)),
                )
            )
            links = list(
                product(
                    ["link"],
                    range(len(self.links)),
                    range(len(self.link_attributes)),
                )
            )
            system = list(product(["sys"], ["sys"], range(len(self.system_attributes))))

            cols = subs + nodes + links + system
            cols.insert(0, ("datetime", 0, 0))

            idx = MultiIndex.from_tuples(cols)
            fmts = "f8" + ",f4" * (len(cols) - 1)

            with open(self._binfile, "rb") as fil:
                fil.seek(self._output_position, 0)
                dat = numpy.core.records.fromfile(fil, formats=fmts)
            self.data = DataFrame(dat)
            self.data.columns = idx

        return True

    def _close(self) -> bool:
        """Close an opened binary file.

        Parameters
        ----------

        Returns
        -------
        bool
            True if binary file was closed successfully.

        """
        if self._loaded:
            output.close(self._handle)
            self._loaded = False
            del self._handle
            self._handle = None
            self._delete_handle = True

        return True

    ###### outfile property getters ######
    @property
    def _output_position(self):
        if not hasattr(self, "__output_position"):
            with open(self._binfile, "rb") as fil:
                fil.seek(-4 * 4, SEEK_END)
                self.__output_position = struct.unpack("i", fil.read(4))[0]

        return self.__output_position

    @property  # type: ignore
    @output_open_handler
    def report(self) -> int:
        """Return the reporting timestep in seconds.

        Returns
        -------
        int
            The reporting timestep in seconds.

        """
        return self._report

    @property  # type: ignore
    @output_open_handler
    def start(self) -> datetime:
        """Return the reporting start datetime.

        Returns
        -------
        datetime
            The reporting start datetime.

        """
        return self._start

    @property  # type: ignore
    @output_open_handler
    def end(self) -> datetime:
        """Return the reporting end datetime.

        Returns
        -------
        datetime
            The reporting end datetime.
        """
        return self._end

    @property  # type: ignore
    @output_open_handler
    def period(self) -> int:
        """Return the number of reporting timesteps in the binary output file.

        Returns
        -------
        int
            The number of reporting timesteps.
        """
        return self._period

    @property  # type: ignore
    def project_size(self) -> list[int]:
        """Returns the number of each model element type available in out binary output file
        in the following order:

        [subcatchment, node, link, system, pollutant]

        Returns
        -------
        list
            A list of numbers of each model type.

            [nSubcatchments, nNodes, nLinks, nSystems(1), nPollutants]

        """
        if not hasattr(self, "_project_size"):
            self._load_project_size()
        return self._project_size

    @output_open_handler
    def _load_project_size(self) -> None:
        """Load model size into self._project_size"""
        self._project_size = output.get_proj_size(self._handle)

    @property
    def pollutants(self) -> tuple[str, ...]:
        """Return a tuple of pollutants available in SWMM binary output file.

        Returns
        -------
         Tuple[str]
           A tuple of pollutant names.

        """

        # chose not to write a pollutant loader method
        # because loading such is kind of imperative to the functionality
        # of other data getter methods, which don't necessarily
        # call pollutants method. Instead, pollutant loading logic is
        # thrown in the _open() method, and this method calls open if
        # pollutants are not available.
        if self._pollutants is None:
            self._open()

        return self._pollutants

    @property  # type: ignore
    @output_open_handler
    def _unit(self) -> tuple[int]:
        """Return SWMM binary output file unit type from `swmm.toolkit.shared_enum.UnitSystem`.

        Returns
        -------
        Tuple[int]
            Tuple of integers indicating system units, flow units, and units for each pollutant.

        """
        return tuple(output.get_units(self._handle))  # type: ignore

    @property
    def units(self) -> list[str]:
        """Return SWMM binary output file unit type from `swmm.toolkit.shared_enum.UnitSystem`.

        Returns
        -------
        List[str]
            List of string names for system units, flow units, and units for each pollutant.

            Values returned are the names from swmm.toolkit.shared_enum:
                UnitSystem
                FlowUnits
                ConcUnits

        """
        return [
            shared_enum.UnitSystem(self._unit[0]).name,
            shared_enum.FlowUnits(self._unit[1]).name,
        ] + [shared_enum.ConcUnits(i).name for i in self._unit[2:]]

    @property  # type: ignore
    @output_open_handler
    def _version(self) -> int:
        """Return SWMM version used to generate SWMM binary output file results.

        Returns
        -------
        int
            Integer representation of SWMM version used to make output file.

        """
        return output.get_version(self._handle)

    @output_open_handler
    def _objectName(self, object_type: int, index: int) -> str:
        """Get object name from SWMM binary output file using object type and object index.

        Parameters
        ----------
        object_type: int
            The object type from swmm.toolkit.shared_enum.ElementType.
        index: int
            The object index.

        Returns
        -------
        str
            object name

        """
        return output.get_elem_name(self._handle, object_type, index)

    ##### timestep setters and getters #####
    def _time2step(
        self,
        dateTime: (
            None
            | str
            | int
            | datetime
            | Timestamp
            | datetime64
            | Sequence[str | int | datetime | Timestamp | datetime64]
        ),
        ifNone: int = 0,
        method: str = "nearest",
    ) -> list[int]:
        """Convert datetime value to SWMM timestep index. By deafult, this returns the nearest timestep to
        to the requested date, so it will always return a time index available in the binary output file.


        Parameters
        ----------
        dateTime: datetime-like or string or sequence of such
            datetime to convert. Must be a datetime-like object or convertable
            with `pd.to_datetime`.

        ifNone: int
            The value to return if dateTime is None, defaults to 0.

        method: str
            The method name to pass to pandas `get_indexer`_, default to "nearest.

            .. _get_indexer: https://pandas.pydata.org/docs/reference/api/pandas.Index.get_indexer.html

        Returns
        -------
        Union[int, np.ndarray]
            SWMM model time step or array of time steps

        """
        if dateTime is None:
            return [ifNone]

        dt = asarray(dateTime).flatten()

        # if passing swmm time step, no indexing necessary
        if dt.dtype in (float, int):
            return dt.astype(int).tolist()

        # ensure datetime value
        dt = to_datetime(dt)
        return self.timeIndex.get_indexer(dt, method=method).tolist()

    @property
    def timeIndex(self) -> DatetimeIndex:
        """Returns DatetimeIndex of reporting timeseries in binary output file.

        Returns
        -------
        pd.DatetimeIndex
            A pandas `DatetimeIndex`_ for each reporting timestep.

            .. _DatetimeIndex: https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html?highlight=datetimeindex#pandas.DatetimeIndex

        """
        if not hasattr(self, "_timeIndex"):
            self._load_timeIndex()
        return self._timeIndex

    @output_open_handler
    def _load_timeIndex(self) -> None:
        """Load model reporting times into self._times"""
        self._timeIndex = DatetimeIndex(
            [
                self._start + timedelta(seconds=self._report) * step
                for step in range(1, self._period + 1)
            ]
        )

    ##### model element setters and getters #####
    def _subcatchmentIndex(
        self, subcatchment: str | int | Sequence[str | int] | None
    ) -> list[int] | int:
        """Get the swmm index for subcatchment.

        Parameters
        ----------
        subcatchment: Union[str, int, Sequence[Union[str, int]]]
            The name(s) of subcatchment(s).

        Returns
        -------
        Union[List[int], int]
           The SWMM index(s) of subcatchment(s).

        """

        if isinstance(subcatchment, (str, int, type(None))):
            return self._elementIndex(subcatchment, self.subcatchments, "subcatchment")

        elif subcatchment is not None:
            return [
                self._elementIndex(sub, self.subcatchments, "subcatchment")
                for sub in subcatchment
            ]
        else:
            raise TypeError("Invalid type for _subcatchmentIndex argument")

    @property
    def subcatchments(self) -> tuple[str, ...]:
        """Return a tuple of subcatchments available in SWMM output binary file.

        Returns
        -------
        Tuple[str]
            A tuple of model subcatchment names.

        """
        if not hasattr(self, "_subcatchments"):
            self._load_subcatchments()
        return self._subcatchments

    @output_open_handler
    def _load_subcatchments(self) -> None:
        """Load model size into self._project_size"""
        total = self.project_size[0]

        self._subcatchments = tuple(
            self._objectName(shared_enum.ElementType.SUBCATCH, index)
            for index in range(total)
        )

    def _nodeIndex(
        self, node: str | int | Sequence[str | int] | None
    ) -> list[int] | int:
        """Get the swmm index for node.

        Parameters
        ----------
        node: Union[str, int, Sequence[Union[str, int]]]
            The name(s) of node(s)

        Returns
        -------
        Union[List[int], int]
            The SWMM index(s) of node(s).

        """

        if isinstance(node, (str, int, type(None))):
            return self._elementIndex(node, self.nodes, "node")

        # elif here because mypy issues
        elif node is not None:
            return [self._elementIndex(nd, self.nodes, "node") for nd in node]

        else:
            raise TypeError("Invalid type for self._nodeIndex argument")

    @property
    def nodes(self) -> tuple[str, ...]:
        """Return a tuple of nodes available in SWMM binary output file.

        Returns
        -------
        Tuple[str]
            A tuple of model node names.

        """
        if not hasattr(self, "_nodes"):
            self._load_nodes()
        return self._nodes

    @output_open_handler
    def _load_nodes(self) -> None:
        """Load model nodes into self._nodes"""
        total = self.project_size[1]

        self._nodes = tuple(
            self._objectName(shared_enum.ElementType.NODE, index)
            for index in range(total)
        )

    def _linkIndex(
        self, link: str | int | Sequence[str | int] | None
    ) -> list[int] | int:
        """Get the swmm index for link.

        Parameters
        ----------
        link: Union[str, int, Sequence[Union[str, int]]]
            The name(s) of link(s)

        Returns
        -------
        Union[List[int], int]
            SWMM index(s) of link(s).

        """
        if isinstance(link, (str, int, type(None))):
            return self._elementIndex(link, self.links, "link")

        # elif here because mypy issues
        elif link is not None:
            return [self._elementIndex(lnk, self.links, "link") for lnk in link]

        else:
            raise TypeError("Invalid type for self._linkIndex argument")

    @property
    def links(self) -> tuple[str, ...]:
        """Return a tuple of links available in SWMM binary output file.

        Returns
        -------
        Tuple[str]
            A tuple of model link names.

        """
        if not hasattr(self, "_links"):
            self._load_links()
        return self._links

    @output_open_handler
    def _load_links(self) -> None:
        """Load model links into self._links"""
        total = self.project_size[2]

        self._links = tuple(
            self._objectName(shared_enum.ElementType.LINK, index)
            for index in range(total)
        )

    ####### series getters #######

    def _memory_series_getter(self, elemType: str) -> Callable:
        if elemType == "sys":

            def getter(  # type: ignore
                _handle, Attr: EnumMeta, startIndex: int, endIndex: int
            ) -> ndarray:
                # col = f"{type};{type};{Attr.value}"
                # return self.data[col][startIndex:endIndex]
                return self.data.loc[
                    startIndex : endIndex - 1,  # type: ignore
                    IndexSlice[elemType, elemType, Attr.value],  # type: ignore
                ].to_numpy()

        else:

            def getter(  # type: ignore
                _handle, elemIdx: int, Attr: EnumMeta, startIndex: int, endIndex: int
            ) -> ndarray:
                # col = f"{type};{elemIdx};{Attr.value}"
                # return self.data[col][startIndex:endIndex]
                return self.data.loc[
                    startIndex : endIndex - 1, IndexSlice[elemType, elemIdx, Attr.value]  # type: ignore
                ].to_numpy()

        return getter

    def _model_series(
        self,
        elementIndexArray: list[int],
        attributeIndexArray: list[EnumMeta],
        startIndex: int,
        endIndex: int,
        columns: str | None,
        getterFunc: Callable,
    ) -> ndarray:
        """
        Base series getter for any attribute. The function consilidates the logic
        necessary to build long or wide timeseries dataframes for each type of swmm
        model element.

        Parameters
        ----------
        elementIndexArray: List[int]
            Array of SWMM model element indicies
        attributeIndexArray: List[enum]
            Array of attribute Enums to pull for each element
        startIndex: int
            SWMM simulation time index to start timeseries
        endIndex: int
            SWMM simulation time index to end timeseries
        columns: Optional[str]
             Decide whether or not to break out elements or attributes as columns. May be one of:

             None: Return long-form data with one column for each data point

            'elem': Return data with a column for each element. If more than one attribute are given, attribute names are added to the index.

            'attr': Return data with a column for each attribute. If more than one element are given, element names are added to the index.

        getterFunc: Callable
            The swmm.toolkit series getter function. Should be one of:

             swmm.toolkit.output.get_subcatch_series
             swmm.toolkit.output.get_node_series
             swmm.toolkit.output.get_link_series

        Returns
        -------
        np.ndarray
            array of SWMM timeseries results

        Raises
        ------
        ValueError
            Value error if columns is not one of "elem", "attr", or None
        """

        if columns not in ("elem", "attr", None):
            raise ValueError(
                f"columns must be one of 'elem','attr', or None. {columns} was given"
            )

        if columns is None:
            return concatenate(
                [
                    concatenate(
                        [
                            getterFunc(
                                self._handle, elemIdx, Attr, startIndex, endIndex
                            )
                            for Attr in attributeIndexArray
                        ],
                        axis=0,
                    )
                    for elemIdx in elementIndexArray
                ],
                axis=0,
            )

        elif columns.lower() == "attr":
            return concatenate(
                [
                    stack(
                        [
                            getterFunc(
                                self._handle, elemIdx, Attr, startIndex, endIndex
                            )
                            for Attr in attributeIndexArray
                        ],
                        axis=1,
                    )
                    for elemIdx in elementIndexArray
                ],
                axis=0,
            )

        elif columns.lower() == "elem":
            return concatenate(
                [
                    stack(
                        [
                            getterFunc(
                                self._handle, elemIdx, Attr, startIndex, endIndex
                            )
                            for elemIdx in elementIndexArray
                        ],
                        axis=1,
                    )
                    for Attr in attributeIndexArray
                ],
                axis=0,
            )
        else:
            raise Exception("Columns must be None, 'attr', or 'elem'")

    def _model_series_index(
        self,
        elementArray: list[str],
        attributeArray: list[str],
        startIndex: int,
        endIndex: int,
        columns: str | None,
    ) -> tuple:
        """
        Base dataframe index getter for model timeseries. The function consilidates the logic
        necessary to build a data frame index for long or wide dataframes built with time series
        getters.

        Parameters
        ----------
        elementArray: List[str]
            Array of SWMM model element names
        attributeArray: List[str]
            Array of attribute names pulled for each element
        startIndex: int
            SWMM simulation time index to start timeseries
        endIndex: int
            SWMM simulation time index to end timeseries
        columns: Optional[str]
             Decide whether or not to break out elements or attributes as columns. May be one of:

             None: Return long-form data with one column for each data point

            'elem': Return data with a column for each element. If more than one attribute are given, attribute names are added to the index.

            'attr': Return data with a column for each attribute. If more than one element are given, element names are added to the index.

        Returns
        -------
        (pd.MultiIndex, Union[list,np.ndarray])
            A pandas MultiIndex for the row indicies and an iterable of column names

        Raises
        ------
        ValueError
            Value error if columns is not one of "elem", "attr", or None

        """

        if columns not in ("elem", "attr", None):
            raise ValueError(
                f"columns must be one of 'elem','attr', or None. {columns} was given"
            )

        if columns is None:
            dtIndex = tile(
                self.timeIndex[startIndex:endIndex],
                len(elementArray) * len(attributeArray),
            )
            indexArrays = [dtIndex]
            names = ["datetime"]
            cols = ["result"]
            if len(elementArray) > 1:
                indexArrays.append(
                    asarray(elementArray).repeat(
                        (endIndex - startIndex) * len(attributeArray)
                    )
                )
                names.append("element")
            if len(attributeArray) > 1:
                indexArrays.append(
                    tile(asarray(attributeArray), len(elementArray)).repeat(
                        endIndex - startIndex
                    )
                )
                names.append("attribute")

        elif columns.lower() == "attr":
            dtIndex = tile(self.timeIndex[startIndex:endIndex], len(elementArray))
            indexArrays = [dtIndex]
            names = ["datetime"]
            cols = attributeArray
            if len(elementArray) > 1:
                indexArrays.append(asarray(elementArray).repeat(endIndex - startIndex))
                names.append("element")

        elif columns.lower() == "elem":
            dtIndex = tile(self.timeIndex[startIndex:endIndex], len(attributeArray))
            indexArrays = [dtIndex]
            names = ["datetime"]
            cols = elementArray

            if len(attributeArray) > 1:
                indexArrays.append(
                    asarray(attributeArray).repeat(endIndex - startIndex)
                )
                names.append("attribute")
        index = (
            MultiIndex.from_arrays(
                indexArrays,
                names=names,
            )
            if len(indexArrays) > 1
            else Index(indexArrays[0], name=names[0])
        )

        return index, cols

    def subcatch_series(
        self,
        subcatchment: int | str | Sequence[int | str] | None,
        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None = (
            "rainfall",
            "runoff_rate",
            "gw_outflow_rate",
        ),
        start: str | int | datetime | None = None,
        end: str | int | datetime | None = None,
        columns: str | None = "attr",
        asframe: bool = True,
    ) -> DataFrame | ndarray:
        """Get one or more time series for one or more subcatchment attributes.
        Specify series start index and end index to get desired time range.

        Parameters
        ----------
        subcatchment: Union[int, str, Sequence[Union[int, str]], None]
            The subcatchment index or name.

        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None,
            The attribute index or name.

            On of:

            **rainfall, snow_depth, evap_loss, infil_loss, runoff_rate, gw_outflow_rate,
            gw_table_elev, soil_moisture**.


            Defaults to: `('rainfall', 'runoff_rate', 'gw_outflow_rate').`


            Can also input the integer index of the attribute you would like to
            pull or the actual enum from Output.subcatch_attributes.

            Setting to None indicates all attributes.

        start: Union[str,int, datetime, None], optional
            The start datetime or index of from which to return series, defaults to None.

            Setting to None indicates simulation start.

        end: Union[str,int, datetime, None], optional
            The end datetime or index of from which to return series, defaults to None.

            Setting to None indicates simulation end.

        columns: Optional[str], optional
            Decide whether or not to break out elements or attributes as columns. May be one of:

            None: Return long-form data with one column for each data point

            'elem': Return data with a column for each element. If more than one attribute are given, attribute names are added to the index.

            'attr': Return data with a column for each attribute. If more than one element are given, element names are added to the index.

            defaults to 'attr'.

        asframe: bool
            A switch to return an indexed DataFrame. Set to False to get an array of values only, defaults to True.

        Returns
        -------
        Union[pd.DataFrame,np.ndarray]
            A DataFrame or ndarray of attribute values in each column for requested
            date range and subcatchments.

        Examples
        ---------

        Pull single time series for a single subcatchment

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.subcatch_series('SUB1', 'runoff_rate')
                             runoff_rate
        datetime
        1900-01-01 00:05:00     0.000000
        1900-01-01 00:10:00     0.000000
        1900-01-01 00:15:00     0.000000
        1900-01-01 00:20:00     0.000000
        1900-01-01 00:25:00     0.000000
        ...                          ...
        1900-01-01 23:40:00     0.025057
        1900-01-01 23:45:00     0.025057
        1900-01-01 23:50:00     0.025057
        1900-01-01 23:55:00     0.025057
        1900-01-02 00:00:00     0.025057
        [288 rows x 1 columns]

        Pull a wide-form dataframe for all parameters for a catchment

        >>> out.subcatch_series('SUB1', out.subcatch_attributes)
                            rainfall  snow_depth  evap_loss  infil_loss  ...  soil_moisture  groundwater  pol_rainfall  sewage
        datetime                                                          ...
        1900-01-01 00:05:00   0.03000         0.0        0.0    0.020820  ...       0.276035          0.0           0.0     0.0
        1900-01-01 00:10:00   0.03000         0.0        0.0    0.020952  ...       0.276053          0.0           0.0     0.0
        1900-01-01 00:15:00   0.03000         0.0        0.0    0.021107  ...       0.276071          0.0           0.0     0.0
        1900-01-01 00:20:00   0.03000         0.0        0.0    0.021260  ...       0.276089          0.0           0.0     0.0
        1900-01-01 00:25:00   0.03000         0.0        0.0    0.021397  ...       0.276107          0.0           0.0     0.0
        ...                       ...         ...        ...         ...  ...            ...          ...           ...     ...
        1900-01-01 23:40:00   0.03224         0.0        0.0    0.027270  ...       0.280026          0.0         100.0     0.0
        1900-01-01 23:45:00   0.03224         0.0        0.0    0.027270  ...       0.280026          0.0         100.0     0.0
        1900-01-01 23:50:00   0.03224         0.0        0.0    0.027270  ...       0.280026          0.0         100.0     0.0
        1900-01-01 23:55:00   0.03224         0.0        0.0    0.027270  ...       0.280026          0.0         100.0     0.0
        1900-01-02 00:00:00   0.00000         0.0        0.0    0.027270  ...       0.280026          0.0         100.0     0.0
        [288 rows x 11 columns]

        Pull a long-form dataframe of all catchments and attributes

        >>> out.subcatch_series(out.subcatchments, out.subcatch_attributes, columns=None)
                                               result
        datetime            element attribute
        1900-01-01 00:05:00 SUB1    rainfall     0.03
        1900-01-01 00:10:00 SUB1    rainfall     0.03
        1900-01-01 00:15:00 SUB1    rainfall     0.03
        1900-01-01 00:20:00 SUB1    rainfall     0.03
        1900-01-01 00:25:00 SUB1    rainfall     0.03
        ...                                       ...
        1900-01-01 23:40:00 SUB3    sewage       0.00
        1900-01-01 23:45:00 SUB3    sewage       0.00
        1900-01-01 23:50:00 SUB3    sewage       0.00
        1900-01-01 23:55:00 SUB3    sewage       0.00
        1900-01-02 00:00:00 SUB3    sewage       0.00
        [9504 rows x 1 columns]

        Pull two parameters for one subcatchment and plot the results

        .. plot::

           import matplotlib.pyplot as plt
           from matplotlib.dates import DateFormatter
           from swmm.pandas import Output,example_out_path

           # read output file in Output object
           out = Output(example_out_path)

           # pull rainfall and runoff_rate timeseries and plot them
           ax = out.subcatch_series('SUB1', ['rainfall', 'runoff_rate']).plot(figsize=(8,4))
           plt.title("SUB1 Params")
           plt.tight_layout()
           plt.show()

        Pull the one parameter for all subcatchments

        .. plot::

           import matplotlib.pyplot as plt
           from matplotlib.dates import DateFormatter
           from swmm.pandas import Output,example_out_path

           # read output file in Output object
           out = Output(example_out_path)

           # pull runoff_rate timeseries for all cathments and plot them
           ax = out.subcatch_series(out.subcatchments, 'runoff_rate', columns='elem').plot(figsize=(8,4))
           plt.title("Runoff Rate")
           plt.tight_layout()
           plt.show()


        """
        subcatchementArray, subcatchmentIndexArray = self._validateElement(
            subcatchment, self.subcatchments
        )

        attributeArray, attributeIndexArray = self._validateAttribute(
            attribute, self.subcatch_attributes
        )

        startIndex = self._time2step(start, 0)[0]
        endIndex = self._time2step(end, self._period)[0]

        getter = (
            self._memory_series_getter("sub")
            if self._preload
            else output.get_subcatch_series
        )

        values = self._model_series(
            subcatchmentIndexArray,
            attributeIndexArray,
            startIndex,
            endIndex,
            columns,
            getter,
        )

        if not asframe:
            return values

        dfIndex, cols = self._model_series_index(
            subcatchementArray, attributeArray, startIndex, endIndex, columns
        )
        return DataFrame(values, index=dfIndex, columns=cols)

    @output_open_handler
    def node_series(
        self,
        node: int | str | Sequence[int | str] | None,
        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None = (
            "invert_depth",
            "flooding_losses",
            "total_inflow",
        ),
        start: str | int | datetime | None = None,
        end: str | int | datetime | None = None,
        columns: str | None = "attr",
        asframe: bool = True,
    ) -> DataFrame | ndarray:
        """Get one or more time series for one or more node attributes.
        Specify series start index and end index to get desired time range.

        Parameters
        ----------
        node: Union[int, str, Sequence[Union[int, str]], None]
            The node index or name.

        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None,
            The attribute index or name.

            On of:

            **invert_depth, hydraulic_head, ponded_volume, lateral_inflow,
            total_inflow, flooding_losses**.

            defaults to: `('invert_depth','flooding_losses','total_inflow')`

            Can also input the integer index of the attribute you would like to
            pull or the actual enum from Output.node_attributes.

            Setting to None indicates all attributes.

        start: Union[str, int, datetime, None], optional
            The start datetime or index of from which to return series, defaults to None.

            Setting to None indicates simulation start.

        end: Union[str, int, datetime, None], optional
            The end datetime or index of from which to return series, defaults to None.

            Setting to None indicates simulation end.

        columns: Optional[str], optional
            Decide whether or not to break out elements or attributes as columns. May be one of:

            None: Return long-form data with one column for each data point

            'elem': Return data with a column for each element. If more than one attribute are given, attribute names are added to the index.

            'attr': Return data with a column for each attribute. If more than one element are given, element names are added to the index.

            defaults to 'attr'.

        asframe: bool
            A switch to return an indexed DataFrame. Set to False to get an array of values only, defaults to True.

        Returns
        -------
        Union[pd.DataFrame,np.ndarray]
            A DataFrame or ndarray of attribute values in each column for requested
            date range and nodes.

        Examples
        ---------

        Pull single time series for a single node

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.node_series('JUNC2', 'invert_depth')
                             invert_depth
        datetime
        1900-01-01 00:05:00      0.334742
        1900-01-01 00:10:00      0.509440
        1900-01-01 00:15:00      0.562722
        1900-01-01 00:20:00      0.602668
        1900-01-01 00:25:00      0.631424
        ...                           ...
        1900-01-01 23:40:00      0.766949
        1900-01-01 23:45:00      0.766949
        1900-01-01 23:50:00      0.766949
        1900-01-01 23:55:00      0.766949
        1900-01-02 00:00:00      0.766949
        [288 rows x 1 columns]

        Pull a wide-form dataframe for all parameters for a node

        >>> out.node_series('JUNC2', out.node_attributes)
                           invert_depth  hydraulic_head  ponded_volume  lateral_inflow  total_inflow  flooding_losses  groundwater  pol_rainfall     sewage
        datetime
        1900-01-01 00:05:00      0.334742       -0.705258            0.0        0.185754      0.185785              0.0     3.935642      0.000000  95.884094
        1900-01-01 00:10:00      0.509440       -0.530560            0.0        0.196764      0.197044              0.0     8.902034      0.000000  90.335831
        1900-01-01 00:15:00      0.562722       -0.477278            0.0        0.198615      0.199436              0.0     9.038609      0.000000  89.253334
        1900-01-01 00:20:00      0.602668       -0.437332            0.0        0.200802      0.202462              0.0     9.259741      0.000000  87.919571
        1900-01-01 00:25:00      0.631424       -0.408576            0.0        0.203108      0.205802              0.0     9.523322      0.000000  86.492836
        ...                           ...             ...            ...             ...           ...              ...          ...           ...        ...
        1900-01-01 23:40:00      0.766949       -0.273052            0.0        0.314470      0.352183              0.0    15.293419     39.303375  45.430920
        1900-01-01 23:45:00      0.766949       -0.273052            0.0        0.314499      0.352183              0.0    15.313400     39.292118  45.430920
        1900-01-01 23:50:00      0.766949       -0.273052            0.0        0.314530      0.352183              0.0    15.333243     39.281300  45.430920
        1900-01-01 23:55:00      0.766949       -0.273052            0.0        0.314559      0.352183              0.0    15.352408     39.271194  45.430920
        1900-01-02 00:00:00      0.766949       -0.273052            0.0        0.314590      0.352183              0.0    15.371475     39.261478  45.430920
        [288 rows x 9 columns]

        Pull a long-form dataframe of all nodes and attributes

        >>> out.node_series('JUNC2', out.node_attributes, columns=None)
                                                    result
        datetime            element attribute
        1900-01-01 00:05:00 JUNC1   invert_depth   0.002143
        1900-01-01 00:10:00 JUNC1   invert_depth   0.010006
        1900-01-01 00:15:00 JUNC1   invert_depth   0.017985
        1900-01-01 00:20:00 JUNC1   invert_depth   0.025063
        1900-01-01 00:25:00 JUNC1   invert_depth   0.031329
        ...                                             ...
        1900-01-01 23:40:00 STOR1   sewage        51.502193
        1900-01-01 23:45:00 STOR1   sewage        51.164684
        1900-01-01 23:50:00 STOR1   sewage        50.905445
        1900-01-01 23:55:00 STOR1   sewage        50.715385
        1900-01-02 00:00:00 STOR1   sewage        50.574486
        [23328 rows x 1 columns]

        Pull flow timeseries and calculate the total flow volume for all nodes


        >>> from swmm.pandas.constants import gal_per_cf
        >>> df = out.node_series(out.nodes, ['lateral_inflow','total_inflow','flooding_losses'])
                                     lateral_inflow  total_inflow  flooding_losses
        datetime            element
        1900-01-01 00:05:00 JUNC1          0.002362      0.002362              0.0
        1900-01-01 00:10:00 JUNC1          0.005792      0.005792              0.0
        1900-01-01 00:15:00 JUNC1          0.006524      0.006524              0.0
        1900-01-01 00:20:00 JUNC1          0.007306      0.007306              0.0
        1900-01-01 00:25:00 JUNC1          0.008039      0.008039              0.0
        ...                                     ...           ...              ...
        1900-01-01 23:40:00 STOR1          0.000000      1.455056              0.0
        1900-01-01 23:45:00 STOR1          0.000000      1.455056              0.0
        1900-01-01 23:50:00 STOR1          0.000000      1.455056              0.0
        1900-01-01 23:55:00 STOR1          0.000000      1.455056              0.0
        1900-01-02 00:00:00 STOR1          0.000000      1.455056              0.0
        [2592 rows x 3 columns]
        #----------------------------------------------------------------------------
        # group by element name and sum,
        # then multiply by reporting timestep in seconds
        # then convert to millions of gallons
        >>> df.groupby('element').sum() * out.report * gal_per_cf / 1e6
                lateral_inflow  total_inflow  flooding_losses
        element
        JUNC1          0.101562      0.101898         0.000053
        JUNC2          0.544891      0.857012         0.000000
        JUNC3          0.000000      0.502078         0.080634
        JUNC4          1.813826      2.096243         0.317929
        JUNC5          0.000000      1.870291         0.073878
        JUNC6          0.000000      1.701455         0.000000
        OUT1           0.000000      1.698081         0.000000
        OUT2           0.000000      0.575617         0.000000
        STOR1          0.000000      1.862843         0.172482
        """
        nodeArray, nodeIndexArray = self._validateElement(node, self.nodes)

        attributeArray, attributeIndexArray = self._validateAttribute(
            attribute, self.node_attributes
        )

        startIndex = self._time2step(start, 0)[0]
        endIndex = self._time2step(end, self._period)[0]

        getter = (
            self._memory_series_getter("node")
            if self._preload
            else output.get_node_series
        )

        values = self._model_series(
            nodeIndexArray,
            attributeIndexArray,
            startIndex,
            endIndex,
            columns,
            getter,
        )

        if not asframe:
            return values

        dfIndex, cols = self._model_series_index(
            nodeArray, attributeArray, startIndex, endIndex, columns
        )

        return DataFrame(values, index=dfIndex, columns=cols)

    @output_open_handler
    def link_series(
        self,
        link: int | str | Sequence[int | str] | None,
        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None = (
            "flow_rate",
            "flow_velocity",
            "flow_depth",
        ),
        start: int | str | datetime | None = None,
        end: int | str | datetime | None = None,
        columns: str | None = "attr",
        asframe: bool = True,
    ) -> DataFrame | ndarray:
        """Get one or more time series for one or more link attributes.
        Specify series start index and end index to get desired time range.

        Parameters
        ----------
        link: Union[int, str, Sequence[Union[int, str]], None]
            The link index or name.

        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None
            The attribute index or name.

            On of:

            **flow_rate, flow_depth, flow_velocity, flow_volume, capacity**.

            defaults to: `('flow_rate','flow_velocity','flow_depth')`

            Can also input the integer index of the attribute you would like to
            pull or the actual enum from output.link_attributes.

            Setting to None indicates all attributes.

        start_index: Union[str,int, datetime, None], optional
            The start datetime or index of from which to return series, defaults to None.

            Setting to None indicates simulation start.

        end_index: Union[str,int, datetime, None], optional
            The end datetime or index of from which to return series, defaults to None.

            Setting to None indicates simulation end.

        columns: Optional[str], optional
            Decide whether or not to break out elements or attributes as columns. May be one of:

            None: Return long-form data with one column for each data point

            'elem': Return data with a column for each element. If more than one attribute are given, attribute names are added to the index.

            'attr': Return data with a column for each attribute. If more than one element are given, element names are added to the index.

            defaults to 'attr'.

        asframe: bool
            A switch to return an indexed DataFrame. Set to False to get an array of values only, defaults to True.

        Returns
        -------
        Union[pd.DataFrame,np.ndarray]
            A DataFrame or ndarray of attribute values in each column for requested
            date range and links.

        Examples
        ---------

        Pull flow rate for two conduits

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.link_series(['COND1','COND6'],out.link_attributes.FLOW_RATE,columns='elem')
                                COND1   COND6
        datetime
        1900-01-01 00:05:00  0.000031  0.0000
        1900-01-01 00:10:00  0.000280  0.0000
        1900-01-01 00:15:00  0.000820  0.0000
        1900-01-01 00:20:00  0.001660  0.0000
        1900-01-01 00:25:00  0.002694  0.0000
        ...                       ...     ...
        1900-01-01 23:40:00  0.037800  1.5028
        1900-01-01 23:45:00  0.037800  1.5028
        1900-01-01 23:50:00  0.037800  1.5028
        1900-01-01 23:55:00  0.037800  1.5028
        1900-01-02 00:00:00  0.037800  1.5028
        [288 rows x 2 columns]

        Pull a wide-form dataframe for all parameters for a link

        >>> out.node_series('COND1', out.link_attributes)
                             flow_rate  flow_depth  ...  pol_rainfall        sewage
        datetime                                    ...
        1900-01-01 00:05:00   0.000031    0.053857  ...      0.000000  0.000000e+00
        1900-01-01 00:10:00   0.000280    0.134876  ...      0.000000  0.000000e+00
        1900-01-01 00:15:00   0.000820    0.165356  ...      0.000000  0.000000e+00
        1900-01-01 00:20:00   0.001660    0.188868  ...      0.000000  0.000000e+00
        1900-01-01 00:25:00   0.002694    0.206378  ...      0.000000  0.000000e+00
        ...                        ...         ...  ...           ...           ...
        1900-01-01 23:40:00   0.037800    0.312581  ...     68.344780  6.173063e-08
        1900-01-01 23:45:00   0.037800    0.312581  ...     68.242958  5.872794e-08
        1900-01-01 23:50:00   0.037800    0.312581  ...     68.144737  5.583060e-08
        1900-01-01 23:55:00   0.037800    0.312581  ...     68.052620  5.311425e-08
        1900-01-02 00:00:00   0.037800    0.312581  ...     67.963829  5.049533e-08

        [288 rows x 8 columns]

        Pull a long-form dataframe of all links and attributes

        >>> out.link_series(out.links, out.link_attributes, columns=None)
                                                  result
        datetime            element attribute
        1900-01-01 00:05:00 COND1   flow_rate   0.000031
        1900-01-01 00:10:00 COND1   flow_rate   0.000280
        1900-01-01 00:15:00 COND1   flow_rate   0.000820
        1900-01-01 00:20:00 COND1   flow_rate   0.001660
        1900-01-01 00:25:00 COND1   flow_rate   0.002694
        ...                                          ...
        1900-01-01 23:40:00 WR1     sewage     45.430920
        1900-01-01 23:45:00 WR1     sewage     45.430920
        1900-01-01 23:50:00 WR1     sewage     45.430920
        1900-01-01 23:55:00 WR1     sewage     45.430920
        1900-01-02 00:00:00 WR1     sewage     45.430920

        [18432 rows x 1 columns]

        Pull flow timeseries and pollutant tracer concentrations for a link and plot

        .. plot::

            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from swmm.pandas import Output,example_out_path

            out = Output(example_out_path)
            df = out.link_series('COND6',['flow_rate','groundwater','pol_rainfall','sewage'])

            # set up figure
            fig,ax = plt.subplots(figsize=(8,4))

            # plot flow rate on primary yaxis
            ax.plot(df.flow_rate,label="flow rate")

            # plot pollutant concentrations on secondary axis
            # rainfall, DWF, and groundwater were given 100 mg/L pollutant
            # concentrations to serve as tracers
            ax1 = ax.twinx()
            ax1.plot(df.groundwater,ls = '--',label="groundwater tracer")
            ax1.plot(df.pol_rainfall,ls = '--',label="rainfall tracer")
            ax1.plot(df.sewage,ls = '--',label="sewage tracer")

            # style axes
            ax.set_ylabel("Flow Rate (cfs)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax1.set_ylabel("Percent")

            # add legend and show figure
            fig.legend(bbox_to_anchor=(1,1),bbox_transform=ax.transAxes)
            fig.tight_layout()

            fig.show()

        """
        linkArray, linkIndexArray = self._validateElement(link, self.links)

        attributeArray, attributeIndexArray = self._validateAttribute(
            attribute, self.link_attributes
        )

        startIndex = self._time2step(start, 0)[0]
        endIndex = self._time2step(end, self._period)[0]

        getter = (
            self._memory_series_getter("link")
            if self._preload
            else output.get_link_series
        )

        values = self._model_series(
            linkIndexArray,
            attributeIndexArray,
            startIndex,
            endIndex,
            columns,
            getter,
        )

        if not asframe:
            return values

        dfIndex, cols = self._model_series_index(
            linkArray, attributeArray, startIndex, endIndex, columns
        )

        return DataFrame(values, index=dfIndex, columns=cols)

    @output_open_handler
    def system_series(
        self,
        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None = None,
        start: str | int | datetime | None = None,
        end: str | int | datetime | None = None,
        asframe: bool = True,
    ) -> DataFrame | ndarray:
        """Get one or more a time series for one or more system attributes.
        Specify series start index and end index to get desired time range.

        Parameters
        ----------
        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None
            The attribute index or name.

            On of:

            **air_temp, rainfall, snow_depth, evap_infil_loss, runoff_flow,
            dry_weather_inflow, gw_inflow, rdii_inflow, direct_inflow, total_lateral_inflow,
            flood_losses, outfall_flows, volume_stored, evap_rate**.

            defaults to `None`.

            Can also input the integer index of the attribute you would like to
            pull or the actual enum from Output.system_attributes.

            Setting to None indicates all attributes.

        start_index: Union[str, int, datetime, None], optional
            The start datetime or index of from which to return series, defaults to None.

            Setting to None indicates simulation start.

        end_index: Union[str, int, datetime, None], optional
            The end datetime or index of from which to return series, defaults to None.

            Setting to None indicates simulation end.

        asframe: bool
            switch to return an indexed DataFrame. Set to False to get an array of values only, defaults to True

        Returns
        -------
        Union[pd.DataFrame,np.ndarray]
            DataFrame or ndarray of attribute values in each column for request date range

        Examples
        ---------

        Pull two system attribute time series

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.system_series(['total_lateral_inflow','rainfall'])
                             total_lateral_inflow  rainfall
        datetime
        1900-01-01 00:05:00              0.902807   0.03000
        1900-01-01 00:10:00              0.902800   0.03000
        1900-01-01 00:15:00              0.902793   0.03000
        1900-01-01 00:20:00              0.902786   0.03000
        1900-01-01 00:25:00              0.902779   0.03000
        ...                                   ...       ...
        1900-01-01 23:40:00              1.431874   0.03224
        1900-01-01 23:45:00              1.431869   0.03224
        1900-01-01 23:50:00              1.431876   0.03224
        1900-01-01 23:55:00              1.431894   0.03224
        1900-01-02 00:00:00              1.431921   0.00000
        [288 rows x 2 columns]

        """

        attributeArray, attributeIndexArray = self._validateAttribute(
            attribute, self.system_attributes
        )

        startIndex = self._time2step(start, 0)[0]
        endIndex = self._time2step(end, self._period)[0]

        getter = (
            self._memory_series_getter("sys")
            if self._preload
            else output.get_system_series
        )

        values = stack(
            [
                getter(self._handle, sysAttr, startIndex, endIndex)
                for sysAttr in attributeIndexArray
            ],
            axis=1,
        )

        if not asframe:
            return values

        dfIndex = Index(self.timeIndex[startIndex:endIndex], name="datetime")
        return DataFrame(values, index=dfIndex, columns=attributeArray)

    ####### attribute getters #######

    @output_open_handler
    def subcatch_attribute(
        self,
        time: str | int | datetime,
        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None = (
            "rainfall",
            "runoff_rate",
            "gw_outflow_rate",
        ),
        asframe: bool = True,
    ) -> DataFrame | ndarray:
        """For all subcatchments at a given time, get a one or more attributes.

        Parameters
        ----------
        time: Union[str, int, datetime]
            The datetime or simulation index for which to pull data, defaults to None.

        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None,
            The attribute index or name.

            On of:

            **rainfall, snow_depth, evap_loss, infil_loss, runoff_rate, gw_outflow_rate,
            gw_table_elev, soil_moisture**.

            Defaults to: `('rainfall','runoff_rate','gw_outflow_rate').`

            You can also input the integer index of the attribute you would like to
            pull or the actual enum from Output.subcatch_attributes.

            Setting to None indicates all attributes.

        asframe: bool
            A switch to return an indexed DataFrame. Set to False to get an array of values only, defaults to True.

        Returns
        -------
        Union[pd.DataFrame, np.ndarray]
            A DataFrame or ndarray of attribute values in each column for requested simulation time.

        Examples
        ---------
        Pull rainfall for all catchments at start of simulation

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.subcatch_attribute(0,'rainfall')
                          rainfall
            subcatchment
            SUB1              0.03
            SUB2              0.03
            SUB3              0.03
        """

        attributeArray, attributeIndexArray = self._validateAttribute(
            attribute, self.subcatch_attributes
        )

        timeIndex = self._time2step([time])[0]

        values = stack(
            [
                output.get_subcatch_attribute(self._handle, timeIndex, scAttr)
                for scAttr in attributeIndexArray
            ],
            axis=1,
        )

        if not asframe:
            return values

        dfIndex = Index(self.subcatchments, name="subcatchment")

        return DataFrame(values, index=dfIndex, columns=attributeArray)

    @output_open_handler
    def node_attribute(
        self,
        time: str | int | datetime,
        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None = (
            "invert_depth",
            "flooding_losses",
            "total_inflow",
        ),
        asframe: bool = True,
    ) -> DataFrame | ndarray:
        """For all nodes at a given time, get one or more attributes.

        Parameters
        ----------
        time: Union[str, int, datetime]
            The datetime or simulation index for which to pull data, defaults to None

        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None,
            The attribute index or name.

            On of:

            **invert_depth, hydraulic_head, ponded_volume, lateral_inflow,
            total_inflow, flooding_losses**.

            defaults to: `('invert_depth','flooding_losses','total_inflow')`

            Can also input the integer index of the attribute you would like to
            pull or the actual enum from Output.node_attributes.

            Setting to None indicates all attributes.

        asframe: bool
            A switch to return an indexed DataFrame. Set to False to get an array of values only, defaults to True.

        Returns
        -------
        Union[pd.DataFrame, np.ndarray]
            A DataFrame or ndarray of attribute values in each column for requested simulation time.

        Examples
        ---------
        Pull all attributes from middle of simulation

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.node_attribute(out.period/2)
                   invert_depth  hydraulic_head  ponded_volume  ...  groundwater  pol_rainfall    sewage
            node                                                ...
            JUNC1      8.677408       10.177408       0.000000  ...     0.260937     99.739067  0.000000
            JUNC2      4.286304        3.246305       0.000000  ...     0.366218     96.767433  2.475719
            JUNC3     11.506939        8.036940      35.862713  ...     0.615687     94.522049  4.862284
            JUNC4     14.936149        9.686150    6107.279785  ...     0.381425     96.532028  3.086555
            JUNC5     11.190232        4.690233       0.000000  ...     0.443388     95.959351  3.597255
            JUNC6      1.650765        1.650765       0.000000  ...     0.963940     91.113075  7.922997
            OUT1       0.946313        1.046313       0.000000  ...     0.969624     91.060143  7.970241
            OUT2       0.000000       -1.040001       0.000000  ...     0.367271     96.756134  2.479369
            STOR1     18.282972        3.032968    7550.865723  ...     0.961457     91.136200  7.902364
            [9 rows x 9 columns]
        """
        attributeArray, attributeIndexArray = self._validateAttribute(
            attribute, self.node_attributes
        )

        timeIndex = self._time2step([time])[0]

        values = stack(
            [
                output.get_node_attribute(self._handle, timeIndex, ndAttr)
                for ndAttr in attributeIndexArray
            ],
            axis=1,
        )

        if not asframe:
            return values

        dfIndex = Index(self.nodes, name="node")

        return DataFrame(values, index=dfIndex, columns=attributeArray)

    @output_open_handler
    def link_attribute(
        self,
        time: str | int | datetime,
        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None = (
            "flow_rate",
            "flow_velocity",
            "flow_depth",
        ),
        asframe: bool = True,
    ) -> DataFrame | ndarray:
        """For all links at a given time, get one or more attributes.

        Parameters
        ----------
        time: Union[str, int, datetime]
            The datetime or simulation index for which to pull data, defaults to None.

        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None
            The attribute index or name.

            On of:

            flow_rate, flow_depth, flow_velocity, flow_volume, capacity,

            defaults to `('flow_rate','flow_velocity','flow_depth')`

            Can also input the integer index of the attribute you would like to
            pull or the actual enum from Output.link_attributes.

            Setting to None indicates all attributes.

        asframe: bool
            A switch to return an indexed DataFrame. Set to False to get an array of values only, defaults to True.

        Returns
        -------
        pd.DataFrame
            A DataFrame of attribute values in each column for requested simulation time.

        Examples
        ---------
        Pull depth. flooding, and total inflow attributes from end of simulation

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.link_attribute(out.period/2)
                   invert_depth  flooding_losses  total_inflow
            node
            JUNC1      8.677408         0.000000      2.665294
            JUNC2      4.286304         0.000000     14.571551
            JUNC3     11.506939         0.341040      2.319820
            JUNC4     14.936149        16.137648     27.521870
            JUNC5     11.190232         0.000000      9.051201
            JUNC6      1.650765         0.000000      5.799996
            OUT1       0.946313         0.000000      5.799996
            OUT2       0.000000         0.000000     14.574173
            STOR1     18.282972         0.000000      9.048394
        """
        attributeArray, attributeIndexArray = self._validateAttribute(
            attribute, self.link_attributes
        )

        timeIndex = self._time2step([time])[0]

        values = stack(
            [
                output.get_link_attribute(self._handle, timeIndex, lnkAttr)
                for lnkAttr in attributeIndexArray
            ],
            axis=1,
        )

        if not asframe:
            return values

        dfIndex = Index(self.links, name="link")

        return DataFrame(values, index=dfIndex, columns=attributeArray)

    @output_open_handler
    def system_attribute(
        self,
        time: str | int | datetime,
        attribute: int | str | EnumMeta | Sequence[int | str | EnumMeta] | None = None,
        asframe=True,
    ) -> DataFrame | ndarray:
        """For all nodes at given time, get a one or more attributes.

        Parameters
        ----------
        time: Union[str, int, datetime]
            The datetime or simulation index for which to pull data, defaults to None.

        attribute: Union[int, str, Sequence[Union[int, str]], None]
            The attribute index or name.

            On of:

            **air_temp, rainfall, snow_depth, evap_infil_loss, runoff_flow,
            dry_weather_inflow, gw_inflow, rdii_inflow, direct_inflow, total_lateral_inflow,
            flood_losses, outfall_flows, volume_stored, evap_rate**.

            defaults to `None`.

            Can also input the integer index of the attribute you would like to
            pull or the actual enum from Output.system_attributes.

            Setting to None indicates all attributes.

        asframe: bool
            A switch to return an indexed DataFrame. Set to False to get an array of values only, defaults to True.

        Returns
        -------
        Union[pd.DataFrame,np.ndarray]
            A DataFrame of attribute values in each column for requested simulation time.

        Examples
        ---------

        Pull all system attributes for the 10th time step

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.system_attribute(10)
                                       result
            attribute
            air_temp                70.000000
            rainfall                 0.030000
            snow_depth               0.000000
            evap_infil_loss          0.015042
            runoff_flow              0.066304
            dry_weather_inflow       0.801000
            gw_inflow                0.101737
            rdii_inflow              0.000000
            direct_inflow            0.000000
            total_lateral_inflow     0.969041
            flood_losses             0.000000
            outfall_flows            0.944981
            volume_stored         1731.835938
            evap_rate                0.000000
            ptnl_evap_rate           0.000000
        """

        attributeArray, attributeIndexArray = self._validateAttribute(
            attribute, self.system_attributes
        )

        timeIndex = self._time2step([time])[0]

        values = asarray(
            [
                output.get_system_attribute(self._handle, timeIndex, sysAttr)
                for sysAttr in attributeIndexArray
            ]
        )

        if not asframe:
            return values

        dfIndex = Index(attributeArray, name="attribute")

        return DataFrame(values, index=dfIndex, columns=["result"])

    ####### result getters #######

    @output_open_handler
    def subcatch_result(
        self,
        subcatchment: int | str | Sequence[int | str] | None,
        time: int | str | Sequence[int | str] | None,
        asframe: bool = True,
    ) -> DataFrame | ndarray:
        """For a subcatchment at one or more given times, get all attributes.

        Only one of `subcatchment` or `time` can be multiple (eg. a list), not both.

        Parameters
        ----------
        subcatchment: Union[int, str, Sequence[Union[int, str]], None],
            The subcatchment(s) name(s) or index(s).

        time: Union[int, str, Sequence[Union[int, str]], None],
            THe datetime(s) or simulation index(s).

        asframe: bool
            A switch to return an indexed DataFrame. Set to False to get an array of values only, defaults to True.

        Returns
        -------
        Union[pd.DataFrame,np.ndarray]
            A DataFrame or ndarray of all attribute values subcatchment(s) at given time(s).

        Examples
        ---------

        Pull all attributes at start, middle, and end time steps for a single catchment

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.subcatch_result("SUB1",[0,out.period/2,out.period-1])
                                 rainfall  snow_depth  evap_loss  infil_loss  ...  soil_moisture  groundwater  pol_rainfall  sewage
            datetime                                                          ...
            1900-01-01 00:05:00     0.030         0.0        0.0    0.020820  ...       0.276035          0.0           0.0     0.0
            1900-01-01 12:05:00     1.212         0.0        0.0    0.594862  ...       0.281631          0.0         100.0     0.0
            1900-01-02 00:00:00     0.000         0.0        0.0    0.027270  ...       0.280026          0.0         100.0     0.0
            [3 rows x 11 columns]

        Pull all attributes for all catchments at the start of the simulation

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.subcatch_result(out.subcatchments,'1900-01-01')
                          rainfall  snow_depth  evap_loss  infil_loss  ...  soil_moisture  groundwater  pol_rainfall  sewage
            subcatchment                                               ...
            SUB1              0.03         0.0        0.0    0.020820  ...       0.276035          0.0           0.0     0.0
            SUB2              0.03         0.0        0.0    0.017824  ...       0.275048          0.0           0.0     0.0
            SUB3              0.03         0.0        0.0    0.011365  ...       0.279013          0.0           0.0     0.0
            [3 rows x 11 columns]
        """

        if isinstance(subcatchment, arrayish) and isinstance(time, arrayish):
            raise Exception("Can only have multiple of one of subcatchment and time")

        elif isinstance(subcatchment, arrayish):
            label = "subcatchment"
            labels, indices = self._validateElement(subcatchment, self.subcatchments)
            timeIndex = self._time2step([time])[0]  # type: ignore

            values = vstack(
                [
                    output.get_subcatch_result(self._handle, timeIndex, idx)
                    for idx in indices
                ]
            )

        else:
            label = "datetime"
            times = self.timeIndex if time is None else atleast_1d(time)
            indices = self._time2step(times)

            # since the timeIndex matches on nearst, we rebuild
            # the label in case it wasn't exact
            labels = self.timeIndex[indices]
            subcatchmentIndex = self._subcatchmentIndex(subcatchment)

            values = atleast_2d(
                vstack(
                    [
                        output.get_subcatch_result(self._handle, idx, subcatchmentIndex)
                        for idx in indices
                    ]
                )
            )

        if not asframe:
            return values

        dfIndex = Index(labels, name=label)

        return DataFrame(
            values, index=dfIndex, columns=_enum_keys(self.subcatch_attributes)
        )

    @output_open_handler
    def node_result(
        self,
        node: int | str | Sequence[int | str] | None,
        time: int | str | Sequence[int | str] | None,
        asframe: bool = True,
    ) -> DataFrame | ndarray:
        """For one or more nodes at one or more given times, get all attributes.

        Only one of `node` or `time` can be multiple (eg. a list), not both.

        Parameters
        ----------
        node: Union[int, str, Sequence[Union[int, str]], None],
            The node(s) name(s) or index(s).

        time: Union[int, str, Sequence[Union[int, str]], None],
            The datetime(s) or simulation index(s).

        asframe: bool
            A switch to return an indexed DataFrame. Set to False to get an array of values only, defaults to True.

        Returns
        -------
        Union[pd.DataFrame,np.ndarray]
            A DataFrame or ndarray of all attribute values nodes(s) at given time(s).

        Examples
        ---------

        Pull all attributes at start, middle, and end time steps for a single node

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.node_result("JUNC1",[0,out.period/2,out.period-1])
                                 invert_depth  hydraulic_head  ponded_volume  lateral_inflow  ...  flooding_losses  groundwater  pol_rainfall  sewage
            datetime                                                                          ...
            1900-01-01 00:05:00      0.002143        1.502143            0.0        0.002362  ...              0.0    84.334671      0.000000     0.0
            1900-01-01 12:05:00      8.677408       10.177408            0.0        2.665294  ...              0.0     0.260937     99.739067     0.0
            1900-01-02 00:00:00      0.108214        1.608214            0.0        0.037889  ...              0.0    33.929119     66.251686     0.0
            [3 rows x 9 columns]

        Pull all attributes for all nodes at the start of the simulation

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.node_result(out.nodes,'1900-01-01')
                   invert_depth  hydraulic_head  ponded_volume  lateral_inflow  total_inflow  flooding_losses  groundwater  pol_rainfall     sewage
            node
            JUNC1      0.002143        1.502143            0.0        0.002362      0.002362              0.0    84.334671           0.0   0.000000
            JUNC2      0.334742       -0.705258            0.0        0.185754      0.185785              0.0     3.935642           0.0  95.884094
            JUNC3      0.000000       -3.470001            0.0        0.000000      0.000000              0.0     0.000000           0.0   0.000000
            JUNC4      0.530241       -4.719759            0.0        0.657521      0.657521              0.0     5.066027           0.0  94.864769
            JUNC5      0.090128       -6.409873            0.0        0.000000      0.027627              0.0     2.723724           0.0  82.198524
            JUNC6      0.000000        0.000000            0.0        0.000000      0.000000              0.0     0.000000           0.0   0.000000
            OUT1       0.000000        0.100000            0.0        0.000000      0.000000              0.0     0.000000           0.0   0.000000
            OUT2       0.000000       -1.040000            0.0        0.000000      0.000000              0.0     0.000000           0.0   0.000000
            STOR1      0.000000      -15.250000            0.0        0.000000      0.000000              0.0     0.000000           0.0   0.000000
        """
        if isinstance(node, arrayish) and isinstance(time, arrayish):
            raise Exception("Can only have multiple of one of node and time")

        elif isinstance(node, arrayish):
            label = "node"
            labels, indices = self._validateElement(node, self.nodes)
            timeIndex = self._time2step([time])[0]
            values = vstack(
                [
                    output.get_node_result(self._handle, timeIndex, idx)
                    for idx in indices
                ]
            )

        else:
            label = "datetime"
            times = self.timeIndex if time is None else atleast_1d(time)
            indices = self._time2step(times)

            # since the timeIndex matches on nearst, we rebuild
            # the label in case it wasn't exact
            labels = self.timeIndex[indices]
            nodeIndex = self._nodeIndex(node)

            values = atleast_2d(
                vstack(
                    [
                        output.get_node_result(self._handle, idx, nodeIndex)
                        for idx in indices
                    ]
                )
            )

        if not asframe:
            return values

        dfIndex = Index(labels, name=label)

        return DataFrame(
            values, index=dfIndex, columns=_enum_keys(self.node_attributes)
        )

    @output_open_handler
    def link_result(
        self,
        link: int | str | Sequence[int | str] | None,
        time: int | str | Sequence[int | str] | None,
        asframe: bool = True,
    ) -> DataFrame | ndarray:
        """For a link at one or more given times, get all attributes.

        Only one of link or time can be multiple.

        Parameters
        ----------
        link: Union[int, str, Sequence[Union[int, str]], None],
            The link(s) name(s) or index(s).

        time: Union[int, str, Sequence[Union[int, str]], None],
            The datetime(s) or simulation index(s).

        asframe: bool
            A switch to return an indexed DataFrame. Set to False to get an array of values only, defaults to True.

        Returns
        -------
        Union[pd.DataFrame,np.ndarray]
            A DataFrame or ndarray of all attribute values link(s) at given time(s).

        Examples
        ---------

        Pull all attributes at start, middle, and end time steps for a single link

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.link_result("COND1",[0,out.period/2,out.period-1])
                                 flow_rate  flow_depth  flow_velocity  flow_volume  capacity  groundwater  pol_rainfall        sewage
            datetime
            1900-01-01 00:05:00   0.000031    0.053857       0.001116    23.910770  0.024351    79.488449      0.000000  0.000000e+00
            1900-01-01 12:05:00   2.665548    1.000000       3.393882   732.276428  1.000000     0.491514     99.142815  2.742904e-01
            1900-01-02 00:00:00   0.037800    0.312581       0.180144   212.443344  0.267168    32.083355     67.963829  5.049533e-08

        Pull all attributes for all links at the start of the simulation

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.link_result(out.links,'1900-01-01')
                   flow_rate  flow_depth  flow_velocity  flow_volume  capacity  groundwater  pol_rainfall     sewage
            link
            COND1   0.000031    0.053857       0.001116    23.910770  0.024351    79.488449           0.0   0.000000
            COND2   0.000000    0.000100       0.000000     0.074102  0.000161     0.000000           0.0   0.000000
            COND3   0.000000    0.000100       0.000000     0.076337  0.000113     0.000000           0.0   0.000000
            COND4   0.027627    0.038128       0.304938    49.596237  0.026561     3.034879           0.0  86.882553
            COND5   0.000000    0.000100       0.000000     0.012962  0.000127     0.000000           0.0   0.000000
            COND6   0.000000    0.000100       0.000000     0.000404  0.000014     0.000000           0.0   0.000000
            PUMP1   0.000000    0.000000       0.000000     0.000000  0.000000     0.000000           0.0   0.000000
            WR1     0.000000    0.000000       0.000000     0.000000  1.000000     3.935642           0.0  95.884094
        """
        if isinstance(link, arrayish) and isinstance(time, arrayish):
            raise Exception("Can only have multiple of one of link and time")

        elif isinstance(link, arrayish):
            label = "link"
            labels, indices = self._validateElement(link, self.links)
            timeIndex = self._time2step([time])[0]

            values = vstack(
                [
                    output.get_link_result(self._handle, timeIndex, idx)
                    for idx in indices
                ]
            )

        else:
            label = "datetime"
            times = self.timeIndex if time is None else atleast_1d(time)
            indices = self._time2step(times)

            # since the timeIndex matches on nearst, we rebuild
            # the label in case it wasn't exact
            labels = self.timeIndex[indices]

            linkIndex = self._linkIndex(link)
            values = atleast_2d(
                vstack(
                    [
                        output.get_link_result(self._handle, idx, linkIndex)
                        for idx in indices
                    ]
                )
            )

        if not asframe:
            return values

        dfIndex = Index(labels, name=label)

        return DataFrame(
            values, index=dfIndex, columns=_enum_keys(self.link_attributes)
        )

    @output_open_handler
    def system_result(
        self,
        time: str | int | datetime,
        asframe=True,
    ) -> DataFrame | ndarray:
        """For a given time, get all system attributes.

        Parameters
        ----------
        time: Union[str, int, datetime]
            The datetime or simulation index.

        asframe: bool
            A switch to return an indexed DataFrame. Set to False to get an array of values only, defaults to True.

        Returns
        -------
        Union[pd.DataFrame,np.ndarray]
            A DataFrame of attribute values in each row for requested simulation time.

        Examples
        ---------

        Pull all attributes at start of simulation

        >>> from swmm.pandas import Output,example_out_path
        >>> out = Output(example_out_path)
        >>> out.system_result('1900-01-01')
                                    result
            attribute
            air_temp               70.000000
            rainfall                0.030000
            snow_depth              0.000000
            evap_infil_loss         0.013983
            runoff_flow             0.000000
            dry_weather_inflow      0.801000
            gw_inflow               0.101807
            rdii_inflow             0.000000
            direct_inflow           0.000000
            total_lateral_inflow    0.902807
            flood_losses            0.000000
            outfall_flows           0.000000
            volume_stored         168.436996
            evap_rate               0.000000
            ptnl_evap_rate          0.000000
        """

        timeIndex = self._time2step([time])[0]

        values = asarray(output.get_system_result(self._handle, timeIndex, 0))

        if not asframe:
            return values

        dfIndex = Index(_enum_keys(self.system_attributes), name="attribute")

        return DataFrame(values, index=dfIndex, columns=["result"])

    def getStructure(self, link, node):
        """
        Return a structure object for a given list of links and nodes.

        Parameters
        ----------
        link: Union[str, Sequence[str]]
            The list of links that belong to the structure.
        node: Union[str, Sequence[str]]
            The list of nodes that below to the structure.

        Returns
        -------
        Structure
            Structure comprised of the given links and nodes.
        """
        return Structure(self, link, node)

    # close outfile when object deleted
    # this doesn't always get called on sys.exit()
    # better to use output object with context
    # manager to ensure _open() and _close() are always closed
    # in some cases, you can get a memory leak message from swig:
    # >>> exit()
    # swig/python detected a memory leak of type 'struct Handle *', no destructor found.
    def __del__(self) -> None:
        """
         Destructor for outfile handle

        :return: Nothing
        :rtype: None
        """
        self._close()

    # method used for context manager with statement
    def __enter__(self):
        self._open()
        return self

    # method used for context manager with statement
    def __exit__(self, *arg) -> None:
        self._close()

    def open(self):
        "open the output file"
        self._open()

    def close(self):
        "close the output file"
        self._close()
