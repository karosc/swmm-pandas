from __future__ import annotations
import re
from io import StringIO

from pandas.core.api import DataFrame, Timestamp, to_datetime, to_timedelta, Series
from pandas.io.parsers import read_csv, read_fwf


class Report:
    _rptfile: str
    """path to swmm rpt file"""

    _rpt_text: str
    """text string of rpt file contents"""

    _sections: dict[str, str]
    """dictionary of SWMM report sections as {section name: section text}"""

    def __init__(self, rptfile: str):
        """Base class for a SWMM simulation report file.

        The report object provides an api for the tables in the the SWMM
        simulation report file. Tables are access as properties of the object
        and returned as pandas DataFrames.

        Parameters
        ----------
        rptfile: str
            model report file path
        """

        self._rptfile = rptfile

        with open(rptfile) as file:
            self._rpt_text = file.read()

        self._sections = {
            self._find_title(section): section
            for section in self._find_sections(self._rpt_text)
        }

    @staticmethod
    def _find_sections(rpt_text: str) -> list[str]:
        r"""
        Function to split the report file text into separate sections using a regex
        pattern match:

        "^\s+$\s+(?=\*|A)": pattern matches blank lines followed by at least
        1 white space followed by a lookhead for a asterisk (demarks section headers)
        or the letter A (looks for the word Analysis at the end of the report file)


        Parameters
        ----------
        rpt_text: str
            Text content of the report file
        Returns
        -------
        List[str]
            A list section texts
        """
        # pattern to match blank lines preceding a line of asterisks
        section_pattern = R"^\s+$\s+(?=\*|A)"
        section_comp = re.compile(section_pattern, re.MULTILINE)
        return list(
            map(lambda x: x.replace("\n  ", "\n"), section_comp.split(rpt_text)[2:-1])
        )

    @staticmethod
    def _find_title(section: str) -> str:
        r"""
        Function to extract the title of section produced by _find_sections using
        regex to match lines between two lines of asterisks.

        "^\*+[\s\S]*?\n([\s\S]*?)\s*\*+": Pattern matches any number white space or non-white
        space characters that are between:
            1. A line starting with a string of asterisks followed by any white space or
               non-whitespace chacter and ending with a new line break
            2. A line starting with a string of asterisks


        Parameters
        ----------
        section: str
            The section text produced by _find_sections

        Returns
        -------
        str
            Title of section

        Raises
        ------
        Exception
            If regex could not find a match
        """
        # pattern to match line between two lines of asterisks
        title_pattern = R"^\*+[\s\S]*?\n([\s\S]*?)\s*\*+"
        title_comp = re.compile(title_pattern, re.MULTILINE)
        s = title_comp.match(section)
        if s:
            # if string is found, split line on more two consecutive spaces and pull the first token
            return s.group(1).split("  ")[0]
        else:
            raise Exception(f"Error finding title for section\n{section}")

    @staticmethod
    def _split_section(section: str) -> tuple[str, str]:
        """
        Function to split a report section into header and data elements. Relies on regex
        matching lines with consecutive dashes indicating header lines.

        Parameters
        ----------
        section: str
            The section text produced by _find_sections

        Returns
        -------
        Tuple[str, str]
            header text and data text

        Raises
        ------
        Exception
            If regex could not find a match
        """
        title = Report._find_title(section)
        subsections = re.split(R"\s*-+\n", section)
        num_subsections = len(subsections)

        if num_subsections == 1:
            header = "Result"
            # split section on line of asterisks
            data = re.split(R"\*+", section)[-1]

        elif num_subsections == 2:
            header, data = subsections

        elif num_subsections == 3:
            notes, header, data = subsections

        elif num_subsections == 4:
            notes, header, data, sytem = subsections

        else:
            raise Exception(f"Error parsing table {title}")

        return header, data

    @staticmethod
    def _parse_header(header: str) -> list[str]:
        """
        Parse header line produced from _split_section into list of column headers. Uses pandas
        read_fwf to automatically parse multi line headers present in report file.


        Parameters
        ----------
        header: str
            Header text string produced from _split_section

        Returns
        -------
        List[str]
            List of column headers
        """

        # substitute single spaces between words with underscores
        # replace asterisks or dashes with spaces
        header = [
            re.sub(R"(?<=\w)[^\S\r\n](?=\w)", "_", field[1].dropna().str.cat(sep="_"))
            for field in read_fwf(
                StringIO(re.sub(R"\*|-", " ", header)), header=None
            ).items()
        ]

        # split day and time into separate fields to be recombined in to datetime object
        # when parsing table
        if "Time_of_Max_Occurrence_days_hr:min" in header:
            max_idx = header.index("Time_of_Max_Occurrence_days_hr:min")
            header[max_idx] = "days"
            header.insert(max_idx + 1, "Time_of_Max")

        return header

    @staticmethod
    def _parse_table(
        header: list[str], data: str, sep: str = R"\s{2,}|\s:\s", index_col: int = 0
    ) -> DataFrame:
        r"""
        Function to parse data string produced from _split_section into pandas DataFrame

        Parameters
        ----------
        header: Sequence[str]
            Sequence of column names to assign to DataFrame. Mostly can be produced from _parse_header.
        data: str
            Data string produced form _split_section
        sep: str, optional
            Delimeter to be fed into pandas read_csv function that operates on data string
            , by default R"\s{2,}|\s:\s"
        index_col: int, optional
            Column in data to be used as DataFrame index, by default 0

        Returns
        -------
        pd.DataFrame
            Report data table
        """

        # remove leading spaces on each line and replace long runs of periods with spaces
        data = re.sub(R"^\s+", "", re.sub(R"\.{2,}", "  ", data), flags=re.MULTILINE)

        # by default read in data with minimum 2-spaces or semicolon flanked by spaces as delimiter
        df = read_csv(
            filepath_or_buffer=StringIO(data),
            header=None,
            engine="python",
            sep=sep,
            index_col=index_col,
            names=header,
        )

        # convert day and time columns into a single datetime column
        if "Time_of_Max" in df.columns:
            # convert time of max to timedelta
            df["Time_of_Max"] = to_timedelta(
                df.pop("days").astype(int), unit="D"
            ) + to_timedelta(
                df["Time_of_Max"] + ":00"
            )  # type: ignore
        return df

    @property
    def analysis_options(self) -> Series:
        """
        Pandas series containing the analysis options listed in the
        report file including units, models, methods, dates, time steps, etc.

        Returns
        -------
        Series
            Series of options.
        """
        if not hasattr(self, "_analysis_options"):
            header, data = self._split_section(self._sections["Analysis Options"])
            df = self._parse_table(["Option", "Setting"], data)["Setting"]
            self._analysis_options = df.dropna()

        return self._analysis_options

    @property
    def runoff_quantity_continuity(self) -> DataFrame:
        """
        Runoff quantity continuity error table in volume and depth units.
        System wide error is show in percent.


        Returns
        -------
        pd.DataFrame
            DataFrame of runoff quantity continuity error table.
        """
        if not hasattr(self, "_runoff_quantity_continuity"):
            header, data = self._split_section(
                self._sections["Runoff Quantity Continuity"]
            )
            # substitute spaces between words with underscore so read_fwf works
            # had to use some regex to not also match new lines
            header = self._parse_header(re.sub(R"(?<=\w)[^\S\r\n](?=\w)", "_", header))
            self._runoff_quantity_continuity = self._parse_table(header, data)
        return self._runoff_quantity_continuity

    @property
    def runoff_quality_continuity(self) -> DataFrame:
        """
        Runoff quality continuity error table in mass units for each pollutant.
        System wide error is show in percent.


        Returns
        -------
        pd.DataFrame
            DataFrame of runoff quality continuity error table
        """
        if not hasattr(self, "_runoff_quality_continuity"):
            header, data = self._split_section(
                self._sections["Runoff Quality Continuity"]
            )
            # substitute spaces between words with underscore so read_fwf works
            # had to use some  regex to not also match new lines
            header = self._parse_header(re.sub(R"(?<=\w)[^\S\r\n](?=\w)", "_", header))
            self._runoff_quality_continuity = self._parse_table(header, data)
        return self._runoff_quality_continuity

    @property
    def groundwater_continuity(self) -> DataFrame:
        """
        Groundwater quantity continuity error table in volume and depth units.
        System wide error is show in percent.


        Returns
        -------
        pd.DataFrame
            DataFrame of groundwater quantity continuity error table
        """
        if not hasattr(self, "_groundwater_continuity"):
            header, data = self._split_section(self._sections["Groundwater Continuity"])
            # substitute spaces between words with underscore so read_fwf works
            # had to use some  regex to not also match new lines
            header = self._parse_header(re.sub(R"(?<=\w)[^\S\r\n](?=\w)", "_", header))
            self._groundwater_continuity = self._parse_table(header, data)
        return self._groundwater_continuity

    @property
    def flow_routing_continuity(self) -> DataFrame:
        """
        Flow routing continuity error table in volume units.
        System wide error is show in percent.


        Returns
        -------
        pd.DataFrame
            DataFrame of flow routing continuity error table
        """
        if not hasattr(self, "_flow_routing_continuity"):
            header, data = self._split_section(
                self._sections["Flow Routing Continuity"]
            )
            # substitute spaces between words with underscore so read_fwf works
            # had to use some  regex to not also match new lines
            header = self._parse_header(re.sub(R"(?<=\w)[^\S\r\n](?=\w)", "_", header))
            self._flow_routing_continuity = self._parse_table(header, data)
        return self._flow_routing_continuity

    @property
    def quality_routing_continuity(self) -> DataFrame:
        """
        Quality routing continuity error table in mass units.
        System wide error is show in percent.


        Returns
        -------
        pd.DataFrame
            DataFrame of quality routing continuity error table
        """
        if not hasattr(self, "_quality_routing_continuity"):
            header, data = self._split_section(
                self._sections["Quality Routing Continuity"]
            )
            # substitute spaces between words with underscore so read_fwf works
            # had to use some  regex to not also match new lines
            header = self._parse_header(re.sub(R"(?<=\w)[^\S\r\n](?=\w)", "_", header))
            self._quality_routing_continuity = self._parse_table(header, data)
        return self._quality_routing_continuity

    @property
    def highest_continuity_errors(self) -> DataFrame:
        """
        Highest continuity error table in percent.
        This table shows the model elements with the highest
        flow routing continuity error.

        Returns
        -------
        pd.DataFrame
            DataFrame of highest continuity errors table
        """
        if not hasattr(self, "_highest_errors"):
            header, data = self._split_section(
                self._sections["Highest Continuity Errors"]
            )
            df = self._parse_table(
                ["object_type", "name", "percent_error"], data, sep=R"\s+", index_col=1
            )
            df["percent_error"] = df["percent_error"].str.strip("()%").astype(float)
            self._highest_errors = df
        return self._highest_errors

    @property
    def time_step_critical_elements(self) -> DataFrame:
        """
        Time-step critical elements table in percent.
        This table shows the model elements that were controlling
        the model time step if a variable one was used.

        Returns
        -------
        pd.DataFrame
            DataFrame of time-step critical elements table
        """

        if not hasattr(self, "_ts_critical"):
            header, data = self._split_section(
                self._sections["Time-Step Critical Elements"]
            )
            df = self._parse_table(
                ["object_type", "name", "percent"], data, sep=R"\s+", index_col=1
            )
            df["percent"] = df["percent"].str.strip("()%").astype(float)
            self._ts_critical = df
        return self._ts_critical

    @property
    def highest_flow_instability_indexes(self) -> DataFrame:
        """
        Highest flow instability indexes.
        This table shows the model elements that have the highest
        flow instability.

        Returns
        -------
        pd.DataFrame
            DataFrame of highest flow instability indexes table
        """
        if not hasattr(self, "_highest_flow_instability_indexes"):
            header, data = self._split_section(
                self._sections["Highest Flow Instability Indexes"]
            )
            if "All links are stable" in data:
                data = ""
            df = self._parse_table(
                ["object_type", "name", "index"], data, sep=R"\s+", index_col=1
            )
            df["index"] = df["index"].str.strip("()").astype(int)
            self._highest_flow_instability_indexes = df
        return self._highest_flow_instability_indexes

    @property
    def routing_time_step_summary(self) -> DataFrame:
        """
        Routing time step summary table that shows the average, minimum,
        and maximum time steps as well as convergance summary.

        Returns
        -------
        pd.DataFrame
            DataFrame of routing time step summary table
        """
        if not hasattr(self, "_routing_time_step_summary"):
            header, data = self._split_section(
                self._sections["Routing Time Step Summary"]
            )
            self._routing_time_step_summary = self._parse_table(
                self._parse_header(header), data, sep=R"\s+:\s+"
            )
        return self._routing_time_step_summary

    @property
    def runoff_summary(self) -> DataFrame:
        """
        Runoff summary table for each subcatchment that details rainfall,
        runon, evap, infil, and runoff.

        Returns
        -------
        pd.DataFrame
            DataFrame of subcatchment runoff summary table
        """
        if not hasattr(self, "_runoff_summary"):
            header, data = self._split_section(
                self._sections["Subcatchment Runoff Summary"]
            )
            self._runoff_summary = self._parse_table(self._parse_header(header), data)
        return self._runoff_summary

    @property
    def groundwater_summary(self) -> DataFrame:
        """
        Groundwater summary table for each subcatchment that details groundwater
        inflow, outflow, moisture, and water table.

        Returns
        -------
        pd.DataFrame
            DataFrame of subcatchment groundwater summary table
        """
        if not hasattr(self, "_groundwater_summary"):
            header, data = self._split_section(self._sections["Groundwater Summary"])
            self._groundwater_summary = self._parse_table(
                self._parse_header(header), data
            )
        return self._groundwater_summary

    @property
    def washoff_summary(self) -> DataFrame:
        """
        Washoff summary table that details the total pollutant load
        that was washed off of each subcatchment.

        Returns
        -------
        pd.DataFrame
            DataFrame of subcatchment washoff summary table
        """
        if not hasattr(self, "_washoff_summary"):
            header, data = self._split_section(
                self._sections["Subcatchment Washoff Summary"]
            )
            self._washoff_summary = self._parse_table(self._parse_header(header), data)
        return self._washoff_summary

    @property
    def node_depth_summary(self) -> DataFrame:
        """
        Node depth summary table that details the average and maximum
        depth and HGL simulated for each node.

        Returns
        -------
        pd.DataFrame
            DataFrame of node depth summary table
        """
        if not hasattr(self, "_node_depth_summary"):
            header, data = self._split_section(self._sections["Node Depth Summary"])
            self._node_depth_summary = self._parse_table(
                self._parse_header(header), data, sep=R"\s{1,}|\s:\s"
            )
        return self._node_depth_summary

    @property
    def node_inflow_summary(self) -> DataFrame:
        """
        Node inflow summary table that details the maximum inflow rates, total
        inflow volumes, and flow balance error percent for each node.

        Returns
        -------
        pd.DataFrame
            DataFrame of node inflow summary table
        """
        if not hasattr(self, "_node_inflow_summary"):
            header, data = self._split_section(self._sections["Node Inflow Summary"])

            self._node_inflow_summary = self._parse_table(
                self._parse_header(header), data
            )
        return self._node_inflow_summary

    @property
    def node_surchage_summary(self) -> DataFrame:
        """
        Node surcharge summary that details the maximum surcharge level and duration
        of surharge for each node.

        Returns
        -------
        pd.DataFrame
            DataFrame of node surcharge summary table
        """
        if not hasattr(self, "_node_surcharge_summary"):
            header, data = self._split_section(self._sections["Node Surcharge Summary"])

            self._node_surcharge_summary = self._parse_table(
                self._parse_header(header), data
            )
        return self._node_surcharge_summary

    @property
    def node_flooding_summary(self) -> DataFrame:
        """
        Node flood summary that details the maximum ponded depth, peak flooding rate, total flood volume,
        total flood duration for each node.

        Returns
        -------
        pd.DataFrame
            DataFrame of node flooding summary table
        """
        if not hasattr(self, "_node_flooding_summary"):
            header, data = self._split_section(self._sections["Node Flooding Summary"])

            self._node_flooding_summary = self._parse_table(
                self._parse_header(header), data
            )
        return self._node_flooding_summary

    @property
    def storage_volume_summary(self) -> DataFrame:
        """
        Storage volume summary that details the frequency of filling, average and peak volumes,
        losses, and outfall rate for each storage unit.

        Returns
        -------
        pd.DataFrame
            DataFrame of storage volume summary table
        """
        if not hasattr(self, "_storage_volume_summary"):
            header, data = self._split_section(self._sections["Storage Volume Summary"])
            header = header.replace("Storage Unit", "Storage     ")
            self._storage_volume_summary = self._parse_table(
                self._parse_header(header), data
            )
        return self._storage_volume_summary

    @property
    def outfall_loading_summary(self) -> DataFrame:
        """
        Outfall loading summary that details the flow frequency, average and peak flow rates,
        total outflow volume, and pollutant mass loads for each outfall.

        Returns
        -------
        pd.DataFrame
            DataFrame of outfall loading summary table
        """
        if not hasattr(self, "_outfall_loading_summary"):
            header, data = self._split_section(
                self._sections["Outfall Loading Summary"]
            )
            header = header.replace("Outfall Node", "Outfall     ")
            self._outfall_loading_summary = self._parse_table(
                self._parse_header(header), data
            )
        return self._outfall_loading_summary

    @property
    def link_flow_summary(self) -> DataFrame:
        """
        Link flow summary that details the peak flow, velocity, depth, and capacity for each link.

        Returns
        -------
        pd.DataFrame
            DataFrame of link flow summary table
        """
        if not hasattr(self, "_link_flow_summary"):
            header, data = self._split_section(self._sections["Link Flow Summary"])
            header = header.replace("|", " ")
            self._link_flow_summary = self._parse_table(
                self._parse_header(header), data, sep=R"\s{1,}|\s:\s"
            )
        return self._link_flow_summary

    @property
    def flow_classification_summary(self) -> DataFrame:
        """
        Flow classification summary that details the amount of conduit lengthening during
        the simualtion and the fraction of simulation time that is dry, subcritical, supercritical,
        or critical flow for each conduit.

        Returns
        -------
        pd.DataFrame
            DataFrame of flow classification summary table
        """
        if not hasattr(self, "_flow_classification_summary"):
            header, data = self._split_section(
                self._sections["Flow Classification Summary"]
            )
            to_remove = "---------- Fraction of Time in Flow Class ----------"
            to_replace = "                                                    "
            header = header.replace(to_remove, to_replace)
            self._flow_classification_summary = self._parse_table(
                self._parse_header(header), data
            )
        return self._flow_classification_summary

    @property
    def conduit_surcharge_summary(self) -> DataFrame:
        """
        Conduit surcharge summary that details the hours of surcharging and
        capacity limited conditions.

        Returns
        -------
        pd.DataFrame
            DataFrame of conduit surcharge summary table
        """
        if not hasattr(self, "_conduit_surcharge_summary"):
            header, data = self._split_section(
                self._sections["Conduit Surcharge Summary"]
            )
            to_remove = "--------- Hours Full --------"
            to_replace = "HrsFull   HoursFull  HrsFull "
            header = header.replace(to_remove, to_replace)
            self._conduit_surcharge_summary = self._parse_table(
                self._parse_header(header), data
            )
        return self._conduit_surcharge_summary

    @property
    def pumping_summary(self) -> DataFrame:
        """
        Pumping summary that details the utilization, peak flow rates, total flow volume,
        power usage, and time off pump curve for each pump.

        Returns
        -------
        pd.DataFrame
            DataFrame of pumping summary table
        """
        if not hasattr(self, "_pumping_summary"):
            header, data = self._split_section(self._sections["Pumping Summary"])
            header = self._parse_header(header)
            header[-1] = "Percent_Time_Off_Pump_Curve_Low"
            header.append("Percent_Time_Off_Pump_Curve_High")
            self._pumping_summary = self._parse_table(header, data)
        return self._pumping_summary

    @property
    def link_pollutant_load_summary(self) -> DataFrame:
        """
        Link pollutant load summary that details the total pollutant mass discharged
        from each link.

        Returns
        -------
        pd.DataFrame
            DataFrame of link pollutant load summary table
        """
        if not hasattr(self, "_link_pollutant_load_summary"):
            header, data = self._split_section(
                self._sections["Link Pollutant Load Summary"]
            )

            self._link_pollutant_load_summary = self._parse_table(
                self._parse_header(header), data
            )
        return self._link_pollutant_load_summary

    @property
    def analysis_begun(self) -> Timestamp:
        """
        Date and time when the simulation was started

        Returns
        -------
        Timestamp
            Simulation start time

        Raises
        ------
        Exception
            if analysis begun text could not be found in the report file
        """
        if not hasattr(self, "_analysis_begun"):
            pattern = R"\s+Analysis begun on:\s+([^\n]+)$"
            s = re.search(pattern, self._rpt_text, flags=re.MULTILINE)
            if s:
                self._analysis_begun = to_datetime(s.group(1))
            else:
                raise Exception("Error finding analysis begun")
        return self._analysis_begun

    @property
    def analysis_end(self) -> Timestamp:
        """
        Date and time when the simulation ended

        Returns
        -------
        Timestamp
            Simulation end time

        Raises
        ------
        Exception
            if analysis ended text could not be found in the report file
        """
        if not hasattr(self, "_analysis_end"):
            pattern = R"\s+Analysis ended on:\s+([^\n]+)$"
            s = re.search(pattern, self._rpt_text, flags=re.MULTILINE)
            if s:
                self._analysis_end = to_datetime(s.group(1))
            else:
                raise Exception("Error finding analysis end")
        return self._analysis_end
