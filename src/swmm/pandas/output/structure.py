from __future__ import annotations
from collections.abc import Sequence

import numpy as np
from pandas.core.api import DataFrame
from pandas._libs.missing import NA

from swmm.pandas.output.tools import arrayish

volumeConstants = {
    "CFS": dict(multiplier=1 * (7.481 / 1e6), volumeUnits="MG"),
    "GPM": dict(multiplier=(1 / 60) * (1 / 1e6), volumeUnits="MG"),
    "MGD": dict(multiplier=(1 / 86400) * 1, volumeUnits="MG"),
    "CMS": dict(multiplier=1 * 1, volumeUnits="CM"),
    "LPS": dict(multiplier=1 * (1 / 1000), volumeUnits="CM"),
    "MLD": dict(multiplier=(1 / 86400) * 1000, volumeUnits="CM"),
}

hour_unit = np.timedelta64(1, "h")

# This class as it is could be DRYer, but it should also integrate with the
# inp file module and rpt file module when they are implemented. Plan to refactor
# and make more DRY when those are added


class Structure:
    """
    A class that represents a particular system structure that may be represented by
    multiple model elements. The outputs from each element are combined into a single
    time series for analysis as if they are a single structure.

    The structure class can be used to summarize flow and flooding at one or more model
    elements from a particular simulation. Parse link flow time series into discrete events
    with the `flowEvents` method, or summarize flooding events with the `floodEvents` method.

    Parameters
    ----------
    outfile: swmm.pandas.Output
        The swmm-pandas outfile object containing the model elements.
    link: Union[str, Sequence[str]]
        The list of links that belong to the structure.
    node: Union[str, Sequence[str]]
        The list of nodes that below to the structure.
    """

    def __init__(
        self,
        outfile,
        link: str | Sequence[str],
        node: str | Sequence[str],
    ):
        self.out = outfile
        """The Output object from which this structure is derived"""
        self.link = link
        """A list of the link(s) that belong to this structure"""
        self.node = node
        """A list of the node(s) that belong to this structure"""

    @property
    def floodFrame(self) -> DataFrame:
        """
        Returns a pandas DataFrame with the flood rates
        of each node in the structure

        Returns
        -------
        pd.DataFrame
            Time series of flooding for each node
        """
        if hasattr(self, "_floodFrame"):
            pass
        else:
            self._floodFrame = self.out.node_series(
                self.node, "flooding_losses", columns="elem"
            )

        return self._floodFrame

    @property
    def flowFrame(self) -> DataFrame:
        """
        Returns a pandas DataFrame with the flow rates
        of each link in the structure

        Returns
        -------
        pd.DataFrame
            Time series of flow for each link
        """
        if hasattr(self, "_flowFrame"):
            pass
        else:
            self._flowFrame = self.out.link_series(
                self.link, "flow_rate", columns="elem"
            )

        return self._flowFrame

    def _aggSeries(
        self,
        df: DataFrame,
        useNegative: bool | Sequence[bool] = False,
        reverse: bool | int | Sequence[bool | int] = False,
        aggFunc: str = "sum",
    ):
        """
        Aggregate a multi element time series into a single element time series.
        This function is used to calculate combined flow rates and flooding rates
        for the structure.

        Parameters
        ----------
        df: pd.DataFrame
            A DataFrame with a time series each column for each element in the structure.
            (e.g. output of self.flowFrame)
        useNegative: Union[bool, Sequence[bool]], optional
            If true, negative values will not be removed from the time series.
            Can either provide a boolean to apply to all columns or a list of
            booleans to apply to each column, by default False
        reverse: Union[bool, Sequence[bool]], optional
            If true, the timeseries will be multiplied by -1.
            Can either priovide a boolean to apply to all columns or a list of
            booleans to apply to each column, by default False
        aggFunc: str, optional
            The aggregation function to apply to all columns in the DataFrame. Should be
            compatible with `pd.DataFrame.agg`_., by default "sum"

        .. _pd.DataFrame.agg: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html


        Returns
        -------
        pd.Series
            Single time series aggregated from input DataFrame.

        Raises
        ------
        ValueError
            If useNegative or reverse arguments are not a compatible type.

        """

        # reverse values if requested
        if isinstance(reverse, arrayish):
            reverse = [-1 if col else 1 for col in reverse]
        elif isinstance(reverse, bool):
            reverse = -1 if reverse else 1
        else:
            raise ValueError(
                f"invert must be either bool or sequence of bool, given {type(reverse)}"
            )
        df = df * reverse

        # screen out negative values if requested
        if isinstance(useNegative, arrayish):
            for i, col in enumerate(df.columns):
                if not useNegative[i]:
                    df.loc[df[col] < 0, col] = 0
        elif isinstance(useNegative, bool):
            if not useNegative:
                for i, col in enumerate(df.columns):
                    df.loc[df[col] < 0, col] = 0
        else:
            raise ValueError(
                f"useNegative must be either bool or sequence of bool, given {type(useNegative)}"
            )

        # return aggregated df according to given aggFunc
        return df.agg(func=aggFunc, axis=1)

    def flowEvents(
        self,
        inter_event_period: float = 6,
        thresholdFlow: float = 0.01,
        useNegativeFlow: bool | Sequence[bool] = False,
        reverseFlow: bool | Sequence[bool] = False,
    ):
        """
        Bin flow data into discrete events based on an inter-event period and threshold flow rate.
        Maximum flowrates, total flow volumes, and duration of each event are returned in a DataFrame.

        Parameters
        ----------
        inter_event_period: float, optional
            The period in hours of flow less than or equal to thresholdFlow that demarks
            flow events, default to 6
        thresholdFlow: float, optional
            The flowrate in model flow units that dry or baseline condition that is not considered
            significant, default to 0.01
        useNegativeFlow: bool, optional
            If true, the method will consider negative flows when calculating flow volumes, defaults  False
        reverseFlow: bool, optional
            If true, the method will calculate the flow in the reverse direction by multiplying
            the timeseries by negative one, defaults to False

        Returns
        -------
        pd.DataFrame
            DataFrame with statistics on each flow event
        """
        # pull aggregated series
        series = self._aggSeries(self.flowFrame, useNegativeFlow, reverseFlow)

        # put series in DataFrame, and add event_num column
        q = DataFrame(
            series[series > thresholdFlow], columns=["flow_rate"]
        ).reset_index()
        q["event_num"] = NA
        # initialize first event
        q.loc[0, "event_num"] = 1

        # calculate period between flows greater than threshold
        hours = q.datetime.diff(1) / hour_unit

        # slice out times demarking a new event
        # assign event numbers to those starting points
        slicer = hours > inter_event_period
        q.loc[slicer, "event_num"] = range(2, sum(slicer) + 2)
        q.event_num.fillna(method="ffill", inplace=True)

        # group by event_num
        gpd = q.groupby("event_num")

        # find indices of max flow timesteps in each event
        maxSer = gpd.flow_rate.idxmax()

        # find event start date
        start_date = gpd.datetime.min().rename("start_datetime")
        # calculate volume for each event
        vol = (
            gpd.flow_rate.sum()
            * self.out.report
            * volumeConstants[self.out.units[1]]["multiplier"]
        )

        # add unit name to column
        vol.name = f"totalVolume_{volumeConstants[self.out.units[1]]['volumeUnits']}"

        # calculate the duration of each event in hours
        durations = (gpd.datetime.count() * self.out.report) / 60 / 60
        durations.name = "hours_duration"

        # join in event volumes and durations with event maxima
        return (
            q.loc[maxSer]
            .join(start_date, on="event_num")
            .join(vol, on="event_num")
            .join(durations, on="event_num")
            .rename({"flow_rate": "maxFlow", "datetime": "time_of_maxFlow"}, axis=1)
            .set_index("event_num")
        )

    def floodEvents(
        self,
        inter_event_period: float = 6,
        thresholdFood: float = 0.01,
    ):
        """
        Bins flooding data into discrete events based on an inter-event period and threshold flooding rate.
        Maximum flooding rates and duration of each flooding event are returned in a DataFrame.

        TODO: add in ponded depth when inp file is integrated. Ponded volume from out file is tough to interpret alone.

        Parameters
        ----------
        inter_event_period: float, optionalep
            The period in hours of flooding less than or equal to thresholdFlood that demarks
            flow events, default to 6
        thresholdFood: float, optional
            The flooding rate in model flow units above which should be considered in
            calculations, default to 0.01

        Returns
        -------
        pd.DataFrame
            DataFrame with statistics on each flooding event
        """
        series = self._aggSeries(self.floodFrame)

        # put series in DataFrame, and add event_num column
        q = DataFrame(
            series[series > thresholdFood],
            columns=["flooding_losses"],
        ).reset_index()
        q["event_num"] = NA
        # initialize first event
        q.loc[0, "event_num"] = 1

        # calculate period between flows greater than threshold
        hours = q.datetime.diff(1) / hour_unit

        # slice out times demarking a new event
        # assign event numbers to those starting points
        slicer = hours > inter_event_period
        q.loc[slicer, "event_num"] = range(2, sum(slicer) + 2)
        q.event_num.fillna(method="ffill", inplace=True)

        # group by event_num
        gpd = q.groupby("event_num")

        # find indices of max flow timesteps in each event
        maxSer = gpd.flooding_losses.idxmax()

        # calculate the duration of each event in hours
        durations = (gpd.datetime.count() * self.out.report) / 60 / 60
        durations.name = "hours_duration"

        # return event maxima joined with durations
        return (
            q.loc[maxSer]
            .join(durations, on="event_num")
            .rename(
                {"flooding_losses": "maxFloodRate", "datetime": "time_of_maxFlow"},
                axis=1,
            )
            .set_index("event_num")
        )
