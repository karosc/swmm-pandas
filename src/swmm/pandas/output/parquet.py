"""Parquet export helpers for SWMM output data."""

from __future__ import annotations

import importlib
import os
import os.path
import posixpath
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy import asarray, ndarray, tile
from swmm.toolkit import shared_enum

import pandas as pd
from pandas.core.api import DataFrame, Timestamp

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem

    from swmm.pandas.output.output import Output


@dataclass(frozen=True)
class _ExportColumn:
    kind: str
    name: str
    attribute: str
    value_index: int


class _ParquetExporter:
    def __init__(self, out: Output):
        self.out = out

    def write(
        self,
        path: str | os.PathLike[str],
        link_attributes: list[str | shared_enum.LinkAttribute] | None = None,
        node_attributes: list[str | shared_enum.NodeAttribute] | None = None,
        subcatchment_attributes: (
            list[str | shared_enum.SubcatchAttribute] | None
        ) = None,
        system_attributes: list[str | shared_enum.SystemAttribute] | None = None,
        links: list[str] | None = None,
        nodes: list[str] | None = None,
        subcatchments: list[str] | None = None,
        row_batch_size: int = 100,
        partition_freq: str | None = None,
        filesystem: AbstractFileSystem | None = None,
    ) -> str:
        if not isinstance(row_batch_size, (int, np.integer)) or row_batch_size < 1:
            raise ValueError("row_batch_size must be a positive integer")

        export_plan = self._build_export_plan(
            link_attributes=link_attributes,
            node_attributes=node_attributes,
            subcatchment_attributes=subcatchment_attributes,
            system_attributes=system_attributes,
            links=links,
            nodes=nodes,
            subcatchments=subcatchments,
        )
        batches = self._iter_long_frame_batches(export_plan, int(row_batch_size))
        partition_mode = self._normalize_partition_freq(partition_freq)

        if partition_mode is None:
            return self._write_parquet_file(path, batches, filesystem=filesystem)

        return self._write_parquet_dataset(
            path,
            batches,
            partition_mode,
            filesystem=filesystem,
        )

    def _build_export_plan(
        self,
        link_attributes: list[str | shared_enum.LinkAttribute] | None = None,
        node_attributes: list[str | shared_enum.NodeAttribute] | None = None,
        subcatchment_attributes: (
            list[str | shared_enum.SubcatchAttribute] | None
        ) = None,
        system_attributes: list[str | shared_enum.SystemAttribute] | None = None,
        links: list[str] | None = None,
        nodes: list[str] | None = None,
        subcatchments: list[str] | None = None,
    ) -> tuple[_ExportColumn, ...]:
        out = self.out
        selected_subcatchments = set(
            out._validateElement(subcatchments, out.subcatchments)[0],
        )
        selected_nodes = set(out._validateElement(nodes, out.nodes)[0])
        selected_links = set(out._validateElement(links, out.links)[0])

        selected_subcatch_attributes = set(
            out._validateAttribute(
                subcatchment_attributes,
                out.subcatch_attributes,
            )[0],
        )
        selected_node_attributes = set(
            out._validateAttribute(node_attributes, out.node_attributes)[0],
        )
        selected_link_attributes = set(
            out._validateAttribute(link_attributes, out.link_attributes)[0],
        )
        selected_system_attributes = set(
            out._validateAttribute(system_attributes, out.system_attributes)[0],
        )

        selected_columns: list[_ExportColumn] = []
        for column in out._result_columns:
            if column.kind == "sub":
                include = (
                    column.name in selected_subcatchments
                    and column.attribute in selected_subcatch_attributes
                )
            elif column.kind == "node":
                include = (
                    column.name in selected_nodes
                    and column.attribute in selected_node_attributes
                )
            elif column.kind == "link":
                include = (
                    column.name in selected_links
                    and column.attribute in selected_link_attributes
                )
            else:
                include = column.attribute in selected_system_attributes

            if include:
                selected_columns.append(column)

        return tuple(selected_columns)

    @staticmethod
    def _normalize_partition_freq(partition_freq: str | None) -> str | None:
        if partition_freq is None:
            return None

        if not isinstance(partition_freq, str):
            raise TypeError("partition_freq must be a pandas frequency string or None")

        normalized = partition_freq.upper()
        partition_modes = {
            "A": "year",
            "Y": "year",
            "YE": "year",
            "YS": "year",
            "M": "month",
            "MS": "month",
            "D": "day",
            "H": "hour",
            "YEAR": "year",
            "MONTH": "month",
            "DAY": "day",
            "HOUR": "hour",
        }

        if normalized not in partition_modes:
            raise ValueError(
                "partition_freq must be one of A, Y, YE, YS, M, MS, D, H, or h",
            )

        return partition_modes[normalized]

    @staticmethod
    def _partition_columns(partition_mode: str) -> list[str]:
        partition_columns = {
            "year": ["year"],
            "month": ["year", "month"],
            "day": ["year", "month", "day"],
            "hour": ["year", "month", "day", "hour"],
        }
        return partition_columns[partition_mode]

    def _empty_export_frame(self) -> DataFrame:
        return DataFrame(
            {
                "time": asarray([], dtype="datetime64[ns]"),
                "element_type": asarray([], dtype=object),
                "element_name": asarray([], dtype=object),
                "attribute": asarray([], dtype=object),
                "value": asarray([], dtype="float32"),
            },
        )

    def _build_long_frame(
        self,
        export_columns: Sequence[_ExportColumn],
        swmm_datetimes: ndarray,
        value_chunk: ndarray,
    ) -> DataFrame:
        if len(export_columns) == 0 or len(swmm_datetimes) == 0:
            return self._empty_export_frame()

        datetime_values = asarray(
            [self.out._datetime_from_swmm(value) for value in swmm_datetimes],
            dtype="datetime64[ns]",
        )

        kind_values = asarray([column.kind for column in export_columns], dtype=object)
        name_values = asarray([column.name for column in export_columns], dtype=object)
        attribute_values = asarray(
            [column.attribute for column in export_columns],
            dtype=object,
        )

        row_count, column_count = value_chunk.shape
        return DataFrame(
            {
                "time": tile(datetime_values, column_count),
                "element_type": kind_values.repeat(row_count),
                "element_name": name_values.repeat(row_count),
                "attribute": attribute_values.repeat(row_count),
                "value": asarray(value_chunk).reshape(-1, order="F"),
            },
        )

    def _iter_long_frame_batches(
        self,
        export_columns: Sequence[_ExportColumn],
        row_batch_size: int,
    ) -> Iterator[DataFrame]:
        out = self.out
        if row_batch_size < 1:
            raise ValueError("row_batch_size must be greater than 0")

        if len(export_columns) == 0 or out._period == 0:
            yield self._empty_export_frame()
            return

        selected_value_indices = [column.value_index for column in export_columns]

        if out._preload:
            data_column_indices = [index + 1 for index in selected_value_indices]
            for start in range(0, out._period, row_batch_size):
                stop = min(start + row_batch_size, out._period)
                swmm_datetimes = out.data.iloc[start:stop, 0].to_numpy(copy=False)
                value_chunk = out.data.iloc[
                    start:stop,
                    data_column_indices,
                ].to_numpy(copy=False)
                yield self._build_long_frame(
                    export_columns, swmm_datetimes, value_chunk,
                )
            return

        record_dtype = out._result_record_dtype
        record_size = record_dtype.itemsize
        with open(out._binfile, "rb") as fil:
            for start in range(0, out._period, row_batch_size):
                count = min(row_batch_size, out._period - start)
                fil.seek(out._output_position + start * record_size, 0)
                chunk = np.fromfile(fil, dtype=record_dtype, count=count)
                swmm_datetimes = chunk["datetime"]
                value_chunk = chunk["values"][:, selected_value_indices]
                yield self._build_long_frame(
                    export_columns, swmm_datetimes, value_chunk,
                )

    @staticmethod
    def _pyarrow_modules() -> tuple[Any, Any, Any]:
        try:
            pyarrow = importlib.import_module("pyarrow")
            dataset = importlib.import_module("pyarrow.dataset")
            parquet = importlib.import_module("pyarrow.parquet")
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for Output.to_parquet(). Install it in your uv environment first.",
            ) from exc

        return pyarrow, dataset, parquet

    @staticmethod
    def _filesystem_path(
        path: str | os.PathLike[str],
        filesystem: AbstractFileSystem | None,
    ) -> str:
        path_str = os.fspath(path)
        return path_str if filesystem is None else path_str.replace("\\", "/")

    @staticmethod
    def _filesystem_join(
        base_path: str,
        *parts: str,
        filesystem: AbstractFileSystem | None,
    ) -> str:
        return (
            os.path.join(base_path, *parts)
            if filesystem is None
            else posixpath.join(base_path, *parts)
        )

    @staticmethod
    def _filesystem_exists(
        path: str,
        filesystem: AbstractFileSystem | None,
    ) -> bool:
        return os.path.exists(path) if filesystem is None else filesystem.exists(path)

    @staticmethod
    def _filesystem_isdir(
        path: str,
        filesystem: AbstractFileSystem | None,
    ) -> bool:
        if filesystem is None:
            return os.path.isdir(path)
        if hasattr(filesystem, "isdir"):
            return bool(filesystem.isdir(path))
        try:
            return filesystem.info(path).get("type") == "directory"
        except FileNotFoundError:
            return False

    @staticmethod
    def _filesystem_makedirs(
        path: str,
        filesystem: AbstractFileSystem | None,
        exist_ok: bool = True,
    ) -> None:
        if filesystem is None:
            os.makedirs(path, exist_ok=exist_ok)
        else:
            filesystem.makedirs(path, exist_ok=exist_ok)

    def _pyarrow_filesystem(
        self,
        filesystem: AbstractFileSystem | None,
    ) -> object | None:
        if filesystem is None:
            return None

        pyarrow, _, _ = self._pyarrow_modules()
        return pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(filesystem))

    def _write_parquet_file(
        self,
        path: str | os.PathLike[str],
        batches: Iterator[DataFrame],
        filesystem: AbstractFileSystem | None = None,
    ) -> str:
        pyarrow, _, parquet = self._pyarrow_modules()
        path_str = self._filesystem_path(path, filesystem)
        parent = (
            os.path.dirname(path_str)
            if filesystem is None
            else posixpath.dirname(path_str)
        )
        if parent:
            self._filesystem_makedirs(parent, filesystem, exist_ok=True)

        arrow_filesystem = self._pyarrow_filesystem(filesystem)

        writer = None
        try:
            for batch in batches:
                table = pyarrow.Table.from_pandas(batch, preserve_index=False)
                if writer is None:
                    writer = parquet.ParquetWriter(
                        path_str,
                        table.schema,
                        filesystem=arrow_filesystem,
                    )
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

        return path_str

    def _add_partition_columns(
        self,
        batch: DataFrame,
        partition_mode: str,
    ) -> DataFrame:
        partitioned = batch.copy()
        datetime_parts = partitioned["time"].dt
        partitioned["year"] = datetime_parts.year
        if partition_mode in ("month", "day", "hour"):
            partitioned["month"] = datetime_parts.month
        if partition_mode in ("day", "hour"):
            partitioned["day"] = datetime_parts.day
        if partition_mode == "hour":
            partitioned["hour"] = datetime_parts.hour
        return partitioned

    @staticmethod
    def _format_partition_timestamp(timestamp: Timestamp | datetime) -> str:
        ts = Timestamp(timestamp)
        if ts is pd.NaT:
            raise ValueError("Cannot format NaT timestamp for parquet partitioning")
        return cast(Timestamp, ts).strftime("%Y%m%d%H%M%S")

    def _partition_file_name(self, batch: DataFrame) -> str:
        start = cast(Timestamp, batch["time"].min())
        end = cast(Timestamp, batch["time"].max()) + timedelta(
            seconds=self.out._report,
        )
        return (
            f"{self._format_partition_timestamp(start)}_"
            f"{self._format_partition_timestamp(end)}.parquet"
        )

    def _write_parquet_dataset(
        self,
        path: str | os.PathLike[str],
        batches: Iterator[DataFrame],
        partition_mode: str,
        filesystem: AbstractFileSystem | None = None,
    ) -> str:
        pyarrow, _, parquet = self._pyarrow_modules()
        path_str = self._filesystem_path(path, filesystem)
        arrow_filesystem = self._pyarrow_filesystem(filesystem)

        if self._filesystem_exists(path_str, filesystem):
            if not self._filesystem_isdir(path_str, filesystem):
                raise FileExistsError(
                    f"Partitioned parquet path '{path_str}' already exists and is not a directory.",
                )
        else:
            self._filesystem_makedirs(path_str, filesystem, exist_ok=True)

        partition_columns = self._partition_columns(partition_mode)

        for batch in batches:
            partitioned = self._add_partition_columns(batch, partition_mode)
            if partitioned.empty:
                continue

            for _, group in partitioned.groupby(partition_columns, sort=True):
                if group.empty:
                    continue

                partition_values = group.iloc[0]
                partition_path = self._filesystem_join(
                    path_str,
                    *[
                        f"{column}={int(partition_values[column])}"
                        for column in partition_columns
                    ],
                    filesystem=filesystem,
                )
                self._filesystem_makedirs(partition_path, filesystem, exist_ok=True)

                file_name = self._partition_file_name(group)
                file_path = self._filesystem_join(
                    partition_path,
                    file_name,
                    filesystem=filesystem,
                )
                table = pyarrow.Table.from_pandas(
                    group.drop(columns=partition_columns),
                    preserve_index=False,
                )
                parquet.write_table(table, file_path, filesystem=arrow_filesystem)

        return path_str
