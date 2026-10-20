# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Options that ``scan_iceberg`` and ``scan_delta`` attach to a parquet scan."""

from __future__ import annotations

import dataclasses
import itertools
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.dsl import expr
from cudf_polars.dsl.to_ast import to_parquet_filter
from cudf_polars.dsl.traversal import traversal
from cudf_polars.dsl.utils.replace import replace

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataType
    from cudf_polars.dsl.utils.io import CachedParquetInfo
    from cudf_polars.typing import Schema

    _ElementLayout = tuple[
        str, plc.io.parquet_metadata.PhysicalType, int, int, tuple[Any, ...] | None
    ]
    _Layout = tuple[_ElementLayout, ...]
    _FieldIds = tuple[int | None, tuple["_FieldIds", ...]]

__all__ = ["IcebergColumn", "LakeScanOptions", "read_lake_files"]

_INTEGER_TYPES = MappingProxyType(
    {
        plc.TypeId.INT8: (8, True),
        plc.TypeId.INT16: (16, True),
        plc.TypeId.INT32: (32, True),
        plc.TypeId.INT64: (64, True),
        plc.TypeId.UINT8: (8, False),
        plc.TypeId.UINT16: (16, False),
        plc.TypeId.UINT32: (32, False),
        plc.TypeId.UINT64: (64, False),
    }
)
_FLOAT_WIDTHS = MappingProxyType({plc.TypeId.FLOAT32: 32, plc.TypeId.FLOAT64: 64})
_TIMESTAMP_DIGITS = MappingProxyType(
    {
        plc.TypeId.TIMESTAMP_MILLISECONDS: 3,
        plc.TypeId.TIMESTAMP_MICROSECONDS: 6,
        plc.TypeId.TIMESTAMP_NANOSECONDS: 9,
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class IcebergColumn:
    """
    A column of an Iceberg schema, identified by its physical field ID.

    Parameters
    ----------
    name
        Name the column has in the table schema, which may differ from the
        name recorded in any given data file.
    physical_id
        Field ID written into the parquet footer for this column.
    kind
        One of ``"primitive"``, ``"list"``, ``"fixed-size-list"``, ``"map"``
        or ``"struct"``.
    children
        Fields of a nested column, empty for a primitive column.
    """

    name: str
    physical_id: int
    kind: str
    children: tuple[IcebergColumn, ...]

    @classmethod
    def from_polars(cls, spec: tuple[str, int, tuple[Any, ...]]) -> IcebergColumn:
        """
        Build from the nested tuples polars uses for an Iceberg column.

        Parameters
        ----------
        spec
            ``(name, physical_id, type_)`` where ``type_`` is one of:
                * ``("primitive", dtype)``
                * ``("list", column)``
                * ``("fixed-size-list", column, width)``
                * ``("map", key, value)``
                * ``("struct", {physical_id: column})``

        Returns
        -------
        The parsed column.
        """
        name, physical_id, type_ = spec
        kind, *rest = type_
        if kind == "primitive":
            children: tuple[IcebergColumn, ...] = ()
        elif kind == "struct":
            children = tuple(cls.from_polars(child) for child in rest[0].values())
        elif kind in ("list", "fixed-size-list", "map"):
            children = tuple(
                cls.from_polars(child) for child in rest if isinstance(child, tuple)
            )
        else:  # pragma: no cover; polars only produces the kinds above
            raise NotImplementedError(f"Unhandled Iceberg column type {kind!r}")
        return cls(name=name, physical_id=physical_id, kind=kind, children=children)

    def walk(self) -> Iterable[IcebergColumn]:
        """Yield this column and every column nested within it."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclasses.dataclass(frozen=True)
class LakeScanOptions:
    """
    The Iceberg and Delta Lake specific parts of a polars ``Scan``.

    Parameters
    ----------
    columns
        Top-level columns of the Iceberg schema, in table order, or ``None``
        when the scan reads columns by name (Delta, and Iceberg tables that
        never evolved).
    position_deletes
        Maps the index of a path in the scan to the Iceberg position delete
        files that apply to it.
    deletion_vectors
        Maps the index of a path in the scan to the Iceberg puffin file
        holding its deletion vector.
    delta_selections
        Maps the index of a path in the scan to the Delta deletion vector
        that applies to it, as a boolean series that is true for the rows
        to keep.
    partition_values
        Maps a physical field ID to the value its identity-transformed
        partition field takes, one row per path.
    partition_errors
        Maps a physical field ID to the reason polars could not supply the
        values of its identity-transformed partition field. Such a field is
        read from the files, and a file that lacks it fails the scan.
    initial_defaults
        Maps a physical field ID to the single-element series holding the
        Iceberg V3 initial default for a column absent from a file.
    missing_columns_policy
        ``"insert"`` to fill columns absent from a file, ``"raise"`` to fail.
    extra_columns_policy
        ``"ignore"`` to drop columns a file has but the schema does not,
        ``"raise"`` to fail.
    cast_columns_policy
        Which casts from the physical to the table type are permitted.
    row_count
        ``(total_rows, deleted_rows)`` for the whole scan when the table
        metadata could supply it.
    expected_keys
        Keys the top-level columns of a data file may have when
        ``extra_columns_policy`` is ``"raise"``: physical field IDs when
        ``columns`` is given, otherwise column names. ``None`` allows any.
    """

    columns: tuple[IcebergColumn, ...] | None
    position_deletes: Mapping[int, tuple[str, ...]]
    deletion_vectors: Mapping[int, str]
    delta_selections: Mapping[int, pl.Series]
    partition_values: Mapping[int, pl.Series]
    partition_errors: Mapping[int, str]
    initial_defaults: Mapping[int, pl.Series]
    missing_columns_policy: str
    extra_columns_policy: str
    cast_columns_policy: Mapping[str, Any]
    row_count: tuple[int, int] | None
    expected_keys: frozenset[Any] | None = None

    def __post_init__(self) -> None:  # noqa: D105
        unhandled_kinds = sorted(
            {
                nested.kind
                for column in self.columns or ()
                for nested in column.walk()
                if nested.kind in ("fixed-size-list", "map")
            }
        )
        if unhandled_kinds:
            raise NotImplementedError(
                f"Iceberg column mapping of {' and '.join(unhandled_kinds)} columns"
            )
        if (self.partition_values or self.partition_errors) and self.columns is None:
            raise NotImplementedError("Iceberg partition values without column mapping")
        if self.extra_columns_policy not in ("ignore", "raise"):
            raise NotImplementedError(  # pragma: no cover; only two policies exist
                f"Extra columns policy {self.extra_columns_policy!r}"
            )
        if self.missing_columns_policy not in ("insert", "raise"):
            raise NotImplementedError(  # pragma: no cover; only two policies exist
                f"Missing columns policy {self.missing_columns_policy!r}"
            )
        top_level_ids = {column.physical_id for column in self.columns or ()}
        unknown_defaults = set(self.initial_defaults) - top_level_ids
        if unknown_defaults:
            raise NotImplementedError(
                f"Iceberg defaults for nested or unmapped fields {sorted(unknown_defaults)}"
            )
        unknown_partitions = (
            set(self.partition_values).union(self.partition_errors) - top_level_ids
        )
        if unknown_partitions:
            raise NotImplementedError(
                f"Iceberg partition values for nested fields {sorted(unknown_partitions)}"
            )

    @classmethod
    def from_file_options(
        cls, file_options: Any, paths: Sequence[str]
    ) -> LakeScanOptions | None:
        """
        Build from the ``file_options`` of a polars ``Scan`` node.

        Parameters
        ----------
        file_options
            The ``PyFileOptions`` of the scan.
        paths
            Data files of the scan, named as polars names them.

        Returns
        -------
        The parsed options, or ``None`` for a scan that is not a lake scan.

        Raises
        ------
        NotImplementedError
            If polars supplies a variant we do not understand. Failing here
            keeps us from silently ignoring information that changes the
            result.

        Notes
        -----
        A Delta scan carries no column mapping, deletions or defaults, but a
        table that has evolved still has files that are missing a column of
        the schema, which a plain parquet read cannot express. Such a scan is
        recognised by its missing columns policy.

        A Delta deletion vector arrives as a callback holding a handle to the
        table, which cannot be sent to another process, so it is resolved
        here rather than in the tasks that read the files.
        """
        column_mapping = file_options.column_mapping
        columns: tuple[IcebergColumn, ...] | None = None
        if column_mapping is not None:
            mapping_kind, mapping = column_mapping
            if (
                mapping_kind != "iceberg-column-mapping"
            ):  # pragma: no cover; only kind polars emits
                raise NotImplementedError(f"Unhandled column mapping {mapping_kind!r}")
            columns = tuple(
                IcebergColumn.from_polars(spec) for spec in mapping.values()
            )

        position_deletes: dict[int, tuple[str, ...]] = {}
        deletion_vectors: dict[int, str] = {}
        delta_selections: dict[int, pl.Series] = {}
        if file_options.deletion_files is not None:
            deletion_kind, deletion_payload = file_options.deletion_files
            if deletion_kind == "iceberg":
                raw_position_deletes, raw_deletion_vectors = deletion_payload
                position_deletes = {
                    index: tuple(delete_files)
                    for index, delete_files in raw_position_deletes.items()
                }
                deletion_vectors = dict(raw_deletion_vectors)
            elif deletion_kind == "delta-deletion-vector":
                frame = deletion_payload(pl.DataFrame({"path": list(paths)}))
                if (
                    frame is None
                ):  # pragma: no cover; the polars callback returns a frame
                    delta_selections = {}
                else:
                    delta_selections = {
                        index: selection
                        for index, selection in enumerate(
                            frame.get_column("selection_vector")
                        )
                        if selection is not None
                    }
            else:
                raise NotImplementedError(  # pragma: no cover; only kinds polars emits
                    f"Unhandled deletion files {deletion_kind!r}"
                )

        partition_values: dict[int, pl.Series] = {}
        partition_errors: dict[int, str] = {}
        initial_defaults: dict[int, pl.Series] = {}
        if file_options.default_values is not None:
            defaults_kind, defaults_payload = file_options.default_values
            if defaults_kind != "iceberg":  # pragma: no cover; only kind polars emits
                raise NotImplementedError(f"Unhandled default values {defaults_kind!r}")
            partition_fields, raw_initial_defaults = defaults_payload
            for physical_id, value in partition_fields.items():
                if isinstance(value, str):
                    partition_errors[physical_id] = value
                else:
                    partition_values[physical_id] = pl.Series._from_pyseries(value)
            initial_defaults = {
                physical_id: pl.Series._from_pyseries(value)
                for physical_id, value in raw_initial_defaults.items()
            }
        if (
            columns is None
            and not position_deletes
            and not deletion_vectors
            and not delta_selections
            and not partition_values
            and not partition_errors
            and not initial_defaults
            and file_options.missing_columns_policy == "raise"
        ):
            return None
        return cls(
            columns=columns,
            position_deletes=position_deletes,
            deletion_vectors=deletion_vectors,
            delta_selections=delta_selections,
            partition_values=partition_values,
            partition_errors=partition_errors,
            initial_defaults=initial_defaults,
            missing_columns_policy=file_options.missing_columns_policy,
            extra_columns_policy=file_options.extra_columns_policy,
            cast_columns_policy=file_options.cast_columns_policy,
            row_count=file_options.row_count,
        )

    def expect_columns(
        self,
        reader_schema: Mapping[str, Any] | None,
        first_path: str,
        hive_names: Iterable[str],
    ) -> LakeScanOptions:
        """
        Record the keys a data file may hold.

        Parameters
        ----------
        reader_schema
            Schema given to the scan, as polars serializes it, or ``None``
            when the scan infers its schema from the first file.
        first_path
            First data file of the scan.
        hive_names
            Names of the hive partition columns, which a file may also hold.

        Returns
        -------
        Options whose ``expected_keys`` are set: the field IDs of
        ``columns`` when it is given, otherwise the names of the schema and
        of the hive partition columns.
        """
        if self.columns is not None:
            return dataclasses.replace(
                self,
                expected_keys=frozenset(column.physical_id for column in self.columns),
            )
        names = (
            reader_schema["fields"]
            if reader_schema is not None
            else _footer(first_path, None, by_field_id=False).names
        )
        return dataclasses.replace(
            self, expected_keys=frozenset(names).union(hive_names)
        )

    def check_field_ids(self, paths: Sequence[str]) -> None:
        """
        Check that every data file can be read by field ID.

        Parameters
        ----------
        paths
            Data files of the scan.

        Raises
        ------
        NotImplementedError
            If the scan reads columns by field ID and any of ``paths`` holds
            no parquet field IDs.
        """
        if self.columns is None:
            return
        metadatas = plc.io.parquet_metadata.read_parquet_footers(
            plc.io.SourceInfo(list(paths))
        )
        for path, metadata in zip(paths, metadatas, strict=True):
            _footer(path, metadata, by_field_id=True)

    @property
    def has_deletions(self) -> bool:
        """Whether any rows have to be dropped after reading."""
        return bool(
            self.position_deletes or self.deletion_vectors or self.delta_selections
        )

    def slice(self, start: int, stop: int) -> LakeScanOptions:
        """
        Restrict to the paths in ``range(start, stop)``.

        Deletions and partition values are keyed by the position of a path
        within the scan, so both have to be re-keyed when a task takes only
        part of the paths.

        Parameters
        ----------
        start
            Index of the first path to keep.
        stop
            Index one past the last path to keep.

        Returns
        -------
        Options covering the selected paths.
        """
        return dataclasses.replace(
            self,
            position_deletes={
                i - start: v
                for i, v in self.position_deletes.items()
                if start <= i < stop
            },
            deletion_vectors={
                i - start: v
                for i, v in self.deletion_vectors.items()
                if start <= i < stop
            },
            delta_selections={
                i - start: v
                for i, v in self.delta_selections.items()
                if start <= i < stop
            },
            partition_values={
                physical_id: series.slice(start, stop - start)
                for physical_id, series in self.partition_values.items()
            },
            row_count=None,
        )

    def __hash__(self) -> int:
        """Hash of every option."""
        return hash(
            (
                self.columns,
                tuple(sorted(self.position_deletes.items())),
                tuple(sorted(self.deletion_vectors.items())),
                tuple(sorted(self.delta_selections)),
                tuple(sorted(self.partition_values)),
                tuple(sorted(self.partition_errors.items())),
                tuple(sorted(self.initial_defaults)),
                self.missing_columns_policy,
                self.extra_columns_policy,
                self.row_count,
                self.expected_keys,
            )
        )

    def __eq__(self, other: Any) -> bool:
        """Whether two sets of options agree."""
        return (
            isinstance(other, LakeScanOptions)
            and self.columns == other.columns
            and self.position_deletes == other.position_deletes
            and self.deletion_vectors == other.deletion_vectors
            and _series_mapping_eq(self.delta_selections, other.delta_selections)
            and _series_mapping_eq(self.partition_values, other.partition_values)
            and self.partition_errors == other.partition_errors
            and _series_mapping_eq(self.initial_defaults, other.initial_defaults)
            and self.missing_columns_policy == other.missing_columns_policy
            and self.extra_columns_policy == other.extra_columns_policy
            and self.cast_columns_policy == other.cast_columns_policy
            and self.row_count == other.row_count
            and self.expected_keys == other.expected_keys
        )


def _series_mapping_eq(
    left: Mapping[int, pl.Series], right: Mapping[int, pl.Series]
) -> bool:
    return left.keys() == right.keys() and all(
        left_series.equals(right_series)
        for left_series, right_series in zip(left.values(), right.values(), strict=True)
    )


def read_lake_files(
    paths: Sequence[str],
    lake_options: LakeScanOptions,
    schema: Schema,
    with_columns: Sequence[str] | None,
    cached_parquet_info: Sequence[CachedParquetInfo] | None = None,
    predicate: expr.Expr | None = None,
    *,
    with_source_index: bool = False,
    stream: Stream,
) -> tuple[DataFrame, list[int], plc.Column | None, bool]:
    """
    Read the data files of an Iceberg or Delta scan onto the table schema.

    Parameters
    ----------
    paths
        Data files of the scan, in scan order.
    lake_options
        Options of the scan.
    schema
        Schema the scan must produce.
    with_columns
        Columns to project, or ``None`` for all of them.
    cached_parquet_info
        Footers already read for ``paths``, in the same order, or ``None`` to
        read them here.
    predicate
        Predicate, in terms of the table schema, to filter the files with
        while reading them, or ``None`` to read every row.
    with_source_index
        Whether to return the index into ``paths`` of the file each row was
        read from when ``predicate`` is given.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    tuple of:
        - The concatenated frame.
        - The number of rows each path holds before ``predicate`` is applied.
        - The source index of each row, or ``None`` if ``with_source_index`` is ``False``.
        - Whether ``predicate`` was applied exactly.
    """
    by_field_id = lake_options.columns is not None
    projected = (
        frozenset(schema)
        if with_columns is None
        else frozenset(schema).intersection(with_columns)
    )
    wanted: dict[Any, str]
    if lake_options.columns is not None:
        wanted = {
            column.physical_id: column.name
            for column in lake_options.columns
            if column.name in projected
            and column.physical_id not in lake_options.partition_values
        }
    else:
        wanted = {name: name for name in schema if name in projected}

    metadatas: Sequence[plc.io.parquet_metadata.FileMetaData | None] = (
        [None] * len(paths)
        if cached_parquet_info is None
        else [info.file_metadata for info in cached_parquet_info]
    )
    footers = [
        _footer(path, metadata, by_field_id=by_field_id)
        for path, metadata in zip(paths, metadatas, strict=True)
    ]
    rows_per_path = [footer.num_rows for footer in footers]
    expected_keys = lake_options.expected_keys
    if expected_keys is not None:
        for path, footer in zip(paths, footers, strict=True):
            extra = [
                name
                for name, key in zip(footer.names, footer.keys, strict=True)
                if key not in expected_keys
            ]
            if extra:
                raise pl.exceptions.SchemaError(
                    f"extra column in file outside of expected schema: {extra[0]}, "
                    "hint: specify this column in the schema, or pass "
                    "extra_columns='ignore' in scan options. "
                    f"File containing extra column: '{path}'."
                )

    predicate_columns = (
        {
            node.name: node
            for node in traversal([predicate])
            if isinstance(node, expr.Col)
        }
        if predicate is not None
        else {}
    )
    track_source = predicate is not None and with_source_index
    frames: list[DataFrame] = []
    source_indices: list[plc.Column] = []
    exact = True
    start = 0
    for (keys, _), group in itertools.groupby(
        footers, key=lambda footer: footer.group_key(wanted)
    ):
        stop = start + len(list(group))
        present = frozenset(keys)
        read_keys = [key for key in wanted if key in present]
        filters = None
        if predicate is not None:
            filters, group_exact = _group_filter(
                predicate,
                predicate_columns,
                paths[start],
                footers[start],
                read_keys,
                wanted,
                schema,
                stream=stream,
            )
            exact = exact and group_exact
        frame, source_index = _read_group(
            paths[start:stop],
            read_keys,
            footers[start],
            rows_per_path[start:stop],
            wanted,
            schema,
            lake_options,
            filters,
            start if track_source else None,
            by_field_id=by_field_id,
            stream=stream,
        )
        frames.append(frame)
        if source_index is not None:
            source_indices.append(source_index)
        start = stop

    source_index = (
        None
        if not track_source
        else source_indices[0]
        if len(source_indices) == 1
        else plc.concatenate.concatenate(source_indices, stream=stream)
    )
    if len(frames) == 1:
        return frames[0], rows_per_path, source_index, exact
    names = list(frames[0].column_map)
    if not names:
        return (
            DataFrame(
                [], num_rows=sum(frame.num_rows for frame in frames), stream=stream
            ),
            rows_per_path,
            source_index,
            exact,
        )
    return (
        DataFrame.from_table(
            plc.concatenate.concatenate(
                [
                    plc.Table([frame.column_map[name].obj for name in names])
                    for frame in frames
                ],
                stream=stream,
            ),
            names,
            [schema[name] for name in names],
            stream=stream,
        ),
        rows_per_path,
        source_index,
        exact,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _Footer:
    """What the parquet footer of one data file says about its columns."""

    names: tuple[str, ...]
    keys: tuple[Any, ...]
    signatures: tuple[_Layout, ...]
    field_ids: tuple[_FieldIds | None, ...]
    num_rows: int

    def group_key(
        self, wanted: Mapping[Any, str]
    ) -> tuple[tuple[Any, ...], tuple[tuple[_Layout, _FieldIds | None], ...]]:
        """
        Key on which consecutive files can be read together.

        Parameters
        ----------
        wanted
            Maps the key of each column the scan must produce to its name in
            the table schema.

        Returns
        -------
        The top-level keys of the file, and the names, physical and logical
        types and field IDs of each wanted column it holds and of its nested
        fields. The parquet reader requires the selected columns of its
        sources to agree on their types and on the names of their nested
        fields. Files that agree on the field IDs as well can be read by
        name even when the scan is keyed by field ID.
        """
        return self.keys, tuple(
            (signature, field_ids)
            for key, signature, field_ids in zip(
                self.keys, self.signatures, self.field_ids, strict=True
            )
            if key in wanted
        )


def _element_layout(
    element: plc.io.parquet_metadata.SchemaElement,
) -> _ElementLayout:
    """Name, physical and logical type of a parquet schema element."""
    logical_type = element.logical_type
    if logical_type is None:
        return (
            element.name,
            element.type,
            element.type_length,
            element.num_children,
            None,
        )
    decimal_type = logical_type.decimal_type
    temporal_type = (
        logical_type.time_type
        if logical_type.time_type is not None
        else logical_type.timestamp_type
    )
    int_type = logical_type.int_type
    return (
        element.name,
        element.type,
        element.type_length,
        element.num_children,
        (
            logical_type.type,
            None
            if decimal_type is None
            else (decimal_type.scale, decimal_type.precision),
            None if temporal_type is None else temporal_type.unit,
            None if int_type is None else (int_type.bit_width, int_type.is_signed),
        ),
    )


def _footer(
    path: str,
    metadata: plc.io.parquet_metadata.FileMetaData | None,
    *,
    by_field_id: bool,
) -> _Footer:
    """
    Top-level columns and row count of a parquet file.

    Raises
    ------
    NotImplementedError
        If ``by_field_id`` and a field that polars identifies by its field ID
        has none, sibling fields share a field ID, or a list uses a legacy
        layout that the parquet reader and polars read differently.
    """
    if metadata is None:
        metadata = plc.io.parquet_metadata.read_parquet_footers(
            plc.io.SourceInfo([path])
        )[0]
    schema = metadata.schema

    def check_unique(children: Sequence[_FieldIds]) -> None:
        ids = [field_id for field_id, _ in children if field_id is not None]
        if len(ids) != len(set(ids)):
            raise NotImplementedError(
                f"Iceberg data file with duplicate field IDs: {path}"
            )

    def nested_field_ids(index: int, *, list_parent: str | None) -> _FieldIds:
        element = schema[index]
        if element.is_stub():
            if list_parent is not None and element.name in (
                "array",
                f"{list_parent}_tuple",
            ):  # pragma: no cover; pyarrow cannot write this layout
                raise NotImplementedError(
                    f"Iceberg data file with a legacy list layout: {path}"
                )
            return nested_field_ids(element.children_idx[0], list_parent=list_parent)
        if element.field_id is None and list_parent is None:
            raise NotImplementedError(
                f"Iceberg data file without parquet field IDs: {path}"
            )
        logical_type = element.logical_type
        is_list = logical_type is not None and logical_type.type in (
            plc.io.parquet_metadata.LogicalTypeId.LIST,
            plc.io.parquet_metadata.LogicalTypeId.MAP,
        )
        children = tuple(
            nested_field_ids(child, list_parent=element.name if is_list else None)
            for child in element.children_idx
        )
        check_unique(children)
        return element.field_id, children

    def subtree(index: int) -> Iterable[plc.io.parquet_metadata.SchemaElement]:
        element = schema[index]
        yield element
        for child in element.children_idx:
            yield from subtree(child)

    top_level = schema[0].children_idx
    names = tuple(schema[index].name for index in top_level)
    signatures = tuple(
        tuple(_element_layout(element) for element in subtree(index))
        for index in top_level
    )
    if not by_field_id:
        return _Footer(
            names=names,
            keys=names,
            signatures=signatures,
            field_ids=(None,) * len(top_level),
            num_rows=metadata.num_rows,
        )
    field_ids = tuple(nested_field_ids(index, list_parent=None) for index in top_level)
    check_unique(field_ids)
    return _Footer(
        names=names,
        keys=tuple(field_id for field_id, _ in field_ids),
        signatures=signatures,
        field_ids=field_ids,
        num_rows=metadata.num_rows,
    )


def _group_filter(
    predicate: expr.Expr,
    predicate_columns: Mapping[str, expr.Col],
    path: str,
    footer: _Footer,
    read_keys: Sequence[Any],
    wanted: Mapping[Any, str],
    schema: Schema,
    *,
    stream: Stream,
) -> tuple[plc.expressions.Expression | None, bool]:
    """
    Translate a predicate into a parquet filter for one run of files.

    Parameters
    ----------
    predicate
        Predicate in terms of the table schema.
    predicate_columns
        Maps the name of each column ``predicate`` refers to onto its
        ``Col`` node in ``predicate``.
    path
        First file of the run, whose metadata gives the column types of the
        whole run. It is only read when ``predicate`` refers to a column in
        ``read_keys``.
    footer
        Footer of ``path``, used to map keys to the names the files hold.
    read_keys
        Keys of the wanted columns that the files hold: field IDs for an
        Iceberg scan read by field ID, otherwise column names.
    wanted
        Maps the key of each column the scan must produce to its name in the
        table schema.
    schema
        Schema the scan must produce.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    filters
        Parquet filter in terms of the names the files hold, or ``None`` when
        no part of ``predicate`` can be applied while reading. The scalars
        it holds are created on ``stream``, and its data is valid on
        ``stream``.
    exact
        Whether ``filters`` is equivalent to ``predicate``. When it is not,
        the rows read are a superset of the rows that satisfy ``predicate``,
        which must be applied again after reading.
    """
    name_by_key = dict(zip(footer.keys, footer.names, strict=True))
    file_names = {
        wanted[key]: name_by_key[key]
        for key in read_keys
        if wanted[key] in predicate_columns
    }
    file_types = (
        plc.io.parquet_metadata.read_parquet_metadata(plc.io.SourceInfo([path]))
        .schema()
        .column_types()
        if file_names
        else {}
    )
    readable = {
        name: file_name
        for name, file_name in file_names.items()
        if schema[name].id() not in _NESTED_TYPES
        and file_types.get(file_name) == schema[name].plc_type
    }
    renames: dict[expr.Expr, expr.Expr] = {
        predicate_columns[name]: expr.Col(predicate_columns[name].dtype, file_name)
        for name, file_name in readable.items()
        if file_name != name
    }
    (renamed,) = replace([predicate], renames)
    filters, residual = to_parquet_filter(
        renamed,
        stream=stream,
        unreadable_columns=frozenset(predicate_columns).difference(readable),
    )
    return filters, filters is not None and residual is None


def _check_cast(
    name: str,
    incoming: plc.DataType,
    target: DataType,
    cast_columns_policy: Mapping[str, Any],
) -> None:
    """
    Check that a column read from a file may be cast to the table type.

    Parameters
    ----------
    name
        Name of the column in the table schema.
    incoming
        Type the column was read with.
    target
        Type of the column in the table schema.
    cast_columns_policy
        Which casts from the physical to the table type are permitted.

    Raises
    ------
    polars.exceptions.SchemaError
        If ``cast_columns_policy`` forbids an integer, float or datetime
        unit cast that the column needs. Other type differences are cast
        without checking, since cudf and polars do not represent every
        type the same way.
    """
    source = incoming.id()
    dest = target.plc_type.id()
    if source == dest:
        return
    hint: str | None
    if source in _INTEGER_TYPES and dest in _INTEGER_TYPES:
        source_bits, source_signed = _INTEGER_TYPES[source]
        dest_bits, dest_signed = _INTEGER_TYPES[dest]
        if not cast_columns_policy["integer_upcast"]:
            hint = _cast_hint("integer_cast='upcast'")
        elif dest_bits > source_bits and (dest_signed or not source_signed):
            hint = None
        else:
            hint = "incoming dtype cannot safely cast to target dtype"
    elif source in _FLOAT_WIDTHS and dest in _FLOAT_WIDTHS:
        policy, option = (
            ("float_upcast", "upcast")
            if _FLOAT_WIDTHS[dest] > _FLOAT_WIDTHS[source]
            else ("float_downcast", "downcast")
        )
        hint = (
            None
            if cast_columns_policy[policy]
            else _cast_hint(f"float_cast='{option}'")
        )
    elif source in _INTEGER_TYPES and dest in _FLOAT_WIDTHS:
        hint = (
            None
            if cast_columns_policy["integer_to_float_cast"]
            else _cast_hint("integer_cast='allow-float'")
        )
    elif source in _TIMESTAMP_DIGITS and dest in _TIMESTAMP_DIGITS:
        source_digits = _TIMESTAMP_DIGITS[source]
        dest_digits = _TIMESTAMP_DIGITS[dest]
        if source_digits == 9:
            policy, option = "datetime_nanoseconds_downcast", "nanosecond-downcast"
        elif source_digits == 6 and dest_digits == 3:
            policy, option = "datetime_microseconds_downcast", "microsecond-downcast"
        elif source_digits == 6:
            policy, option = "datetime_microseconds_upcast", "microsecond-upcast"
        else:
            policy, option = "datetime_milliseconds_upcast", "millisecond-upcast"
        hint = (
            None
            if cast_columns_policy[policy]
            else _cast_hint(f"datetime_cast='{option}'")
        )
    else:
        return
    if hint is not None:
        raise pl.exceptions.SchemaError(
            f"data type mismatch for column {name}: "
            f"incoming: {source.name} != target: {target.polars_type}, {hint}"
        )


def _cast_hint(option: str) -> str:
    return f"hint: pass cast_options=pl.ScanCastOptions({option})"


_NESTED_TYPES = frozenset({plc.TypeId.STRUCT, plc.TypeId.LIST})
_LIST_ELEMENT_DEFAULT_ID = 2**32 - 1


def _cast_to_table_type(
    name: str,
    column: plc.Column,
    children: Sequence[plc.io.types.ColumnNameSpec],
    target: DataType,
    cast_columns_policy: Mapping[str, Any],
    mapping: tuple[_FieldIds, IcebergColumn] | None,
    *,
    stream: Stream,
) -> plc.Column:
    """
    Cast a column read from a file to its type in the table schema.

    Parameters
    ----------
    name
        Name of the top-level column in the table schema.
    column
        Column as read from the file.
    children
        Names of the children of ``column``, as the parquet reader
        reports them.
    target
        Type of the column in the table schema.
    cast_columns_policy
        Which casts from the physical to the table type are permitted.
    mapping
        For a scan read by field ID, the field IDs of ``column`` and its
        nested fields in the file, and the Iceberg column that ``target``
        was built from. ``None`` to match nested fields by name.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    The column with the type of ``target``. Struct fields are matched by
    field ID when ``mapping`` is given and by name otherwise, put in the
    order of ``target`` and cast field by field.

    Raises
    ------
    polars.exceptions.SchemaError
        If ``column`` and ``target`` do not nest in the same way, if a
        struct field is missing or extra and ``cast_columns_policy`` forbids
        that, if the element of a list has a different field ID than in
        the table schema, or if ``_check_cast`` rejects the cast of a leaf.
    """
    source_id = column.type().id()
    target_id = target.id()
    if source_id not in _NESTED_TYPES and target_id not in _NESTED_TYPES:
        _check_cast(name, column.type(), target, cast_columns_policy)
        return Column(column, dtype=target).astype(target, stream).obj
    if source_id != target_id:
        raise pl.exceptions.SchemaError(
            f"data type mismatch for column {name}: "
            f"incoming: {source_id.name} != target: {target.polars_type}"
        )
    if target_id == plc.TypeId.LIST:
        (_, (_, element_children)) = children
        element_mapping = None
        if mapping is not None:
            (_, file_elements), iceberg_column = mapping
            (iceberg_element,) = iceberg_column.children
            file_element = file_elements[0] if file_elements else (None, ())
            element_ids = {file_element[0], iceberg_element.physical_id} - {
                None,
                _LIST_ELEMENT_DEFAULT_ID,
            }
            if len(element_ids) > 1:
                raise pl.exceptions.SchemaError(
                    f"data type mismatch for column {name}: "
                    "physical ID mismatch for list values column"
                )
            element_mapping = (file_element, iceberg_element)
        new_children = [
            column.child(0),
            _cast_to_table_type(
                name,
                column.child(1),
                element_children,
                target.children[0],
                cast_columns_policy,
                element_mapping,
                stream=stream,
            ),
        ]
    else:
        fields = cast("pl.Struct", target.polars_type).fields
        file_fields: Sequence[_FieldIds | None]
        incoming_keys: list[Any]
        target_fields: list[tuple[Any, IcebergColumn | None]]
        if mapping is None:
            file_fields = [None] * len(children)
            incoming_keys = [field_name for field_name, _ in children]
            target_fields = [(field.name, None) for field in fields]
        else:
            (_, file_fields), iceberg_column = mapping
            incoming_keys = [field_id for field_id, _ in file_fields]
            target_fields = [
                (iceberg_field.physical_id, iceberg_field)
                for iceberg_field in iceberg_column.children
            ]
        position = {key: index for index, key in enumerate(incoming_keys)}
        if cast_columns_policy["extra_struct_fields"] == "raise":
            target_keys = frozenset(key for key, _ in target_fields)
            extra = [
                field_name
                for (field_name, _), key in zip(children, incoming_keys, strict=True)
                if key not in target_keys
            ]
            if extra:
                raise pl.exceptions.SchemaError(
                    f"data type mismatch for column {name}: "
                    f"encountered extra struct field: {extra[0]}, "
                    "hint: specify this field in the schema, or pass "
                    "cast_options=pl.ScanCastOptions(extra_struct_fields='ignore')"
                )
        new_children = []
        for field, field_dtype, (key, iceberg_field) in zip(
            fields, target.children, target_fields, strict=True
        ):
            index = position.get(key)
            if index is not None:
                _, field_children = children[index]
                file_field = file_fields[index]
                new_children.append(
                    _cast_to_table_type(
                        name,
                        column.child(index),
                        field_children,
                        field_dtype,
                        cast_columns_policy,
                        None
                        if file_field is None or iceberg_field is None
                        else (file_field, iceberg_field),
                        stream=stream,
                    )
                )
            elif cast_columns_policy["missing_struct_fields"] == "raise":
                raise pl.exceptions.SchemaError(
                    f"data type mismatch for column {name}: "
                    f"encountered missing struct field: {field.name}, "
                    + _cast_hint("missing_struct_fields='insert'")
                )
            else:
                new_children.append(
                    _null_column(
                        field_dtype, column.offset() + column.size(), stream=stream
                    )
                )
    return plc.Column(
        column.type(),
        column.size(),
        None,
        column.null_mask(),
        column.null_count(),
        column.offset(),
        new_children,
    )


def _null_column(dtype: DataType, size: int, *, stream: Stream) -> plc.Column:
    """An all-null column of any type, including nested types."""
    type_id = dtype.id()
    if type_id not in _NESTED_TYPES:
        return plc.Column.from_scalar(
            plc.Scalar.from_py(None, dtype.plc_type, stream=stream),
            size,
            stream=stream,
        )
    if type_id == plc.TypeId.LIST:
        children = [
            plc.Column.from_scalar(
                plc.Scalar.from_py(0, plc.DataType(plc.TypeId.INT32), stream=stream),
                size + 1,
                stream=stream,
            ),
            _null_column(dtype.children[0], 0, stream=stream),
        ]
    else:
        children = [
            _null_column(child, size, stream=stream) for child in dtype.children
        ]
    return plc.Column(
        dtype.plc_type,
        size,
        None,
        plc.null_mask.create_null_mask(
            size, plc.types.MaskState.ALL_NULL, stream=stream
        ),
        size,
        0,
        children,
    )


def _read_group(
    paths: Sequence[str],
    present: Sequence[Any],
    footer: _Footer,
    rows_per_path: Sequence[int],
    wanted: Mapping[Any, str],
    schema: Schema,
    lake_options: LakeScanOptions,
    filters: plc.expressions.Expression | None,
    source_offset: int | None,
    *,
    by_field_id: bool,
    stream: Stream,
) -> tuple[DataFrame, plc.Column | None]:
    """
    Read one run of same-shaped files and map it onto the table schema.

    Parameters
    ----------
    paths
        Data files of the run, in scan order. They hold the same top-level
        columns, and the wanted ones have the same types and the same names
        and field IDs throughout, so the run is read by column name.
    present
        Keys of the wanted columns that the files hold: field IDs when
        ``by_field_id``, otherwise column names. The files are only read
        when this is non-empty.
    footer
        Footer of the first file of the run, used to map the column names
        the reader returns back to their keys.
    rows_per_path
        Number of rows each file holds before ``filters`` is applied.
    wanted
        Maps the key of each column the scan must produce to its name in the
        table schema, in output order.
    schema
        Schema the scan must produce.
    lake_options
        Options of the scan, giving the policy and defaults for columns the
        files do not hold.
    filters
        Parquet filter to apply while reading, in terms of the names the
        files hold, or ``None`` to read every row.
    source_offset
        Index within the scan of the first file of the run, or ``None`` when
        the source of each row is not needed.
    by_field_id
        Whether ``present`` and ``wanted`` are keyed by Iceberg field ID
        rather than by column name.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    frame
        The rows read, with the columns of ``wanted`` renamed and cast to
        the table schema. Columns the files do not hold are filled with
        their initial default, or with nulls.
    source_index
        For each row of ``frame``, the index within the scan of the file it
        was read from, or ``None`` when ``source_offset`` is ``None``. Its
        data is valid on ``stream``.
    """
    read: dict[Any, tuple[plc.Column, Sequence[plc.io.types.ColumnNameSpec]]] = {}
    num_rows = sum(rows_per_path)
    source_index: plc.Column | None = None
    if present:
        options = (
            plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo(list(paths)))
            .allow_mismatched_pq_schemas(val=True)
            .build()
        )
        name_by_key = dict(zip(footer.keys, footer.names, strict=True))
        options.set_column_names([name_by_key[key] for key in present])
        if filters is not None:
            options.set_filter(filters)
        if source_offset is not None:
            options.enable_prepend_source_index_column(val=True)
        table = plc.io.parquet.read_parquet(options, stream=stream)
        num_rows = table.tbl.num_rows()
        names = table.column_names(include_children=True)
        columns = table.tbl.columns()
        if source_offset is not None:
            source_index = (
                columns[0]
                if source_offset == 0
                else plc.binaryop.binary_operation(
                    columns[0],
                    plc.Scalar.from_py(
                        source_offset, plc.DataType(plc.TypeId.INT32), stream=stream
                    ),
                    plc.binaryop.BinaryOperator.ADD,
                    plc.DataType(plc.TypeId.INT32),
                    stream=stream,
                )
            )
            names = names[1:]
            columns = columns[1:]
        key_by_name = dict(zip(footer.names, footer.keys, strict=True))
        read = {
            key_by_name[name]: (column, children)
            for (name, children), column in zip(names, columns, strict=True)
        }
    elif source_offset is not None:
        (source_index,) = plc.filling.repeat(
            plc.Table(
                [
                    plc.filling.sequence(
                        len(rows_per_path),
                        plc.Scalar.from_py(
                            source_offset, plc.DataType(plc.TypeId.INT32), stream=stream
                        ),
                        plc.Scalar.from_py(
                            1, plc.DataType(plc.TypeId.INT32), stream=stream
                        ),
                        stream=stream,
                    )
                ]
            ),
            plc.Column.from_arrow(
                pl.Series(values=rows_per_path, dtype=pl.Int32()), stream=stream
            ),
            stream=stream,
        ).columns()

    file_field_ids = dict(zip(footer.keys, footer.field_ids, strict=True))
    iceberg_columns = {
        column.physical_id: column for column in lake_options.columns or ()
    }
    frame_columns = []
    for key, name in wanted.items():
        dtype = schema[name]
        if key in read:
            column, children = read[key]
            field_ids = file_field_ids[key]
            obj = _cast_to_table_type(
                name,
                column,
                children,
                dtype,
                lake_options.cast_columns_policy,
                None if field_ids is None else (field_ids, iceberg_columns[key]),
                stream=stream,
            )
            frame_columns.append(Column(obj, name=name, dtype=dtype))
        elif by_field_id and key in lake_options.partition_errors:
            raise pl.exceptions.ComputeError(
                "failed to resolve Iceberg column mapping: "
                "error loading identity transform value from metadata for "
                f"missing field: {lake_options.partition_errors[key]}"
            )
        elif lake_options.missing_columns_policy == "raise":
            raise pl.exceptions.ColumnNotFoundError(f"did not find column {name}")
        else:
            default = lake_options.initial_defaults.get(key) if by_field_id else None
            if default is None:
                obj = _null_column(dtype, num_rows, stream=stream)
                frame_columns.append(Column(obj, name=name, dtype=dtype))
            else:
                (obj,) = plc.filling.repeat(
                    plc.Table([plc.Column.from_arrow(default, stream=stream)]),
                    num_rows,
                    stream=stream,
                ).columns()
                frame_columns.append(
                    Column(obj, name=name, dtype=dtype).astype(dtype, stream)
                )
    if not frame_columns:
        return DataFrame([], num_rows=num_rows, stream=stream), source_index
    return DataFrame(frame_columns, stream=stream), source_index
