# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa

import pylibcudf as plc


class TestApplyBooleanMask:
    def test_non_null_mask(self):
        table = plc.Table(
            [
                plc.Column.from_arrow(
                    pa.array([10, 40, 70, 5, 2, 10], type=pa.int32())
                ),
                plc.Column.from_arrow(
                    pa.array([10, 40, 70, 5, 2, 10], type=pa.float64())
                ),
            ]
        )
        mask = plc.Column.from_arrow(
            pa.array([True, False, True, False, True, False], type=pa.bool_())
        )
        result = plc.stream_compaction.apply_boolean_mask(table, mask)
        got = result.to_arrow()
        assert got.num_rows == 3
        assert got.column(0).to_pylist() == [10, 70, 2]
        assert got.column(1).to_pylist() == [10, 70, 2]

    def test_null_mask(self):
        table = plc.Table(
            [
                plc.Column.from_arrow(
                    pa.array([10, 40, 70, 5, 2, 10], type=pa.int32())
                )
            ]
        )
        mask = plc.Column.from_arrow(
            pa.array([None, False, True, False, True, False], type=pa.bool_())
        )
        result = plc.stream_compaction.apply_boolean_mask(table, mask)
        got = result.to_arrow()
        assert got.num_rows == 2
        assert got.column(0).to_pylist() == [70, 2]

    def test_empty_mask(self):
        table = plc.Table(
            [plc.Column.from_arrow(pa.array([10, 40, 70], type=pa.int32()))]
        )
        mask = plc.Column.from_arrow(pa.array([], type=pa.bool_()))
        result = plc.stream_compaction.apply_boolean_mask(table, mask)
        assert result.num_rows() == 0

    def test_all_true(self):
        table = plc.Table(
            [plc.Column.from_arrow(pa.array([1, 2, 3], type=pa.int32()))]
        )
        mask = plc.Column.from_arrow(
            pa.array([True, True, True], type=pa.bool_())
        )
        result = plc.stream_compaction.apply_boolean_mask(table, mask)
        got = result.to_arrow()
        assert got.num_rows == 3
        assert got.column(0).to_pylist() == [1, 2, 3]

    def test_all_false(self):
        table = plc.Table(
            [plc.Column.from_arrow(pa.array([1, 2, 3], type=pa.int32()))]
        )
        mask = plc.Column.from_arrow(
            pa.array([False, False, False], type=pa.bool_())
        )
        result = plc.stream_compaction.apply_boolean_mask(table, mask)
        assert result.num_rows() == 0

    def test_string_column(self):
        table = plc.Table(
            [
                plc.Column.from_arrow(
                    pa.array(
                        [
                            "This",
                            "is",
                            "the",
                            "a",
                            "k12",
                            None,
                            "table",
                            "column",
                        ],
                        type=pa.string(),
                    )
                )
            ]
        )
        mask = plc.Column.from_arrow(
            pa.array(
                [True, True, None, True, False, True, False, True],
                type=pa.bool_(),
            )
        )
        result = plc.stream_compaction.apply_boolean_mask(table, mask)
        got = result.to_arrow()
        assert got.column(0).to_pylist() == ["This", "is", "a", None, "column"]

    def test_no_null_input(self):
        values = [
            9668,
            9590,
            9526,
            9205,
            9434,
            9347,
            9160,
            9569,
            9143,
            9807,
            9606,
            9446,
            9279,
            9822,
            9691,
        ]
        mask_vals = [
            False,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            False,
            True,
            False,
            True,
        ]
        table = plc.Table(
            [plc.Column.from_arrow(pa.array(values, type=pa.int32()))]
        )
        mask = plc.Column.from_arrow(pa.array(mask_vals, type=pa.bool_()))
        result = plc.stream_compaction.apply_boolean_mask(table, mask)
        expected = [v for v, m in zip(values, mask_vals, strict=True) if m]
        assert result.to_arrow().column(0).to_pylist() == expected


class TestListsApplyBooleanMask:
    def test_basic(self):
        input_col = plc.Column.from_arrow(
            pa.array(
                [[0, 1, 2, 3], [4, 5], [6, 7, 8, 9]],
                type=pa.list_(pa.int32()),
            )
        )
        mask_col = plc.Column.from_arrow(
            pa.array(
                [
                    [True, False, True, False],
                    [True, False],
                    [True, False, True, False],
                ],
                type=pa.list_(pa.bool_()),
            )
        )
        result = plc.lists.apply_boolean_mask(input_col, mask_col)
        assert result.to_arrow().to_pylist() == [[0, 2], [4], [6, 8]]

    def test_null_list_rows(self):
        input_col = plc.Column.from_arrow(
            pa.array(
                [[0, 1, 2, 3], None, [6, 7, 8, 9]],
                type=pa.list_(pa.int32()),
            )
        )
        mask_col = plc.Column.from_arrow(
            pa.array(
                [[True, False, True, False], None, [True, False, True, False]],
                type=pa.list_(pa.bool_()),
            )
        )
        result = plc.lists.apply_boolean_mask(input_col, mask_col)
        assert result.to_arrow().to_pylist() == [[0, 2], None, [6, 8]]

    def test_empty(self):
        input_col = plc.Column.from_arrow(
            pa.array([], type=pa.list_(pa.int32()))
        )
        mask_col = plc.Column.from_arrow(
            pa.array([], type=pa.list_(pa.bool_()))
        )
        result = plc.lists.apply_boolean_mask(input_col, mask_col)
        assert result.to_arrow().to_pylist() == []

    def test_all_true(self):
        input_col = plc.Column.from_arrow(
            pa.array([[1, 2, 3], [4, 5]], type=pa.list_(pa.int32()))
        )
        mask_col = plc.Column.from_arrow(
            pa.array(
                [[True, True, True], [True, True]],
                type=pa.list_(pa.bool_()),
            )
        )
        result = plc.lists.apply_boolean_mask(input_col, mask_col)
        assert result.to_arrow().to_pylist() == [[1, 2, 3], [4, 5]]

    def test_all_false(self):
        input_col = plc.Column.from_arrow(
            pa.array([[1, 2, 3], [4, 5]], type=pa.list_(pa.int32()))
        )
        mask_col = plc.Column.from_arrow(
            pa.array(
                [[False, False, False], [False, False]],
                type=pa.list_(pa.bool_()),
            )
        )
        result = plc.lists.apply_boolean_mask(input_col, mask_col)
        assert result.to_arrow().to_pylist() == [[], []]


class TestApplyDeletionMask:
    def test_basic(self):
        table = plc.Table(
            [plc.Column.from_arrow(pa.array([1, 2, 3, 4, 5], type=pa.int32()))]
        )
        mask = plc.Column.from_arrow(
            pa.array([True, False, True, False, True], type=pa.bool_())
        )
        result = plc.stream_compaction.apply_deletion_mask(table, mask)
        got = result.to_arrow()
        assert got.num_rows == 2
        assert got.column(0).to_pylist() == [2, 4]

    def test_null_mask(self):
        table = plc.Table(
            [plc.Column.from_arrow(pa.array([1, 2, 3, 4, 5], type=pa.int32()))]
        )
        mask = plc.Column.from_arrow(
            pa.array([False, None, False, None, False], type=pa.bool_())
        )
        result = plc.stream_compaction.apply_deletion_mask(table, mask)
        got = result.to_arrow()
        assert got.num_rows == 3
        assert got.column(0).to_pylist() == [1, 3, 5]

    def test_all_true(self):
        table = plc.Table(
            [plc.Column.from_arrow(pa.array([1, 2, 3], type=pa.int32()))]
        )
        mask = plc.Column.from_arrow(
            pa.array([True, True, True], type=pa.bool_())
        )
        result = plc.stream_compaction.apply_deletion_mask(table, mask)
        assert result.num_rows() == 0

    def test_all_false(self):
        table = plc.Table(
            [plc.Column.from_arrow(pa.array([1, 2, 3], type=pa.int32()))]
        )
        mask = plc.Column.from_arrow(
            pa.array([False, False, False], type=pa.bool_())
        )
        result = plc.stream_compaction.apply_deletion_mask(table, mask)
        got = result.to_arrow()
        assert got.num_rows == 3
        assert got.column(0).to_pylist() == [1, 2, 3]

    def test_empty(self):
        table = plc.Table(
            [plc.Column.from_arrow(pa.array([], type=pa.int32()))]
        )
        mask = plc.Column.from_arrow(pa.array([], type=pa.bool_()))
        result = plc.stream_compaction.apply_deletion_mask(table, mask)
        assert result.num_rows() == 0


class TestListsApplyDeletionMask:
    def test_basic(self):
        input_col = plc.Column.from_arrow(
            pa.array([[0, 1, 2], [3, 4], [5, 6, 7]], type=pa.list_(pa.int32()))
        )
        mask_col = plc.Column.from_arrow(
            pa.array(
                [[True, False, True], [False, True], [True, True, False]],
                type=pa.list_(pa.bool_()),
            )
        )
        result = plc.lists.apply_deletion_mask(input_col, mask_col)
        got = result.to_arrow().to_pylist()
        assert got == [[1], [3], [7]]

    def test_all_true(self):
        input_col = plc.Column.from_arrow(
            pa.array([[1, 2, 3], [4, 5]], type=pa.list_(pa.int32()))
        )
        mask_col = plc.Column.from_arrow(
            pa.array(
                [[True, True, True], [True, True]],
                type=pa.list_(pa.bool_()),
            )
        )
        result = plc.lists.apply_deletion_mask(input_col, mask_col)
        got = result.to_arrow().to_pylist()
        assert got == [[], []]

    def test_all_false(self):
        input_col = plc.Column.from_arrow(
            pa.array([[1, 2, 3], [4, 5]], type=pa.list_(pa.int32()))
        )
        mask_col = plc.Column.from_arrow(
            pa.array(
                [[False, False, False], [False, False]],
                type=pa.list_(pa.bool_()),
            )
        )
        result = plc.lists.apply_deletion_mask(input_col, mask_col)
        got = result.to_arrow().to_pylist()
        assert got == [[1, 2, 3], [4, 5]]
