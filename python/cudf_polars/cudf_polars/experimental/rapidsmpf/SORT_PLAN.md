# Plan: Sort support for RapidsMPF runtime

This document outlines a plan to implement multi-partition Sort for the rapidsmpf streaming runtime, enabling `Sort` to be lowered to `ShuffleSorted` and executed via a new `sort_actor` instead of falling back to single-partition execution.

---

## 1. Current state

- **Tasks runtime** (`experimental/sort.py`):
  - `Sort` is lowered to: local sort → `ShuffleSorted` → final local sort.
  - `ShuffleSorted` uses: per-partition sample candidates → allgather → global boundaries → partition by boundaries (find_sort_splits + split_and_pack) → shuffle.
  - Helpers: `find_sort_splits`, `_select_local_split_candidates`, `_get_final_sort_boundaries`, `_sort_boundaries_graph`.

- **RapidsMPF today**:
  - In `lower_ir_node.register(Sort)`, when `runtime == "rapidsmpf"` we always `_lower_ir_fallback` (single partition).
  - In `lower_ir_node.register(ShuffleSorted)`, when streaming + rapidsmpf we also fall back.
  - `ShuffleManager` (in `collectives/shuffle.py`) uses **hash** partitioning only: `partition_and_pack(table, columns_to_hash, num_partitions)`.

- **Reusable pieces**:
  - `ShufflerAsync.insert(chunks: Mapping[int, PackedData])` — accepts partition_id → PackedData. So boundary-based partitioning (split_and_pack) can feed the same shuffler; only the way we produce the map changes.
  - `split_and_pack(table, splits, stream, br)` in rapidsmpf gives `dict[int, PackedData]`, same shape as hash partition output.
  - `allgather_reduce` in `utils.py` does allgather of packed scalars and local sum; for "global histogram" we can allgather local statistics (e.g. sampled boundaries or histogram bins) and reduce/merge in Python.
  - `fanout_node_unbounded` buffers into `SpillableMessages`; the sort actor can use a single `SpillableMessages` to buffer all incoming chunks.

---

## 2. High-level design

- **New file**: `rapidsmpf/sort.py` (or `rapidsmpf/collectives/sort.py`).
- **Two main changes elsewhere**:
  1. **Lowering** (`experimental/sort.py`): Allow lowering `Sort` → `ShuffleSorted` when runtime is rapidsmpf (remove the rapidsmpf fallback that forces single partition).
  2. **Shuffle path**: Support boundary-based partitioning in addition to hash. Options:
     - **A)** Extend `ShuffleManager` with e.g. `insert_chunk_sorted(chunk, splits)` that uses `split_and_pack(chunk.table_view(), splits, ...)` and then `shuffler.insert(partitioned_chunks)`.
     - **B)** Keep a separate "sort shuffle" path that uses the same `ShufflerAsync` but a different insert path (sort_actor produces the partition map via split_and_pack and calls the shuffler directly).

Option **A** keeps one ShuffleManager; the sort_actor would compute splits per chunk and call something like `shuffle.insert_chunk_sorted(chunk, splits)`. Option **B** avoids changing ShuffleManager: sort_actor holds a ShufflerAsync (or ShuffleManager) and does split_and_pack + insert itself. Either is fine; **A** is slightly cleaner for reuse.

---

## 3. Sort actor: data flow

1. **Phase 1 – Buffer and collect statistics**
   - Single input channel `ch_in`, single output channel `ch_out`.
   - Receive metadata; send output metadata (partitioning after sort is by "sort boundaries", not hash; may need a new `Partitioning` variant or reuse "inherit" after defining boundaries).
   - Loop: for each message from `ch_in`, extract `TableChunk`, optionally make_available_and_spill.
   - **Before** pushing into the spillable buffer: compute and accumulate **min/max** (and optionally a local histogram or sample) for the sort-by columns only. Store in a lightweight structure (e.g. per-chunk min/max, or sampled rows).
   - Push chunk into a **SpillableMessages** container (same idea as `fanout_node_unbounded`), so all chunks are buffered and can be replayed.

2. **Phase 2 – Global boundaries**
   - When input is finished (recv returns None):
     - Build a **local** representation of the sort key distribution:
       - **Option (a)** Reuse tasks semantics: from all buffered sort-column data, produce "split candidates" (e.g. rows at indices `i * local_rows // num_partitions`). That implies concatenating sort columns from all chunks (or sampling from each chunk and merging). Then allgather these candidates and run `_get_final_sort_boundaries` to get a global boundaries table.
       - **Option (b)** Use min/max + local histogram bins: define local bins, count per bin, allgather bin counts (or boundaries), then compute global boundaries so each partition gets ~equal share.
   - **Allreduce**: Python side has `allgather_reduce` (scalar sum). For (a) we need **allgather** of small tables (split candidates), then one rank (or all) computes `_get_final_sort_boundaries`. So we use **AllGather** of packed "split candidate" tables, then local `_get_final_sort_boundaries`. No need for a true allreduce if we allgather and then compute boundaries locally on each rank (so everyone has the same boundaries).

3. **Phase 3 – Sort, split, shuffle**
   - For each buffered chunk (in order):
     - Sort the chunk locally (by the same `by` / `order` / `null_order`).
     - Run `find_sort_splits(chunk_sort_columns, sort_boundaries, my_part_id, column_order, null_order, stream)` to get split indices.
     - Call `split_and_pack(chunk.table, splits, stream, br)` to get `dict[partition_id, PackedData]`.
     - Insert into the shuffler (either via extended `ShuffleManager.insert_chunk_sorted` or directly into `ShufflerAsync`).
   - Call `insert_finished()` on the shuffler.

4. **Phase 4 – Extract and send**
   - Same as hash shuffle: for each local partition id, `extract_chunk(partition_id, stream)` → unpack_and_concat (or use ShuffleManager's extract path) → send as one (or more) `TableChunk` messages on `ch_out`.
   - Drain `ch_out`.

---

## 4. Boundary-based shuffle (ShuffleManager)

- **Current**: `insert_chunk` does `partition_and_pack(..., columns_to_hash, num_partitions)` then `shuffler.insert(partitioned_chunks)`.
- **Addition**: Either:
  - Add `insert_chunk_sorted(self, chunk: TableChunk, splits: list[int])` that:
    - Calls `split_and_pack(chunk.table_view(), splits, stream, br)` (from `rapidsmpf.integrations.cudf.partition`).
    - Calls `self.shuffler.insert(partitioned_chunks)`.
  - Or keep ShuffleManager hash-only and have sort_actor own a ShufflerAsync and do split_and_pack + insert itself (and use the same `extract_chunk` / unpack_and_concat pattern).

Recommendation: add `insert_chunk_sorted` (or a more generic "insert pre-partitioned chunks" method) so the shuffler remains the single place that owns the collective.

---

## 5. Collectives: allgather for boundaries

- **allreduce**: Not exposed in Python today; you suggested "allgather_reduce until we can use that API." For boundaries we don't need a true allreduce: we need every rank to have the **same global boundaries**.
- **Approach**: Use **AllGather** of each rank's "local split candidates" (small table of sort key values + partition_id + row_number, as in `_select_local_split_candidates`). After allgather, each rank runs `_get_final_sort_boundaries` on the concatenated candidates to get the same global boundaries. So:
  - Pack local candidates table to bytes (or use existing PackedData if schema is fixed).
  - AllGatherManager (or raw AllGather) with one collective ID.
  - Unpack and concat all candidates, then `_get_final_sort_boundaries(concat, column_order, null_order, num_partitions)`.
- **allgather_reduce** in utils is for scalar sums; for sort we only need allgather of small tables. Reuse the same AllGather pattern as in repartition/join (AllGatherManager or equivalent).

---

## 6. Implementation tasks (checklist)

- [ ] **Reserve collective ID for ShuffleSorted**
  In `collectives/common.py`, add `ShuffleSorted` to `collective_types` (when rapidsmpf runtime) and reserve one collective ID per ShuffleSorted node (for the boundary allgather and for the shuffle; if we use one ShufflerAsync per sort, one ID may suffice for shuffle — confirm whether we need a second ID for allgather). Likely: 1 ID for shuffle, 1 for allgather = 2 IDs (similar to GroupBy).

- [ ] **Allow Sort → ShuffleSorted lowering for rapidsmpf**
  In `experimental/sort.py`:
  - In `lower_ir_node.register(Sort)`, remove or relax the branch that does `_lower_ir_fallback` when `config_options.executor.runtime == "rapidsmpf"` so that multi-partition Sort lowers to ShuffleSorted.
  - In `lower_ir_node.register(ShuffleSorted)`, remove or relax the branch that falls back when streaming + rapidsmpf, so ShuffleSorted is actually used in the plan.

- [ ] **Implement sort_actor** (new `rapidsmpf/sort.py` or under `rapidsmpf/collectives/sort.py`)
  - Buffer: receive all chunks into a `SpillableMessages`; on each chunk, compute min/max (and optionally samples) for sort-by columns before pushing to buffer.
  - Boundaries: when input ends, build local split candidates (either by concatenating sort columns from buffered chunks and sampling, or from min/max + histogram). Allgather candidates; run `_get_final_sort_boundaries` to get `sort_boundaries` table.
  - Sort/split/shuffle: for each buffered chunk, local sort → `find_sort_splits` → `split_and_pack` → insert into ShufflerAsync (via ShuffleManager or directly).
  - Extract: same as shuffle_node (extract each local partition, send as TableChunk on ch_out).

- [ ] **ShuffleManager: boundary-based insert**
  - Add `insert_chunk_sorted(chunk, splits)` (or similar) that uses `split_and_pack` and then `shuffler.insert`. Alternatively, keep ShuffleManager hash-only and have sort_actor call split_and_pack and shuffler.insert directly (and still use the same ShufflerAsync for consistency).

- [ ] **Register ShuffleSorted in rapidsmpf dispatch**
  - Implement `generate_ir_sub_network.register(ShuffleSorted)` that:
    - Takes the single child (locally sorted input).
    - Builds one sort_actor with ch_in = child output, ch_out = ShuffleSorted output.
    - Uses collective_id_map[ShuffleSorted] for allgather and shuffle IDs.
  - Wire sort_actor into the pipeline (inputs/outputs and channel managers).

- [ ] **Output metadata**
  - After a sort shuffle, output partitioning is "by sort boundaries" (not hash). Set output ChannelMetadata so downstream nodes know partitioning (e.g. for optimizations or correctness). This may require a new Partitioning variant (e.g. "range" or "sort") or document that we use "inherit" with the understanding that boundaries were used.

- [ ] **Tests**
  - Single rank: multi-partition sort (e.g. 2 partitions) and compare to tasks/single-partition result.
  - Multi-rank: sort with multiple workers; verify global order and partition boundaries.

---

## 7. Optional: histogram-based boundaries

If we want to avoid concatenating all sort columns for sampling (to save memory):
- Maintain per-chunk min/max; optionally, a fixed number of histogram bins (e.g. 256) and count rows per bin from each chunk as it arrives.
- Allgather: each rank sends (min, max, bin_counts) or just bin_counts with a global min/max agreed from an allgather of min/max.
- Compute global boundaries so each partition gets ~1/P of the global count. Then for each buffered chunk we still need to **find_sort_splits** against those boundaries; boundaries would be represented as a small table of "representative" values (e.g. one row per partition boundary). That's a bit more involved than reusing `_select_local_split_candidates` + `_get_final_sort_boundaries`. So for a first version, reusing the existing "sample at indices" approach (Option (a)) is simpler; we can add histogram-based boundaries later if needed.

---

## 8. File layout (suggested)

- `cudf_polars/experimental/rapidsmpf/sort.py` — sort_actor, boundary allgather, and `generate_ir_sub_network.register(ShuffleSorted)`.
- Or split as:
  - `rapidsmpf/collectives/sort.py` — sort_actor and boundary logic.
  - Register in same file or in `rapidsmpf/nodes.py` / `rapidsmpf/dispatch.py`.

Import of `find_sort_splits`, `_select_local_split_candidates`, `_get_final_sort_boundaries` from `cudf_polars.experimental.sort` to avoid duplication.

---

## 9. Summary

| Piece | Action |
|-------|--------|
| Lowering | Allow Sort → ShuffleSorted and ShuffleSorted in plan for rapidsmpf. |
| Collective IDs | Reserve 1–2 IDs for ShuffleSorted (allgather + shuffle). |
| sort_actor | Buffer chunks (SpillableMessages), collect min/max/samples, allgather candidates, _get_final_sort_boundaries, then per-chunk sort → find_sort_splits → split_and_pack → shuffler.insert. |
| ShuffleManager | Add insert_chunk_sorted(splits) or equivalent so the same ShufflerAsync is used for boundary-based partitioning. |
| Dispatch | generate_ir_sub_network(ShuffleSorted) → one sort_actor. |

This reuses the existing tasks semantics for boundaries and split points, uses the existing ShufflerAsync for the actual exchange, and defers a possible future allreduce API by using allgather + local _get_final_sort_boundaries instead.
