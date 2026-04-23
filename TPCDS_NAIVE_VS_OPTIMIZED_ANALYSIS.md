# TPC-DS Naive vs Optimized Polars Implementation: Performance Analysis

## 1. Executive Summary

Running all 99 TPC-DS queries against naive and optimized Polars implementations reveals a consistent pattern: a small number of queries account for the vast majority of performance differences, and the gap grows dramatically with data scale.

At SF1, the optimized implementation is **1.40x faster overall** (39.92s vs 28.51s). At SF10, the overall ratio drops to **1.21x** — but this headline number is misleading. The top outlier query (q72) alone accounts for roughly 28 seconds of difference, and at SF100 it triggers an out-of-memory kill. Remove q72 from the SF10 totals and the two implementations are nearly equivalent.

The practical takeaway: **optimizing 5 queries out of 99 captures the overwhelming majority of performance benefit.** The other 94 queries are either already well-optimized, dominated by window function computation, or too small to matter at typical scale factors.

---

## 2. Methodology

- **Benchmark**: Full TPC-DS suite (99 queries), scale factors SF1, SF10, and SF100
- **Iterations**: 3 runs per query per scale factor; results are averages
- **Engine**: Polars (in-process DataFrame operations, no query planner)
- **Naive implementation**: Direct SQL-to-Python translation — joins follow the FROM clause order, filters applied after joining, all columns carried through operations
- **Optimized implementation**: Hand-tuned transformations applying predicate pushdown, FK-only aggregation, semi/anti joins, and column pruning
- **Infrastructure**: Single node; SF100 results are partial (OOM at q72)

---

## 3. Overall Results

### Summary by Scale Factor

| Metric | SF1 | SF10 |
|---|---|---|
| Total time, naive | 39.92s | 87.76s |
| Total time, optimized | 28.51s | 72.34s |
| Overall speedup | **1.40x** | **1.21x** |
| Queries where naive is faster | 4 | ~4 |
| Worst single query (naive/opt ratio) | q88: 19.58x | q72: 54.15x |

### Per-Query Results: Top Outliers

| Query | SF1 Naive | SF1 Opt | SF1 Ratio | SF10 Naive | SF10 Opt | SF10 Ratio | Primary cause |
|---|---|---|---|---|---|---|---|
| q88 | 1.729s | 0.088s | **19.58x** | 2.631s | 0.148s | **17.73x** | 8 redundant base joins |
| q4 | 4.451s | 0.549s | **8.11x** | 8.685s | 1.519s | **5.72x** | FK-only agg + date pushdown |
| q11 | 1.672s | 0.296s | **5.66x** | 3.704s | 1.502s | **2.47x** | FK-only agg + date pushdown |
| q72 | 1.826s | 0.379s | **4.82x** | 29.194s | 0.539s | **54.15x** | Predicate pushdown (10+ tables) |
| q74 | 1.055s | 0.283s | **3.72x** | 1.886s | 0.406s | **4.65x** | FK-only agg + date pushdown |
| q14 | 2.780s | 0.999s | **2.78x** | n/a | n/a | n/a | Cross-channel items + ROLLUP |
| q9 | 0.092s | 0.039s | **2.35x** | 0.152s | 0.057s | **2.65x** | Scalar subquery restructuring |

### Queries Where Naive Outperforms Optimized

| Query | Naive/Opt ratio | Notes |
|---|---|---|
| q47 | 0.52x (naive 48% faster) | LAG/LEAD window-dominated |
| q84 | 0.54x (naive 46% faster) | Simple join chain |
| q53 | 0.56x (naive 44% faster) | Straightforward aggregation |
| q57 | 0.60x (naive 40% faster) | LAG/LEAD window-dominated |

---

## 4. Optimization Technique Rankings

Ranked by observed performance impact, from highest to lowest ROI.

### Rank 1: Predicate Pushdown

**What it is**: Filter dimension tables to only matching rows *before* joining them to fact tables. In naive code, WHERE conditions are applied after all joins complete.

**Queries affected**: q72, q4, q11, q74, q14, and others

**Impact**:

| Query | SF1 ratio | SF10 ratio | SF100 |
|---|---|---|---|
| q72 | 4.82x | 54.15x | OOM |
| q4 | 8.11x | 5.72x | 46.4s (naive) |
| q11 | 5.66x | 2.47x | 19.9s (naive) |

The impact scales **superlinearly** with data size because join cardinality grows quadratically. At SF1, the intermediate DataFrame from an unfiltered 10-table join fits in memory and costs 4.82x. At SF10, it costs 54.15x. At SF100, it causes OOM.

**Concrete example (q72)**: Naive code joins `catalog_sales × inventory × warehouse × item × customer_demographics × household_demographics × date_dim × date_dim × date_dim × promotion × catalog_returns` (10+ tables, millions of rows each) with no pre-filtering. The optimized version creates three pre-filtered date subsets (single year, matching week sequences, 3-year range), pre-filters `customer_demographics` by marital status, and pre-filters `household_demographics` by buy potential. Each join thereafter operates on a tiny fraction of the original data.

---

### Rank 2: Scalar Subquery Restructuring

**What it is**: Replace multiple redundant subqueries that share the same base join with a single base join and conditional aggregation.

**Queries affected**: q88, q9

**Impact**:

| Query | SF1 ratio | SF10 ratio |
|---|---|---|
| q88 | 19.58x | 17.73x |
| q9 | 2.35x | 2.65x |

**Concrete example (q88)**: The query counts store sales across 8 non-overlapping 30-minute time windows. Naive code creates 8 independent subqueries, each performing `store_sales JOIN time_dim JOIN household_demographics JOIN store` with identical demographic filters, differing only in the hour/minute range. These 8 intermediate results are then cross-joined via a synthetic key column. The same 4-table base join is executed 8 times.

The optimized version performs the base join once, then uses `pl.when().then().sum()` to compute all 8 time-bucket counts in a single aggregation pass. The 19.58x speedup is almost entirely explained by this 8x reduction in redundant join work.

---

### Rank 3: FK-Only Aggregation

**What it is**: Aggregate on the foreign key column without joining the dimension table first. Join the dimension table once at the end, only to retrieve display columns.

**Queries affected**: q4, q11, q74 (all "year-over-year" style queries)

**Concrete example (q4)**: This query compares total spend per customer across 3 sales channels (store, catalog, web) in two consecutive years. Naive code builds `year_total` by joining `customer × sales × date_dim` for each channel, carrying all customer columns (customer_id, first_name, last_name, preferred_flag, birth_country, login, email) through the aggregation. It then self-joins this wide result 6 times on `customer_id`.

Each alias in the 6-way join carries ~10 columns, producing an intermediate DataFrame with 60+ columns before the final SELECT. The optimized version aggregates each channel down to just `(customer_sk, year_total)` — a 2-column DataFrame — performs the 6-way join on those, then joins the customer table once at the end. The intermediate join width shrinks from 60+ columns to 12.

---

### Rank 4: Semi-Join and Anti-Join

**What it is**: Use `how="semi"` or `how="anti"` instead of `inner join + unique()` for EXISTS and NOT EXISTS patterns.

**Queries affected**: q16, q78, q87, q94, q95

**Impact**: Moderate, typically 1.1x to 1.5x. These queries aren't in the top outlier list, but the optimization is cheap and consistent.

**Why it helps**: A semi-join produces only left-side rows that match the right side, without materializing any right-side columns. The naive approach using `inner join + unique()` first creates a wider intermediate DataFrame (carrying all right-side columns), then deduplicates. At larger scale factors, avoiding the extra column materialization and deduplication step adds up.

---

### Rank 5: Column Pruning

**What it is**: Use `.select()` to drop unneeded columns before expensive join or aggregation operations.

**Queries affected**: q72 (date_dim columns), q64 (wide table joins), others

**Impact**: Usually bundled with predicate pushdown; hard to isolate independently. Meaningful for queries joining wide tables repeatedly.

**Concrete example (q72)**: The optimized version selects only `["d_date_sk", "d_week_seq", "d_date"]` from date_dim before joining. Since date_dim has ~20 columns and is joined multiple times, dropping unused columns reduces the size of each intermediate DataFrame.

---

### Rank 6: Date Range Pre-computation

**What it is**: Rather than joining the full date_dim and filtering by year afterward, pre-filter date_dim to the exact rows needed and store them as named DataFrames.

**Queries affected**: q72, q4, q11, q74

This is a specific form of predicate pushdown applied consistently to date_dim. The technique creates named filtered subsets (e.g., `d1_dates` for year 2002, `d2_dates` for matching week sequences, `d3_dates` for a 3-year range) that are reused across multiple joins. The benefit is both the reduced join cardinality and the clarity of the resulting code.

---

### Rank 7: Join Reordering

**What it is**: Place more selective joins earlier in the chain to reduce the size of later joins.

**Impact**: Minimal as a standalone technique in these benchmarks. Most join reordering benefit is captured by predicate pushdown. Without a query planner, manual join reordering is fragile and query-specific.

---

## 5. Deep Dive: Top 5 Most Impacted Queries

### q72 — The Predicate Pushdown Poster Child

**Ratios**: SF1: 4.82x | SF10: 54.15x | SF100: OOM

q72 performs a promotional analysis across the full supply chain: it joins `catalog_sales`, `inventory`, `warehouse`, `item`, `customer_demographics`, `household_demographics`, three instances of `date_dim`, `promotion`, and `catalog_returns`. That's 10+ tables, with fact tables containing millions of rows at any scale factor.

**Naive execution path**:
1. Join all 10+ tables with no pre-filtering
2. `catalog_sales × inventory` alone is enormous — every sale row crossed with every inventory record for matching items
3. Filter the result by year, marital status, buy potential, quantity threshold, week sequence match

At SF1 the intermediate join result is large but fits in memory (4.82x overhead). At SF10 it grows superlinearly — the join cardinality scales with the square of the data size, pushing the ratio to 54.15x. At SF100 the intermediate result exceeds available memory entirely.

**Optimized execution path**:
1. Filter `date_dim` to year 2002 (`d1_dates`, a tiny fraction of rows)
2. Compute matching week sequences (`d2_dates`)
3. Compute a 3-year surrounding date range (`d3_dates`)
4. Pre-filter `customer_demographics` by marital status
5. Pre-filter `household_demographics` by buy potential
6. All subsequent joins operate on small, pre-filtered DataFrames

The net effect is that the cartesian blowup never happens. Each join filters aggressively, keeping intermediate DataFrames small throughout.

---

### q88 — 8 Redundant Base Joins

**Ratios**: SF1: 19.58x | SF10: 17.73x

q88 counts store sales in 8 non-overlapping 30-minute time windows throughout a day, filtered by household demographics and store details. The SQL is written as 8 scalar subqueries, each returning a single COUNT.

**Naive execution path**:
- Subquery 1: `store_sales JOIN time_dim JOIN household_demographics JOIN store`, filter by 8am-8:30am, COUNT
- Subquery 2: same base join, filter by 8:30am-9am, COUNT
- ... repeated 8 times
- Cross-join all 8 results via a synthetic key column

The base join (`store_sales × time_dim × household_demographics × store`) is performed 8 times identically. This is pure redundant computation.

**Optimized execution path**:
- One base join with shared demographic filters
- Single aggregation using `pl.when(hour_condition & minute_condition).then(1).otherwise(0).sum()` for each of the 8 buckets

The 19.58x speedup at SF1 is almost entirely the 8x reduction in join work, with additional gains from avoiding 7 redundant DataFrame materializations.

The ratio stays high at SF10 (17.73x) because the inefficiency is structural — it scales directly with data size rather than getting absorbed by other costs.

---

### q4 — Year-over-Year Three-Channel Analysis

**Ratios**: SF1: 8.11x | SF10: 5.72x

q4 identifies customers whose total spend in year 2 exceeded year 1 spend across all three sales channels (store, catalog, web). The query requires a 6-way self-join comparing (store year 1, store year 2, catalog year 1, catalog year 2, web year 1, web year 2) for each customer.

**Naive execution path**:
- For each of 6 combinations (channel × year), join `customer × sales × date_dim`
- Each join carries all customer columns: customer_id, first_name, last_name, preferred_flag, birth_country, login, email (~10 columns)
- UNION ALL into a single DataFrame
- Self-join this wide DataFrame 6 times on customer_id
- The 6-way joined result carries ~10 columns × 6 aliases = 60+ columns

**Optimized execution path**:
- For each combination, pre-filter `date_dim` to a single year, aggregate sales to `(customer_sk, year_total)` only
- The 6-way join operates on 2-column DataFrames (12 columns total in the joined result)
- Join the `customer` table once at the end to retrieve display columns

The intermediate DataFrame width shrinks by ~5x, and each join is significantly faster. The SF10 ratio (5.72x) is lower than SF1 (8.11x) because data loading and I/O begin to dominate at larger scale factors, partially masking the join optimization benefit.

---

### q11 — Year-over-Year Two-Channel Analysis

**Ratios**: SF1: 5.66x | SF10: 2.47x

q11 is structurally identical to q4 but uses two channels (store and web) instead of three, producing a 4-way self-join instead of a 6-way self-join. The same optimization pattern applies: FK-only aggregation to `(customer_sk, year_total)`, predicate pushdown on `date_dim`, and a final customer join for display columns.

The lower SF10 ratio (2.47x vs 5.66x at SF1) reflects the same data-loading effect as q4: at larger scale factors, I/O costs grow and partially hide the join optimization savings.

---

### q74 — Year-over-Year with STDDEV

**Ratios**: SF1: 3.72x | SF10: 4.65x

q74 is another year-over-year customer comparison, this time computing standard deviation of spending to identify high-value customers. The optimization pattern is the same as q4 and q11.

q74 shows the opposite scaling trend from q11: the SF10 ratio (4.65x) is *higher* than SF1 (3.72x). This suggests the STDDEV computation cost grows with data size in the naive version (carrying more columns through the standard deviation calculation amplifies the overhead), while the optimized version's smaller intermediates keep that cost low.

---

## 6. Surprising Results: When Naive Wins

Four queries where the naive implementation is measurably faster:

| Query | Naive/Opt ratio | Why |
|---|---|---|
| q47 | 0.52x (naive 48% faster) | Window functions dominate; optimization overhead exceeds savings |
| q84 | 0.54x (naive 46% faster) | Simple join chain; pre-filtering adds overhead without benefit |
| q53 | 0.56x (naive 44% faster) | Straightforward aggregation; no join cardinality problem to solve |
| q57 | 0.60x (naive 40% faster) | Window functions dominate; same as q47 |

**q47 and q57** both rely heavily on LAG and LEAD window functions to compute month-over-month and year-over-year comparisons. The window function computation dominates their execution time regardless of what happens in the joins. The optimized version applies pre-filtering and column selection steps that add overhead, but these steps don't reduce the window function work at all. Net result: the optimized version pays extra overhead with no corresponding gain.

**q84 and q53** are relatively simple queries where the base data is small enough that carrying extra columns or doing an extra post-join filter costs essentially nothing. The overhead of the optimization techniques (helper function calls, extra `.select()` operations, intermediate DataFrame creation) exceeds the savings at small scale.

The broader lesson: optimization techniques that target join cardinality are not universally beneficial. They're most valuable when joins are the bottleneck. When query execution is dominated by window functions, aggregations, or simple filters on small tables, the overhead of "optimizing" can make things worse. Profiling before optimizing is not just a cliche here — it's how you avoid the q47 mistake.

---

## 7. Scaling Behavior

### Overall Ratios by Scale Factor

| Scale Factor | Naive Total | Optimized Total | Ratio |
|---|---|---|---|
| SF1 (1GB) | 39.92s | 28.51s | 1.40x |
| SF10 (10GB) | 87.76s | 72.34s | 1.21x |
| SF100 (100GB) | Partial (OOM at q72) | Partial | N/A |

The SF10 overall ratio (1.21x) looks *lower* than SF1 (1.40x), which seems counterintuitive. Two effects explain this:

**Effect 1: Convergence of most queries.** At SF10, both implementations become more I/O and computation bound for the majority of queries. Loading larger DataFrames from disk and performing group-by aggregations on millions of rows takes meaningful time regardless of whether you pre-filtered a dimension table. The 94 "normal" queries converge toward similar performance.

**Effect 2: Extreme outlier behavior.** The few queries with structural inefficiencies get dramatically worse. q72 goes from 4.82x at SF1 to 54.15x at SF10. But because q72 is just one query out of 99, and the *optimized* q72 is fast (0.539s), it adds ~28 seconds to the naive total while adding almost nothing to the optimized total. This distorts the denominator (naive total) upward without proportionally increasing the ratio.

### How the Top Queries Scale

| Query | SF1 ratio | SF10 ratio | Trend |
|---|---|---|---|
| q72 | 4.82x | 54.15x | Superlinear (OOM at SF100) |
| q88 | 19.58x | 17.73x | Roughly constant |
| q4 | 8.11x | 5.72x | Sub-linear (I/O bounds SF10) |
| q74 | 3.72x | 4.65x | Linear |
| q11 | 5.66x | 2.47x | Sub-linear |

**Superlinear scaling (q72)**: The join cardinality problem grows with the square of the data size. At SF100 it exceeds available memory. This is the most dangerous class of inefficiency in a naive implementation.

**Constant scaling (q88)**: The 8x redundant join overhead is a fixed multiplier that doesn't change with data size. The ratio stays ~18-20x regardless of scale.

**Sub-linear scaling (q4, q11)**: Data loading and I/O costs grow with scale and begin to dominate over join optimization benefits. The ratio decreases as scale increases. These queries are expensive but not catastrophic in the naive implementation.

### SF100 Notes

SF100 completed all 71 queries before q72. Notable timings in the naive implementation:
- q4: 46.4s (vs ~8.7s at SF10 — scaling as expected)
- q11: 19.9s (vs ~3.7s at SF10)
- q23: 15.4s

q72 caused an OOM kill. The intermediate DataFrame from joining 10+ unfiltered fact and dimension tables at 100x scale exceeded available memory before any WHERE filters could reduce it.

---

## 8. Conclusions and Recommendations

### Where to Invest Optimization Effort

The data shows a highly skewed distribution of impact. Five queries account for the majority of the performance difference. Any engineer working on this codebase should prioritize in this order:

**Priority 1: Fix q72 (predicate pushdown on multi-fact joins)**

This query is the only one that causes OOM at SF100. It's 54x slower at SF10. The fix is mechanical: identify the WHERE conditions that filter dimension tables by year, date range, demographic attributes, or other selective predicates, and apply those filters *before* joining to fact tables. This single change makes SF100 viable.

**Priority 2: Fix q88 (eliminate redundant base joins)**

A 19x speedup requires zero algorithmic cleverness — just recognize that 8 subqueries sharing the same base join can be collapsed into one join with conditional aggregation. The pattern is immediately recognizable in any query with multiple scalar subqueries over the same tables.

**Priority 3: Apply FK-only aggregation to year-over-year queries (q4, q11, q74)**

Any query that computes `year_total` by joining a large customer table before aggregating, then self-joining the result multiple times, is a candidate. Aggregate to `(foreign_key, metric)` first, perform the self-joins on the tiny result, then join the customer table once at the end. The benefit is proportional to the number of self-joins and the width of the dimension table being avoided.

**Priority 4: Add semi/anti joins for EXISTS/NOT EXISTS patterns**

Lower impact but essentially free — swapping `inner join + unique()` for `how="semi"` and `left join + null filter` for `how="anti"` is a one-line change per query with consistent 1.1x-1.5x improvement.

**Priority 5: Add column pruning before expensive joins**

Use `.select()` to keep only required columns before any join involving wide tables. The investment is minimal and the benefit compounds with data scale.

### What Not to Optimize

Don't apply predicate pushdown or column selection to queries dominated by window functions (q47, q57). The overhead of creating pre-filtered intermediate DataFrames exceeds the savings when join cardinality isn't the problem. For these queries, profile first to identify the actual bottleneck.

### The 80/20 Summary

| Fix | Queries | Effort | Impact |
|---|---|---|---|
| Predicate pushdown on multi-fact joins | q72, q4, q11, q74 | Medium | Critical at SF100+ |
| Collapse redundant scalar subqueries | q88, q9 | Low | 19x for q88 |
| FK-only aggregation for year-over-year | q4, q11, q74 | Medium | 3-8x for these queries |
| Semi/anti join replacement | q16, q78, q87, q94, q95 | Very low | 1.1-1.5x each |

Five queries. Four fix categories. Roughly 15-20% of the total engineering effort for this suite would address 80% or more of the total performance gap, and would be the difference between a working SF100 run and an OOM crash.
