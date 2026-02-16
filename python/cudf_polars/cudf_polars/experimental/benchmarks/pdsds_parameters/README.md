# PDS-DS Query Parameters

Pre-generated parameters for PDS-DS queries (1-11) across all scale factors.

## Configuration

- **RNG Seed**: 42
- **Queries**: 1-11
- **Scale Factors**: 1, 10, 50, 100, 200, 400, 1000, 3000, 10000, 30000, 100000

## File

`parameter_substitutions.json` - Single file containing all parameters for all queries and scale factors.

## Parameterized Queries

- **Query 1**: `county`, `state`, `year`, `agg_field`
- **Query 4**: `year`
- **Queries 2, 3, 5-11**: No parameters

## Usage

```python
from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters

# Load parameters for a specific query and scale factor
params = load_parameters(scale_factor=100, query_id=1)
# Returns: {'county': 1, 'state': 'TN', 'year': 2001, 'agg_field': 'SR_RETURN_AMT'}

# Queries without parameters return None
params = load_parameters(scale_factor=100, query_id=3)
# Returns: None
```

## File Format

```json
{
  "rng_seed": 42,
  "scale_factors": {
    "1": {
      "1": {"county": 1, "state": "TN", "year": 2001, "agg_field": "SR_RETURN_AMT"},
      "4": {"year": 2000}
    },
    "10": {
      "1": {"county": 3, "state": "AL", "year": 2001, "agg_field": "SR_RETURN_AMT"},
      "4": {"year": 2000}
    },
    "100": {
      "1": {"county": 9, "state": "MI", "year": 2001, "agg_field": "SR_RETURN_AMT"},
      "4": {"year": 2000}
    }
  }
}
```
