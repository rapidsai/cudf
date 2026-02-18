# PDS-DS Query Parameters

Pre-generated parameters for PDS-DS queries across a set of scale factors.

## Configuration

- **RNG Seed**: 42
- **Scale Factors**: 1, 10, 50, 100, 200, 400, 1000, 3000, 10000, 30000, 100000

## File

`parameter_substitutions.json` - Single file containing all parameters for all queries and scale factors.

## Parameterized Queries

- **Query 1**: `county`, `state`, `year`, `agg_field`
- **Query 2**: `year`
- **Query 3**: `aggc`, `month`, `manufact`
- **Query 4**: `year`, `select_one`
- **Query 5**: `year`, `sales_date`
- **Query 6**: `year`, `month`
- **Query 7**: `year`, `gender`, `marital_status`, `education_status`, `promo_channel`
- **Query 8**: `year`, `qoy`, `zip_codes` (list of 400 zip codes)
- **Query 9**: `aggcthen`, `aggcelse`, `rc` (list of 5 count thresholds)
- **Query 10**: `county` (list of 10 counties), `month`, `year`
- **Query 11**: `year`, `select_one`

## Usage

```python
from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters

# Load parameters for a specific query and scale factor
params = load_parameters(scale_factor=100, query_id=1)
# Returns: {'county': 9, 'state': 'MI', 'year': 2001, 'agg_field': 'SR_RETURN_AMT'}

params = load_parameters(scale_factor=100, query_id=7)
# Returns: {'year': 2002, 'gender': 'M', 'marital_status': 'U',
#           'education_status': 'College', 'promo_channel': 'N'}

params = load_parameters(scale_factor=100, query_id=10)
# Returns: {'county': ['Lake County', 'Terrell County', ...], 'month': 4, 'year': 2001}
```

## File Format

```json
{
  "rng_seed": 42,
  "scale_factors": {
    "1": {
      "1": {"county": 1, "state": "TN", "year": 2001, "agg_field": "SR_RETURN_AMT"},
      "2": {"year": 2001},
      "3": {"aggc": "ss_quantity", "month": 11, "manufact": 436},
      "4": {"year": 2000, "select_one": "t_s_secyear.customer_preferred_cust_flag"},
      "5": {"year": 2002, "sales_date": "2002-08-09"},
      "6": {"year": 2001, "month": 1},
      "7": {"year": 2002, "gender": "M", "marital_status": "U",
            "education_status": "College", "promo_channel": "N"},
      "8": {"year": 2002, "qoy": 2, "zip_codes": ["81435", "22224", ...]},
      "9": {"aggcthen": "ss_ext_tax", "aggcelse": "ss_net_paid_inc_tax",
            "rc": [880052, 199641, 97203, 2250570, 4595589]},
      "10": {"county": ["Lake County", "Terrell County", ...], "month": 4, "year": 2001},
      "11": {"year": 2000, "select_one": "t_s_secyear.customer_preferred_cust_flag"}
    },
    "100": {
      "1": {"county": 9, "state": "MI", "year": 2001, "agg_field": "SR_RETURN_AMT"},
      ...
    }
  }
}
```

## Parameter Generation

Parameters were generated using the official TPC-DS `dsqgen` tool (version 4.0.0) with:
- RNG seed: 42
- Scale factors: 1, 10, 50, 100, 200, 400, 1000, 3000, 10000, 30000, 100000
- Dialect: netezza

Parameter types:
- **Simple values**: Generated using `random()` functions (e.g., year, month)
- **Distribution values**: Generated using `dist()` functions (e.g., gender, marital_status, state)
- **Lists**: Generated using `ulist()` functions (e.g., zip_codes, county list, rc thresholds)
