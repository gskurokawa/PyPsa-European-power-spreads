# PyPsa European power spread models and study

**How much does the way a model represents transmission limits change the
day-ahead price spreads it produces?**

An hourly zonal production-cost model of eight continental European bidding
zones, built in PyPSA. Three versions of the same model are compared. They are
identical in fleet, demand, fuel and carbon prices, renewable profiles and
acceptance criteria, and differ only in how cross-zonal transfer is limited:

| model | the limit applies to | recomputed hourly | source |
|---|---|---|---|
| **NTC** | a zone's net exports | no, one pair per year | estimated here from past flows |
| **maxNetPos** | a zone's net exports | yes | published by JAO |
| **CNEC** | individual network elements | yes | published by JAO |

Parameters are estimated on 2024. All three are evaluated on 2025, a year no
parameter was fitted to.

## Headline result

German and French prices differed in 86% of the hours of 2025. The CNEC model
reproduces that at 84%. The NTC model reaches 53% — in the remaining hours it
gives both countries an identical price, so the spread it reports there is not
too small, it is zero. Between Germany and Poland all three models agree.

Under identical gas and carbon price shocks the CNEC model responds 1.3 to 2.7
times more strongly than the NTC model on DE–FR, and the same as it on DE–PL, in
both years. The third model separates the two features that differ, and they
matter for different questions: getting the *number* of congested hours right
depends mostly on limiting elements rather than zones, while getting the
*response to a price change* right depends at least as much on the limit being
recomputed hourly.

**The full study is in [RESULTS.md](RESULTS.md)**, including the model
specification, the fifteen acceptance criteria, what the results do not support,
and the allocation regime on each border since 2015.

## Reproducing it

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # then paste your ENTSO-E token into .env
```

Check the token works, then run the pipeline:

```
python scripts/pipeline/01_pull_entsoe.py --test
python run_pipeline.py --year 2025
```

`run_pipeline.py` runs the sixteen stages in order — pull the raw data, build
the fleet and fuel prices, assemble the flow-based domain, then build and solve
the model — and stops at the first failure, so a half-built set of inputs is
never passed to the next stage.

```
python run_pipeline.py --list              what the stages are
python run_pipeline.py --year 2025 --from 10   skip the data pull
python run_pipeline.py --dry-run           print the commands, run nothing
```

The first run takes hours, almost all of it the ENTSO-E pull, which is cached
per dataset and month and can be interrupted and resumed.

To build one version of the model on its own:

```
python scripts/pipeline/10_build_network.py --year 2025 --chunk-days 30 --bid-ladder 20
```

Add `--flow-based` for the CNEC model or `--max-net-pos` for the maxNetPos
model; omit both for the NTC model. `run_shocks.py` drives the full perturbation
experiment.

## Layout

```
run_pipeline.py        the stages, in order
run_shocks.py          the perturbation experiment
src/spread/            the library: network construction, processing, validation
scripts/pipeline/      the sixteen stages, numbered in execution order
scripts/analysis/      the analyses behind the numbers in RESULTS.md
config/                fleet, technology and zone configuration
```

Each model run reports its own statistics — the fifteen acceptance criteria, the
spread means and standard deviations, and how often each border separates — into
`logs/run_<tag>.txt`, using `src/spread/validate.py`. The scripts in
`scripts/analysis/` are the comparisons *across* runs.

Nothing under `data/` is committed — every input is public and is rebuilt by the
scripts. The raw flow-based domain alone is about 120 MB a year.

All timestamps are stored in UTC; conversion to market time happens once, in
processing.

## Data sources

All inputs are public, and nothing proprietary is used anywhere in this
repository.

- **ENTSO-E Transparency Platform** — load, prices, generation, installed
  capacity, cross-border flows, outages
- **JAO publication tool** — the Core flow-based domain, published net-position
  limits, and the record of which constraints were binding
- **EEX auction reports and futures settlements**, and the **IMF primary
  commodity database** — fuel and carbon prices

An ENTSO-E API token is required and is read from `.env`, which is gitignored.
Tokens are redacted from log output.

## Two notes on the data

European day-ahead prices moved from hourly to quarter-hourly market time units
on 1 October 2025. Prices after that date are averaged to hourly; the model is
hourly throughout.

2023 is excluded deliberately. German nuclear ran until April 2023 and French
nuclear was recovering from the 2022 stress-corrosion outages, so evaluating on
it would mix fleet change with model error.
