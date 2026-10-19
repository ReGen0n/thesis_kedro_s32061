# Configuration

- `base/` — shared, committed configuration (catalog, parameters, logging).
- `local/` — machine-specific settings and **secrets** (e.g. `credentials.yml`). Ignored by Git; overrides `base/`.
- `test/` — small environment that runs on a sample CSV: `kedro run --env=test`.

Never put credentials in `conf/base/`.
