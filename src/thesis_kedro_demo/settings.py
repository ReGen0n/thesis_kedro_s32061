"""Project settings. https://docs.kedro.org/en/stable/kedro_project_setup/settings.html"""

from kedro.config import OmegaConfigLoader

CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # `soft` merge lets conf/test/parameters.yml override single keys
    # (e.g. model_options.n_estimators) instead of replacing the whole block.
    "merge_strategy": {"parameters": "soft"},
}
