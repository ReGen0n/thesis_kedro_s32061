from kedro.pipeline import Pipeline, node

from .nodes import build_features, validate_patients


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=validate_patients,
                inputs=["patients_raw", "params:validation"],
                outputs="patients_validated",
                name="validate_patients_node",
            ),
            node(
                func=build_features,
                inputs=["patients_validated", "params:features"],
                outputs="model_input_table",
                name="build_features_node",
            ),
        ]
    )
