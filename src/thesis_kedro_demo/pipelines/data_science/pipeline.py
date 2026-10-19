from kedro.pipeline import Pipeline, node

from .nodes import evaluate_model, explain_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:features.target", "params:split"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:model"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test", "y_train", "params:model"],
                outputs=["test_predictions", "metrics"],
                name="evaluate_model_node",
            ),
            node(
                func=explain_model,
                inputs=["regressor", "X_test", "y_test", "params:evaluation"],
                outputs="feature_importance",
                name="explain_model_node",
            ),
        ]
    )
