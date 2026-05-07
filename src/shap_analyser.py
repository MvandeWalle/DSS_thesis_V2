import shap
import pandas as pd
from splitter import Splitter
from trainers import XGBoostTrainer
import logging


logger = logging.getLogger(__name__)


class SHAPAnalyser:
    """Compute and summarise SHAP feature importance for fitted classifiers.

    Supports tree-based models (Random Forest, XGBoost) via
    ``shap.TreeExplainer`` and linear models wrapped in a scikit-learn
    ``Pipeline`` (Logistic Regression) via ``shap.LinearExplainer``.
    Features are grouped into three categories: daily weather, sliding
    window, and calendar features.

    Parameters
    ----------
    model : sklearn estimator or Pipeline
        A fitted scikit-learn model or Pipeline to explain.
    data : pd.DataFrame
        Dataset to compute SHAP values on, typically the out-of-sample
        test set.
    feature_names : list of str
        Names of the feature columns used during training.
    calendar_features : list of str
        Names of calendar-based feature columns, used to assign group
        labels.
    """

    def __init__(
        self,
        model,
        data: pd.DataFrame,
        feature_names: list[str],
        calendar_features: list[str],
    ):
        self.model = model
        self.data = data
        self.feature_names = feature_names
        self.calendar_features = calendar_features

    def _assign_group(self, feature_name: str) -> str:
        """Assign a feature group label based on the feature name.

        Parameters
        ----------
        feature_name : str
            Name of a single feature column.

        Returns
        -------
        str
            One of ``"Calendar"``, ``"Sliding window"``, or
            ``"Daily weather"``.
        """

        if feature_name in self.calendar_features:
            return "Calendar"
        elif "_mean_" in feature_name or "_sum_" in feature_name:
            return "Sliding window"
        else:
            return "Daily weather"

    def _compute_shap_values(self):  # internal function
        """Compute SHAP values for all features on ``self.data``.

        Selects the appropriate SHAP explainer based on the model type:
        ``LinearExplainer`` for Pipeline models and ``TreeExplainer`` for
        tree-based models. For binary tree classifiers, the positive class
        SHAP values are extracted.

        Returns
        -------
        pd.DataFrame
            DataFrame of shape ``(n_samples, n_features)`` containing raw
            SHAP values, with feature names as column headers.
        """

        # SHAP has different explainers for different types of models. Therefore I need to check which model SHAP is currently working with.
        # My LogisticRegressionTrainer is built in a Pipeline and a Pipeline has "steps", while my tree-based models do not.
        if hasattr(self.model, "steps"):
            logger.debug("Starting SHAP analysis for Logistic Regression.")
            classifier = self.model.named_steps["classifier"]
            # I still need transformed data, because in the Pipeline it was transformed as well before training the LogReg model.
            transformed_data = self.model.named_steps["scaler"].transform(
                self.data[self.feature_names]
            )
            explainer = shap.LinearExplainer(classifier, transformed_data)
            shap_values = explainer.shap_values(transformed_data)

        else:
            logger.debug("Starting SHAP analysis for a Tree-based model.")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.data[self.feature_names])

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Older SHAP versions
                shap_values = shap_values[1]

            elif len(shap_values.shape) == 3:
                # Some multiclass/tree outputs
                shap_values = shap_values[:, :, 1]

            # Otherwise already correct 2D array

        values_df = pd.DataFrame(shap_values, columns=self.feature_names)
        return values_df

    def summarise_shap(self, decimal: int = None):
        """Summarise SHAP values as mean absolute importance per feature.

        Calls ``_compute_shap_values`` internally, computes the mean
        absolute SHAP value per feature, assigns group labels, and returns
        a sorted summary DataFrame.

        Parameters
        ----------
        decimal : int, optional
            Number of decimal places to round results to. If ``None``,
            no rounding is applied.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``["group", "feature", "mean_abs_shap"]``

        Examples
        --------
        >>> analyser = SHAPAnalyser(model, test_data, features, calendar_features)
        >>> summary = analyser.summarise_shap(decimal=3)
        >>> print(summary.head())
        """

        values_df = self._compute_shap_values()
        # Calculate the absolute mean values for all observations from the same feature.
        mean_abs_shap = values_df.abs().mean(axis=0).reset_index()
        mean_abs_shap.columns = ["feature", "mean_abs_shap"]

        # Add group labels
        mean_abs_shap["group"] = mean_abs_shap["feature"].apply(self._assign_group)

        # Reorder dataset for clarity
        reordered_shap = mean_abs_shap.iloc[:, [2, 0, 1]]

        # A parameter to optionally limit the decimals
        if decimal is not None:
            reordered_shap = reordered_shap.round(decimal)

        reordered_shap = reordered_shap.reset_index(drop=True)

        return reordered_shap


if __name__ == "__main__":
    train_data = pd.read_csv("data/processed/example_dataset.csv")
    test_data = pd.read_csv("data/processed/test_dataset.csv")
    folds = Splitter(train_data, year_col="year")
    calendar_features = [
        "holiday_NL",
        "workday",
        "vacation_NL_North",
        "vacation_NL_Central",
        "vacation_NL_South",
        "vacation_BE",
        "vacation_GE",
    ]
    features = train_data.columns.drop(["date", "year", "binary_wf", "numeric_wf"])

    model_trainer = XGBoostTrainer(
        data=train_data,
        target_col="binary_wf",
        feature_cols=features,
        year_col="year",
        binary_col="binary_wf",
    )

    model_trainer.run(folds)
    model_trainer.train_final(testing_data=test_data)
    shaped = SHAPAnalyser(
        model=model_trainer.model,
        data=test_data,
        feature_names=features,
        calendar_features=calendar_features,
    )

    output = shaped.summarise_shap(decimal=3)
    print(output.groupby("group")["mean_abs_shap"].mean().round(3))
