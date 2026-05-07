import shap
import pandas as pd
from splitter import Splitter
from trainers import RandomForestTrainer, XGBoostTrainer, LogisticRegressionTrainer


class SHAPAnalyser:
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
        self.daily_weather_features = []
        self.sliding_window_features = []
        for feature_name in self.feature_names:
            if feature_name in self.calendar_features:
                continue
            elif "_mean_" in feature_name or "_sum_" in feature_name:
                self.sliding_window_features.append(feature_name)
            else:
                self.daily_weather_features.append(feature_name)

    def _assign_group(self, feature_name: str) -> str:
        if feature_name in self.calendar_features:
            return "Calendar"
        elif "_mean_" in feature_name or "_sum_" in feature_name:
            return "Sliding window"
        else:
            return "Daily weather"

    def _compute_shap_values(self):  # internal function
        if hasattr(self.model, "steps"):
            print("LogRegShap")
            classifier = self.model.named_steps["classifier"]
            # also need transformed data, not raw data
            transformed_data = self.model.named_steps["scaler"].transform(
                self.data[self.feature_names]
            )
            explainer = shap.LinearExplainer(classifier, transformed_data)
            shap_values = explainer.shap_values(transformed_data)

        else:
            print("TreeShap")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.data[self.feature_names])
            # Select positive class
            shap_values = shap_values[:, :, 1]

        values_df = pd.DataFrame(shap_values, columns=self.feature_names)
        return values_df

    def summarise_shap(self):
        values_df = self._compute_shap_values()
        mean_abs_shap = values_df.abs().mean(axis=0)
        mean_abs_shap = mean_abs_shap.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"})
    
        # Add group labels
        mean_abs_shap["group"] = mean_abs_shap["feature"].apply(self._assign_group)

        # Reorder dataset for clarity
        reordered_shap = mean_abs_shap.iloc[:, [2,0,1]]

        return reordered_shap


if __name__ == "__main__":
    data = pd.read_csv("data/processed/example_dataset.csv")
    train_data = data[data["year"].isin([2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021])]
    validation_data = data[data["year"].isin([2022,2023])]
    test_data = pd.read_csv("data/processed/test_dataset.csv")
    folds = Splitter(data, year_col="year")
    calendar_features = [
        "holiday_NL",
        "workday",
        "vacation_NL_North",
        "vacation_NL_Central",
        "vacation_NL_South",
        "vacation_BE",
        "vacation_GE",
    ]
    features = data.columns.drop(["date", "year", "binary_wf", "numeric_wf"])

    model_trainer = RandomForestTrainer(
        data=data,
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

    output = shaped.summarise_shap()
    # print(output[daily_weather_features].sort_values(ascending=False))
    # print(output[rolling_window_features].sort_values(ascending=False))
    # print(output[calendar_features].sort_values(ascending=False))
    print(output.round(3))
    print(output.groupby("group")["mean_abs_shap"].mean().to_latex())
    