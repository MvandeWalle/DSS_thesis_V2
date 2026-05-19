import logging
import pandas as pd
from data_prep_v2 import (
    WeatherPrep,
    WildfirePrep,
    CalendarPrep,
    DataMerger,
)
from splitter import Splitter
from trainers import (
    DummyTrainer,
    LogisticRegressionTrainer,
    RandomForestTrainer,
    XGBoostTrainer,
)

from shap_analyser import SHAPAnalyser

from evaluator import Evaluator, evaluate_final

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Run dataprep
## Weather
def main():
    weather = WeatherPrep(
        data_path="data/raw/weather_data_original.txt", skip_rows=22, separator=","
    )
    weather.clean_data()
    weather.feature_engineering(window_length=[8, 16, 24, 32, 40, 48, 56, 64, 72, 80])
    weather_path = weather.write_file(folder="data/processed")

    ## Calendar
    calendar = CalendarPrep(data_path="data/raw/calendar_features.csv", separator=";")
    calendar.clean_data()
    calendar_path = calendar.write_file(folder="data/processed")

    ## Wildfire
    wildfire = WildfirePrep(data_path="data/raw/wildfire_data.csv", separator=";")
    wildfire.clean_data()
    wildfire.feature_engineering()
    wildfire_path = wildfire.write_file(folder="data/processed")

    # Merge the data
    example_data, train_data, test_data = DataMerger(
        weather_path=weather_path,
        calendar_path=calendar_path,
        wildfire_path=wildfire_path,
        output_folder="data/processed",
    )

    current_dataset = example_data

    # Split data
    ## Binary and numeric datasets have the same structure and the same dates, so the folds can be extracted from one, not from both.
    folds = Splitter(current_dataset, year_col="year")
    logger.debug(folds)

    ## Create a list of relevant features
    features = current_dataset.columns.drop(["date", "year", "binary_wf", "numeric_wf"])

    calendar_features = [
        "holiday_NL",
        "workday",
        "vacation_NL_North",
        "vacation_NL_Central",
        "vacation_NL_South",
        "vacation_BE",
        "vacation_GE",
    ]

    # Train and evaluate model
    validation_output = pd.DataFrame()
    test_output = []
    shap_df = None

    for target, bi_col in zip(["binary_wf", "numeric_wf"], [None, "binary_wf"]):
        models = [
            DummyTrainer,
            RandomForestTrainer,
            XGBoostTrainer,
            LogisticRegressionTrainer,
        ]

        if (target != "binary_wf") and (LogisticRegressionTrainer in models):
            models.remove(LogisticRegressionTrainer)

        for TrainerClass in models:
            model_trainer = TrainerClass(
                data=current_dataset,
                target_col=target,
                feature_cols=features,
                year_col="year",
                binary_col=bi_col,
            )

            model_name = TrainerClass.__name__
            model_name = model_name.replace("Trainer", "")
            target_name = target.replace("_wf", "")
            logger.info(f"{model_name} with {target_name} target started.")

            val, pms = model_trainer.run(folds)
            evaluation = Evaluator(
                validation_results=val,
                best_parameters=pms,
                model=model_name,
                target_type=target,
            )
            ev = evaluation.evaluate()

            validation_output = pd.concat([validation_output, ev], ignore_index=True)

            # Predict on the test set
            output_per_model = model_trainer.train_final(testing_data=test_data)
            output_per_model['date'] = test_data['date'].values
            evaluation_per_model = evaluate_final(
                output_per_model, target_type=target, model_name=model_name
            )
            
            output_name = f"data/output/test_predictions/test_pred_{model_name}_{target}.csv"
            output_per_model.to_csv(output_name, index=False)

            test_output.append(evaluation_per_model)

            if TrainerClass.__name__ != "DummyTrainer":
                shaped = SHAPAnalyser(
                    model=model_trainer.model,
                    data=test_data,
                    feature_names=features,
                    calendar_features=calendar_features,
                )

                # Save raw SHAP values for plotting later
                shap_values_path = f"data/output/shap_values_{model_name}_{target_name}.csv"
                shap_output = shaped.summarise_shap(
                    decimal=4,
                    save_raw_shap=True,
                    raw_shap_path=shap_values_path,
                )

                col_name = f"{model_name}_{target_name}"
                shap_output = shap_output.rename(columns={"mean_abs_shap": col_name})

                if shap_df is None:
                    shap_df = shap_output
                else:
                    shap_df = shap_df.merge(
                        shap_output[["feature", col_name]],
                        on="feature",
                        how="left"
                    )
                


    test_output = pd.DataFrame(test_output)

    logger.debug(f"The validation output per fold per model: \n {validation_output}")
    logger.info(f"The test output per model: \n {test_output}")
    logger.debug(f"The SHAP values per model are: \n {shap_df}")
    shap_cols = shap_df.columns.drop(["group", "feature"])
    grouped_shap_df = shap_df.groupby("group")[shap_cols].mean().round(4)
    grouped_shap_df["overall_mean"] = grouped_shap_df.mean(axis=1)
    grouped_shap_df = grouped_shap_df.sort_values("overall_mean", ascending=False)
    logger.info(
        f"The SHAP values per model per feature group are: \n {grouped_shap_df}"
    )

    # Save the values to a CSV file
    validation_output.to_csv("data/output/validation_output.csv", index=False)
    test_output.to_csv("data/output/test_output.csv", index=False)
    shap_df.to_csv("data/output/shap_data.csv", index=False)

    # Log results


if __name__ == "__main__":
    main()
