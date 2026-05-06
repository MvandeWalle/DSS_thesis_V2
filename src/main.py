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

    current_dataset = train_data

    # Split data
    ## Binary and numeric datasets have the same structure and the same dates, so the folds can be extracted from one, not from both.
    folds = Splitter(current_dataset, year_col="year")

    ## Create a list of relevant features
    features = current_dataset.columns.drop(["date", "year", "binary_wf", "numeric_wf"])

    # Train and evaluate model
    validation_output = pd.DataFrame()
    test_output = []
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

            logger.info(f"{TrainerClass.__name__} with target {target} started.")

            val, pms = model_trainer.run(folds)
            evaluation = Evaluator(
                validation_results=val,
                best_parameters=pms,
                model=TrainerClass.__name__,
                target_type=target,
            )
            ev = evaluation.evaluate()

            validation_output = pd.concat([validation_output, ev], ignore_index=True)

            # Predict on the test set
            output_per_model = model_trainer.train_final(testing_data=test_data)
            evaluation_per_model = evaluate_final(
                output_per_model, target_type=target, model_name=TrainerClass.__name__
            )

            test_output.append(evaluation_per_model)

    test_output = pd.DataFrame(test_output)

    logger.info(f"The validation output per fold per model: \n {validation_output}")
    logger.info(f"The test output per model: \n {test_output}")

    # Evaluate model_selection

    # Evaluate features

    # Log results


if __name__ == "__main__":
    main()
