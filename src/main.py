import logging
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

from evaluator import Evaluator

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
    print(folds)

    ## Create a list of relevant features
    features = train_data.columns.drop(["date", "year", "binary_wf", "numeric_wf"])

    # Train and evaluate model
    ## Binary
    output = []
    for TrainerClass in [
        DummyTrainer,
        RandomForestTrainer,
        XGBoostTrainer,
    ]:
        model_trainer = TrainerClass(
            data=current_dataset,
            target_col="binary_wf",
            feature_cols=features,
            year_col="year",
        )

        val, pms = model_trainer.run(folds)
        evaluation = Evaluator(
            validation_results=val, best_parameters=pms, model=TrainerClass.__name__
        )
        ev = evaluation.evaluate()

        # output[f"{TrainerClass.__name__}_val_results"] = ev.merge(pms)

        print(ev)



# Predict

# Evaluate model_selection

# Evaluate features

# Log results


if __name__ == "__main__":
    main()
