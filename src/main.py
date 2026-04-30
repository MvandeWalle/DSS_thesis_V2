import logging
from data_prep import (
    WeatherPrep,
    WildfirePrep,
    CalendarPrep,
    DataMerger,
)
from splitter import Splitter
from trainers import (
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
    weather.feature_engineering(window_length=[3, 7, 28])
    weather_path = weather.write_file(folder="data/processed")

    ## Calendar
    calendar = CalendarPrep(data_path="data/raw/calendar_features.csv", separator=";")
    calendar.clean_data()
    calendar_path = calendar.write_file(folder="data/processed")

    ## Wildfire
    wildfire = WildfirePrep(data_path="data/raw/wildfire_data.csv", separator=";")
    wildfire.clean_data()
    wildfire.feature_engineering()
    binary_wildfire_path, numeric_wildfire_path = wildfire.write_file(
        folder="data/processed"
    )


    # Merge the data
    ## Binary dataset
    binary_train, binary_test = DataMerger(
        weather_path=weather_path,
        calendar_path=calendar_path,
        wildfire_path=binary_wildfire_path,
        wf_type="binary",
        output_folder="data/processed",
    )

    ## Numeric dataset
    numeric_train, numeric_test = DataMerger(
        weather_path=weather_path,
        calendar_path=calendar_path,
        wildfire_path=numeric_wildfire_path,
        wf_type="numeric",
        output_folder="data/processed",
    )

    # Quick check
    if not (binary_train["date"].values == numeric_train["date"].values).all():
        raise ValueError("Date mismatch between binary and numeric datasets.")

    # Split data
    ## Binary and numeric datasets have the same structure and the same dates, so the folds can be extracted from one, not from both.
    folds = Splitter(binary_train, year_col="year")

    ## Create a list of relevant features
    features = binary_train.columns.drop(["date", "year", "wildfire"])

    # Train and evaluate model
    ## Binary 
    def xxx():
        output = {}
        for trainer, name in zip([RandomForestTrainer, XGBoostTrainer], ["RandomForest", "XGBoost"]):
            model_trainer = trainer(data=binary_train, target_col="wildfire", feature_cols=features, year_col="year")
            val, pms = model_trainer.run(folds)
            evaluation = Evaluator(val)
            ev = evaluation.evaluate()
            
            output[f"{name}_val_results"] = ev.merge(pms)

            print(output)

    output = {}
    rf_trainer = RandomForestTrainer(data=binary_train, target_col="wildfire", feature_cols=features, year_col="year")
    val, pms = rf_trainer.run(folds)
    logger.info("The model has been trained, starting evaluation")
    evalu = Evaluator(val, pms)
    ev = evalu.evaluate()
    output["RF_val_results"] = ev.merge(pms)
    print(output)


def unused():
    ### Logistic Regression
    LogReg = LogisticRegressionTrainer(
        data=binary_train, target_col="wildfire", feature_cols=features, year_col="year"
    )
    v, t = LogReg.run(folds)
    LogRegEvaluation = Evaluator(v, t)

    ### Random Forest
    RanFor = RandomForestTrainer(
        data=binary_train, target_col="wildfire", feature_cols=features, year_col="year"
    )
    v, t = RanFor.run(folds)
    RanForEvaluation = Evaluator(v, t)

    ### XGBoost
    xgb = XGBoostTrainer(
        data=binary_train, target_col="wildfire", feature_cols=features, year_col="year"
    )
    v, t = xgb.run(folds)
    xgbEvaluation = Evaluator(v, t)


# Predict

# Evaluate

# Log results



if __name__ == "__main__":
    main()