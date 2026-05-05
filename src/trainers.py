import pandas as pd
import logging
from splitter import Splitter
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    make_scorer,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        year_col: str,
        binary_col: str = None,
    ):
        self.data = data
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.year_col = year_col
        self.model = None  # specified in child classes
        self.binary_col = binary_col

        logger.debug(f"{self.__class__.__name__} initialised.")

    def _get_fold_data(self, years: list[int]):
        # Filter data to a list of years, return X and y
        data = self.data[self.data[self.year_col].isin(years)]
        X = data[self.feature_cols]
        y = data[self.target_col]
        return X, y

    def _tune_hyperparameters(self, X_train, y_train):

        # In _tune_hyperparameters, before creating the search:
        labels = sorted(y_train.unique())
        scorer = make_scorer(
            log_loss,
            response_method="predict_proba",
            labels=labels,
            greater_is_better=False,
        )

        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_grid,  # See in model-specific trainer class
            n_jobs=-1,
            scoring=scorer,
            cv=TimeSeriesSplit(n_splits=3),
            n_iter=20,
            random_state=2026,
        )

        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        return search.best_params_  # dictionary

    def train_fold(self, fold_name: str, fold: dict):

        # Takes one fold dict from Splitter, fits the model, returns predictions
        X_train, y_train = self._get_fold_data(fold["train"])
        X_val, y_val = self._get_fold_data(fold["val"])

        # Extract the best parameters for the model
        best_prms = self._tune_hyperparameters(X_train=X_train, y_train=y_train)

        # Predict probability and put the results in a DataFrame
        all_probs = self.model.predict_proba(X_val)

        if all_probs.shape[1] > 2:
            # numeric_wf: sum all columns from index 1 onwards
            prob_val = all_probs[:, 1:].sum(axis=1)
        else:
            # binary_wf
            prob_val = all_probs[:, 1]

        val_results = pd.DataFrame(
            {"Model": self.model.__class__.__name__, "Fold": fold_name, "y_true": y_val, "y_pred": prob_val}
        )

        if self.binary_col is not None:
            binary_data = self.data[self.data[self.year_col].isin(fold["val"])]
            val_results["y_true_binary"] = binary_data[self.binary_col]

        # Add the parameters to the dataframe
        params_results = pd.DataFrame([{"Fold": fold_name, **best_prms}])

        return val_results, params_results

    def run(self, folds: dict):
        # Loops over all folds and collects results
        val_results = []
        best_params_per_fold = []
        for fold_name, fold_dict in folds.items():
            val_df, params_dict = self.train_fold(fold_name=fold_name, fold=fold_dict)
            val_results.append(val_df)
            best_params_per_fold.append(params_dict)
            logger.debug(f"{fold_name} was completed.")
        self.best_params_per_fold = pd.concat(best_params_per_fold)
        logger.debug(
            f"Evaluation metrics per fold for {self.model.__class__.__name__}: \n {self.best_params_per_fold}"
        )
        return pd.concat(val_results), self.best_params_per_fold

    def train_final(self, testing_data: pd.DataFrame):
        # Define the features and target columns
        ## Training data
        X_train = self.data[self.feature_cols]
        y_train = self.data[self.target_col]

        ## Testing data
        X_test = testing_data[self.feature_cols]
        y_test = testing_data[self.target_col]

        # I take the most frequent parameters across all folds and implement these in the final model.
        best_params = (
            self.best_params_per_fold.drop(columns="Fold").mode().iloc[0].to_dict()
        )
        best_params = {
            k: int(v) if isinstance(v, float) and v.is_integer() else v
            for k, v in best_params.items()
        }
        # Replace NaN with None (pandas converts None to NaN in DataFrames)
        best_params = {k: None if pd.isna(v) else v for k, v in best_params.items()}

        self.model.set_params(**best_params)

        # Fit the model
        self.model.fit(X=X_train, y=y_train)

        # Predict probabilities on the testing data
        all_probs_test = self.model.predict_proba(X_test)
        if all_probs_test.shape[1] > 2:
            prob = all_probs_test[:, 1:].sum(axis=1)
        else:
            prob = all_probs_test[:, 1]
        final_results = pd.DataFrame({"y_true": y_test, "y_pred": prob})

        if self.target_col != "binary_wf":
            binary_data = testing_data[testing_data[self.year_col].isin([2024, 2025])]
            final_results["y_true_binary"] = binary_data[self.binary_col]

        return final_results


class DummyTrainer(BaseTrainer):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        year_col: str,
        binary_col: str,
    ):

        super().__init__(data, target_col, feature_cols, year_col, binary_col)
        self.model = DummyClassifier(random_state=2026, strategy="prior")

    def train_fold(self, fold_name: str, fold: dict):
        X_train, y_train = self._get_fold_data(fold["train"])
        X_val, y_val = self._get_fold_data(fold["val"])

        # No hyperparameter tuning needed for DummyClassifier
        self.model.fit(X_train, y_train)

        prob_val = self.model.predict_proba(X_val)[:, 1]
        val_results = pd.DataFrame(
            {"y_true": y_val, "y_pred": prob_val, "Fold": fold_name}
        )

        # Add binary labels for numeric target evaluation
        if self.binary_col is not None:
            binary_data = self.data[self.data[self.year_col].isin(fold["val"])]
            val_results["y_true_binary"] = binary_data[self.binary_col]

        # Empty params since there's nothing to tune
        params_results = pd.DataFrame([{"Fold": fold_name}])

        return val_results, params_results

    def train_final(self, testing_data: pd.DataFrame):
        if not hasattr(self, "best_params_per_fold"):
            raise RuntimeError("train_final() must be called after run().")

        X_train = self.data[self.feature_cols]
        y_train = self.data[self.target_col]
        X_test = testing_data[self.feature_cols]
        y_test = testing_data[self.target_col]

        # No hyperparameters to set for DummyClassifier
        self.model.fit(X_train, y_train)

        all_probs_test = self.model.predict_proba(X_test)
        if all_probs_test.shape[1] > 2:
            prob = all_probs_test[:, 1:].sum(axis=1)
        else:
            prob = all_probs_test[:, 1]

        final_results = pd.DataFrame({"y_true": y_test, "y_pred": prob})

        if self.target_col != "binary_wf":
            binary_data = testing_data[testing_data[self.year_col].isin([2024, 2025])]
            final_results["y_true_binary"] = binary_data[self.binary_col]

        return final_results


class RandomForestTrainer(BaseTrainer):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        year_col: str,
        binary_col: str,
    ):

        self.param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 5, 10, 20],
        }

        super().__init__(data, target_col, feature_cols, year_col, binary_col)
        self.model = RandomForestClassifier(
            random_state=2026, class_weight="balanced", n_jobs=-1
        )


class XGBoostTrainer(BaseTrainer):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        year_col: str,
        binary_col: str,
    ):

        self.param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
        }

        super().__init__(data, target_col, feature_cols, year_col, binary_col)
        self.model = XGBClassifier(random_state=2026, n_jobs=-1)


class LogisticRegressionTrainer(BaseTrainer):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        year_col: str,
        binary_col: str,
    ):

        self.param_grid = {
            "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "classifier__class_weight": ["balanced", None],
            "classifier__penalty": ["l1", "l2", "elasticnet"],
        }

        super().__init__(data, target_col, feature_cols, year_col, binary_col)
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=2026,
                        max_iter=1000,
                        solver="saga",
                    ),
                ),
            ]
        )

    def _tune_hyperparameters(self, X_train, y_train):
        search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            n_jobs=-1,
            scoring="average_precision",
            cv=TimeSeriesSplit(n_splits=3),
        )
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        return search.best_params_


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Load training data and define features
    df = pd.read_csv("data/processed/example_dataset.csv")
    fc = df.columns.drop(["date", "year", "binary_wf", "numeric_wf"])

    # Load  testing data
    testing_data = pd.read_csv("data/processed/test_dataset.csv")

    # Define folds
    folds = Splitter(data=df, year_col="year")

    # Create an empty list to append the evaluation metrics of each model
    metrics = []

    # Train and evaluate
    for TrainerClass in [
        DummyTrainer,
        RandomForestTrainer,
        XGBoostTrainer,
        # LogisticRegressionTrainer,
    ]:
        logger.debug(f"Running {TrainerClass.__name__}...")
        model = TrainerClass(
            data=df, target_col="binary_wf", feature_cols=fc, year_col="year", binary_col="binary_wf"
        )

        # Train the model
        model.run(folds=folds)

        # Run it on the test set
        output = model.train_final(testing_data=testing_data)

        # Compute the evaluation statistics
        auprc = average_precision_score(
            y_true=output["y_true"], y_score=output["y_pred"]
        )
        brier = brier_score_loss(y_true=output["y_true"], y_proba=output["y_pred"])
        f1 = f1_score(y_true=output["y_true"], y_pred=output["y_pred"].round())

        eval_metrics = {
            "Model": TrainerClass.__name__,
            "AUPRC": round(auprc, 3),
            "Brier Score": round(brier, 3),
            "F1-score": round(f1, 3),
        }

        metrics.append(eval_metrics)

    print(pd.DataFrame(metrics))
