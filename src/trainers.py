import pandas as pd
import logging
from splitter import Splitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
from evaluator import Evaluator

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        year_col: str,
    ):
        self.data = data
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.year_col = year_col
        self.model = None  # child classes will set this

    def _get_fold_data(self, years: list[int]):
        # Filter data to a list of years, return X and y
        data = self.data[self.data[self.year_col].isin(years)]
        X = data[self.feature_cols]
        y = data[self.target_col]
        return X, y

    def _tune_hyperparameters(self, X_train, y_train):
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_grid,  # See in model-specific trainer class
            n_jobs=-1,
            scoring="average_precision",
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
        prob_val = self.model.predict_proba(X_val)[:, 1]
        val_results = pd.DataFrame(
            {"y_true": y_val, "y_pred": prob_val, "Fold": fold_name}
        )

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
        self.model.set_params(**best_params)

        # Fit the model
        self.model.fit(X=X_train, y=y_train)

        # Predict probabilities on the testing data
        prob = self.model.predict_proba(X_test)[:, 1]
        final_results = pd.DataFrame({"y_true": y_test, "y_pred": prob})

        return final_results


class RandomForestTrainer(BaseTrainer):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        year_col: str,
    ):

        self.param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 5, 10, 20],
        }

        super().__init__(data, target_col, feature_cols, year_col)
        self.model = RandomForestClassifier(
            random_state=2026, class_weight="balanced", n_jobs=-1
        )

        logging.debug("RF trainer initialised...")


class XGBoostTrainer(BaseTrainer):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        year_col: str,
    ):

        self.param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
        }

        super().__init__(data, target_col, feature_cols, year_col)
        self.model = XGBClassifier(random_state=2026, n_jobs=-1)


class LogisticRegressionTrainer(BaseTrainer):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        year_col: str,
    ):
        self.param_grid = {"C": [0.01, 0.1, 1, 10, 100]}

        super().__init__(data, target_col, feature_cols, year_col)
        self.model = LogisticRegression(
            random_state=2026, class_weight="balanced", max_iter=1000, solver="lbfgs"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    df = pd.read_csv("data/processed/train_binary_dataset.csv")
    print(df.head())
    fc = df.columns.drop(["date", "year", "wildfire"])
    model_list = [XGBoostTrainer]
    for m in model_list:
        model = m(
            data=df, target_col="wildfire", feature_cols=fc, year_col="year"
        )
        logging.debug("Model initialised.")
        flds = Splitter(data=df, year_col="year")
        v, p = model.run(folds=flds)
        # evaluation = Evaluator(validation_results=v, best_parameters=p)
        # ev = evaluation.evaluate()
        testing_data = pd.read_csv("data/processed/test_binary_dataset.csv")
        output = model.train_final(testing_data=testing_data)

        from sklearn.metrics import average_precision_score, brier_score_loss, f1_score

        auprc = average_precision_score(y_true=output["y_true"], y_score=output["y_pred"])
        brier = brier_score_loss(y_true=output["y_true"], y_proba=output["y_pred"])
        f1 = f1_score(y_true=output["y_true"], y_pred=output["y_pred"].round())

        eval_metrics = {
            "AUPRC": round(auprc, 3),
            "Brier Score": round(brier, 3),
            "F1-score": round(f1, 3),
        }
        
        print(eval_metrics)