import pandas as pd
from splitter import Splitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from evaluator import Evaluator


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

    def train_fold(self, fold_name: str, fold: dict):

        # Takes one fold dict from Splitter, fits the model, returns predictions
        X_train, y_train = self._get_fold_data(fold["train"])
        X_val, y_val = self._get_fold_data(fold["val"])

        self.model.fit(X_train, y_train)
        prob_val = self.model.predict_proba(X_val)[:, 1]
        val_results = pd.DataFrame(
            {"y_true": y_val, "y_pred": prob_val, "fold": fold_name}
        )
        return val_results

    def run(self, folds: dict):
        # Loops over all folds and collects results
        val_results = []
        for fold_name, fold_dict in folds.items():
            val_df = self.train_fold(fold_name=fold_name, fold=fold_dict)
            val_results.append(val_df)
        return pd.concat(val_results)


class LogisticRegressionTrainer(BaseTrainer):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        year_col: str,
    ):
        super().__init__(data, target_col, feature_cols, year_col)
        self.model = LogisticRegression(
            random_state=2026, class_weight="balanced", max_iter=1000, n_jobs=3
        )


class RandomForestTrainer(BaseTrainer):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        year_col: str,
    ):
        super().__init__(data, target_col, feature_cols, year_col)
        self.model = RandomForestClassifier(
            random_state=2026, class_weight="balanced", n_jobs=3
        )


class XGBoostTrainer(BaseTrainer):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        year_col: str,
    ):
        super().__init__(data, target_col, feature_cols, year_col)
        self.model = XGBClassifier(random_state=2026, n_jobs=3)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/train_binary_dataset.csv")
    fc = df.columns.drop(["date", "year", "wildfire"])
    model = XGBoostTrainer(
        data=df, target_col="wildfire", feature_cols=fc, year_col="year"
    )
    flds = Splitter(data=df, year_col="year")
    v, t = model.run(folds=flds)
    evaluation = Evaluator(v, t)
    ev, et = evaluation.evaluate()
    print(ev)
    print(et)
