import pandas as pd 
from sklearn.model_selection import predict_proba
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class BaseTrainer:
    def __init__(self, data: pd.DataFrame, target_col: str, feature_cols: list[str], year_col: str):
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
        val_results = {}
        test_results = {}

        # Takes one fold dict from Splitter, fits the model, returns predictions
        X_train, y_train = self._get_fold_data(fold["train"])
        X_val, y_val     = self._get_fold_data(fold["val"])
        X_test, y_test   = self._get_fold_data(fold["test"])

        self.model.fit(X_train, y_train)
        prob_val = self.model.predict_proba(X_val)[:,1]
        val_results[fold_name] = pd.DataFrame({
            "y_true": y_val,
            "y_pred": prob_val
        })

        prob_test = self.model.predict_proba(X_test)[:,1]
        test_results[fold_name] = pd.DataFrame({
            "y_true": y_test,
            "y_pred": prob_test
        })

        return val_results, test_results

    def run(self, folds: dict):
        # Loops over all folds and collects results
        val_results = []
        test_results = []
        for fold_name, fold_dict in folds.items():
            val_df, test_df = self.train_fold(fold_name = fold_name, fold = fold_dict)
            val_results.append(val_df)
            test_results.append(test_df)
        
        return pd.concat(val_results), pd.concat(test_results)
    

class LogisticRegressionTrainer(BaseTrainer):
    def __init__(self, data: pd.DataFrame, target_col: str, feature_cols: list[str], year_col: str):
        super().__init__(data, target_col, feature_cols, year_col)
        self.model = LogisticRegression(random_state = 2026, class_weight = "balanced")


class RandomForestTrainer(BaseTrainer):
    def __init__(self, data: pd.DataFrame, target_col: str, feature_cols: list[str], year_col: str):
        super().__init__(data, target_col, feature_cols, year_col)
        self.model = RandomForestClassifier(random_state = 2026, class_weight = "balanced")


class XGBoostTrainer(BaseTrainer):
    def __init__(self, data: pd.DataFrame, target_col: str, feature_cols: list[str], year_col: str):
        super().__init__(data, target_col, feature_cols, year_col)
        self.model = XGBClassifier(random_state = 2026)



if __name__ == "__main__":
    print("beep")