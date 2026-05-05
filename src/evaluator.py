import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from splitter import Splitter
from trainers import DummyTrainer, RandomForestTrainer, XGBoostTrainer, LogisticRegressionTrainer


class Evaluator:
    def __init__(
        self,
        validation_results: pd.DataFrame,
        best_parameters: pd.DataFrame,
        model: str,
        target_type: str = "binary_wf",
    ):

        if target_type not in ["binary_wf", "numeric_wf"]:
            raise ValueError(
                f"Target type should be 'binary_wf' or 'numeric_wf', got {target_type} instead."
            )

        self.val_results = validation_results
        self.best_parameters = best_parameters
        self.model_name = model
        self.target_type = target_type

    def evaluate_fold(self, fold_df: pd.DataFrame, fold_name: str = "empty"):
        if self.target_type == "numeric_wf":
            y_true = fold_df["y_true_binary"]
        else:
            y_true = fold_df["y_true"]

        auprc = average_precision_score(y_true=y_true, y_score=fold_df["y_pred"])
        brier = brier_score_loss(y_true=y_true, y_proba=fold_df["y_pred"])
        f1 = f1_score(y_true=y_true, y_pred=fold_df["y_pred"].round())

        eval_metrics = {
            "Model": self.model_name,
            "Target type": self.target_type,
            "Fold": fold_name,
            "AUPRC": auprc,
            "Brier Score": brier,
            "F1-score": f1,
        }

        return eval_metrics

    def evaluate(self):
        val_eval = []

        for fold_name, fold_df in self.val_results.groupby("Fold"):
            eval_metrics = self.evaluate_fold(fold_df, fold_name)
            val_eval.append(eval_metrics)

        evaluation_df = pd.DataFrame(val_eval)
        output = evaluation_df.merge(self.best_parameters, on="Fold", how="left")

        return output


if __name__ == "__main__":
    data = pd.read_csv("data/processed/example_dataset.csv")
    folds = Splitter(data, year_col="year")
    features = data.columns.drop(["date", "year", "binary_wf", "numeric_wf"])

    for target, bi_col in zip(["binary_wf", "numeric_wf"], [None, "binary_wf"]): 
        rft = DummyTrainer(
            data=data,
            target_col=target,
            feature_cols=features,
            year_col="year",
            binary_col=bi_col
        )

        val, pms = rft.run(folds=folds)

        evaluation = Evaluator(
            validation_results=val,
            best_parameters=pms,
            model="RandomForest",
            target_type=target,
        )
        ev = evaluation.evaluate()

        print(ev)
