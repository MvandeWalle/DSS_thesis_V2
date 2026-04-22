import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score


class Evaluator:
    def __init__(self, validation_results: pd.DataFrame):
        self.val_results = validation_results

    def evaluate_fold(self, fold_df: pd.DataFrame, fold_name: str = "empty"):
        auprc = average_precision_score(
            y_true=fold_df["y_true"], y_score=fold_df["y_pred"]
        )
        brier = brier_score_loss(y_true=fold_df["y_true"], y_proba=fold_df["y_pred"])
        f1 = f1_score(y_true=fold_df["y_true"], y_pred=fold_df["y_pred"].round())

        eval_metrics = {
            "Fold": fold_name,
            "AUPRC": auprc,
            "Brier Score": brier,
            "F1-score": f1,
        }

        return eval_metrics

    def evaluate(self):
        val_eval = []
        for data, output in zip(
            [self.val_results], [val_eval]
        ):
            for fold_name, fold_df in data.groupby("fold"):
                eval_metrics = self.evaluate_fold(fold_df, fold_name)
                output.append(eval_metrics)

        return pd.DataFrame(val_eval)
