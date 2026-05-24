# This script is used for the creation of tables and figures for the Results section
# Imports
import pandas as pd
import matplotlib
import shap

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("data/processed/train_dataset.csv")
# Add a column with months
dataset["month"] = pd.to_datetime(dataset["date"]).dt.month
months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

plt.rcParams.update({"font.size": 9})


def monthly_plot():
    # Create plot with the distribution of wildfires per month

    monthly_data = dataset.groupby("month")[["binary_wf", "numeric_wf"]].sum()
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

    # binary_wf
    axes[0].bar(monthly_data.index, monthly_data["binary_wf"])
    axes[0].set_title("Days with wildfires per month")
    axes[0].set_ylabel("Days with wildfires")

    # numeric_wf
    axes[1].bar(monthly_data.index, monthly_data["numeric_wf"])
    axes[1].set_title("Wildfires per month")
    axes[1].set_ylabel("Wildfires")

    for ax in axes:
        ax.set_xlabel("Month")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months, rotation=90, ha="center")

    plt.tight_layout()
    plt.savefig("figures/plots/monthly_count.pdf", format="pdf", bbox_inches="tight")

    plt.show()


def yearly_plot():
    # Create plot with the distribution of wildfires per month

    monthly_data = dataset.groupby("year")[["binary_wf", "numeric_wf"]].sum()
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

    # binary_wf
    axes[0].bar(monthly_data.index, monthly_data["binary_wf"])
    axes[0].set_title("Days with wildfires per year")
    axes[0].set_ylabel("Count")

    # numeric_wf
    axes[1].bar(monthly_data.index, monthly_data["numeric_wf"])
    axes[1].set_title("Wildfires per year")
    axes[1].set_ylabel("Count")

    for ax in axes:
        ax.set_xlabel("Year")
        ax.set_xticks(range(2011, 2024))
        ax.set_xticklabels(labels=range(2011, 2024), rotation=90, ha="center")

    plt.tight_layout()
    plt.savefig("figures/plots/yearly_count.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def weather_descriptives():
    weather_data = dataset.groupby("month")[
        [
            "tempDailyAvg",
            "tempDailyMin",
            "tempDailyMax",
            "precipDailySum",
            "airpresDailyAvg",
            "humidDailyAvg",
            "humidDailyMax",
            "humidDailyMin",
        ]
    ].mean()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 4))
    # Temperature
    axes[0].plot(weather_data.index, weather_data["tempDailyMin"], label="min temp")
    axes[0].plot(weather_data.index, weather_data["tempDailyAvg"], label="mean temp")
    axes[0].plot(weather_data.index, weather_data["tempDailyMax"], label="max temp")

    axes[0].set_title("Temperature per month")
    axes[0].set_ylabel("°C")
    axes[0].legend()

    # Rainfall
    axes[1].bar(weather_data.index, weather_data["precipDailySum"])

    axes[1].set_title("Precipitation per month")
    axes[1].set_ylabel("mm")

    for ax in axes:
        ax.set_xlabel("Month")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months, rotation=90, ha="center")

    plt.tight_layout()
    plt.savefig(
        "figures/plots/weather_descriptives.pdf", format="pdf", bbox_inches="tight"
    )

    plt.show()


def shap_plots():
    # Load test data
    test_data = pd.read_csv("data/processed/test_dataset.csv")
    feature_names = test_data.columns.drop(["date", "year", "binary_wf", "numeric_wf"])
    X_test = test_data[feature_names]

    # Define model and target names
    model_names = ["RandomForest", "XGBoost", "LogisticRegression"]
    target_names = ["binary", "numeric"]

    plot_dir = "figures/plots/shap_plots"

    for model_name in model_names:
        for target_name in target_names:
            if model_name == "LogisticRegression" and target_name == "numeric":
                continue
            shap_values_path = f"data/output/shap_values_{model_name}_{target_name}.csv"

            shap_values = pd.read_csv(shap_values_path)

            # Ensure SHAP values are in the correct format (numpy array)
            shap_values_array = shap_values.values

            # --- CLEAN FIX: Wrap arrays into a SHAP Explanation Object ---
            explanation = shap.Explanation(
                values=shap_values_array,
                data=X_test.values,
                feature_names=list(feature_names),
            )

            # --- Generate a Bar Plot (Feature Importance) ---
            plt.figure()
            # This modern function automatically handles the data labels
            shap.plots.bar(explanation, show=False)

            plt.title(f"SHAP Feature Importance: {model_name} ({target_name})")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/shap_bar_{model_name}_{target_name}.pdf")
            plt.close()

            # --- Generate a Beeswarm Plot (Summary Plot) ---
            plt.figure()
            # The modern beeswarm plot also accepts the Explanation object
            shap.plots.beeswarm(explanation, show=False)

            plt.title(f"SHAP Summary: {model_name} ({target_name})")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/shap_summary_{model_name}_{target_name}.pdf")
            plt.close()


def prob_plot():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    models = ["RandomForest", "XGBoost", "LogisticRegression"]
    targets = ["binary_wf", "numeric_wf"]

    label_map = {
        "RandomForest": "Random Forest",
        "XGBoost": "XGBoost",
        "LogisticRegression": "Logistic Regression",
    }

    # Collect valid combinations
    combinations = []
    for model in models:
        for target in targets:
            if model == "LogisticRegression" and target == "numeric_wf":
                continue
            combinations.append((model, target))

    binary_combinations = [(m, t) for m, t in combinations if t == "binary_wf"]
    numeric_combinations = [(m, t) for m, t in combinations if t == "numeric_wf"]

    for combo_list, filename, target_label in [
        (binary_combinations, "figures/pred_prob_binary.pdf", "Binary WF"),
        (numeric_combinations, "figures/pred_prob_numeric.pdf", "Numeric WF"),
    ]:
        fig, axes = plt.subplots(
            len(combo_list), 1, figsize=(6, 3 * len(combo_list)), sharex=False
        )

        # If only one subplot, axes is not a list — wrap it for consistency
        if len(combo_list) == 1:
            axes = [axes]

        for ax, (model, target) in zip(axes, combo_list):
            filepath = f"data/output/test_predictions/test_pred_{model}_{target}.csv"
            df = pd.read_csv(filepath, parse_dates=["date"])

            # Predicted probability line
            ax.plot(
                df["date"],
                df["y_pred"],
                color="#2c7bb6",
                linewidth=1,
                label="Predicted probability",
                zorder=2,
            )

            # Actual wildfire days
            if target == "numeric_wf":
                fire_dates = df[df["y_true_binary"] == 1]["date"]
            else:
                fire_dates = df[df["y_true"] == 1]["date"]

            ax.vlines(
                fire_dates,
                ymin=0,
                ymax=0.1,
                color="#d7191c",
                linewidth=0.8,
                alpha=0.6,
                label="Wildfire observed",
                zorder=3,
            )

            # Formatting
            ax.set_title(
                f"{label_map[model]} — {target_label}", loc="left", fontsize=10
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Predicted probability")
            ax.set_ylim(0, 1)
            ax.set_xlim(df["date"].min(), df["date"].max())
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(frameon=False, loc="upper left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.show()


def shap_table():
    data = pd.read_csv("data/output/shap_data.csv")
    data_per_group = data.groupby("group")[
        [
            "RandomForest_binary",
            "RandomForest_numeric",
            "XGBoost_binary",
            "XGBoost_numeric",
            "LogisticRegression_binary",
        ]
    ].sum()
    data_per_group["Mean"] = data_per_group.mean(axis=1)
    data_per_group["St. Dev."] = data_per_group.std(axis=1)
    data_per_group = data_per_group.sort_values("Mean", ascending=False).T
    data_per_group = data_per_group[["Daily weather", "Sliding window", "Calendar"]]

    print(
        data_per_group.to_latex(
            index=True,
            float_format="%.3f",  # 3 decimal places
            caption="SHAP values for each group and model combination",
            label="tab: SHAP model x group",
        )
    )
    print(data_per_group)


def summarized_model_comparison():
    validation_output = pd.read_csv("data/output/validation_output.csv")
    test_output = pd.read_csv("data/output/test_output.csv")

    # Target type comparison
    tt_val_comparison = validation_output[validation_output["Model"] != "Dummy"]
    tt_val_comparison = (
        tt_val_comparison.groupby("Target type")[["ROC-AUC", "AUPRC", "Brier Score"]]
        .mean()
        .round(3)
    )
    tt_test_comparison = test_output[test_output["Model"] != "Dummy"]
    tt_test_comparison = (
        tt_test_comparison.groupby("Target type")[["ROC-AUC", "AUPRC", "Brier Score"]]
        .mean()
        .round(3)
    )

    # Model type comparison
    model_val_comparison = validation_output[validation_output["Model"] != "Dummy"]
    model_val_comparison = (
        model_val_comparison.groupby("Model")[["ROC-AUC", "AUPRC", "Brier Score"]]
        .mean()
        .round(3)
    )
    model_test_comparison = test_output[test_output["Model"] != "Dummy"]
    model_test_comparison = (
        model_test_comparison.groupby("Model")[["ROC-AUC", "AUPRC", "Brier Score"]]
        .mean()
        .round(3)
    )

    print(
        model_val_comparison.to_latex(
            index=True,
            float_format="%.3f",  # 3 decimal places
            caption="Model comparison with validation results",
            label="tab: model validation",
        )
    )
    print(
        model_test_comparison.to_latex(
            index=True,
            float_format="%.3f",  # 3 decimal places
            caption="Model comparison with test results",
            label="tab: model test",
        )
    )

def fold_table():
    data = pd.read_csv("data/output/validation_output.csv")
    data = data[data["Model"] != "Dummy"]
    fold_data = data.groupby("Fold")[["ROC-AUC","AUPRC","Brier Score"]].mean().round(3)
    print(fold_data.to_latex(
            index=True,
            float_format="%.3f",  # 3 decimal places
            caption="Model comparison with test results",
            label="tab: model test",
        ))

prob_plot()