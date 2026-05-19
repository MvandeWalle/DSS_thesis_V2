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
    weather_data = dataset.groupby("month")[['tempDailyAvg', 'tempDailyMin', 'tempDailyMax', 'precipDailySum', 'airpresDailyAvg', 'humidDailyAvg', 'humidDailyMax', 'humidDailyMin']].mean()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,4))
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
    plt.savefig("figures/plots/weather_descriptives.pdf", format="pdf", bbox_inches="tight")

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

            # --- Generate a Bar Plot (Feature Importance) ---
            plt.figure()
            shap.summary_plot(
                shap_values_array,
                X_test,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
            )
            plt.title(f"SHAP Feature Importance: {model_name} ({target_name})")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/shap_bar_{model_name}_{target_name}.pdf")
            plt.close()

            # --- Generate a Beeswarm Plot (Summary Plot) ---
            plt.figure()
            shap.summary_plot(
                shap_values_array,
                X_test,
                feature_names=feature_names,
                show=False,
            )
            plt.title(f"SHAP Summary: {model_name} ({target_name})")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/shap_summary_{model_name}_{target_name}.pdf")
            plt.close()

shap_plots()