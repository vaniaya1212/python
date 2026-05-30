import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42


def print_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    print(f"Mean absolute error: {mae:.2f} kWh")
    print(f"Percent error: {mape:.2f}%")
    print(f"R2 score: {r2:.4f}")


def save_actual_vs_predicted_plot(y_test, y_pred, filename, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.75)

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")

    plt.title(title)
    plt.xlabel("Actual consumption, kWh")
    plt.ylabel("Predicted consumption, kWh")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def task_1_linear_regression():
    print("Task 1. Linear regression with numeric features")

    df = pd.read_csv("energy_usage.csv")

    x = df[["temperature", "humidity", "hour", "is_weekend"]]
    y = df["consumption"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print_metrics(y_test, y_pred)
    save_actual_vs_predicted_plot(
        y_test,
        y_pred,
        "plot1.png",
        "Actual vs Predicted Consumption",
    )
    print("Plot saved to plot1.png")
    print()


def task_2_regression_with_categories():
    print("Task 2. Regression with OneHotEncoder for categorical features")

    df = pd.read_csv("energy_usage_plus.csv")

    numeric_features = ["temperature", "humidity", "hour", "is_weekend"]
    categorical_features = ["season", "district_type"]

    x = df[numeric_features + categorical_features]
    y = df["consumption"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numbers", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regression", LinearRegression()),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print_metrics(y_test, y_pred)
    save_actual_vs_predicted_plot(
        y_test,
        y_pred,
        "plot2.png",
        "Actual vs Predicted Consumption with Categorical Features",
    )
    print("Plot saved to plot2.png")


def main():
    sns.set_theme(style="whitegrid")
    task_1_linear_regression()
    task_2_regression_with_categories()


if __name__ == "__main__":
    main()
