from fastapi import APIRouter
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from services.paths_service import get_parsed_data_path, get_ml_results_path

router = APIRouter()

@router.get("/run-ml")
def run_ml(car_brand: str, car_model: str, date_max: str, count_pages: int):
    """
    Линейная регрессия
    http://127.0.0.1:8000/run-ml?car_brand=toyota&car_model=camry&date_max=2015&count_pages=5
    """
    csv_dir = get_parsed_data_path()
    csv_filename = f"{car_brand}_{car_model}_{date_max}_{count_pages}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    if not os.path.exists(csv_path):
        return {"error": f"CSV file not found: {csv_path}. Сначала выполните /save_local."}

    df = pd.read_csv(csv_path)
    df['Price'] = df['Price'].replace(r'\D+', '', regex=True).astype(float)
    df['Year'] = df['Year'].astype(float)
    # Удалим некорректные данные
    df = df[(df['Price'] > 0) & (df['Year'] > 1900)]

    X = df[['Year']]
    y = df['Price']

    model = LinearRegression()
    model.fit(X, y)

    X_sorted = np.sort(X['Year'].unique())
    X_sorted_2d = X_sorted.reshape(-1, 1)
    y_pred = model.predict(X_sorted_2d)

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label="Данные", alpha=0.5)
    plt.plot(X_sorted, y_pred, color='red', label="Линейная регрессия")
    plt.xlabel("Год")
    plt.ylabel("Цена")
    plt.title(f"Linear Regression: {car_brand} {car_model}, год <= {date_max}")
    plt.legend()

    ml_dir = get_ml_results_path()
    plot_filename = f"{car_brand}_{car_model}_{date_max}_{count_pages}.png"
    plot_path = os.path.join(ml_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return {
        "message": "Linear Regression обучение завершено",
        "png_path": plot_path
    }


@router.get("/run-ml2")
def run_ml2(car_brand: str, car_model: str, date_max: str, count_pages: int):
    """
    Более точный подход — Gradient Boosting Regressor
    http://127.0.0.1:8000/run-ml2?car_brand=toyota&car_model=camry&date_max=2015&count_pages=5
    """
    csv_dir = get_parsed_data_path()
    csv_filename = f"{car_brand}_{car_model}_{date_max}_{count_pages}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    if not os.path.exists(csv_path):
        return {"error": f"CSV file not found: {csv_path}. Сначала выполните /save_local."}

    df = pd.read_csv(csv_path)
    df['Price'] = df['Price'].replace(r'\D+', '', regex=True).astype(float)
    df['Year'] = df['Year'].astype(float)
    df = df[(df['Price'] > 0) & (df['Year'] > 1900)]

    X = df[['Year']]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    years_range = np.arange(int(df['Year'].min()), int(df['Year'].max()) + 1)
    years_range_2d = years_range.reshape(-1, 1)
    y_range_pred = model.predict(years_range_2d)

    plt.figure(figsize=(8, 6))
    # TRAIN
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label="Train")
    # TEST
    plt.scatter(X_test, y_test, color='green', alpha=0.5, label="Test")

    # Линия предсказаний
    plt.plot(years_range, y_range_pred, color='red', label="GB Predictions")

    plt.xlabel("Год")
    plt.ylabel("Цена")
    plt.title(f"Gradient Boosting: {car_brand} {car_model}, год <= {date_max}")
    plt.legend()

    ml_dir = get_ml_results_path()
    plot_filename = f"improved_{car_brand}_{car_model}_{date_max}_{count_pages}.png"
    plot_path = os.path.join(ml_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return {
        "message": "Gradient Boosting обучение завершено",
        "MSE": mse,
        "R^2": r2,
        "png_path": plot_path
    }
