# services/logistic_service.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Dict

def run_logistic_regression(
    df: pd.DataFrame, price_threshold: float
) -> Dict:
    """
    Принимает DataFrame с колонками:
      - Price (строка вида '19 500 000', '₸', и т.д.),
      - Year, Mileage, EngineVolume (строки, могут содержать пробелы, '4.6', '230 000'),
      - Fuel, Transmission (строки),
    и порог price_threshold (float).

    Возвращает словарь с метриками (accuracy, confusion_matrix и т.д.).
    """

    # 0) Функция, убирающая все символы, кроме цифр и точки,
    #    чтобы можно было корректно парсить '4.6' или '230 000' => '230000'
    #    а '4.6' => '4.6'
    def clean_numeric(val: str) -> str:
        # Оставляем только цифры и точку
        import re
        cleaned = re.sub(r"[^0-9.]+", "", val)
        return cleaned

    # Приведём все нужные колонки к float

    # --- Price ---
    df["Price"] = df["Price"].astype(str).apply(clean_numeric)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    # --- Year ---
    df["Year"] = df["Year"].astype(str).apply(clean_numeric)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    # --- Mileage ---
    df["Mileage"] = df["Mileage"].astype(str).apply(clean_numeric)
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
    # --- EngineVolume ---
    df["EngineVolume"] = df["EngineVolume"].astype(str).apply(clean_numeric)
    df["EngineVolume"] = pd.to_numeric(df["EngineVolume"], errors="coerce")

    # Убираем строки, где Price не получилось сконвертировать
    df.dropna(subset=["Price"], inplace=True)

    # 1) Создаём целевой признак is_expensive (0 или 1)
    df["is_expensive"] = (df["Price"] > price_threshold).astype(int)

    # 2) Убираем строки, где нет нужных данных
    df.dropna(subset=["Price", "Year", "Mileage", "EngineVolume", "Fuel", "Transmission"], inplace=True)

    print("Price threshold:", price_threshold)
    print("Количество записей:", len(df))
    print("Min price:", df["Price"].min())
    print("Max price:", df["Price"].max())
    print("Value counts for is_expensive:")
    print(df["is_expensive"].value_counts())

    # Сравниваем с int/float
    df = df[
        (df["Year"] > 1900) &
        (df["Mileage"] > 0) &
        (df["EngineVolume"] > 0)
    ]
    if df.empty:
        raise ValueError("После фильтрации не осталось записей для обучения логистической регрессии.")

    # 3) Формируем X (признаки) и y (целевой признак)
    numeric_cols = ["Year", "Mileage", "EngineVolume"]
    cat_cols = ["Fuel", "Transmission"]

    # Создаем dummy для категориальных
    df_dummies = pd.get_dummies(df[cat_cols], drop_first=True)

    X = pd.concat([df[numeric_cols], df_dummies], axis=1)
    y = df["is_expensive"]

    if X.empty or len(X) < 2:
        raise ValueError("Слишком мало данных для обучения логистической регрессии.")

    # 4) Обучаем LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Предсказание на тех же данных (для демонстрации)
    y_pred = model.predict(X)

    # 5) Метрики
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    # Результат
    results = {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "coefs": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "features": X.columns.tolist(),
    }

    return results
