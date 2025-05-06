# services/logistic_service.py
import os
import json
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier

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

    # 0. Убираем всё, кроме цифр и точки
    def clean_numeric(val: str) -> str:
        import re
        return re.sub(r"[^0-9.]+", "", str(val))

    for col in ["Price", "Year", "Mileage", "EngineVolume"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).apply(clean_numeric),
            errors="coerce"
        )

    df.dropna(subset=["Price"], inplace=True)

    # 1. Обрезаем порог в рамки [min_price, max_price]
    min_price, max_price = df["Price"].min(), df["Price"].max()
    price_threshold = max(min_price, min(price_threshold, max_price))

    # 2. Формируем целевой признак
    df["is_expensive"] = (df["Price"] > price_threshold).astype(int)

    # 3. Убираем строки с пропусками в ключевых колонках
    df.dropna(subset=["Year", "Mileage", "EngineVolume", "Fuel", "Transmission"], inplace=True)

    # 4. Фильтрация по годам/пробегу/объёму
    df = df[
        (df["Year"] > 1900) &
        (df["Mileage"] > 0) &
        (df["EngineVolume"] > 0)
    ]
    if df.empty:
        raise ValueError("После фильтрации не осталось записей для обучения логистической регрессии.")

    # 5. Собираем X и y
    numeric_cols = ["Year", "Mileage", "EngineVolume"]
    cat_cols = ["Fuel", "Transmission"]
    df_dummies = pd.get_dummies(df[cat_cols], drop_first=True)
    X = pd.concat([df[numeric_cols], df_dummies], axis=1)
    y = df["is_expensive"]

    # 6. Выбираем модель: Dummy, если только один класс или мало данных
    is_dummy = (len(y.unique()) < 2) or (len(X) < 2)
    if is_dummy:
        model = DummyClassifier(strategy="most_frequent")
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X, y)
    y_pred = model.predict(X)

    # 7. Считаем метрики
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    # 8. Формируем коэффициенты
    if is_dummy:
        coefs = [[0.0] * X.shape[1]]
        intercept = [0.0]
    else:
        coefs = model.coef_.tolist()
        intercept = model.intercept_.tolist()

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "coefs": coefs,
        "intercept": intercept,
        "features": X.columns.tolist(),
    }