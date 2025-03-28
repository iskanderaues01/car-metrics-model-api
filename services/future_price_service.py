# services/future_price_service.py

import pandas as pd
import re
from sklearn.linear_model import LinearRegression
from typing import Dict


def train_price_model(df: pd.DataFrame):
    """
    Обучаем модель множественной линейной регрессии на основе
    столбцов:
      - Price (строка или число, возможно с пробелами),
      - Year, Mileage, EngineVolume (строки/числа),
      - Fuel, Transmission (категориальные).

    Возвращаем обученную модель (LinearRegression) и список фич (feature_names).
    """

    # 1) Очистка Price
    df["Price"] = df["Price"].astype(str).replace(r"[^0-9.]+", "", regex=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    # 2) Year, Mileage, EngineVolume -> float
    for col in ["Year", "Mileage", "EngineVolume"]:
        df[col] = df[col].astype(str).replace(r"[^0-9.]+", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Удаляем строки, где нет нужных данных
    df.dropna(subset=["Price", "Year", "Mileage", "EngineVolume", "Fuel", "Transmission"], inplace=True)
    df = df[df["Price"] > 0]
    if df.empty:
        raise ValueError("Нет корректных данных для обучения (после очистки и фильтрации)")

    # 3) Формируем X, y
    numeric_cols = ["Year", "Mileage", "EngineVolume"]
    cat_cols = ["Fuel", "Transmission"]

    # dummy
    df_dummies = pd.get_dummies(df[cat_cols], drop_first=True)

    X = pd.concat([df[numeric_cols], df_dummies], axis=1)
    y = df["Price"]

    if len(X) < 2:
        raise ValueError("Слишком мало данных для обучения.")

    # 4) Обучаем линейную регрессию
    model = LinearRegression()
    model.fit(X, y)

    feature_names = list(X.columns)
    return model, feature_names


def predict_future_price(
        model,
        feature_names,
        future_data: Dict[str, str]
):
    """
    Принимает обученную модель, список фич (feature_names),
    и словарь future_data, где указаны Year, Mileage, EngineVolume, Fuel, Transmission.

    Возвращает предсказанную цену (float).
    """

    # Преобразуем future_data в DataFrame из 1 строки
    # Например, future_data = {"Year": "2020", "Mileage": "50000", "EngineVolume": "1.6", "Fuel": "бензин", "Transmission": "механика"}
    temp_df = pd.DataFrame([future_data])

    # 1) Готовим числовые колонки (Year, Mileage, EngineVolume)
    for col in ["Year", "Mileage", "EngineVolume"]:
        if col in temp_df.columns:
            temp_df[col] = temp_df[col].astype(str).replace(r"[^0-9.]+", "", regex=True)
            temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")
        else:
            temp_df[col] = 0  # или NaN, как вариант

    # 2) dummy для категориальных
    cat_cols = ["Fuel", "Transmission"]
    dummies_df = pd.get_dummies(temp_df[cat_cols], drop_first=True)

    # Собираем все фичи
    numeric_cols = ["Year", "Mileage", "EngineVolume"]
    X_future = pd.concat([temp_df[numeric_cols], dummies_df], axis=1)

    # Возможно, некоторые дамми-колонки отсутствуют,
    # приводим X_future к тем же столбцам, что были у модели
    for col in feature_names:
        if col not in X_future.columns:
            X_future[col] = 0
    # и убираем лишние столбцы, если в X_future что-то есть неподходящее
    X_future = X_future[feature_names]

    # 3) Предсказываем
    pred = model.predict(X_future)[0]
    return float(pred)
