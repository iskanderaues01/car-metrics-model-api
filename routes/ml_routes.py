import json

from fastapi import APIRouter
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from services.logistic_service import run_logistic_regression
from services.paths_service import get_parsed_data_path, get_ml_results_path
from services.paths_service import get_parsed_data_path, get_ml_results_path
from services.future_price_service import train_price_model, predict_future_price
router = APIRouter()

@router.get("/analysis-linear_by_year")
def analysis_linear_by_year(filename: str):
    """
    Принимает:
      - filename: название JSON-файла, например "toyota_camry_2014_2015_3.json"

    1) Открывает JSON-файл в папке parsed_data
    2) Считывает первый объект, из поля "Title" извлекает марку (car_brand) и модель (car_model)
    3) Приводит Price, Year к числам
    4) Выполняет линейную регрессию (Price ~ Year)
    5) Формирует график с заголовком "Цена в зависимости от года выпуска\n{car_brand} {car_model}"
    6) Сохраняет PNG
    7) Возвращает метрики и путь к графику
    """

    # 1) Путь к файлу
    parsed_dir = get_parsed_data_path()
    file_path = os.path.join(parsed_dir, filename)

    if not os.path.exists(file_path):
        return {"error": f"Файл не найден: {file_path}"}

    # 2) Читаем JSON
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return {"error": "Файл JSON пустой или некорректный."}

    # Извлекаем марку и модель из первого объекта
    # Предположим, что поле "Title" содержит что-то вроде "Toyota Camry"
    first_title = data[0].get("Title", "").strip()
    if not first_title:
        return {"error": "Не удалось определить марку и модель из Title первого объекта."}

    # Разделим строку "Toyota Camry" на ["Toyota", "Camry"]
    parts = first_title.split(maxsplit=1)
    car_brand = parts[0] if len(parts) >= 1 else "UnknownBrand"
    car_model = parts[1] if len(parts) >= 2 else "UnknownModel"

    # 3) Приводим к DataFrame и фильтруем
    df = pd.DataFrame(data)

    # Price -> float (убираем нецифровые символы)
    df["Price"] = df["Price"].replace(r"\D+", "", regex=True).astype(float)

    # Year -> число
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Фильтруем
    df.dropna(subset=["Year", "Price"], inplace=True)
    df = df[(df["Price"] > 0) & (df["Year"] > 1900)]

    if df.empty:
        return {"error": "После фильтрации не осталось записей для анализа."}

    # 4) Линейная регрессия: Price ~ Year
    X = df[["Year"]]
    y = df["Price"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    slope = model.coef_[0]
    intercept = model.intercept_

    # 5) График
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Year"], df["Price"], alpha=0.5, label="Данные")

    x_sorted = np.sort(df["Year"].unique())
    y_sorted = model.predict(x_sorted.reshape(-1, 1))
    plt.plot(x_sorted, y_sorted, color="red", label="Линейная регрессия")

    plt.xlabel("Год выпуска")
    plt.ylabel("Цена")
    plt.title(f"Цена в зависимости от года выпуска\n{car_brand} {car_model}")
    plt.legend()

    # 6) Сохраняем график
    ml_dir = get_ml_results_path()
    base_name, _ = os.path.splitext(filename)
    plot_filename = f"{base_name}_analysis_linear_year.png"
    plot_path = os.path.join(ml_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    # 7) Возврат метрик и пути
    return {
        "message": "Анализ завершён (Linear Regression): Цена ~ Год.",
        "FileAnalyzed": filename,
        "CountRecords": len(df),
        "MSE": mse,
        "R^2": r2,
        "Equation": f"Price = {slope:.2f} * Year + {intercept:.2f}",
        "PlotPath": plot_path,
        "CarBrand": car_brand,
        "CarModel": car_model
    }

@router.get("/analysis-linear-by-mileage")
def analysis_linear_by_mileage(filename: str):
    """
    Пример запроса:
      GET /analysis-linear-by-mileage?filename=toyota_camry_2014_2015_3.json

    1) Ищет JSON-файл (filename) в директории parsed_data.
    2) Извлекает из первого объекта поля "Title" => car_brand, car_model (разделяем строку).
    3) Превращает список объявлений в DataFrame, приводит 'Price' и 'Mileage' к float.
    4) Фильтрует некорректные записи (Price <= 0, Mileage <= 0, пустые).
    5) Строит линейную регрессию: Price ~ Mileage.
    6) Рисует график "Цена в зависимости от пробега\n{car_brand} {car_model}".
    7) Сохраняет под именем "{base_name}_analysis_linear_mileage.png" в ml_results.
    8) Возвращает метрики, путь к изображению и пр.
    """

    # Шаг 1: Путь к JSON-файлу
    parsed_dir = get_parsed_data_path()
    file_path = os.path.join(parsed_dir, filename)
    if not os.path.exists(file_path):
        return {"error": f"Файл не найден: {file_path}"}

    # Шаг 2: Считываем JSON
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return {"error": "Файл JSON пуст или некорректен."}

    # Извлекаем бренд и модель из первого объекта (поле "Title")
    first_title = data[0].get("Title", "").strip()
    if not first_title:
        return {"error": "Не удалось определить марку и модель из Title первого объекта."}

    parts = first_title.split(maxsplit=1)
    car_brand = parts[0] if len(parts) >= 1 else "UnknownBrand"
    car_model = parts[1] if len(parts) >= 2 else "UnknownModel"

    # Шаг 3: Превращаем в DataFrame
    df = pd.DataFrame(data)

    # Приводим Price к float
    df["Price"] = df["Price"].replace(r"\D+", "", regex=True).astype(float)

    # Приводим Mileage к float (в JSON она уже строка, но там должны быть цифры)
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")

    # Шаг 4: Фильтр (убираем NaN, нули, отрицательные)
    df.dropna(subset=["Mileage", "Price"], inplace=True)
    df = df[(df["Price"] > 0) & (df["Mileage"] > 0)]

    if df.empty:
        return {"error": "После фильтрации не осталось записей для анализа (Price/Mileage невалидны)."}

    # Шаг 5: Линейная регрессия Price ~ Mileage
    X = df[["Mileage"]]
    y = df["Price"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    slope = model.coef_[0]
    intercept = model.intercept_

    # Шаг 6: Рисуем график
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Mileage"], df["Price"], alpha=0.5, label="Данные")

    # Чтобы построить "линию", сортируем значения пробега
    x_sorted = np.sort(df["Mileage"].unique())
    y_sorted = model.predict(x_sorted.reshape(-1, 1))
    plt.plot(x_sorted, y_sorted, color="red", label="Линейная регрессия")

    plt.xlabel("Пробег (км)")
    plt.ylabel("Цена")
    plt.title(f"Цена в зависимости от пробега\n{car_brand} {car_model}")
    plt.legend()

    # Шаг 7: Сохраняем
    ml_dir = get_ml_results_path()
    base_name, _ = os.path.splitext(filename)
    plot_filename = f"{base_name}_analysis_linear_mileage.png"
    plot_path = os.path.join(ml_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    # Шаг 8: Возвращаем результаты
    return {
        "message": "Линейная регрессия: Цена ~ Пробег. Анализ завершён.",
        "FileAnalyzed": filename,
        "CountRecords": len(df),
        "MSE": mse,
        "R^2": r2,
        "Equation": f"Price = {slope:.2f} * Mileage + {intercept:.2f}",
        "PlotPath": plot_path,
        "CarBrand": car_brand,
        "CarModel": car_model
    }

@router.get("/analysis-linear-by-engine-volume")
def analysis_linear_by_engine_volume(filename: str):
    """
    Пример запроса:
      GET /analysis-linear-by-engine-volume?filename=toyota_camry_2014_2015_3.json

    1) Ищет JSON-файл (filename) в директории parsed_data.
    2) Извлекает из первого объекта поля "Title" => car_brand, car_model.
    3) Превращает список объявлений в DataFrame, приводит 'Price' и 'EngineVolume' к float.
    4) Фильтрует некорректные записи (Price <= 0, EngineVolume <= 0).
    5) Строит линейную регрессию: Price ~ EngineVolume.
    6) Рисует график: "Цена в зависимости от объёма двигателя\n{car_brand} {car_model}".
    7) Сохраняет изображение в ml_results под именем "{base_name}_analysis_linear_enginevolume.png".
    8) Возвращает JSON с метриками (MSE, R^2), уравнением, путём к графику и т.д.
    """

    # 1) Путь к файлу
    parsed_dir = get_parsed_data_path()
    file_path = os.path.join(parsed_dir, filename)
    if not os.path.exists(file_path):
        return {"error": f"Файл не найден: {file_path}"}

    # 2) Считываем JSON
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return {"error": "Файл JSON пуст или некорректен."}

    # Извлекаем бренд и модель из первого объекта (поле "Title")
    first_title = data[0].get("Title", "").strip()
    if not first_title:
        return {"error": "Не удалось определить марку и модель из Title первого объекта."}

    parts = first_title.split(maxsplit=1)
    car_brand = parts[0] if len(parts) >= 1 else "UnknownBrand"
    car_model = parts[1] if len(parts) >= 2 else "UnknownModel"

    # 3) Превращаем в DataFrame
    df = pd.DataFrame(data)

    # Price -> float (убираем нецифровые символы)
    df["Price"] = df["Price"].replace(r"\D+", "", regex=True).astype(float)

    # EngineVolume -> float (например, "2.5" или "3" и т.д.)
    df["EngineVolume"] = pd.to_numeric(df["EngineVolume"], errors="coerce")

    # 4) Фильтр (убираем NaN, нули, отрицательные)
    df.dropna(subset=["EngineVolume", "Price"], inplace=True)
    df = df[(df["Price"] > 0) & (df["EngineVolume"] > 0)]

    if df.empty:
        return {"error": "После фильтрации не осталось записей для анализа (Price/EngineVolume невалидны)."}

    # 5) Линейная регрессия Price ~ EngineVolume
    X = df[["EngineVolume"]]
    y = df["Price"]

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    slope = model.coef_[0]
    intercept = model.intercept_

    # 6) Рисуем график
    plt.figure(figsize=(8, 6))
    plt.scatter(df["EngineVolume"], df["Price"], alpha=0.5, label="Данные")

    # Чтобы построить линию регрессии, отсортируем значения
    x_sorted = np.sort(df["EngineVolume"].unique())
    y_sorted = model.predict(x_sorted.reshape(-1, 1))
    plt.plot(x_sorted, y_sorted, color="red", label="Линейная регрессия")

    plt.xlabel("Объём двигателя (л)")
    plt.ylabel("Цена")
    plt.title(f"Цена в зависимости от объёма двигателя\n{car_brand} {car_model}")
    plt.legend()

    # 7) Сохраняем изображение
    ml_dir = get_ml_results_path()
    base_name, _ = os.path.splitext(filename)
    plot_filename = f"{base_name}_analysis_linear_enginevolume.png"
    plot_path = os.path.join(ml_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    # 8) Возвращаем результаты
    return {
        "message": "Линейная регрессия: Цена ~ Объём двигателя. Анализ завершён.",
        "FileAnalyzed": filename,
        "CountRecords": len(df),
        "MSE": mse,
        "R^2": r2,
        "Equation": f"Price = {slope:.2f} * EngineVolume + {intercept:.2f}",
        "PlotPath": plot_path,
        "CarBrand": car_brand,
        "CarModel": car_model
    }


@router.get("/analysis-multiple-linear")
def analysis_multiple_linear(filename: str):
    """
    Множественная линейная регрессия: Цена ~ Год + Пробег + Объём двигателя.

    Пример вызова:
      GET /analysis-multiple-linear?filename=toyota_camry_2014_2015_3.json

    1) Берём JSON-файл из parsed_data
    2) Парсим поля Price, Year, Mileage, EngineVolume
    3) Фильтруем неверные или пустые данные
    4) Обучаем модель Price ~ Year + Mileage + EngineVolume
    5) Рисуем три subplot частичных зависимостей на русском
    6) Сохраняем результат и возвращаем JSON с метриками
    """

    # 1) Путь к файлу JSON
    parsed_dir = get_parsed_data_path()
    file_path = os.path.join(parsed_dir, filename)
    if not os.path.exists(file_path):
        return {"error": f"Файл не найден: {file_path}"}

    # 2) Читаем JSON
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return {"error": "Файл JSON пуст или некорректен."}

    # Извлекаем марку и модель (для заголовка) из первого объекта
    first_title = data[0].get("Title", "").strip() if data else ""
    parts = first_title.split(maxsplit=1)
    car_brand = parts[0] if len(parts) >= 1 else "UnknownBrand"
    car_model = parts[1] if len(parts) >= 2 else "UnknownModel"

    # 3) DataFrame + приведение типов
    df = pd.DataFrame(data)

    df["Price"] = df["Price"].replace(r"\D+", "", regex=True).astype(float)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
    df["EngineVolume"] = pd.to_numeric(df["EngineVolume"], errors="coerce")

    # 4) Фильтрация
    df.dropna(subset=["Price", "Year", "Mileage", "EngineVolume"], inplace=True)
    df = df[
        (df["Price"] > 0) &
        (df["Year"] > 1900) &
        (df["Mileage"] > 0) &
        (df["EngineVolume"] > 0)
        ]
    if df.empty:
        return {"error": "После фильтрации не осталось валидных записей для анализа."}

    # 5) Модель множественной линейной регрессии
    X = df[["Year", "Mileage", "EngineVolume"]]
    y = df["Price"]
    model = LinearRegression()
    model.fit(X, y)

    # Предсказания
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    coef_year = model.coef_[0]
    coef_mileage = model.coef_[1]
    coef_engine_volume = model.coef_[2]
    intercept = model.intercept_

    # 6) Визуализация частичных зависимостей
    mean_year = df["Year"].mean()
    mean_mileage = df["Mileage"].mean()
    mean_engine_volume = df["EngineVolume"].mean()

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Заголовок на русском
    fig.suptitle(
        f"Множественная регрессия: Цена ~ Год + Пробег + Объём двигателя\n{car_brand} {car_model}",
        fontsize=14
    )

    # --- (1) Цена ~ Год ---
    x_min, x_max = int(df["Year"].min()), int(df["Year"].max())
    x_range = np.arange(x_min, x_max + 1)
    X_line = pd.DataFrame({
        "Year": x_range,
        "Mileage": mean_mileage,
        "EngineVolume": mean_engine_volume
    })
    y_line = model.predict(X_line)

    axs[0].scatter(df["Year"], df["Price"], alpha=0.5, label="Данные")
    axs[0].plot(x_range, y_line, color="red", label="Лин. регрессия")
    axs[0].set_xlabel("Год")
    axs[0].set_ylabel("Цена")
    axs[0].legend()
    axs[0].set_title("При фикс.\nПробег, Объём двигателя")

    # --- (2) Цена ~ Пробег ---
    x_min2, x_max2 = int(df["Mileage"].min()), int(df["Mileage"].max())
    x_range2 = np.linspace(x_min2, x_max2, 50)
    X_line2 = pd.DataFrame({
        "Year": mean_year,
        "Mileage": x_range2,
        "EngineVolume": mean_engine_volume
    })
    y_line2 = model.predict(X_line2)

    axs[1].scatter(df["Mileage"], df["Price"], alpha=0.5, label="Данные")
    axs[1].plot(x_range2, y_line2, color="red", label="Лин. регрессия")
    axs[1].set_xlabel("Пробег")
    axs[1].set_ylabel("Цена")
    axs[1].legend()
    axs[1].set_title("При фикс.\nГод, Объём двигателя")

    # --- (3) Цена ~ Объём двигателя ---
    x_min3, x_max3 = df["EngineVolume"].min(), df["EngineVolume"].max()
    x_range3 = np.linspace(x_min3, x_max3, 50)
    X_line3 = pd.DataFrame({
        "Year": mean_year,
        "Mileage": mean_mileage,
        "EngineVolume": x_range3
    })
    y_line3 = model.predict(X_line3)

    axs[2].scatter(df["EngineVolume"], df["Price"], alpha=0.5, label="Данные")
    axs[2].plot(x_range3, y_line3, color="red", label="Лин. регрессия")
    axs[2].set_xlabel("Объём двигателя")
    axs[2].set_ylabel("Цена")
    axs[2].legend()
    axs[2].set_title("При фикс.\nГод, Пробег")

    plt.tight_layout()

    # 7) Сохраняем рисунок
    ml_dir = get_ml_results_path()
    base_name, _ = os.path.splitext(filename)
    plot_filename = f"{base_name}_analysis_multiple_linear.png"
    plot_path = os.path.join(ml_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    # 8) Возвращаем результат
    return {
        "message": "Множественная регрессия: Цена ~ Год + Пробег + Объём двигателя. Анализ завершён.",
        "FileAnalyzed": filename,
        "CarBrand": car_brand,
        "CarModel": car_model,
        "CountRecords": len(df),
        "MSE": mse,
        "R^2": r2,
        "Equation": (
            f"Price = {intercept:.2f} "
            f"+ ({coef_year:.2f} * Год) "
            f"+ ({coef_mileage:.2f} * Пробег) "
            f"+ ({coef_engine_volume:.2f} * ОбъёмДвигателя)"
        ),
        "PlotPath": plot_path
    }


# routes/ml_routes.py
import os
import json
from services.paths_service import get_parsed_data_path, get_ml_results_path

@router.get("/analysis-multiple-dummies")
def analysis_multiple_dummies(filename: str):
    """
    Пример запроса:
      GET /analysis-multiple-dummies?filename=toyota_camry_2014_2015_3.json

    Выполняем множественную линейную регрессию с учётом категориальных признаков:
    Цена ~ Год + Пробег + ОбъёмДвигателя + Топливо + Трансмиссия.
    (Fuel и Transmission будут преобразованы в dummy-переменные).
    """

    print("[LOG] => analysis_multiple_dummies STARTED")
    print(f"[LOG] => Received filename: {filename}")

    # 1) Путь к JSON
    parsed_dir = get_parsed_data_path()
    print(f"[LOG] => Parsed data directory: {parsed_dir}")
    file_path = os.path.join(parsed_dir, filename)
    print(f"[LOG] => Full file path: {file_path}")

    if not os.path.exists(file_path):
        print(f"[LOG] => File not found: {file_path}")
        return {"error": f"Файл не найден: {file_path}"}

    # 2) Загружаем JSON
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[LOG] => JSON loaded, total records: {len(data) if data else 0}")
    except Exception as e:
        print(f"[LOG] => Error reading JSON: {e}")
        return {"error": f"Ошибка чтения JSON: {str(e)}"}

    if not data:
        print("[LOG] => JSON is empty or invalid.")
        return {"error": "Файл JSON пуст или некорректен."}

    # Извлекаем название (Brand, Model) из первого объявления
    first_title = data[0].get("Title", "").strip()
    parts = first_title.split(maxsplit=1)
    car_brand = parts[0] if len(parts) >= 1 else "UnknownBrand"
    car_model = parts[1] if len(parts) >= 2 else "UnknownModel"
    print(f"[LOG] => Extracted Brand: {car_brand}, Model: {car_model}")

    # 3) DataFrame
    df = pd.DataFrame(data)
    print(f"[LOG] => Created DataFrame, shape: {df.shape}")

    # Приводим Price к float, Year, Mileage, EngineVolume к числу
    print("[LOG] => Converting Price, Year, Mileage, EngineVolume to numeric...")
    df["Price"] = df["Price"].replace(r"\D+", "", regex=True).astype(float)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
    df["EngineVolume"] = pd.to_numeric(df["EngineVolume"], errors="coerce")

    # 4) Фильтрация
    print("[LOG] => Dropping NaN for required columns and filtering out invalid rows...")
    df.dropna(subset=["Price", "Year", "Mileage", "EngineVolume", "Fuel", "Transmission"], inplace=True)
    df = df[
        (df["Price"] > 0) &
        (df["Year"] > 1900) &
        (df["Mileage"] > 0) &
        (df["EngineVolume"] > 0)
        ]
    print(f"[LOG] => DataFrame after filtering, shape: {df.shape}")

    if df.empty:
        print("[LOG] => No valid records left after filtering.")
        return {"error": "После фильтрации не осталось валидных записей."}

    # 5) Формируем X, y для регрессии
    print("[LOG] => Preparing numeric and categorical columns for regression.")
    numeric_cols = ["Year", "Mileage", "EngineVolume"]
    cat_cols = ["Fuel", "Transmission"]

    # Для категориальных делаем dummy-переменные
    print("[LOG] => Generating dummy variables (one-hot encoding)...")
    df_dummies = pd.get_dummies(df[cat_cols], drop_first=True)
    print(f"[LOG] => Dummy columns created: {list(df_dummies.columns)}")

    # Объединяем числовые и дамми-колонки
    X = pd.concat([df[numeric_cols], df_dummies], axis=1)
    y = df["Price"]
    print("[LOG] => Final feature set shape:", X.shape)

    # 6) Обучение модели
    print("[LOG] => Training LinearRegression model...")
    model = LinearRegression()
    model.fit(X, y)

    # Предсказание
    print("[LOG] => Making predictions...")
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    intercept = model.intercept_
    coefs = model.coef_
    feature_names = X.columns.tolist()

    print(f"[LOG] => Model trained. MSE={mse:.2f}, R^2={r2:.2f}, intercept={intercept:.2f}")

    # 7) Визуализация (упрощённая, на русском)
    print("[LOG] => Building bar chart of coefficients...")
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, coefs, color="cadetblue")
    plt.xlabel("Значение коэффициента")
    plt.title(
        f"Множественная регрессия (dummy-переменные)\n{car_brand} {car_model}\n"
        "Цена ~ Год + Пробег + ОбъёмДвигателя + Топливо + Трансмиссия"
    )
    plt.tight_layout()

    # 8) Сохраняем
    ml_dir = get_ml_results_path()
    print(f"[LOG] => ML results directory: {ml_dir}")
    base_name, _ = os.path.splitext(filename)
    plot_filename = f"{base_name}_analysis_multiple_dummies.png"
    plot_path = os.path.join(ml_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"[LOG] => Plot saved to: {plot_path}")

    # 9) Формируем строку уравнения (для наглядности)
    eq_parts = [f"{intercept:.2f}"]
    for feat, c in zip(feature_names, coefs):
        eq_parts.append(f"({c:.2f} * {feat})")
    eq_str = " + ".join(eq_parts)

    print("[LOG] => analysis_multiple_dummies FINISHED - returning results.")

    return {
        "message": (
            "Множественная линейная регрессия с категориальными переменными (dummy): "
            "Цена ~ Год + Пробег + ОбъёмДвигателя + Топливо + Трансмиссия. Анализ завершён."
        ),
        "FileAnalyzed": filename,
        "CarBrand": car_brand,
        "CarModel": car_model,
        "CountRecords": len(df),
        "MSE": mse,
        "R^2": r2,
        "Equation": f"Price = {eq_str}",
        "PlotPath": plot_path,
        "DummyFeatures": df_dummies.columns.tolist()
    }

# Логистическая регрессия
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from fastapi import APIRouter
from services.paths_service import get_parsed_data_path, get_ml_results_path
from services.logistic_service import run_logistic_regression
@router.get("/analysis-logistic")
def analysis_logistic(filename: str, price_threshold: float):
    """
    Пример запроса:
      GET /analysis-logistic?filename=toyota_camry_2014_2015_3.json&price_threshold=5000000

    1) Проверяем наличие JSON-файла (filename) в parsed_data
    2) Загружаем DataFrame
    3) Запускаем логистическую регрессию (Price ~ Year + Mileage + EngineVolume + Fuel + Transmission),
       где класс is_expensive определяется price_threshold.
    4) Строим confusion matrix, сохраняем её в PNG (с подписями на русском).
    5) Сохраняем результат (метрики + описание) в JSON (подписи тоже на русском).
    6) Возвращаем пути к сохранённому PNG, JSON и итоги анализа.
    """

    # 1) Путь к файлу
    parsed_dir = get_parsed_data_path()
    file_path = os.path.join(parsed_dir, filename)
    if not os.path.exists(file_path):
        return {"error": f"Файл не найден: {file_path}"}

    # 2) Читаем JSON => DataFrame
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return {"error": "JSON-файл пуст или некорректен."}

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "После загрузки из JSON нет данных."}

    # 3) Запускаем логистическую регрессию
    try:
        results = run_logistic_regression(df, price_threshold)
    except Exception as e:
        return {"error": f"Ошибка при выполнении логистической регрессии: {str(e)}"}

    # Достаём нужные метрики
    accuracy = results["accuracy"]
    cm_list = results["confusion_matrix"]  # это list, пригодный для JSON
    cm = np.array(cm_list)                 # превращаем в np.array для ConfusionMatrixDisplay

    # 4) Строим confusion matrix и сохраняем (русские подписи)
    plt.figure(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Недорого (0)", "Дорого (1)"]  # русские метки классов
    )
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Матрица ошибок (Порог цены > {price_threshold})\nТочность (Accuracy): {accuracy:.2f}")
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")

    ml_dir = get_ml_results_path()
    base_name, _ = os.path.splitext(filename)
    cm_plot_name = f"{base_name}_analysis_logistic_cm.png"
    cm_plot_path = os.path.join(ml_dir, cm_plot_name)
    plt.savefig(cm_plot_path)
    plt.close()

    # 5) Формируем краткое пояснение (на русском)
    explanation = (
        f"Логистическая регрессия: задача «Дорого ли авто?» (Порог цены = {price_threshold}).\n"
        "Если Price выше порога, класс 1 (Дорого), иначе 0 (Недорого).\n"
        "Матрица ошибок (confusion matrix) показывает, сколько объектов верно/неверно отнесены к классам.\n"
        "Точность (accuracy) указывает долю верных классификаций."
    )

    # Собираем словарь с итоговой информацией (русские ключи там, где уместно)
    analysis_result = {
        "message": "Логистическая регрессия успешно выполнена.",
        "fileAnalyzed": file_path,
        "priceThreshold": price_threshold,
        "accuracy": accuracy,
        "confusionMatrix": cm_list,
        "coefficients": results["coefs"],
        "intercept": results["intercept"],
        "features": results["features"],
        "explanation": explanation
    }

    # 6) Сохраняем результат анализа в JSON (тоже с русскими ключами)
    json_filename = f"{base_name}_analysis_logistic.json"
    analysis_path = os.path.join(ml_dir, json_filename)
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)

    # Возвращаем ответ
    return {
        "message": "Логистическая регрессия выполнена. PNG и JSON сохранены.",
        "analysisResult": analysis_result,
        "confusionMatrixPlotPath": cm_plot_path,
        "savedResultPath": analysis_path
    }

# routes/price_forecast_routes.py
@router.get("/analysis-future-price")
def analysis_future_price(
    filename: str,
    future_year: str,
    future_mileage: str,
    future_engine_volume: str,
    future_fuel: str,
    future_transmission: str
):
    """
    Пример запроса:
      GET /analysis-future-price?filename=today_cars.json
        &future_year=2020
        &future_mileage=90000
        &future_engine_volume=1.6
        &future_fuel=бензин
        &future_transmission=автомат

    1) Загружаем DataFrame из JSON (список объявлений на сегодня).
    2) Обучаем линейную регрессию (Price ~ Year + Mileage + EngineVolume + Fuel + Transmission).
    3) Предсказываем цену для "будущего" авто (задаётся параметрами).
    4) Сохраняем график (гистограмму) + JSON с результатами.
    5) Возвращаем итоговый ответ.
    """

    parsed_dir = get_parsed_data_path()
    file_path = os.path.join(parsed_dir, filename)
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return {"error": "JSON file is empty or invalid."}

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data after loading JSON."}

    # Обучаем модель
    try:
        model, feature_names = train_price_model(df)
    except Exception as e:
        return {"error": f"Error training model: {str(e)}"}

    # Собираем словарь характеристик для "будущего" авто
    future_data = {
        "Year": future_year,
        "Mileage": future_mileage,
        "EngineVolume": future_engine_volume,
        "Fuel": future_fuel,
        "Transmission": future_transmission
    }

    # Предсказываем цену
    try:
        predicted_price = predict_future_price(model, feature_names, future_data)
    except Exception as e:
        return {"error": f"Error predicting future price: {str(e)}"}

    # Считаем среднюю и медиану
    avg_price = df["Price"].mean()
    median_price = df["Price"].median()

    # Рисуем гистограмму по Price
    plt.figure(figsize=(8,5))
    df["Price"].plot(kind="hist", bins=20, alpha=0.6, label="Текущее распределение цен")

    # Линия для predicted_price
    plt.axvline(x=predicted_price, color="red", linestyle="--",
                label=f"Прогнозируемая будущая цена: ~ {int(predicted_price)}")

    # Линии для средней и медианы
    plt.axvline(x=avg_price, color="green", linestyle=":",
                label=f"Средняя цена: ~ {int(avg_price)}")
    plt.axvline(x=median_price, color="blue", linestyle=":",
                label=f"Медиана: ~ {int(median_price)}")

    plt.title("Текущее распределение цен и примерная будущая оценка")
    plt.xlabel("Цена")
    plt.ylabel("Частота (сколько объявлений попадает в данный диапазон цен)")
    plt.legend()

    ml_dir = get_ml_results_path()
    base_name, _ = os.path.splitext(filename)
    plot_name = f"{base_name}_future_price.png"
    plot_path = os.path.join(ml_dir, plot_name)
    plt.savefig(plot_path)
    plt.close()

    # Формируем более подробное пояснение
    explanation = (
        "Здесь мы рассматриваем объявления, имеющиеся на сегодня, и обучаем модель линейной регрессии, "
        "которая оценивает цену (Price) на основе параметров Year, Mileage, EngineVolume, Fuel и Transmission. "
        "Затем мы рассчитываем примерную стоимость для будущего авто, характеристики которого указаны в параметрах запроса.\n"
        "Гистограмма на графике показывает, как распределяются текущие цены (ось X — цена, ось Y — частота, т.е. "
        "сколько объявлений попало в тот или иной ценовой диапазон). Красной линией отмечена прогнозируемая цена "
        "для указанного будущего авто, зелёной — средняя цена по выборке, синей — медианная."
    )

    analysis_result = {
        "message": "Future price analysis done.",
        "fileAnalyzed": file_path,
        "futureData": future_data,
        "predictedPrice": predicted_price,
        "averagePrice": float(avg_price),
        "medianPrice": float(median_price),
        "explanation": explanation
    }

    # Сохраняем результат в JSON
    json_filename = f"{base_name}_future_price.json"
    analysis_path = os.path.join(ml_dir, json_filename)
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)

    return {
        "message": "Future price predicted. Graph and JSON are saved.",
        "analysisResult": analysis_result,
        "priceDistributionPlot": plot_path,
        "savedResultPath": analysis_path
    }


# routes/epoch_training_routes.py
from services.paths_service import get_parsed_data_path, get_ml_results_path
from services.epoch_training_service import train_sgd_regressor_with_epochs

@router.get("/analysis-epochs")
def analysis_epochs(
    filename: str,
    epochs: int = 5,
    batch_size: int = 32
):
    """
    Пример вызова:
      GET /analysis-epochs?filename=car_data.json&epochs=10&batch_size=32

    - Загружаем JSON (car_data.json), где есть Year, Mileage, EngineVolume, Price
    - Очищаем, приводим к float
    - Вызываем train_sgd_regressor_with_epochs (метод partial_fit на каждую эпоху)
    - Получаем epoch_losses, final_mse, final_r2
    - Рисуем график (epoch vs. loss)
    - Сохраняем PNG и JSON
    - Возвращаем результат
    """

    parsed_dir = get_parsed_data_path()
    file_path = os.path.join(parsed_dir, filename)
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return {"error": "JSON file empty or invalid."}

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data after loading JSON."}

    # Предположим, что есть Year, Mileage, EngineVolume, Price
    for col in ["Price", "Year", "Mileage", "EngineVolume"]:
        if col not in df.columns:
            return {"error": f"Column {col} not found in data."}
        df[col] = df[col].astype(str).replace(r"[^0-9.]+", "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Price", "Year", "Mileage", "EngineVolume"], inplace=True)
    df = df[df["Price"] > 0]
    if df.empty:
        return {"error": "No valid numeric data after cleaning."}

    X = df[["Year","Mileage","EngineVolume"]]
    y = df["Price"]

    # Обучаем
    try:
        training_result = train_sgd_regressor_with_epochs(X, y, epochs=epochs, batch_size=batch_size)
    except Exception as e:
        return {"error": f"Error in epoch training: {str(e)}"}

    epoch_losses = training_result["epoch_losses"]
    final_mse = training_result["final_mse"]
    final_r2 = training_result["final_r2"]

    # Рисуем график epoch vs. loss
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs+1), epoch_losses, marker="o")
    plt.title("SGDRegressor: epoch vs. loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss per epoch (train set)")
    plt.grid(True)

    ml_dir = get_ml_results_path()
    base_name, _ = os.path.splitext(filename)
    plot_name = f"{base_name}_epoch_training.png"
    plot_path = os.path.join(ml_dir, plot_name)
    plt.savefig(plot_path)
    plt.close()

    # Дополнительное описание
    explanation = (
        "Обучение линейной регрессии методом стохастического градиентного спуска "
        f"в течение {epochs} эпох, с размером батча = {batch_size}.\n"
        "Loss в каждой эпохе усредняется по батчам.\n"
        f"По окончании обучения модель оценена на всей выборке:\n"
        f"MSE (mean squared error) = {final_mse:.2f}\n"
        f"R^2 (коэффициент детерминации) = {final_r2:.3f}\n"
        "Чем выше R^2, тем лучше модель объясняет дисперсию цены. MSE показывает "
        "среднеквадратичную ошибку, чем меньше, тем лучше."
    )

    analysis_result = {
        "message": "Epoch-based training done.",
        "fileAnalyzed": file_path,
        "epochs": epochs,
        "batchSize": batch_size,
        "epochLosses": epoch_losses,
        "finalMSE": final_mse,
        "finalR2": final_r2,
        "explanation": explanation
    }

    json_filename = f"{base_name}_epoch_training.json"
    analysis_path = os.path.join(ml_dir, json_filename)
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)

    return {
        "message": "SGDRegressor training with epochs completed, PNG and JSON saved.",
        "analysisResult": analysis_result,
        "epochTrainingPlot": plot_path,
        "savedResultPath": analysis_path
    }


