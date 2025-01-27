import os
import random
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import StringIO
from bs4 import BeautifulSoup
from fastapi import FastAPI, Response
from typing import List, Dict

# Для ML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = FastAPI()

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Firefox/89.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Edge/91.0.864.59",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Safari/537.36"
]

# Количество страниц, которые хотим спарсить
NUM_PAGES = 10


def get_random_user_agent() -> Dict[str, str]:
    """Выбираем случайный User-Agent из списка."""
    return {"User-Agent": random.choice(USER_AGENTS)}


def fetch_car_data(page_url: str) -> List[Dict[str, str]]:
    """Забираем данные по объявлениям с конкретной страницы."""
    headers = get_random_user_agent()
    response = requests.get(page_url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch data from {page_url}. "
                        f"Status code: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    car_elements = soup.select(".a-list__item")

    cars = []
    for car in car_elements:
        title_elem = car.select_one(".a-card__title")
        title = title_elem.text.strip() if title_elem else "N/A"

        link_elem = car.select_one(".a-card__link")
        link = f"https://kolesa.kz{link_elem['href']}" if link_elem else "N/A"

        price_elem = car.select_one(".a-card__price")
        price = price_elem.text.strip().replace("\u00a0", " ") if price_elem else "N/A"
        price = price.replace("₸", "").strip()

        description_elem = car.select_one(".a-card__description")
        description = description_elem.text.strip() if description_elem else "N/A"

        # Пропускаем объявления с "На заказ"
        if "На заказ" in description:
            continue

        # Примерный парсинг года из описания (грубо берёт всё до "г.,"):
        year = description.split("г.,")[0].strip() if "г." in description else "N/A"

        # Пропускаем некорректные объявления
        if title == "N/A" or price == "N/A" or year == "N/A":
            continue

        cars.append({
            "Title": title,
            "Price": price,
            "Year": year,
            "Link": link
        })

    return cars


def scrape_multiple_pages(base_url: str, num_pages: int) -> List[Dict[str, str]]:
    """Парсим несколько страниц и собираем данные об автомобилях."""
    all_cars = []
    for page in range(1, num_pages + 1):
        page_url = f"{base_url}&page={page}" if page > 1 else base_url
        print(f"Fetching page {page} -> {page_url}")
        try:
            cars = fetch_car_data(page_url)
            all_cars.extend(cars)
        except Exception as e:
            print(f"Error fetching page {page}: {e}")

        # Рандомная задержка между запросами
        time.sleep(random.uniform(1, 3))

    return all_cars

def get_ml_results_path() -> str:
    """
    Возвращает путь к директории для сохранения результатов ML.
    При необходимости здесь можно сменить стандартный путь.
    """
    ml_dir = "/home/chaplin/Desktop/ml_results"
    os.makedirs(ml_dir, exist_ok=True)
    return ml_dir

def get_parsed_data_path() -> str:
    parsed_dir = "/home/chaplin/Desktop/parsed_data"
    os.makedirs(parsed_dir, exist_ok=True)
    return parsed_dir

@app.get("/download-csv")
def download_csv(
        car_brand: str,
        car_model: str,
        date_max: str
):
    """
    Пример запроса:
    http://127.0.0.1:8000/download-csv?car_brand=toyota&car_model=camry&date_max=2015

    Параметры:
    - car_brand (str): название бренда, например "toyota"
    - car_model (str): модель, например "camry"
    - date_max (str): ограничение по году (year[to])

    Возвращает CSV-файл (attachment) с данными об автомобилях.
    """
    # Формируем динамически URL для парсинга
    base_url = f"https://kolesa.kz/cars/{car_brand}/{car_model}/?year[to]={date_max}"

    # Получаем данные
    car_data = scrape_multiple_pages(base_url, NUM_PAGES)

    if not car_data:
        return {"message": "No data found."}

    # Формируем DataFrame и сохраняем в буфер (StringIO) как CSV
    df = pd.DataFrame(car_data)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Отдаём CSV в ответе как вложение
    response = Response(content=csv_buffer.getvalue(), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=cars.csv"
    return response

@app.get("/save_local")
def save_local(
        car_brand: str,
        car_model: str,
        date_max: str,
        count_pages: int
):
    """
    Пример запроса:
    http://127.0.0.1:8001/save_local?car_brand=toyota&car_model=camry&date_max=2015&count_pages=5

    Параметры:
    - car_brand (str): название бренда, например "toyota"
    - car_model (str): модель, например "camry"
    - date_max (str): ограничение по году (year[to])
    - count_pages (int): количество страниц для парсинга

    Результат:
    - CSV-файл сохраняется локально в директорию "parsed_data"
      с именем {car_brand}_{car_model}_{date_max}_{count_pages}.csv
    - Также возвращается как вложение в HTTP-ответе.
    """
    # Формируем динамически URL для парсинга
    base_url = f"https://kolesa.kz/cars/{car_brand}/{car_model}/?year[to]={date_max}"

    # Запускаем парсинг
    car_data = scrape_multiple_pages(base_url, count_pages)
    if not car_data:
        return {"message": "No data found."}

    # Создаём директорию для сохранения, если её нет
    dir_path = get_parsed_data_path()
    os.makedirs(dir_path, exist_ok=True)

    # Формируем название файла
    filename = f"{car_brand}_{car_model}_{date_max}_{count_pages}.csv"
    file_path = os.path.join(dir_path, filename)

    # Сохраняем результат парсинга в DataFrame и экспортируем в CSV
    df = pd.DataFrame(car_data)
    df.to_csv(file_path, index=False)

    # Дополнительно возвращаем CSV в ответе (как вложение)
    # Можно сформировать CSV "на лету" через StringIO
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    response_content = csv_buffer.getvalue()

    response = Response(content=response_content, media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response

@app.get("/run-ml")
def run_ml(
    car_brand: str,
    car_model: str,
    date_max: str,
    count_pages: int
):
    """
    Запускает линейную регрессию по ранее скачанным данным и сохраняет итоговый график в PNG.
    Пример запроса:
    http://127.0.0.1:8000/run-ml?car_brand=toyota&car_model=camry&date_max=2015&count_pages=5
    """
    # Ищем соответствующий CSV
    csv_dir = get_parsed_data_path()
    csv_filename = f"{car_brand}_{car_model}_{date_max}_{count_pages}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    if not os.path.exists(csv_path):
        return {"error": f"CSV file not found: {csv_path}. Сперва выполните /download-csv2 для получения данных."}

    # Читаем CSV и готовим данные для линейной регрессии
    df = pd.read_csv(csv_path)

    # Приводим Price к числу, удаляя нецифровые символы (если остались)
    df['Price'] = df['Price'].replace(r'\D+', '', regex=True).astype(float)
    # Приводим Year к числу
    df['Year'] = df['Year'].astype(float)

    # Фильтрация или чистка данных при необходимости
    # (Например, если встречаются некорректные года или цены, можно отфильтровать)

    # Готовим X, y для обучения
    # Предположим, что хотим предсказать Price по Year
    X = df[['Year']]  # нужно 2D, поэтому двойные скобки
    y = df['Price']

    # Создаём и обучаем модель
    model = LinearRegression()
    model.fit(X, y)

    # Предсказываем, чтобы построить линию
    # Для красоты построения отсортируем по году
    X_sorted = np.sort(X['Year'].unique())
    X_sorted_2d = X_sorted.reshape(-1, 1)
    y_pred = model.predict(X_sorted_2d)

    # Строим график: scatter + линия
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label="Данные", alpha=0.5)
    plt.plot(X_sorted, y_pred, color='red', label="Линейная регрессия")
    plt.xlabel("Год")
    plt.ylabel("Цена")
    plt.title(f"Линейная регрессия для {car_brand} {car_model}, год <= {date_max}")
    plt.legend()

    # Сохраняем график в отдельную директорию
    ml_dir = get_ml_results_path()
    os.makedirs(ml_dir, exist_ok=True)
    plot_filename = f"{car_brand}_{car_model}_{date_max}_{count_pages}.png"
    plot_path = os.path.join(ml_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    return {
        "message": "ML модель успешно обучена и график сохранён.",
        "png_path": plot_path
    }

@app.get("/run-ml2")
def run_ml2(
    car_brand: str,
    car_model: str,
    date_max: str,
    count_pages: int
):
    """
    Новый, более точный подход — например, Gradient Boosting Regressor.
    http://127.0.0.1:8001/run-ml2?car_brand=toyota&car_model=camry&date_max=2015&count_pages=5
    """
    # Путь к CSV
    csv_dir = get_parsed_data_path()
    csv_filename = f"{car_brand}_{car_model}_{date_max}_{count_pages}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    if not os.path.exists(csv_path):
        return {"error": f"CSV file not found: {csv_path}. Сначала выполните /download-csv2."}

    df = pd.read_csv(csv_path)

    # Приводим Price и Year к числу
    df['Price'] = df['Price'].replace(r'\D+', '', regex=True).astype(float)
    df['Year'] = df['Year'].astype(float)

    # Удалим возможные выбросы или некорректные значения
    df = df[(df['Price'] > 0) & (df['Year'] > 1900)]

    # X, y
    X = df[['Year']]
    y = df['Price']

    # Разделим на train/test, чтобы оценить качество
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Модель Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Предсказания
    y_pred = model.predict(X_test)

    # Оценка качества
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Построим для графика предсказания на диапазоне годов
    # Сортируем уникальные года во всей выборке
    years_range = np.arange(int(df['Year'].min()), int(df['Year'].max()) + 1)
    years_range_2d = years_range.reshape(-1, 1)
    y_range_pred = model.predict(years_range_2d)

    # Построим график: рассеяние train + test + линия GB
    plt.figure(figsize=(8, 6))

    # Отобразим TRAIN
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label="Train")
    # Отобразим TEST
    plt.scatter(X_test, y_test, color='green', alpha=0.5, label="Test")

    # Линия предсказания
    plt.plot(years_range, y_range_pred, color='red', label="GB Predictions")

    plt.xlabel("Год")
    plt.ylabel("Цена")
    plt.title(f"Gradient Boosting: {car_brand} {car_model}, год <= {date_max}")
    plt.legend()

    # Получаем путь для сохранения
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