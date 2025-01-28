import json

from fastapi import APIRouter, Response
from io import StringIO
import pandas as pd
import os

from services.parsing_service import scrape_multiple_pages
from services.paths_service import get_parsed_data_path

router = APIRouter()

# Количество страниц, которые хотим спарсить по умолчанию
NUM_PAGES = 10

@router.get("/download-csv")
def download_csv(car_brand: str, car_model: str, date_max: str):
    """
    Пример запроса:
    http://127.0.0.1:8000/download-csv?car_brand=toyota&car_model=camry&date_max=2015

    Параметры:
    - car_brand (str): название бренда
    - car_model (str): модель
    - date_max (str): ограничение по году (year[to])

    Возвращает CSV-файл (attachment) с данными об автомобилях.
    """
    base_url = f"https://kolesa.kz/cars/{car_brand}/{car_model}/?year[to]={date_max}"
    car_data = scrape_multiple_pages(base_url, NUM_PAGES)

    if not car_data:
        return {"message": "No data found."}

    df = pd.DataFrame(car_data)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    response = Response(content=csv_buffer.getvalue(), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=cars.csv"
    return response


@router.get("/save_local")
def save_local(car_brand: str, car_model: str, date_max: str, count_pages: int):
    """
    Пример запроса:
    http://127.0.0.1:8000/save_local?car_brand=toyota&car_model=camry&date_max=2015&count_pages=5
    """
    base_url = f"https://kolesa.kz/cars/{car_brand}/{car_model}/?year[to]={date_max}"
    car_data = scrape_multiple_pages(base_url, count_pages)
    if not car_data:
        return {"message": "No data found."}

    dir_path = get_parsed_data_path()
    filename = f"{car_brand}_{car_model}_{date_max}_{count_pages}.csv"
    file_path = os.path.join(dir_path, filename)

    df = pd.DataFrame(car_data)
    df.to_csv(file_path, index=False)

    # Возвращаем CSV "на лету"
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    response_content = csv_buffer.getvalue()

    response = Response(content=response_content, media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response

@router.get("/parse-json")
def parse_json(car_brand: str, car_model: str, date_max: str, count_pages: int):
    """
    Пример: /parse-json?car_brand=toyota&car_model=camry&date_max=2015&count_pages=1
    Парсит указанные страницы, возвращает список объявлений в JSON (без сохранения).
    При этом каждое объявление содержит расширенные поля:
    Title, Price, Year, Link, ConditionBody, EngineVolume, Fuel, Transmission, Mileage, RawDescription
    """
    base_url = f"https://kolesa.kz/cars/{car_brand}/{car_model}/?year[to]={date_max}"
    car_data = scrape_multiple_pages(base_url, count_pages)

    if not car_data:
        return {"message": "No data found."}

        # 1. Сохраняем JSON локально
    dir_path = get_parsed_data_path()
    filename = f"{car_brand}_{car_model}_{date_max}_{count_pages}.json"
    file_path = os.path.join(dir_path, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False, чтобы сохранять юникодные символы (кириллица, т.д.) в читабельном виде
        json.dump(car_data, f, ensure_ascii=False, indent=2)

    # Просто возвращаем список словарей => FastAPI сам сериализует в JSON
    return car_data