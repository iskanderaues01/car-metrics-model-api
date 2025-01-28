import random
import time
import requests
import pandas as pd
from typing import List, Dict
from bs4 import BeautifulSoup

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

def get_random_user_agent() -> Dict[str, str]:
    """Выбираем случайный User-Agent из списка."""
    return {"User-Agent": random.choice(USER_AGENTS)}

# def fetch_car_data(page_url: str) -> List[Dict[str, str]]:
#     """Забираем данные по объявлениям с конкретной страницы."""
#     headers = get_random_user_agent()
#     response = requests.get(page_url, headers=headers)
#
#     if response.status_code != 200:
#         raise Exception(f"Failed to fetch data from {page_url}. "
#                         f"Status code: {response.status_code}")
#
#     soup = BeautifulSoup(response.text, "html.parser")
#     car_elements = soup.select(".a-list__item")
#
#     cars = []
#     for car in car_elements:
#         title_elem = car.select_one(".a-card__title")
#         title = title_elem.text.strip() if title_elem else "N/A"
#
#         link_elem = car.select_one(".a-card__link")
#         link = f"https://kolesa.kz{link_elem['href']}" if link_elem else "N/A"
#
#         price_elem = car.select_one(".a-card__price")
#         price = price_elem.text.strip().replace("\u00a0", " ") if price_elem else "N/A"
#         price = price.replace("₸", "").strip()
#
#         description_elem = car.select_one(".a-card__description")
#         description = description_elem.text.strip() if description_elem else "N/A"
#
#         # Пропускаем объявления с "На заказ"
#         if "На заказ" in description:
#             continue
#
#         # Примерный парсинг года
#         year = description.split("г.,")[0].strip() if "г." in description else "N/A"
#
#         # Пропускаем некорректные объявления
#         if title == "N/A" or price == "N/A" or year == "N/A":
#             continue
#
#         cars.append({
#             "Title": title,
#             "Price": price,
#             "Year": year,
#             "Link": link
#         })
#
#     return cars

def parse_description_fields(description: str) -> Dict[str, str]:
    """
    Парсим из описания более детальную информацию, например:
    "2012 г., Б/у седан, 1.6 л, бензин, КПП механика, с пробегом 220 390 км, серый, металлик..."
    Извлекаем год, состояние/тип кузова, объём, топливо, трансмиссию, пробег.
    """
    # Удалим лишние пробелы
    raw = description.strip()
    # Разделяем запятыми
    parts = [p.strip() for p in raw.split(',')]
    # Результирующий словарь
    result = {
        "ConditionBody": None,
        "EngineVolume": None,
        "Fuel": None,
        "Transmission": None,
        "Mileage": None
    }
    # parts[0] обычно содержит "2012 г."
    # parts[1] -> "Б/у седан"
    # parts[2] -> "1.6 л"
    # parts[3] -> "бензин"
    # parts[4] -> "КПП механика"
    # parts[5] -> "с пробегом 220 390 км"
    # и т.д.

    # Пытаемся по индексам забрать нужную инфу (в реальной жизни желательно более гибко/надёжно)
    if len(parts) >= 2:
        result["ConditionBody"] = parts[1]  # "Б/у седан" etc.
    if len(parts) >= 3:
        result["EngineVolume"] = parts[2].replace(" л", "").strip()  # "1.6"
    if len(parts) >= 4:
        result["Fuel"] = parts[3]  # "бензин"
    if len(parts) >= 5:
        # "КПП механика" -> уберём "КПП"
        result["Transmission"] = parts[4].replace("КПП", "").strip()
    if len(parts) >= 6:
        # "с пробегом 220 390 км" -> извлечём число
        mileage_str = parts[5].replace("с пробегом", "").replace("км", "").strip()
        result["Mileage"] = mileage_str

    return result

def fetch_car_data(page_url: str) -> List[Dict[str, str]]:
    """Забираем данные по объявлениям с конкретной страницы + расширенный парсинг."""
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

        # Грубый парсинг года из описания (до "г.,"):
        # Но мы также берём часть полей из description
        year_parsed = "N/A"
        if "г." in description:
            year_parsed = description.split("г.")[0].strip()

        if title == "N/A" or price == "N/A":
            continue

        # Парсим расширенные поля
        extra_info = parse_description_fields(description)

        # Формируем финальный словарь
        cars.append({
            "Title": title,          # "Toyota Camry"
            "Price": price,          # "4 800 000"
            "Year": year_parsed,     # "2012" и т.д.
            "Link": link,
            "ConditionBody": extra_info["ConditionBody"],
            "EngineVolume": extra_info["EngineVolume"],
            "Fuel": extra_info["Fuel"],
            "Transmission": extra_info["Transmission"],
            "Mileage": extra_info["Mileage"],
            "RawDescription": description
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
