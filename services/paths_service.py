import os

def get_ml_results_path() -> str:
    """
    Возвращает путь к директории для сохранения результатов ML.
    При необходимости здесь можно сменить стандартный путь.
    """
    ml_dir = "/home/chaplin/Desktop/ml_results"
    os.makedirs(ml_dir, exist_ok=True)
    return ml_dir

def get_parsed_data_path() -> str:
    """
    Возвращает путь к директории для сохранения результатов парсинга (CSV).
    """
    parsed_dir = "/home/chaplin/Desktop/parsed_data"
    os.makedirs(parsed_dir, exist_ok=True)
    return parsed_dir
