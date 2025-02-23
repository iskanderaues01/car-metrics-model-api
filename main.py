from fastapi import FastAPI
from numpy.random import logistic

from routes.parser_routes import router as parser_router
from routes.ml_routes import router as ml_router


def create_app() -> FastAPI:
    app = FastAPI()
    # Подключаем роуты
    app.include_router(parser_router, prefix="", tags=["Parser"])
    app.include_router(ml_router, prefix="", tags=["ML"])

    return app

app = create_app()

# Если хотите запускать как "python main.py", можете добавить:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
