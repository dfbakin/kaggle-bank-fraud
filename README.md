# Добавление датасета

Необходимо создать директорию `data` и поместить туда `train.csv`, `test.csv` и `sample_submission.csv` из набора данных [Kaggle Bank Fraud Detection](https://www.kaggle.com/competitions/teta-ml-1-2025/).


# Обучение модели

## Install poetry

Например, через `pipx`
```bash
python3 -m pip install pipx
pipx ensurepath
pipx install poetry
```

## Установка зависимостей

```bash 
poetry install --no-root
```

## Запуск обучения модели

```bash
poetry run python src/train.py
```

# Создание посылки на Kaggle (ДЗ)

## Запуск контейнера

```bash
docker compose up
```

В папке `data` появится файл `submission.csv`, который можно отправить на Kaggle.

# Использоние argparse

На данный момент все пути явно прописаны в коде. Стоит использовать `argparse`, чтобы запускать скрипты с командными утилитами, что позволится кастомизировать пути к моделям и данным.
